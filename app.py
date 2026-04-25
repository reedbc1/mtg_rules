from __future__ import annotations

import os
import textwrap
from functools import lru_cache
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix

from rag_pipeline import answer_query, load_index


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = BASE_DIR / "data" / "mtg_rules_index.json"
SOUNDS_DIR = BASE_DIR / "sounds"
DEFAULT_TOP_K = 10
DEFAULT_CHAT_MODEL = "gpt-5.4-mini"
DEFAULT_RATE_LIMIT_STORAGE_URI = "memory://"

app = Flask(__name__, static_folder="static", template_folder="templates")


def get_index_path() -> Path:
    configured_path = os.environ.get("MTG_RULES_INDEX_PATH")
    return Path(configured_path) if configured_path else DEFAULT_INDEX_PATH


def get_chat_model() -> str:
    return os.environ.get("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)


def get_rate_limit_storage_uri() -> str:
    return os.environ.get(
        "RATELIMIT_STORAGE_URI",
        os.environ.get("REDIS_URL", DEFAULT_RATE_LIMIT_STORAGE_URI),
    )


def get_trusted_proxy_hops() -> int:
    configured_hops = os.environ.get("TRUSTED_PROXY_HOPS", "0")
    try:
        return max(0, int(configured_hops))
    except ValueError:
        return 0


def get_top_k() -> int:
    configured_top_k = os.environ.get("MTG_RAG_TOP_K")
    if not configured_top_k:
        return DEFAULT_TOP_K

    try:
        return max(1, int(configured_top_k))
    except ValueError:
        return DEFAULT_TOP_K


@lru_cache(maxsize=1)
def load_cached_index() -> dict:
    return load_index(get_index_path())


def get_rate_limit_key() -> str:
    return get_remote_address() or "unknown"


trusted_proxy_hops = get_trusted_proxy_hops()
if trusted_proxy_hops > 0:
    app.wsgi_app = ProxyFix(
        app.wsgi_app,
        x_for=trusted_proxy_hops,
        x_proto=trusted_proxy_hops,
        x_host=trusted_proxy_hops,
    )


limiter = Limiter(
    key_func=get_rate_limit_key,
    app=app,
    default_limits=[],
    storage_uri=get_rate_limit_storage_uri(),
)


def build_source_payload(results: list[tuple[float, dict]]) -> list[dict]:
    sources: list[dict] = []
    for rank, (score, chunk) in enumerate(results, start=1):
        excerpt = textwrap.shorten(
            chunk["text"].replace("\n", " "),
            width=420,
            placeholder="...",
        )
        sources.append(
            {
                "rank": rank,
                "score": round(score, 4),
                "id": chunk["id"],
                "title": chunk["title"],
                "source_line": chunk["source_line"],
                "excerpt": excerpt,
            }
        )
    return sources


@app.get("/")
def home():
    return render_template("index.html")


@app.get("/sounds/<path:filename>")
def serve_sound(filename: str):
    return send_from_directory(SOUNDS_DIR, filename)


@app.errorhandler(429)
def handle_rate_limit_exceeded(_exc):
    message = "Limit reached, please try again later."
    if request.path.startswith("/api/"):
        return jsonify({"error": message}), 429
    return message, 429


@app.post("/api/ai-answer")
@limiter.limit("10 per minute; 100 per day")
def api_ai_answer():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()
    if not query:
        return jsonify({"error": "Query is required."}), 400

    try:
        index = load_cached_index()
        answer, results = answer_query(
            index=index,
            query=query,
            top_k=get_top_k(),
            chat_model=get_chat_model(),
            conversation_history=None,
        )
    except (FileNotFoundError, EnvironmentError) as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"AI request failed: {exc}"}), 500

    return jsonify(
        {
            "query": query,
            "answer": answer,
            "sources": build_source_payload(results),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
