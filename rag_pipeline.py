from __future__ import annotations

import argparse
import json
import math
import os
import re
import textwrap
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TOP_LEVEL_RULE_RE = re.compile(r"^(?P<rule>\d{3})\. (?P<title>.+)$")
SUBRULE_RE = re.compile(r"^(?P<rule_id>\d{3}\.\d+[a-z]?\.?)\s")


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def clean_lines(text: str) -> list[str]:
    return [line.rstrip() for line in text.splitlines()]


def trim_blank_lines(lines: list[str]) -> list[str]:
    start = 0
    end = len(lines)

    while start < end and not lines[start].strip():
        start += 1
    while end > start and not lines[end - 1].strip():
        end -= 1

    return lines[start:end]


def join_block(lines: list[str]) -> str:
    return "\n".join(trim_blank_lines(lines))


def find_body_start(lines: list[str]) -> int:
    for index, line in enumerate(lines):
        if SUBRULE_RE.match(line):
            for heading_index in range(index, -1, -1):
                if TOP_LEVEL_RULE_RE.match(lines[heading_index]):
                    return heading_index
    raise ValueError("Could not locate the start of the numbered rules body.")


def find_marker(lines: list[str], marker: str, start_index: int = 0) -> int | None:
    for index in range(start_index, len(lines)):
        if lines[index].strip() == marker:
            return index
    return None


def split_rule_blocks(section_lines: list[str]) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []

    for line in section_lines:
        if SUBRULE_RE.match(line) and current:
            block_text = join_block(current)
            if block_text:
                blocks.append(block_text)
            current = [line]
            continue
        current.append(line)

    if current:
        block_text = join_block(current)
        if block_text:
            blocks.append(block_text)

    return blocks


def split_long_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    for delimiter in (r"\n\s*\n", r"(?<=[.!?])\s+"):
        units = [part.strip() for part in re.split(delimiter, text) if part.strip()]
        if len(units) <= 1:
            continue

        parts: list[str] = []
        current: list[str] = []

        for unit in units:
            if len(unit) > max_chars:
                if current:
                    parts.append("\n\n".join(current))
                    current = []
                parts.extend(split_long_text(unit, max_chars))
                continue

            candidate = "\n\n".join(current + [unit]) if current else unit
            if len(candidate) > max_chars and current:
                parts.append("\n\n".join(current))
                current = [unit]
            else:
                current.append(unit)

        if current:
            parts.append("\n\n".join(current))

        return parts

    return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]


def coalesce_blocks(
    title: str,
    blocks: list[str],
    chunk_id_prefix: str,
    kind: str,
    target_chars: int,
    max_chars: int,
    start_line: int,
) -> list[dict]:
    chunks: list[dict] = []
    current_blocks: list[str] = []
    chunk_number = 1

    def flush() -> None:
        nonlocal chunk_number, current_blocks
        if not current_blocks:
            return

        chunk_text = f"{title}\n\n" + "\n\n".join(current_blocks)
        chunks.append(
            {
                "id": f"{chunk_id_prefix}-{chunk_number:03d}",
                "kind": kind,
                "title": title,
                "text": chunk_text,
                "source_line": start_line,
            }
        )
        chunk_number += 1
        current_blocks = []

    for block in blocks:
        if len(block) > max_chars:
            flush()
            for part in split_long_text(block, max_chars):
                chunks.append(
                    {
                        "id": f"{chunk_id_prefix}-{chunk_number:03d}",
                        "kind": kind,
                        "title": title,
                        "text": f"{title}\n\n{part}",
                        "source_line": start_line,
                    }
                )
                chunk_number += 1
            continue

        candidate = "\n\n".join(current_blocks + [block]) if current_blocks else block
        if current_blocks and len(candidate) > target_chars:
            flush()

        current_blocks.append(block)

    flush()
    return chunks


def build_rule_chunks(lines: list[str], start_index: int, end_index: int, target_chars: int, max_chars: int) -> list[dict]:
    heading_indexes = [
        index for index in range(start_index, end_index) if TOP_LEVEL_RULE_RE.match(lines[index])
    ]
    heading_indexes.append(end_index)

    chunks: list[dict] = []
    for heading_position, next_heading_position in zip(heading_indexes, heading_indexes[1:]):
        title = lines[heading_position].strip()
        section_lines = trim_blank_lines(lines[heading_position + 1 : next_heading_position])
        if not section_lines:
            continue

        blocks = split_rule_blocks(section_lines)
        if not blocks:
            continue

        rule_code = title.split(".", maxsplit=1)[0]
        chunks.extend(
            coalesce_blocks(
                title=title,
                blocks=blocks,
                chunk_id_prefix=f"rule-{rule_code}",
                kind="rule",
                target_chars=target_chars,
                max_chars=max_chars,
                start_line=heading_position + 1,
            )
        )

    return chunks


def build_glossary_chunks(lines: list[str], glossary_index: int, end_index: int) -> list[dict]:
    entries: list[dict] = []
    current: list[str] = []
    entry_number = 1
    entry_start_line = glossary_index + 2

    def flush() -> None:
        nonlocal current, entry_number, entry_start_line
        block = trim_blank_lines(current)
        current = []
        if len(block) < 2:
            return

        term = block[0].strip()
        definition = "\n".join(block[1:]).strip()
        if not definition:
            return

        entries.append(
            {
                "id": f"glossary-{entry_number:04d}",
                "kind": "glossary",
                "title": f"Glossary: {term}",
                "text": f"{term}\n\n{definition}",
                "source_line": entry_start_line,
            }
        )
        entry_number += 1

    for index in range(glossary_index + 1, end_index):
        line = lines[index]
        if not line.strip():
            flush()
            entry_start_line = index + 2
            continue
        if not current:
            entry_start_line = index + 1
        current.append(line)

    flush()
    return entries


def chunk_document(source_path: Path, target_chars: int = 2400, max_chars: int = 3200) -> list[dict]:
    text = read_text(source_path)
    lines = clean_lines(text)

    body_start = find_body_start(lines)
    glossary_start = find_marker(lines, "Glossary", start_index=body_start)
    credits_start = find_marker(lines, "Credits", start_index=glossary_start or body_start)

    rules_end = glossary_start if glossary_start is not None else len(lines)
    glossary_end = credits_start if credits_start is not None else len(lines)

    chunks = build_rule_chunks(lines, body_start, rules_end, target_chars=target_chars, max_chars=max_chars)

    if glossary_start is not None:
        chunks.extend(build_glossary_chunks(lines, glossary_start, glossary_end))

    if not chunks:
        raise ValueError("No chunks were produced from the document.")

    return chunks


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def vector_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def cosine_similarity(query_vector: list[float], query_norm: float, chunk_vector: list[float], chunk_norm: float) -> float:
    if not query_norm or not chunk_norm:
        return 0.0
    dot_product = sum(left * right for left, right in zip(query_vector, chunk_vector))
    return dot_product / (query_norm * chunk_norm)


def get_client() -> OpenAI:
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return OpenAI()


def embed_texts(client: OpenAI, texts: list[str], model: str, batch_size: int, pause_seconds: float = 0.0) -> list[list[float]]:
    embeddings: list[list[float]] = []

    for batch in batched(texts, batch_size):
        response = client.embeddings.create(model=model, input=batch)
        embeddings.extend(item.embedding for item in response.data)
        if pause_seconds:
            time.sleep(pause_seconds)

    return embeddings


def build_index(source_path: Path, index_path: Path, model: str, batch_size: int, target_chars: int, max_chars: int) -> dict:
    chunks = chunk_document(source_path, target_chars=target_chars, max_chars=max_chars)
    client = get_client()
    embeddings = embed_texts(client, [chunk["text"] for chunk in chunks], model=model, batch_size=batch_size)

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        chunk["embedding_norm"] = vector_norm(embedding)

    index = {
        "source_path": str(source_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": model,
        "chunk_count": len(chunks),
        "chunks": chunks,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index), encoding="utf-8")
    return index


def load_index(index_path: Path) -> dict:
    return json.loads(index_path.read_text(encoding="utf-8"))


def search_index(index: dict, query: str, top_k: int) -> list[tuple[float, dict]]:
    client = get_client()
    query_vector = embed_texts(client, [query], model=index["embedding_model"], batch_size=1)[0]
    query_norm = vector_norm(query_vector)

    scored_chunks = []
    for chunk in index["chunks"]:
        score = cosine_similarity(
            query_vector=query_vector,
            query_norm=query_norm,
            chunk_vector=chunk["embedding"],
            chunk_norm=chunk["embedding_norm"],
        )
        scored_chunks.append((score, chunk))

    scored_chunks.sort(key=lambda item: item[0], reverse=True)
    return scored_chunks[:top_k]


def format_retrieval_context(results: list[tuple[float, dict]]) -> str:
    sections = []
    for rank, (score, chunk) in enumerate(results, start=1):
        sections.append(
            "\n".join(
                [
                    f"[Document {rank}]",
                    f"score: {score:.4f}",
                    f"id: {chunk['id']}",
                    f"title: {chunk['title']}",
                    f"source_line: {chunk['source_line']}",
                    "content:",
                    chunk["text"],
                ]
            )
        )
    return "\n\n".join(sections)


def answer_query(
    index: dict,
    query: str,
    top_k: int,
    chat_model: str = "gpt-4o",
    conversation_history: list[dict[str, str]] | None = None,
) -> tuple[str, list[tuple[float, dict]]]:
    results = search_index(index=index, query=query, top_k=top_k)
    context = format_retrieval_context(results)

    system_prompt = (
        "You are an MTG Comprehensive Rules assistant. "
        "Answer the user's question using the retrieved rules context first. "
        "Be precise, avoid inventing rules, and say when the retrieved context is insufficient. "
        "When helpful, cite the retrieved rule titles or glossary entries in plain language."
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    if conversation_history:
        messages.extend(conversation_history)

    messages.append(
        {
            "role": "user",
            "content": (
                "Retrieved MTG rules context:\n"
                f"{context}\n\n"
                "User question:\n"
                f"{query}"
            ),
        }
    )

    client = get_client()
    response = client.chat.completions.create(
        model=chat_model,
        messages=messages,
    )
    answer = response.choices[0].message.content or ""
    return answer, results


def print_chunk_report(chunks: list[dict], preview_count: int) -> None:
    counts = Counter(chunk["kind"] for chunk in chunks)
    print(f"Prepared {len(chunks)} chunks.")
    print(f"Rule chunks: {counts.get('rule', 0)}")
    print(f"Glossary chunks: {counts.get('glossary', 0)}")
    print()

    for chunk in chunks[:preview_count]:
        print(f"{chunk['id']} | {chunk['title']} | {len(chunk['text'])} chars")


def build_command(args: argparse.Namespace) -> None:
    source_path = Path(args.source)
    index_path = Path(args.index)
    chunks = chunk_document(source_path, target_chars=args.target_chars, max_chars=args.max_chars)

    if args.dry_run:
        print_chunk_report(chunks, preview_count=args.preview)
        return

    index = build_index(
        source_path=source_path,
        index_path=index_path,
        model=args.model,
        batch_size=args.batch_size,
        target_chars=args.target_chars,
        max_chars=args.max_chars,
    )
    print(f"Wrote {index['chunk_count']} embedded chunks to {index_path}.")


def search_command(args: argparse.Namespace) -> None:
    index = load_index(Path(args.index))
    results = search_index(index, query=args.query, top_k=args.top_k)

    for rank, (score, chunk) in enumerate(results, start=1):
        excerpt = textwrap.shorten(chunk["text"].replace("\n", " "), width=args.preview_chars, placeholder="...")
        print(f"{rank}. score={score:.4f} | {chunk['title']} | {chunk['id']}")
        print(f"   {excerpt}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chunk, embed, and search the Magic comprehensive rules.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Chunk the document and build an embeddings index.")
    build_parser.add_argument("--source", default="MagicCompRules.txt", help="Path to the source rules document.")
    build_parser.add_argument("--index", default="data/mtg_rules_index.json", help="Where to write the embeddings index.")
    build_parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model.")
    build_parser.add_argument("--batch-size", type=int, default=32, help="How many chunks to embed per API request.")
    build_parser.add_argument("--target-chars", type=int, default=2400, help="Preferred chunk size before starting a new chunk.")
    build_parser.add_argument("--max-chars", type=int, default=3200, help="Maximum size for a single chunk before splitting it further.")
    build_parser.add_argument("--dry-run", action="store_true", help="Preview chunking without calling the OpenAI API.")
    build_parser.add_argument("--preview", type=int, default=10, help="How many chunks to preview during a dry run.")
    build_parser.set_defaults(func=build_command)

    search_parser = subparsers.add_parser("search", help="Search an embeddings index for the most relevant chunks.")
    search_parser.add_argument("--index", default="data/mtg_rules_index.json", help="Path to the saved embeddings index.")
    search_parser.add_argument("--query", required=True, help="User query to embed and search.")
    search_parser.add_argument("--top-k", type=int, default=5, help="How many results to return.")
    search_parser.add_argument("--preview-chars", type=int, default=280, help="How much of each matching chunk to print.")
    search_parser.set_defaults(func=search_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
