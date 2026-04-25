from __future__ import annotations

import textwrap
from pathlib import Path

import streamlit as st

from rag_pipeline import answer_query, load_index


DEFAULT_INDEX_PATH = "data/mtg_rules_index.json"
DEFAULT_CHAT_MODEL = "gpt-5.4-mini"


@st.cache_data(show_spinner=False)
def cached_load_index(index_path: str) -> dict:
    return load_index(Path(index_path))


def build_history_for_model(messages: list[dict[str, str]], max_turns: int = 6) -> list[dict[str, str]]:
    conversational_messages = [
        {"role": message["role"], "content": message["content"]}
        for message in messages
        if message["role"] in {"user", "assistant"}
    ]
    return conversational_messages[-max_turns:]


def render_sources(results: list[tuple[float, dict]]) -> None:
    st.subheader("Retrieved Sources")
    for rank, (score, chunk) in enumerate(results, start=1):
        preview = textwrap.shorten(chunk["text"].replace("\n", " "), width=400, placeholder="...")
        with st.expander(f"{rank}. {chunk['title']} ({score:.4f})", expanded=(rank == 1)):
            st.caption(f"Chunk ID: {chunk['id']} | Source line: {chunk['source_line']}")
            st.write(preview)
            st.code(chunk["text"], language="text")


def main() -> None:
    st.set_page_config(page_title="MTG Rules Chat", page_icon=":game_die:", layout="wide")
    st.title("MTG Comprehensive Rules Chat")
    st.write("Ask a question about the Magic comprehensive rules. The app retrieves the closest rule chunks and uses `gpt-5.4-mini` to answer.")

    with st.sidebar:
        st.header("Settings")
        index_path = st.text_input("Index path", value=DEFAULT_INDEX_PATH)
        top_k = st.slider("Top-k retrieved chunks", min_value=1, max_value=10, value=5)
        chat_model = st.text_input("Chat model", value=DEFAULT_CHAT_MODEL)
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    try:
        index = cached_load_index(index_path)
    except FileNotFoundError:
        st.error(
            "The embeddings index was not found. Build it first with "
            "`python rag_pipeline.py build --index data/mtg_rules_index.json`."
        )
        return
    except Exception as exc:
        st.error(f"Could not load index: {exc}")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    user_query = st.chat_input("Ask about layers, combat, timing, commander rules, and more...")
    if not user_query:
        return

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving rules and drafting an answer..."):
            try:
                answer, results = answer_query(
                    index=index,
                    query=user_query,
                    top_k=top_k,
                    chat_model=chat_model,
                    conversation_history=build_history_for_model(st.session_state.messages[:-1]),
                )
            except Exception as exc:
                st.error(f"Request failed: {exc}")
                return

        st.markdown(answer)
        render_sources(results)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": results,
        }
    )


if __name__ == "__main__":
    main()
