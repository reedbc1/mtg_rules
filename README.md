# mtg_rag
Retrieval augmented generation for MTG comprehensive rules.

## Static rules viewer

`index.html` provides a readable browser version of `MagicCompRules.txt` with:

- a linked table of contents
- anchored chapter and rule navigation
- client-side keyword search across rules text and glossary entries

Open the folder through a local web server so the browser can fetch `MagicCompRules.txt` correctly.

## RAG pipeline

`rag_pipeline.py` chunks `MagicCompRules.txt`, creates OpenAI embeddings for each chunk, saves a local index, and searches that index by semantic similarity.

Build the index:

```bash
python rag_pipeline.py build --source MagicCompRules.txt --index data/mtg_rules_index.json
```

Preview the chunking without using the API:

```bash
python rag_pipeline.py build --dry-run
```

Search the index:

```bash
python rag_pipeline.py search --index data/mtg_rules_index.json --query "How does first strike work in combat?"
```

Run the Streamlit chat app:

```bash
streamlit run app.py
```

The app loads the saved index, retrieves the top-k most relevant chunks for each question, and sends those chunks to `gpt-4o` to answer in a chat interface.

Set `OPENAI_API_KEY` before running the build, search, or chat commands.
