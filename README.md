# mtg_rag
Retrieval augmented generation for MTG comprehensive rules.

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

Set `OPENAI_API_KEY` before running the build or search commands.
