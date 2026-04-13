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

Run the Flask app locally in development:

```bash
flask --app app run --debug
```

Run the Flask app with Gunicorn:

```bash
gunicorn -c gunicorn.conf.py wsgi:app
```

The `/api/ai-answer` endpoint is rate limited per client IP to `10/minute` and `100/day`
using Redis-backed `Flask-Limiter`. Configure Redis with `RATELIMIT_STORAGE_URI` or
`REDIS_URL`. If you are running behind a reverse proxy or load balancer, set
`TRUSTED_PROXY_HOPS` to the number of trusted proxy hops so rate limiting uses the
real client IP instead of the proxy IP. This repo's Gunicorn config sets
`TRUSTED_PROXY_HOPS=1`, which matches a `cloudflared -> gunicorn` deployment.

Run the Streamlit app:

```bash
streamlit run streamlit/app.py
```

The Flask app loads the saved index, retrieves the top-k most relevant chunks for each question, and sends those chunks to `gpt-4o` to answer in a chat interface.

Set `OPENAI_API_KEY` before running the build, search, or chat commands.
