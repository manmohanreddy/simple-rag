# Simple RAG

RAG pipeline: Qdrant for vectors, sentence-transformers for local embeddings,
Claude for generation, files ingested from `./data`. Runs either as a local CLI
(`python -m src.query`) or as an HTTP service (`src/api.py`) behind Docker Compose.

## Setup - local CLI (dev/experimentation)

```bash
docker compose up -d qdrant   # starts Qdrant on localhost:6333
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
copy .env.example .env         # then fill in ANTHROPIC_API_KEY and API_KEY
```

```bash
python -m src.ingest                     # chunk + embed + index everything in data/
python -m src.query "What is this project about?"
```

Add your own `.txt`/`.md`/`.pdf` files to `data/` and re-run ingest to update the index.

## Setup - HTTP service (team use)

```bash
copy .env.example .env
# fill in ANTHROPIC_API_KEY, and generate a shared team key:
python -c "import secrets; print(secrets.token_urlsafe(32))"   # -> API_KEY in .env

docker compose up -d --build   # builds the app image, starts app + Qdrant together
docker compose exec app python -m src.ingest   # ingest data/ into the running stack
```

```bash
curl http://localhost:8000/health
curl http://localhost:8000/query \
  -X POST -H "Content-Type: application/json" -H "X-API-Key: <your API_KEY>" \
  -d '{"question": "What is this project about?"}'
```

`POST /query` returns `{answer, sources, timings, warning}` - `warning` is set
if Claude's response hit `max_tokens` (truncated); a `refusal` stop reason
returns HTTP 502 instead of a truncated/empty answer. `GET /health` reports
Qdrant reachability without requiring the API key.

**Cold-start note:** the first request after the app container starts pays a
one-time cost loading the embedding + reranker models into memory (tens of
seconds in a fresh container - slower than a warm local venv). Every request
after that is steady-state (sub-2s retrieval). This is expected, not a bug -
don't read the first request's latency as representative.

## Evaluation

`eval/cases.jsonl` holds the test questions. Running the eval calls the real
`src.query.answer_with_meta()` entry point for each case, grades the answer with a
`claude-sonnet-5` judge on 4 rubric criteria (`correct`, `grounded`, `cites_source`,
`complete`), and records cost/retrieval-score plus a per-stage latency breakdown:
embed, dense search, BM25 search, fusion, rerank, and the LLM call. The first case
in a run includes one-time model cold-load cost (MiniLM + cross-encoder loading into
memory); later cases show steady-state latency - don't read case 1's numbers as
representative.

```bash
python -m eval.runner
node "<claude-api skill base dir>/shared/evals/report/build-report-lite.mjs" .claude/hillclimb/rag-qa/
```

Open `.claude/hillclimb/rag-qa/report.html` for per-case scores and links to full
transcripts (`.claude/hillclimb/rag-qa/baseline/traces/`). Failed calls land in
`errors.jsonl`, not `results.jsonl`.

15 cases (5 hand-written, 10 pulled from the source doc's own Q&A content,
paraphrased, spanning topics beyond the original 5) - stable enough for a real
signal. Grow further in `eval/cases.jsonl` as needed.

## How it works

1. `ingest.py` walks `data/`, splits each file into overlapping token-based chunks
   sized to the embedding model's max sequence length (`chunking.py`), embeds them
   locally with `sentence-transformers` (`embeddings.py`), and upserts vectors + text
   into a Qdrant collection (`store.py`).
2. `query.py` runs hybrid retrieval: dense cosine search (`store.py`) and BM25
   keyword search (`bm25_search.py`) each fetch `RETRIEVE_K` candidates, merged by
   rank position via Reciprocal Rank Fusion (`fusion.py`) since their scores aren't
   on comparable scales. The merged set is reranked with a cross-encoder for
   precision (`reranker.py`) down to `TOP_K`, then sent to Claude as context, which
   answers grounded in that context and cites sources.

## Production readiness

Phase 1 done: HTTP API, API key auth, Dockerized app, request timeout +
`stop_reason` handling. Remaining phases (tests/CI, incremental ingestion,
structured logging/observability, ops hardening) are tracked as a plan - ask
about "production readiness" to pick it back up.

## Next steps (advanced RAG / fine-tuning track)

- Try a hosted embedding model (Voyage AI) for quality comparison
- Persist the BM25 index instead of rebuilding it from a full Qdrant scroll each run
