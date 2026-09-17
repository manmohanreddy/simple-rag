# Simple RAG

Local RAG pipeline: Qdrant (Docker) for vectors, sentence-transformers for local
embeddings, Claude for generation, files ingested from `./data`.

## Setup

```bash
docker compose up -d          # starts Qdrant on localhost:6333
python -m venv .venv
.venv\Scripts\activate         # Windows
pip install -r requirements.txt
copy .env.example .env         # then fill in ANTHROPIC_API_KEY
```

## Usage

```bash
python -m src.ingest                     # chunk + embed + index everything in data/
python -m src.query "What is this project about?"
```

Add your own `.txt`/`.md`/`.pdf` files to `data/` and re-run ingest to update the index.

## Evaluation

`eval/cases.jsonl` holds the test questions. Running the eval calls the real
`src.query.answer_with_meta()` entry point for each case, grades the answer with a
`claude-sonnet-5` judge on 4 rubric criteria (`correct`, `grounded`, `cites_source`,
`complete`), and records latency/cost/retrieval-score per case.

```bash
python -m eval.runner
node "<claude-api skill base dir>/shared/evals/report/build-report-lite.mjs" .claude/hillclimb/rag-qa/
```

Open `.claude/hillclimb/rag-qa/report.html` for per-case scores and links to full
transcripts (`.claude/hillclimb/rag-qa/baseline/traces/`). Failed calls land in
`errors.jsonl`, not `results.jsonl`.

Only 5 seed cases right now - fine to start, but expect a noisy headline score;
grow the set in `eval/cases.jsonl` for a more stable number.

## How it works

1. `ingest.py` walks `data/`, splits each file into overlapping character chunks
   (`chunking.py`), embeds them locally with `sentence-transformers`
   (`embeddings.py`), and upserts vectors + text into a Qdrant collection
   (`store.py`).
2. `query.py` embeds your question with the same model, retrieves the top-k
   nearest chunks from Qdrant, and sends them as context to Claude, which
   answers grounded in that context and cites sources.

## Next steps (advanced RAG / fine-tuning track)

- Swap fixed-size chunking for semantic/recursive chunking
- Add hybrid search (BM25 + vector) or reranking
- Try a hosted embedding model (Voyage AI) for quality comparison
- Add eval set to measure retrieval + answer quality (see `/claude-api build-eval`)
