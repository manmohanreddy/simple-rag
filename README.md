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

## Next steps (advanced RAG / fine-tuning track)

- Try a hosted embedding model (Voyage AI) for quality comparison
- Persist the BM25 index instead of rebuilding it from a full Qdrant scroll each run
