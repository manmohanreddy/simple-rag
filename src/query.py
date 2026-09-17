import sys
import time

import anthropic

from . import bm25_search
from . import config
from .embeddings import embed
from .fusion import reciprocal_rank_fusion
from .reranker import rerank
from .store import get_client, search

SYSTEM_PROMPT = """You are a RAG assistant. Answer the user's question using ONLY the
provided context chunks. If the context does not contain the answer, say so plainly
instead of guessing. Cite sources by their [source] tag after each claim."""


def build_context(chunks) -> str:
    parts = []
    for point in chunks:
        payload = point.payload
        parts.append(f"[{payload['source']}#{payload['chunk_index']}]\n{payload['text']}")
    return "\n\n---\n\n".join(parts)


def _elapsed(start: float) -> float:
    return round(time.monotonic() - start, 4)


def answer_with_meta(question: str) -> dict:
    """Runs the real RAG flow and returns retrieval + generation metadata alongside
    the answer text, so callers (CLI, eval runner) can inspect what happened without
    reimplementing the request. Also times each pipeline stage separately, since
    "retrieval" and "generation" have very different latency profiles and bottlenecks
    (local compute vs. network round-trip to Claude)."""
    timings = {}
    client = get_client()

    t0 = time.monotonic()
    query_vector = embed([question])[0]
    timings["embed_s"] = _elapsed(t0)

    t0 = time.monotonic()
    dense_candidates = search(client, query_vector, config.RETRIEVE_K)
    timings["dense_search_s"] = _elapsed(t0)

    t0 = time.monotonic()
    # First call in a process also builds the BM25 index (full Qdrant scroll +
    # tokenize corpus) - that one-time cost is folded into this number.
    keyword_candidates = bm25_search.search(client, question, config.RETRIEVE_K)
    timings["bm25_search_s"] = _elapsed(t0)

    t0 = time.monotonic()
    fused = reciprocal_rank_fusion(dense_candidates, keyword_candidates, top_k=config.RETRIEVE_K)
    timings["fusion_s"] = _elapsed(t0)

    t0 = time.monotonic()
    hits = rerank(question, fused, config.TOP_K)
    timings["rerank_s"] = _elapsed(t0)

    timings["retrieval_total_s"] = round(sum(timings.values()), 4)

    if not hits:
        return {
            "text": "No documents ingested yet. Run `python -m src.ingest` first.",
            "hits": [],
            "response": None,
            "timings": timings,
        }

    context = build_context(hits)
    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

    t0 = time.monotonic()
    llm = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = llm.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    timings["llm_s"] = _elapsed(t0)
    timings["total_s"] = round(timings["retrieval_total_s"] + timings["llm_s"], 4)

    text = next((b.text for b in response.content if b.type == "text"), "(no text response)")

    return {
        "text": text,
        "hits": hits,
        "context": context,
        "user_message": user_message,
        "response": response,
        "timings": timings,
    }


def answer(question: str) -> str:
    return answer_with_meta(question)["text"]


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.query "your question"')
        return
    question = " ".join(sys.argv[1:])
    result = answer_with_meta(question)
    print(result["text"])
    t = result["timings"]
    print(
        f"\n[embed {t.get('embed_s', 0)}s | dense {t.get('dense_search_s', 0)}s | "
        f"bm25 {t.get('bm25_search_s', 0)}s | fusion {t.get('fusion_s', 0)}s | "
        f"rerank {t.get('rerank_s', 0)}s | retrieval total {t.get('retrieval_total_s', 0)}s | "
        f"llm {t.get('llm_s', '-')}s | total {t.get('total_s', '-')}s]"
    )


if __name__ == "__main__":
    main()
