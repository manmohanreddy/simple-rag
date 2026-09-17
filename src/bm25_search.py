import re

from rank_bm25 import BM25Okapi

from .store import ScoredChunk, scroll_all_chunks

TOKEN_RE = re.compile(r"[a-z0-9]+")

_index = None
_chunks: list[ScoredChunk] = []


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _ensure_index(client) -> None:
    global _index, _chunks
    if _index is not None:
        return
    _chunks = scroll_all_chunks(client)
    corpus = [_tokenize(c.payload["text"]) for c in _chunks]
    _index = BM25Okapi(corpus)


def search(client, query: str, top_k: int) -> list[ScoredChunk]:
    """Keyword search - scores chunks by term frequency / inverse document
    frequency, not meaning. Catches exact terms (numbers, acronyms, law names)
    that dense embeddings can blur."""
    _ensure_index(client)
    if not _chunks:
        return []
    scores = _index.get_scores(_tokenize(query))
    ranked = sorted(zip(_chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [
        ScoredChunk(id=chunk.id, score=float(score), payload=chunk.payload)
        for chunk, score in ranked[:top_k]
    ]
