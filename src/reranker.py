from sentence_transformers import CrossEncoder

from . import config
from .store import ScoredChunk

_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


def rerank(query: str, hits: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    if not hits:
        return []
    model = get_reranker()
    pairs = [(query, h.payload["text"]) for h in hits]
    scores = model.predict(pairs)
    reranked = sorted(
        (ScoredChunk(id=h.id, score=float(s), payload=h.payload) for s, h in zip(scores, hits)),
        key=lambda c: c.score,
        reverse=True,
    )
    return reranked[:top_k]
