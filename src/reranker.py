from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from . import config

_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(config.RERANK_MODEL)
    return _model


@dataclass
class RerankedHit:
    score: float
    payload: dict


def rerank(query: str, hits: list, top_k: int) -> list[RerankedHit]:
    if not hits:
        return []
    model = get_reranker()
    pairs = [(query, h.payload["text"]) for h in hits]
    scores = model.predict(pairs)
    reranked = sorted(
        (RerankedHit(score=float(s), payload=h.payload) for s, h in zip(scores, hits)),
        key=lambda r: r.score,
        reverse=True,
    )
    return reranked[:top_k]
