from collections import defaultdict

from .store import ScoredChunk

RRF_K = 60  # standard constant; dampens the impact of any single rank-1 hit


def reciprocal_rank_fusion(*ranked_lists: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
    """Merges result lists from different retrieval methods (dense cosine, BM25)
    by rank position rather than raw score - their scores live on incomparable
    scales (cosine is 0-1, BM25 is unbounded), but "how high did this chunk rank"
    is comparable across any method."""
    fused_scores: dict[int, float] = defaultdict(float)
    payloads: dict[int, dict] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            fused_scores[chunk.id] += 1.0 / (RRF_K + rank + 1)
            payloads[chunk.id] = chunk.payload

    ranked_ids = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
    return [
        ScoredChunk(id=chunk_id, score=score, payload=payloads[chunk_id])
        for chunk_id, score in ranked_ids[:top_k]
    ]
