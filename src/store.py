from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from . import config


@dataclass
class ScoredChunk:
    """Common shape for a retrieved chunk regardless of which stage produced it
    (dense search, BM25, RRF fusion, cross-encoder rerank) - only `score` and
    `payload` are needed downstream (build_context, eval retrieval_score)."""
    id: int
    score: float
    payload: dict


def get_client() -> QdrantClient:
    return QdrantClient(url=config.QDRANT_URL)


def ensure_collection(client: QdrantClient, dim: int) -> None:
    if client.collection_exists(config.COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )


def upsert_chunks(client: QdrantClient, ids: list[int], vectors: list[list[float]], payloads: list[dict]) -> None:
    points = [
        PointStruct(id=i, vector=v, payload=p)
        for i, v, p in zip(ids, vectors, payloads)
    ]
    client.upsert(collection_name=config.COLLECTION_NAME, points=points)


def search(client: QdrantClient, query_vector: list[float], top_k: int) -> list[ScoredChunk]:
    points = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points
    return [ScoredChunk(id=p.id, score=p.score, payload=p.payload) for p in points]


def scroll_all_chunks(client: QdrantClient) -> list[ScoredChunk]:
    """Pulls every chunk's text + payload for building the BM25 corpus. Fine at
    this project's scale (hundreds of chunks); a bigger corpus would persist the
    BM25 index instead of rebuilding it from a full scroll each run."""
    points, _ = client.scroll(
        collection_name=config.COLLECTION_NAME,
        limit=100_000,
        with_payload=True,
        with_vectors=False,
    )
    return [ScoredChunk(id=p.id, score=0.0, payload=p.payload) for p in points]
