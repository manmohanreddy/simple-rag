from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from . import config


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


def search(client: QdrantClient, query_vector: list[float], top_k: int):
    return client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
    ).points
