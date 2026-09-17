from sentence_transformers import SentenceTransformer

from . import config

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    model = get_model()
    return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()


def embedding_dim() -> int:
    return get_model().get_embedding_dimension()


def get_tokenizer():
    return get_model().tokenizer
