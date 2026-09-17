import itertools
from pathlib import Path

import pymupdf

from . import config
from .chunking import chunk_text
from .embeddings import embed, embedding_dim, get_tokenizer
from .store import ensure_collection, get_client, upsert_chunks

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def read_pdf(path: Path) -> str:
    with pymupdf.open(str(path)) as doc:
        return "\n".join(page.get_text() for page in doc)


def load_files(data_dir: str) -> list[tuple[str, str]]:
    root = Path(data_dir)
    docs = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue
        text = read_pdf(path) if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
        docs.append((str(path.relative_to(root)), text))
    return docs


def main() -> None:
    docs = load_files(config.DATA_DIR)
    if not docs:
        print(f"No .txt/.md files found in {config.DATA_DIR}")
        return

    client = get_client()
    ensure_collection(client, embedding_dim())
    tokenizer = get_tokenizer()

    id_counter = itertools.count()
    total_chunks = 0

    for source, text in docs:
        chunks = chunk_text(text, tokenizer, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        if not chunks:
            continue
        vectors = embed(chunks)
        ids = [next(id_counter) for _ in chunks]
        payloads = [
            {"text": chunk, "source": source, "chunk_index": i}
            for i, chunk in enumerate(chunks)
        ]
        upsert_chunks(client, ids, vectors, payloads)
        total_chunks += len(chunks)
        print(f"Ingested {source}: {len(chunks)} chunks")

    print(f"Done. {total_chunks} chunks across {len(docs)} files -> collection '{config.COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
