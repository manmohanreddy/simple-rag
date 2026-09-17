import sys

import anthropic

from . import config
from .embeddings import embed
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


def answer_with_meta(question: str) -> dict:
    """Runs the real RAG flow and returns retrieval + generation metadata alongside
    the answer text, so callers (CLI, eval runner) can inspect what happened without
    reimplementing the request."""
    client = get_client()
    query_vector = embed([question])[0]
    hits = search(client, query_vector, config.TOP_K)

    if not hits:
        return {
            "text": "No documents ingested yet. Run `python -m src.ingest` first.",
            "hits": [],
            "response": None,
        }

    context = build_context(hits)
    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

    llm = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = llm.messages.create(
        model=config.CLAUDE_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "(no text response)")

    return {
        "text": text,
        "hits": hits,
        "context": context,
        "user_message": user_message,
        "response": response,
    }


def answer(question: str) -> str:
    return answer_with_meta(question)["text"]


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.query "your question"')
        return
    question = " ".join(sys.argv[1:])
    print(answer(question))


if __name__ == "__main__":
    main()
