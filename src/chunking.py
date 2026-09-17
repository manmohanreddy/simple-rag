def chunk_text(text: str, tokenizer, chunk_size: int, overlap: int) -> list[str]:
    """Splits text into chunks by token count (using the embedding model's own
    tokenizer), not character count - keeps every chunk under the model's
    max_seq_length and makes chunk size consistent regardless of how dense the
    text's punctuation/whitespace is."""
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    n = len(token_ids)
    while start < n:
        end = start + chunk_size
        window = token_ids[start:end]
        chunks.append(tokenizer.decode(window, skip_special_tokens=True))
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]
