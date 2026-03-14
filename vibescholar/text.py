from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Collapse all whitespace (tabs, newlines, runs of spaces) to single spaces and strip."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks of up to *chunk_size* characters.

    The text is cleaned first via ``clean_text``.  Chunks advance by
    ``chunk_size - overlap`` characters, so consecutive chunks share
    *overlap* characters of context.
    """
    text = clean_text(text)
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be larger than overlap")

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        piece = text[start : start + chunk_size]
        if piece:
            chunks.append(piece)
        if start + chunk_size >= len(text):
            break
    return chunks
