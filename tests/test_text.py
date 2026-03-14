"""Unit tests for vibescholar.text utilities."""
from __future__ import annotations

import pytest

from vibescholar.text import clean_text, chunk_text


class TestCleanText:
    def test_collapses_whitespace(self):
        assert clean_text("hello   world") == "hello world"
        assert clean_text("hello\t\tworld") == "hello world"
        assert clean_text("hello\n\nworld") == "hello world"
        assert clean_text("hello \t\n world") == "hello world"

    def test_strips(self):
        assert clean_text("  hello  ") == "hello"
        assert clean_text("\nhello\n") == "hello"

    def test_empty(self):
        assert clean_text("") == ""
        assert clean_text("   ") == ""

    def test_already_clean(self):
        assert clean_text("hello world") == "hello world"


class TestChunkText:
    def test_basic(self):
        text = "a" * 100
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        assert len(chunks) >= 3
        assert all(len(c) <= 30 for c in chunks)
        # First chunk starts at beginning
        assert chunks[0] == "a" * 30

    def test_overlap_content(self):
        text = "abcdefghij" * 10  # 100 chars
        chunks = chunk_text(text, chunk_size=30, overlap=10)
        # Each chunk's last 10 chars should match next chunk's first 10
        for i in range(len(chunks) - 1):
            assert chunks[i][-10:] == chunks[i + 1][:10]

    def test_short_text(self):
        chunks = chunk_text("hello", chunk_size=100, overlap=10)
        assert chunks == ["hello"]

    def test_empty(self):
        assert chunk_text("", chunk_size=100, overlap=10) == []

    def test_whitespace_only(self):
        assert chunk_text("   ", chunk_size=100, overlap=10) == []

    def test_overlap_validation(self):
        with pytest.raises(ValueError, match="chunk_size must be larger"):
            chunk_text("hello world", chunk_size=10, overlap=10)
        with pytest.raises(ValueError, match="chunk_size must be larger"):
            chunk_text("hello world", chunk_size=5, overlap=10)

    def test_cleans_before_chunking(self):
        text = "hello   world\n\nfoo   bar"
        chunks = chunk_text(text, chunk_size=100, overlap=10)
        assert chunks == ["hello world foo bar"]
