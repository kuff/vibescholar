"""Unit tests for FTS5 query parser in SearchService."""
from __future__ import annotations

from vibescholar.search import SearchService


class TestFTS5QueryParser:
    def test_plain_query(self):
        assert SearchService._fts5_query("neural network") == "neural network"

    def test_and_operator(self):
        assert SearchService._fts5_query("neural AND network") == "neural AND network"

    def test_or_operator(self):
        assert SearchService._fts5_query("neural OR network") == "neural OR network"

    def test_not_between_terms(self):
        result = SearchService._fts5_query("neural NOT adversarial")
        assert result == "neural NOT adversarial"

    def test_leading_not_stripped(self):
        # FTS5 NOT requires a left operand; leading NOT is invalid
        result = SearchService._fts5_query("NOT adversarial")
        assert result == "adversarial"

    def test_quoted_phrase(self):
        result = SearchService._fts5_query('"optimal transport" learning')
        assert result == '"optimal transport" learning'

    def test_quoted_phrase_only(self):
        result = SearchService._fts5_query('"optimal transport"')
        assert result == '"optimal transport"'

    def test_prefix_wildcard(self):
        result = SearchService._fts5_query("optim*")
        assert result == "optim*"

    def test_mixed_operators(self):
        result = SearchService._fts5_query('"deep learning" OR reinforcement')
        assert '"deep learning"' in result
        assert "OR" in result
        assert "reinforcement" in result

    def test_consecutive_operators_cleaned(self):
        result = SearchService._fts5_query("AND OR neural")
        assert result == "neural"

    def test_trailing_operator_cleaned(self):
        result = SearchService._fts5_query("neural AND")
        assert result == "neural"

    def test_empty_query(self):
        assert SearchService._fts5_query("") == ""

    def test_whitespace_only(self):
        assert SearchService._fts5_query("   ") == ""

    def test_special_chars_stripped(self):
        result = SearchService._fts5_query("hello@world!!")
        assert "hello" in result
        assert "world" in result
        assert "@" not in result
        assert "!" not in result

    def test_case_insensitive_operators(self):
        result = SearchService._fts5_query("neural and network")
        assert "AND" in result

    def test_or_case_insensitive(self):
        result = SearchService._fts5_query("neural or network")
        assert "OR" in result

    def test_single_char_tokens_dropped(self):
        # _tokens() drops single-char tokens
        result = SearchService._fts5_query("a neural b network c")
        assert result == "neural network"

    def test_unclosed_quote_handled(self):
        # Unclosed quote should not crash
        result = SearchService._fts5_query('"unclosed phrase')
        assert "unclosed" in result or '"unclosed phrase' in result
