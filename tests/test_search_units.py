"""Unit tests for search.py static/algorithmic methods.

These tests call static methods directly — no model loading, no DB, no fixtures.
"""

from __future__ import annotations

import json

import pytest

from vibescholar.search import SearchService


# ── TestReciprocalRankFusion ──────────────────────────────────────


class TestReciprocalRankFusion:
    def test_overlapping_lists(self):
        """Chunk appearing in both lists should get the highest fused score."""
        list_a = [(0.9, 5), (0.8, 10), (0.7, 15)]
        list_b = [(0.9, 5), (0.8, 20), (0.7, 25)]
        result = SearchService._reciprocal_rank_fusion([list_a, list_b])
        # result is [(fused_score, chunk_id), ...]
        ids_by_score = [cid for _, cid in result]
        assert ids_by_score[0] == 5  # chunk 5 in both -> highest

    def test_disjoint_lists(self):
        """All items from both lists should appear in output."""
        list_a = [(0.9, 1), (0.8, 2)]
        list_b = [(0.9, 3), (0.8, 4)]
        result = SearchService._reciprocal_rank_fusion([list_a, list_b])
        result_ids = {cid for _, cid in result}
        assert result_ids == {1, 2, 3, 4}

    def test_single_list(self):
        """Scores should be 1/(k+rank+1) with default k=60."""
        items = [(0.9, 10), (0.8, 20), (0.7, 30)]
        result = SearchService._reciprocal_rank_fusion([items])
        scores = {cid: score for score, cid in result}
        assert scores[10] == pytest.approx(1.0 / 61)  # rank 0
        assert scores[20] == pytest.approx(1.0 / 62)  # rank 1
        assert scores[30] == pytest.approx(1.0 / 63)  # rank 2

    def test_empty_lists(self):
        result = SearchService._reciprocal_rank_fusion([[], []])
        assert result == []

    def test_k_parameter(self):
        """Different k values should produce different scores."""
        items = [(0.9, 10)]
        result_k10 = SearchService._reciprocal_rank_fusion([items], k=10)
        result_k100 = SearchService._reciprocal_rank_fusion([items], k=100)
        score_k10 = result_k10[0][0]
        score_k100 = result_k100[0][0]
        assert score_k10 == pytest.approx(1.0 / 11)
        assert score_k100 == pytest.approx(1.0 / 101)
        assert score_k10 > score_k100


# ── TestRankedPreview ─────────────────────────────────────────────


class TestRankedPreview:
    def test_empty_text(self):
        assert SearchService._ranked_preview("", "query") == ""

    def test_whitespace_only(self):
        assert SearchService._ranked_preview("   \n\t  ", "query") == ""

    def test_short_text_returned_whole(self):
        text = "This is a short text about transformers."
        result = SearchService._ranked_preview(text, "transformers")
        assert "transformers" in result
        # Short text is returned as-is (cleaned)
        assert len(result) <= 600

    def test_long_text_with_match(self):
        # Build a 2000-char text with "transformer" at specific positions
        filler = "Lorem ipsum dolor sit amet. " * 50
        text = filler[:800] + " transformer architecture " + filler[:800] + " transformer model " + filler[:200]
        result = SearchService._ranked_preview(text, "transformer")
        assert "transformer" in result
        assert " ... " in result  # multiple segments joined

    def test_long_text_no_match(self):
        text = "The quick brown fox jumps over the lazy dog. " * 50
        result = SearchService._ranked_preview(text, "xyznonexistent")
        assert len(result) > 0  # fallback: returns segments anyway

    def test_output_within_target_length(self):
        text = "A " * 2000
        result = SearchService._ranked_preview(text, "query", target_chars=600)
        # Should be at most target + "..." = 603
        assert len(result) <= 603

    def test_truncation_adds_ellipsis(self):
        """When joined segments exceed target, result should end with '...'."""
        # Create text where many segments will be selected
        text = "important keyword " * 200
        result = SearchService._ranked_preview(text, "important keyword", target_chars=100)
        if len(result) > 100:
            assert result.endswith("...")


# ── TestSegmentScore ──────────────────────────────────────────────


class TestSegmentScore:
    def test_exact_phrase_match(self):
        score = SearchService._segment_score(
            "The attention mechanism is all you need.",
            query_tokens=["attention", "mechanism"],
            query_phrase="attention mechanism",
        )
        # Should include +4.0 for phrase match
        assert score >= 4.0

    def test_multiple_token_matches(self):
        score = SearchService._segment_score(
            "Deep learning and residual networks for image classification.",
            query_tokens=["deep", "learning", "residual"],
            query_phrase="deep learning residual",
        )
        assert score > 0.0

    def test_no_matches(self):
        score = SearchService._segment_score(
            "The weather is nice today.",
            query_tokens=["transformer", "attention"],
            query_phrase="transformer attention",
        )
        assert score == 0.0

    def test_empty_query(self):
        score = SearchService._segment_score(
            "Some text here.",
            query_tokens=[],
            query_phrase="",
        )
        assert score == 0.0

    def test_diversity_bonus(self):
        """When all query tokens match, diversity bonus = token_hits / len(tokens) = 1.0."""
        tokens = ["deep", "learning"]
        score = SearchService._segment_score(
            "Deep learning is transforming AI.",
            query_tokens=tokens,
            query_phrase="irrelevant phrase that wont match",
        )
        # Each token: min(2, 1) * 1.2 = 1.2, plus diversity: 2/2 = 1.0
        # Total: 1.2 + 1.2 + 1.0 = 3.4
        assert score == pytest.approx(3.4)

    def test_multiple_occurrences_capped(self):
        """Token occurring 5 times should be capped at min(2, 5) * 1.2 = 2.4."""
        score = SearchService._segment_score(
            "deep deep deep deep deep",
            query_tokens=["deep"],
            query_phrase="no match here",
        )
        # min(2, 5) * 1.2 = 2.4, diversity: 1/1 = 1.0 -> 3.4
        assert score == pytest.approx(3.4)


# ── TestFocusBboxFromSegmentMap ───────────────────────────────────


class TestFocusBboxFromSegmentMap:
    def test_matching_segment(self):
        items = [
            {"t": "attention mechanism for transformers", "x1": 10, "y1": 20, "x2": 300, "y2": 40},
        ]
        result = SearchService._focus_bbox_from_segment_map(
            json.dumps(items),
            query_tokens=["attention"],
            query_phrase="attention",
        )
        assert result == (10.0, 20.0, 300.0, 40.0)

    def test_best_score_wins(self):
        items = [
            {"t": "unrelated weather text", "x1": 10, "y1": 20, "x2": 100, "y2": 40},
            {"t": "attention mechanism paper", "x1": 200, "y1": 300, "x2": 500, "y2": 320},
        ]
        result = SearchService._focus_bbox_from_segment_map(
            json.dumps(items),
            query_tokens=["attention", "mechanism"],
            query_phrase="attention mechanism",
        )
        # Second item matches better -> its bbox returned
        assert result == (200.0, 300.0, 500.0, 320.0)

    def test_malformed_json(self):
        result = SearchService._focus_bbox_from_segment_map(
            "not valid json{{{",
            query_tokens=["test"],
            query_phrase="test",
        )
        assert result is None

    def test_empty_string(self):
        result = SearchService._focus_bbox_from_segment_map(
            "",
            query_tokens=["test"],
            query_phrase="test",
        )
        assert result is None

    def test_invalid_bbox_skipped(self):
        """Items with x2 <= x1 should be filtered out."""
        items = [
            {"t": "attention", "x1": 100, "y1": 20, "x2": 50, "y2": 40},  # invalid: x2 < x1
            {"t": "other", "x1": 10, "y1": 20, "x2": 10, "y2": 40},  # invalid: x2 == x1
        ]
        result = SearchService._focus_bbox_from_segment_map(
            json.dumps(items),
            query_tokens=["attention"],
            query_phrase="attention",
        )
        assert result is None  # all items invalid

    def test_missing_coords_skipped(self):
        items = [
            {"t": "attention", "x1": 10, "y1": 20},  # missing x2, y2
        ]
        result = SearchService._focus_bbox_from_segment_map(
            json.dumps(items),
            query_tokens=["attention"],
            query_phrase="attention",
        )
        assert result is None

    def test_non_list_json(self):
        result = SearchService._focus_bbox_from_segment_map(
            '{"t": "attention"}',
            query_tokens=["attention"],
            query_phrase="attention",
        )
        assert result is None


# ── TestTokens ────────────────────────────────────────────────────


class TestTokens:
    def test_basic(self):
        assert SearchService._tokens("Hello World 123") == ["hello", "world", "123"]

    def test_single_char_dropped(self):
        assert SearchService._tokens("a b cd ef") == ["cd", "ef"]

    def test_special_chars(self):
        assert SearchService._tokens("deep-learning v2.0") == ["deep", "learning", "v2"]

    def test_empty_string(self):
        assert SearchService._tokens("") == []

    def test_all_single_chars(self):
        assert SearchService._tokens("a b c d") == []
