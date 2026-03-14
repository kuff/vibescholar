from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict
from dataclasses import dataclass

from .db import IndexDatabase
from .embeddings import Embedder
from .vectors import FaissStore
from .reranker import CrossEncoderReranker
from .text import clean_text

logger = logging.getLogger(__name__)


@dataclass
class SearchHit:
    chunk_id: int
    file_path: str
    page_number: int
    score: float
    snippet: str
    bbox_x1: float | None
    bbox_y1: float | None
    bbox_x2: float | None
    bbox_y2: float | None


@dataclass
class FileSearchResult:
    file_path: str
    hits: list[SearchHit]


class SearchService:
    def __init__(self, db: IndexDatabase, embedder: Embedder, store: FaissStore):
        self.db = db
        self.embedder = embedder
        self.store = store
        self._reranker: CrossEncoderReranker | None = None

    # ------------------------------------------------------------------
    # FTS5 keyword search
    # ------------------------------------------------------------------

    def _fts5_search(self, query: str, top_k: int) -> list[tuple[float, int]]:
        """Return (score, chunk_id) pairs from SQLite FTS5."""
        import sqlite3

        fts_query = self._fts5_query(query)
        if not fts_query:
            return []
        try:
            return self.db.fts5_search(fts_query, top_k)
        except sqlite3.OperationalError:
            # Malformed FTS5 query — fall back to plain tokens
            tokens = self._tokens(query)
            if not tokens:
                return []
            return self.db.fts5_search(" ".join(tokens), top_k)

    _FTS5_OPERATORS = {"AND", "OR", "NOT", "NEAR"}

    @classmethod
    def _fts5_query(cls, query: str) -> str:
        """Parse a query string into a valid FTS5 match expression.

        Preserves: AND, OR, NOT, NEAR operators (case-insensitive),
        ``"quoted phrases"``, and ``prefix*`` wildcards.  All other
        text is tokenized through ``_tokens()`` (alphanumeric, lowercased,
        single-char tokens dropped).  Leading/trailing/consecutive operators
        are stripped to prevent FTS5 parse errors.
        """
        parts: list[str] = []
        i = 0
        while i < len(query):
            # Skip whitespace
            if query[i].isspace():
                i += 1
                continue
            # Quoted phrase — pass through
            if query[i] == '"':
                end = query.find('"', i + 1)
                if end == -1:
                    end = len(query)
                phrase = query[i : end + 1]
                parts.append(phrase)
                i = end + 1
                continue
            # Read a word
            j = i
            while j < len(query) and not query[j].isspace() and query[j] != '"':
                j += 1
            word = query[i:j]
            i = j
            # Check for operators
            if word.upper() in cls._FTS5_OPERATORS:
                parts.append(word.upper())
                continue
            # Prefix wildcard (e.g. optim*)
            if word.endswith("*"):
                stem = re.sub(r"[^a-z0-9*]", "", word.lower())
                if stem and stem != "*":
                    parts.append(stem)
                continue
            # Regular word — tokenize
            for token in cls._tokens(word):
                parts.append(token)

        # Clean up: remove leading/trailing/consecutive operators
        cleaned: list[str] = []
        for part in parts:
            if part in cls._FTS5_OPERATORS:
                if not cleaned or cleaned[-1] in cls._FTS5_OPERATORS:
                    continue
                cleaned.append(part)
            else:
                cleaned.append(part)
        # Remove trailing operator
        if cleaned and cleaned[-1] in cls._FTS5_OPERATORS:
            cleaned.pop()

        return " ".join(cleaned)

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: list[list[tuple[float, int]]], k: int = 60
    ) -> list[tuple[float, int]]:
        """Fuse multiple ranked lists using Reciprocal Rank Fusion.

        For each chunk, the fused score is ``sum(1 / (k + rank + 1))`` across
        all input lists.  This combines rankings without needing to calibrate
        score distributions between FAISS and FTS5.  Returns ``(fused_score,
        chunk_id)`` sorted descending.
        """
        scores: dict[int, float] = {}
        for ranked in ranked_lists:
            for rank, (_, chunk_id) in enumerate(ranked):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        fused = [(score, cid) for cid, score in scores.items()]
        fused.sort(key=lambda x: x[0], reverse=True)
        return fused

    # ------------------------------------------------------------------
    # Cross-encoder re-ranking
    # ------------------------------------------------------------------

    def _cross_encoder_rerank(
        self,
        query: str,
        candidate_ids: list[int],
        row_map: dict[int, "sqlite3.Row"],
        top_k: int,
    ) -> list[int]:
        """Re-rank candidate_ids using a cross-encoder. Returns reordered chunk_ids."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        passages = []
        for cid in candidate_ids:
            row = row_map.get(cid)
            if row is None:
                continue
            passages.append({"id": cid, "text": str(row["text"])})
        if not passages:
            return candidate_ids[:top_k]
        reranked = self._reranker.rerank(query, passages, top_k=top_k)
        return [p["id"] for p in reranked]

    # ------------------------------------------------------------------
    # Parent context expansion
    # ------------------------------------------------------------------

    def _expand_with_neighbors(self, row: "sqlite3.Row", context_window: int = 1) -> str:
        """Return chunk text expanded with neighboring chunks on the same page."""
        if context_window <= 0:
            return str(row["text"])
        try:
            neighbors = self.db.get_neighboring_chunks(
                file_id=int(row["file_id"]),
                page_number=int(row["page_number"]),
                chunk_number=int(row["chunk_number"]),
                window=context_window,
            )
            if neighbors:
                return " ".join(str(n["text"]) for n in neighbors)
        except Exception:
            pass
        return str(row["text"])

    # ------------------------------------------------------------------
    # Main search pipeline
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 40,
        active_only: bool = True,
        rerank: bool = True,
        expand_context: bool = True,
        context_window: int = 1,
        directory_ids: set[int] | None = None,
    ) -> list[FileSearchResult]:
        """Run the full hybrid search pipeline.

        1. FAISS semantic search  2. FTS5 keyword search  3. RRF fusion
        4. Directory & active-only filtering  5. Cross-encoder reranking
        6. Context expansion  7. Snippet generation

        Args:
            query: Natural language search query (supports FTS5 operators).
            top_k: Max hits per file in the results.
            active_only: If True, exclude chunks from inactive directories.
            rerank: If True, apply cross-encoder reranking.
            expand_context: If True, pull neighboring chunks for richer snippets.
            context_window: Number of neighboring chunks to include on each side.
            directory_ids: If set, restrict results to these directory IDs.
        """
        if self.store.ntotal == 0:
            return []

        query_vector = self.embedder.embed_texts([query])[0]
        request_k = min(max(top_k * 6, top_k), max(self.store.ntotal, top_k))

        # --- FAISS vector search ---
        scores, chunk_ids = self.store.search(query_vector, request_k)
        faiss_ranked: list[tuple[float, int]] = [
            (float(s), int(cid)) for s, cid in zip(scores, chunk_ids) if int(cid) >= 0
        ]

        # --- FTS5 keyword search ---
        fts5_ranked = self._fts5_search(query, request_k)

        # --- Reciprocal Rank Fusion ---
        if fts5_ranked:
            fused = self._reciprocal_rank_fusion([faiss_ranked, fts5_ranked])
        else:
            fused = faiss_ranked

        # Take top candidates for re-ranking (wider pool than final top_k)
        rerank_pool_size = min(len(fused), max(top_k * 3, 60))
        candidate_ids = [cid for _, cid in fused[:rerank_pool_size]]

        if not candidate_ids:
            return []

        rows = self.db.get_chunks_by_ids(candidate_ids, active_only=active_only)
        row_map = {int(row["chunk_id"]): row for row in rows}
        # Filter to only ids that survived active_only filtering
        candidate_ids = [cid for cid in candidate_ids if cid in row_map]

        # --- Directory filter ---
        if directory_ids is not None:
            candidate_ids = [
                cid for cid in candidate_ids
                if int(row_map[cid]["directory_id"]) in directory_ids
            ]

        # --- Cross-encoder re-ranking ---
        if rerank and candidate_ids:
            try:
                candidate_ids = self._cross_encoder_rerank(query, candidate_ids, row_map, top_k=len(candidate_ids))
            except Exception:
                logger.warning("Cross-encoder re-rank failed; using fused order", exc_info=True)

        # Build a score map — use position-based scoring so order from re-ranker is preserved
        score_map: dict[int, float] = {}
        for rank, cid in enumerate(candidate_ids):
            score_map[cid] = 1.0 / (rank + 1)

        # --- Build results ---
        grouped: "OrderedDict[str, list[SearchHit]]" = OrderedDict()
        query_tokens = self._tokens(query)
        query_phrase = clean_text(query).lower()

        for cid in candidate_ids:
            row = row_map.get(cid)
            if row is None:
                continue
            file_path = str(row["file_path"])

            # Context expansion
            if expand_context and context_window > 0:
                expanded_text = self._expand_with_neighbors(row, context_window)
            else:
                expanded_text = str(row["text"])

            focus_bbox = self._focus_bbox_from_segment_map(
                str(row["segment_map"] or ""),
                query_tokens=query_tokens,
                query_phrase=query_phrase,
            )
            if focus_bbox is None and row["bbox_x1"] is not None:
                focus_bbox = (
                    float(row["bbox_x1"]),
                    float(row["bbox_y1"]),
                    float(row["bbox_x2"]),
                    float(row["bbox_y2"]),
                )
            hit = SearchHit(
                chunk_id=cid,
                file_path=file_path,
                page_number=int(row["page_number"]),
                score=score_map.get(cid, 0.0),
                snippet=self._ranked_preview(expanded_text, query),
                bbox_x1=focus_bbox[0] if focus_bbox else None,
                bbox_y1=focus_bbox[1] if focus_bbox else None,
                bbox_x2=focus_bbox[2] if focus_bbox else None,
                bbox_y2=focus_bbox[3] if focus_bbox else None,
            )
            grouped.setdefault(file_path, []).append(hit)

        for file_hits in grouped.values():
            file_hits.sort(key=lambda hit: hit.score, reverse=True)

        sorted_files = sorted(
            grouped.items(),
            key=lambda pair: pair[1][0].score if pair[1] else 0.0,
            reverse=True,
        )

        results: list[FileSearchResult] = []
        for file_path, file_hits in sorted_files:
            results.append(FileSearchResult(file_path=file_path, hits=file_hits[:top_k]))
        return results

    @staticmethod
    def _ranked_preview(
        text: str,
        query: str,
        target_chars: int = 600,
        window_chars: int = 150,
        stride_chars: int = 75,
    ) -> str:
        normalized = clean_text(text)
        if not normalized:
            return ""
        if len(normalized) <= target_chars:
            return normalized

        segments: list[tuple[int, str]] = []
        for start in range(0, len(normalized), stride_chars):
            piece = normalized[start : start + window_chars]
            if piece:
                segments.append((start // stride_chars, piece))
            if start + window_chars >= len(normalized):
                break

        query_tokens = SearchService._tokens(query)
        query_phrase = clean_text(query).lower()

        scored: list[tuple[float, int, str]] = []
        for idx, segment in segments:
            score = SearchService._segment_score(segment, query_tokens, query_phrase)
            scored.append((score, idx, segment))

        scored.sort(key=lambda item: (-item[0], item[1]))

        selected: list[tuple[int, str]] = []
        for _, idx, segment in scored:
            # Avoid stacking nearly identical neighboring windows.
            if any(abs(idx - existing_idx) <= 1 for existing_idx, _ in selected):
                continue
            selected.append((idx, segment))
            if len(selected) >= 6:
                break

        if not selected:
            return normalized[:target_chars]

        ordered_segments = [segment for _, segment in selected]
        preview = " ... ".join(ordered_segments)
        if len(preview) > target_chars:
            preview = preview[: target_chars - 3] + "..."

        return preview

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1]

    @staticmethod
    def _segment_score(segment: str, query_tokens: list[str], query_phrase: str) -> float:
        lower = segment.lower()
        if not query_tokens and not query_phrase:
            return 0.0

        score = 0.0
        if query_phrase and query_phrase in lower:
            score += 4.0

        token_hits = 0
        for token in query_tokens:
            occurrences = lower.count(token)
            if occurrences:
                token_hits += 1
                score += min(2, occurrences) * 1.2

        # Prefer segments covering more distinct query terms.
        if query_tokens:
            score += token_hits / len(query_tokens)

        return score

    @staticmethod
    def _focus_bbox_from_segment_map(
        segment_map: str,
        query_tokens: list[str],
        query_phrase: str,
    ) -> tuple[float, float, float, float] | None:
        if not segment_map:
            return None
        try:
            items = json.loads(segment_map)
        except Exception:
            return None
        if not isinstance(items, list):
            return None

        best: tuple[float, tuple[float, float, float, float]] | None = None
        for item in items:
            if not isinstance(item, dict):
                continue
            text = str(item.get("t", ""))
            x1 = item.get("x1")
            y1 = item.get("y1")
            x2 = item.get("x2")
            y2 = item.get("y2")
            if None in (x1, y1, x2, y2):
                continue
            try:
                rect = (float(x1), float(y1), float(x2), float(y2))
            except Exception:
                continue
            if rect[2] <= rect[0] or rect[3] <= rect[1]:
                continue
            score = SearchService._segment_score(text, query_tokens, query_phrase)
            if best is None or score > best[0]:
                best = (score, rect)

        return best[1] if best else None
