from __future__ import annotations

import logging
from typing import Any

from . import config

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Lazy-loaded cross-encoder re-ranker backed by FlashRank."""

    def __init__(self) -> None:
        self._ranker: Any | None = None

    def _ensure_model(self) -> Any:
        if self._ranker is None:
            from flashrank import Ranker

            self._ranker = Ranker(
                cache_dir=str(config.MODEL_CACHE_DIR),
            )
        return self._ranker

    def rerank(
        self,
        query: str,
        passages: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Re-rank passages using the cross-encoder.

        Each passage dict must have an ``"id"`` and ``"text"`` key.
        Returns passages sorted by cross-encoder score (descending),
        with a ``"rerank_score"`` key added.  On failure, returns the
        original list unchanged.
        """
        if not passages:
            return passages
        try:
            ranker = self._ensure_model()
            from flashrank import RerankRequest

            rerank_request = RerankRequest(query=query, passages=passages)
            results = ranker.rerank(rerank_request)
            # flashrank returns list of dicts with "score" added
            for r in results:
                r["rerank_score"] = r.get("score", 0.0)
            results.sort(key=lambda r: r.get("rerank_score", 0.0), reverse=True)
            if top_k is not None:
                results = results[:top_k]
            return results
        except Exception:
            logger.warning("Cross-encoder re-ranking failed; using original order", exc_info=True)
            return passages[:top_k] if top_k is not None else passages
