"""Tests for FAISS HNSW vector store with soft deletion."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vibescholar.vectors import FaissStore

DIM = 16  # Small dimension for fast tests


@pytest.fixture
def store(tmp_path: Path) -> FaissStore:
    """Fresh FaissStore with HNSW index."""
    return FaissStore(tmp_path / "test.faiss", DIM)


def _random_vectors(n: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.random((n, dim), dtype=np.float32)


class TestAddAndSearch:
    def test_add_and_search(self, store: FaissStore):
        vecs = _random_vectors(5)
        store.add_embeddings(vecs, [10, 20, 30, 40, 50])
        assert store.ntotal == 5

        scores, ids = store.search(vecs[0], k=3)
        assert len(ids) == 3
        assert 10 in ids  # Should find itself

    def test_empty_add(self, store: FaissStore):
        store.add_embeddings(np.empty((0, DIM), dtype=np.float32), [])
        assert store.ntotal == 0

    def test_empty_search(self, store: FaissStore):
        vecs = _random_vectors(3)
        store.add_embeddings(vecs, [1, 2, 3])
        scores, ids = store.search(vecs[0], k=3)
        assert len(ids) == 3


class TestSoftDeletion:
    def test_remove_filters_results(self, store: FaissStore):
        vecs = _random_vectors(5)
        store.add_embeddings(vecs, [10, 20, 30, 40, 50])
        store.remove_ids([20, 40])

        scores, ids = store.search(vecs[0], k=10)
        assert 20 not in ids
        assert 40 not in ids

    def test_ntotal_reflects_deletions(self, store: FaissStore):
        vecs = _random_vectors(5)
        store.add_embeddings(vecs, [10, 20, 30, 40, 50])
        assert store.ntotal == 5
        store.remove_ids([20])
        assert store.ntotal == 4

    def test_soft_delete_persists(self, tmp_path: Path):
        path = tmp_path / "persist.faiss"
        vecs = _random_vectors(5)

        # Create, add, delete, save
        store1 = FaissStore(path, DIM)
        store1.add_embeddings(vecs, [10, 20, 30, 40, 50])
        store1.remove_ids([30])
        store1.save()

        # Reload
        store2 = FaissStore(path, DIM)
        assert store2.ntotal == 4
        scores, ids = store2.search(vecs[0], k=10)
        assert 30 not in ids

    def test_re_add_clears_deleted(self, store: FaissStore):
        vecs = _random_vectors(5)
        store.add_embeddings(vecs, [10, 20, 30, 40, 50])
        store.remove_ids([20])
        assert store.ntotal == 4

        # Re-add the deleted ID — clears it from the soft-deleted set.
        # Note: HNSW keeps the old vector internally, so the raw index
        # count goes up, but 20 is no longer filtered from results.
        store.add_embeddings(vecs[1:2], [20])
        assert 20 not in store._deleted_ids
        scores, ids = store.search(vecs[1], k=5)
        assert 20 in ids

    def test_search_still_returns_k(self, store: FaissStore):
        """With soft-deleted entries, search should still return k results."""
        vecs = _random_vectors(10)
        store.add_embeddings(vecs, list(range(10)))
        store.remove_ids([3, 5, 7])

        scores, ids = store.search(vecs[0], k=5)
        assert len(ids) == 5
        assert all(i not in [3, 5, 7] for i in ids)

    def test_empty_remove(self, store: FaissStore):
        store.remove_ids([])  # Should not crash
        assert store.ntotal == 0
