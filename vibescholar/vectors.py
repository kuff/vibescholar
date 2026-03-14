from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

logger = logging.getLogger(__name__)

# HNSW parameters
_HNSW_M = 32              # connections per node (memory vs recall)
_HNSW_EF_CONSTRUCTION = 40  # beam width during build (quality vs speed)
_HNSW_EF_SEARCH = 64       # beam width during query (recall vs latency)


class FaissStore:
    """FAISS vector store using HNSW for sub-linear approximate nearest neighbor search.

    Uses IndexIDMap2 wrapping IndexHNSWFlat so vectors can be addressed by
    custom chunk IDs rather than sequential internal IDs.

    HNSW does not support hard vector removal.  When ``remove_ids`` is called
    and hard removal fails, the IDs are added to a soft-deletion set that is
    persisted alongside the index (as ``.deleted.npy``).  ``search`` over-fetches
    and filters out soft-deleted IDs before returning results.
    """

    def __init__(self, index_path: Path, dim: int):
        self.index_path = index_path
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._deleted_ids: set[int] = set()
        self._deleted_path = index_path.with_suffix(".deleted.npy")

        if self.index_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            if self._index.d != dim:
                raise ValueError(
                    f"FAISS index dimension mismatch: index={self._index.d}, model={dim}"
                )
            # Load soft-deleted IDs if present
            if self._deleted_path.exists():
                self._deleted_ids = set(
                    np.load(str(self._deleted_path)).tolist()
                )
        else:
            base = faiss.IndexHNSWFlat(dim, _HNSW_M)
            base.hnsw.efConstruction = _HNSW_EF_CONSTRUCTION
            base.hnsw.efSearch = _HNSW_EF_SEARCH
            self._index = faiss.IndexIDMap2(base)

    @property
    def ntotal(self) -> int:
        return int(self._index.ntotal) - len(self._deleted_ids)

    def add_embeddings(self, vectors: np.ndarray, ids: Iterable[int]) -> None:
        vectors_2d = np.asarray(vectors, dtype=np.float32)
        if vectors_2d.ndim == 1:
            vectors_2d = vectors_2d.reshape(1, -1)

        id_array = np.asarray(list(ids), dtype=np.int64)
        if id_array.size == 0:
            return

        faiss.normalize_L2(vectors_2d)
        self._index.add_with_ids(vectors_2d, id_array)

        # Clear any soft-deleted entries that are being re-added
        re_added = self._deleted_ids & set(id_array.tolist())
        if re_added:
            self._deleted_ids -= re_added

    def remove_ids(self, ids: Iterable[int]) -> None:
        id_array = np.asarray(list(ids), dtype=np.int64)
        if id_array.size == 0:
            return
        try:
            self._index.remove_ids(id_array)
        except RuntimeError:
            # HNSW doesn't support hard removal — use soft deletion
            self._deleted_ids.update(id_array.tolist())

    def search(self, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        vector = np.asarray(query_vector, dtype=np.float32)
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        faiss.normalize_L2(vector)

        if not self._deleted_ids:
            scores, ids = self._index.search(vector, k)
            return scores[0], ids[0]

        # Over-fetch to compensate for soft-deleted entries, then filter
        fetch_k = min(k + len(self._deleted_ids), self._index.ntotal)
        scores, ids = self._index.search(vector, fetch_k)
        scores, ids = scores[0], ids[0]

        mask = np.array(
            [(i not in self._deleted_ids) and (i >= 0) for i in ids],
            dtype=bool,
        )
        return scores[mask][:k], ids[mask][:k]

    def save(self) -> None:
        faiss.write_index(self._index, str(self.index_path))
        if self._deleted_ids:
            np.save(
                str(self._deleted_path),
                np.array(list(self._deleted_ids), dtype=np.int64),
            )
        elif self._deleted_path.exists():
            self._deleted_path.unlink()
