"""Tests for IndexDatabase operations."""
from __future__ import annotations

import pytest

from vibescholar.db import IndexDatabase


class TestMetadata:
    def test_roundtrip(self, empty_db: IndexDatabase):
        empty_db.set_metadata("test_key", "test_value")
        assert empty_db.get_metadata("test_key") == "test_value"

    def test_missing_key(self, empty_db: IndexDatabase):
        assert empty_db.get_metadata("nonexistent") is None

    def test_upsert(self, empty_db: IndexDatabase):
        empty_db.set_metadata("k", "v1")
        empty_db.set_metadata("k", "v2")
        assert empty_db.get_metadata("k") == "v2"


class TestDirectories:
    def test_upsert_and_list(self, empty_db: IndexDatabase):
        dir_id = empty_db.upsert_directory("/test/path")
        assert dir_id > 0
        dirs = empty_db.list_directories()
        assert len(dirs) == 1
        assert dirs[0]["path"] == "/test/path"

    def test_active_flag(self, empty_db: IndexDatabase):
        dir_id = empty_db.upsert_directory("/test/path")
        empty_db.set_directory_active(dir_id, False)
        d = empty_db.get_directory(dir_id)
        assert d["active"] == 0

        empty_db.set_directory_active(dir_id, True)
        d = empty_db.get_directory(dir_id)
        assert d["active"] == 1


class TestChunks:
    def _setup_dir_and_file(self, db: IndexDatabase) -> tuple[int, int]:
        dir_id = db.upsert_directory("/test")
        file_id = db.upsert_file(dir_id, "/test/paper.pdf", mtime_ns=0, size_bytes=100)
        return dir_id, file_id

    def test_insert_and_retrieve(self, empty_db: IndexDatabase):
        _, file_id = self._setup_dir_and_file(empty_db)
        chunk_ids = empty_db.insert_chunks(file_id, [
            (1, 0, "hello world", None, None, None, None, None),
            (1, 1, "foo bar", None, None, None, None, None),
        ])
        assert len(chunk_ids) == 2
        assert empty_db.total_chunk_count() == 2

        rows = empty_db.get_chunks_by_ids(chunk_ids, active_only=False)
        texts = {str(r["text"]) for r in rows}
        assert texts == {"hello world", "foo bar"}

    def test_delete_chunks_for_file(self, empty_db: IndexDatabase):
        _, file_id = self._setup_dir_and_file(empty_db)
        empty_db.insert_chunks(file_id, [
            (1, 0, "text one", None, None, None, None, None),
        ])
        assert empty_db.total_chunk_count() == 1
        empty_db.delete_chunks_for_file(file_id)
        assert empty_db.total_chunk_count() == 0

    def test_neighboring_chunks(self, empty_db: IndexDatabase):
        _, file_id = self._setup_dir_and_file(empty_db)
        empty_db.insert_chunks(file_id, [
            (1, 0, "chunk zero", None, None, None, None, None),
            (1, 1, "chunk one", None, None, None, None, None),
            (1, 2, "chunk two", None, None, None, None, None),
        ])
        neighbors = empty_db.get_neighboring_chunks(file_id, page_number=1, chunk_number=1, window=1)
        assert len(neighbors) == 3  # chunks 0, 1, 2


class TestFTS5Sync:
    def _fts5_count(self, db: IndexDatabase) -> int:
        return db._conn.execute("SELECT COUNT(*) AS c FROM chunks_fts").fetchone()["c"]

    def _setup_dir_and_file(self, db: IndexDatabase) -> tuple[int, int]:
        dir_id = db.upsert_directory("/test")
        file_id = db.upsert_file(dir_id, "/test/paper.pdf", mtime_ns=0, size_bytes=100)
        return dir_id, file_id

    def test_fts5_sync_on_insert(self, empty_db: IndexDatabase):
        _, file_id = self._setup_dir_and_file(empty_db)
        empty_db.insert_chunks(file_id, [
            (1, 0, "neural network deep learning", None, None, None, None, None),
            (1, 1, "optimization convergence", None, None, None, None, None),
        ])
        assert self._fts5_count(empty_db) == 2

    def test_fts5_sync_on_delete(self, empty_db: IndexDatabase):
        _, file_id = self._setup_dir_and_file(empty_db)
        empty_db.insert_chunks(file_id, [
            (1, 0, "text to delete", None, None, None, None, None),
        ])
        assert self._fts5_count(empty_db) == 1
        empty_db.delete_chunks_for_file(file_id)
        assert self._fts5_count(empty_db) == 0

    def test_fts5_search_finds_inserted(self, empty_db: IndexDatabase):
        _, file_id = self._setup_dir_and_file(empty_db)
        empty_db.insert_chunks(file_id, [
            (1, 0, "quantum entanglement superposition", None, None, None, None, None),
            (1, 1, "classical mechanics gravity", None, None, None, None, None),
        ])
        results = empty_db.fts5_search("quantum", limit=10)
        assert len(results) >= 1
        # Score should be positive
        assert results[0][0] > 0

    def test_fts5_empty_query(self, empty_db: IndexDatabase):
        assert empty_db.fts5_search("", limit=10) == []
        assert empty_db.fts5_search("   ", limit=10) == []

    def test_delete_directory_cascades_fts5(self, empty_db: IndexDatabase):
        dir_id, file_id = self._setup_dir_and_file(empty_db)
        empty_db.insert_chunks(file_id, [
            (1, 0, "cascading delete test", None, None, None, None, None),
        ])
        assert self._fts5_count(empty_db) == 1
        empty_db.delete_directory(dir_id)
        assert self._fts5_count(empty_db) == 0

    def test_active_only_filter(self, empty_db: IndexDatabase):
        dir_id, file_id = self._setup_dir_and_file(empty_db)
        chunk_ids = empty_db.insert_chunks(file_id, [
            (1, 0, "active test", None, None, None, None, None),
        ])
        # Active — should return chunks
        rows = empty_db.get_chunks_by_ids(chunk_ids, active_only=True)
        assert len(rows) == 1

        # Deactivate directory
        empty_db.set_directory_active(dir_id, False)
        rows = empty_db.get_chunks_by_ids(chunk_ids, active_only=True)
        assert len(rows) == 0

        # Without active filter — still returns
        rows = empty_db.get_chunks_by_ids(chunk_ids, active_only=False)
        assert len(rows) == 1
