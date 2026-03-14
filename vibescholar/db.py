from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable


class IndexDatabase:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        self._migrate()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self):
        self._conn.execute("BEGIN")
        try:
            yield
        except Exception:
            self._conn.rollback()
            raise
        self._conn.commit()

    def _migrate(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS directories (
                id INTEGER PRIMARY KEY,
                path TEXT NOT NULL UNIQUE,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_indexed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                directory_id INTEGER NOT NULL REFERENCES directories(id) ON DELETE CASCADE,
                path TEXT NOT NULL UNIQUE,
                mtime_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL,
                indexed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                page_number INTEGER NOT NULL,
                chunk_number INTEGER NOT NULL,
                text TEXT NOT NULL,
                bbox_x1 REAL,
                bbox_y1 REAL,
                bbox_x2 REAL,
                bbox_y2 REAL,
                segment_map TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_files_directory_id ON files(directory_id);
            CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);

            -- FTS5 full-text index (external-content, synced via triggers)
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content='chunks',
                content_rowid='id',
                tokenize='porter unicode61'
            );

            CREATE TRIGGER IF NOT EXISTS chunks_fts_insert AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_fts_update AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES('delete', old.id, old.text);
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            """
        )
        self._ensure_chunk_columns()
        self._ensure_fts5()
        self._conn.commit()

    def _ensure_fts5(self) -> None:
        """One-time bulk population of FTS5 for existing databases."""
        if self.get_metadata("fts5_version") is not None:
            return
        self._conn.execute(
            "INSERT INTO chunks_fts(rowid, text) SELECT id, text FROM chunks"
        )
        self.set_metadata("fts5_version", "1")

    def _ensure_chunk_columns(self) -> None:
        rows = self._conn.execute("PRAGMA table_info(chunks)").fetchall()
        existing = {str(row["name"]) for row in rows}
        for column in ["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "segment_map"]:
            if column in existing:
                continue
            column_type = "TEXT" if column == "segment_map" else "REAL"
            self._conn.execute(f"ALTER TABLE chunks ADD COLUMN {column} {column_type}")

    def set_metadata(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO metadata(key, value)
            VALUES(?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self._conn.commit()

    def get_metadata(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def upsert_directory(self, path: str) -> int:
        self._conn.execute(
            """
            INSERT INTO directories(path, active)
            VALUES(?, 1)
            ON CONFLICT(path) DO UPDATE SET path = excluded.path
            """,
            (path,),
        )
        self._conn.commit()
        row = self._conn.execute("SELECT id FROM directories WHERE path = ?", (path,)).fetchone()
        assert row is not None
        return int(row["id"])

    def touch_directory_indexed(self, directory_id: int) -> None:
        self._conn.execute(
            "UPDATE directories SET last_indexed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (directory_id,),
        )
        self._conn.commit()

    def set_directory_active(self, directory_id: int, active: bool) -> None:
        self._conn.execute(
            "UPDATE directories SET active = ? WHERE id = ?",
            (1 if active else 0, directory_id),
        )
        self._conn.commit()

    def get_directory(self, directory_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT id, path, active FROM directories WHERE id = ?", (directory_id,)
        ).fetchone()

    def list_directories(self) -> list[sqlite3.Row]:
        rows = self._conn.execute(
            """
            SELECT
                d.id,
                d.path,
                d.active,
                d.last_indexed_at,
                (SELECT COUNT(*) FROM files f WHERE f.directory_id = d.id) AS file_count,
                (
                    SELECT COUNT(*)
                    FROM chunks c
                    JOIN files f ON f.id = c.file_id
                    WHERE f.directory_id = d.id
                ) AS chunk_count
            FROM directories d
            ORDER BY d.path
            """
        ).fetchall()
        return list(rows)

    def get_files_for_directory(self, directory_id: int) -> list[sqlite3.Row]:
        rows = self._conn.execute(
            """
            SELECT id, path, mtime_ns, size_bytes
            FROM files
            WHERE directory_id = ?
            """,
            (directory_id,),
        ).fetchall()
        return list(rows)

    def get_chunk_ids_for_directory(self, directory_id: int) -> list[int]:
        rows = self._conn.execute(
            """
            SELECT c.id
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            WHERE f.directory_id = ?
            """,
            (directory_id,),
        ).fetchall()
        return [int(row["id"]) for row in rows]

    def delete_directory(self, directory_id: int) -> tuple[str | None, list[int]]:
        """Delete a directory and all its files and chunks.

        Chunks are deleted explicitly before files/directory so that FTS5
        sync triggers fire (SQLite cascade deletes do NOT fire triggers).
        """
        row = self._conn.execute(
            "SELECT path FROM directories WHERE id = ?",
            (directory_id,),
        ).fetchone()
        chunk_ids = self.get_chunk_ids_for_directory(directory_id)
        # Explicitly delete chunks then files before the directory so that
        # FTS5 sync triggers fire (cascade deletes do NOT fire triggers).
        self.delete_chunks_by_ids(chunk_ids)
        self._conn.execute(
            "DELETE FROM files WHERE directory_id = ?", (directory_id,)
        )
        self._conn.execute("DELETE FROM directories WHERE id = ?", (directory_id,))
        self._conn.commit()
        path = str(row["path"]) if row else None
        return (path, chunk_ids)

    def get_file_by_path(self, path: str) -> sqlite3.Row | None:
        return self._conn.execute(
            """
            SELECT id, directory_id, path, mtime_ns, size_bytes
            FROM files
            WHERE path = ?
            """,
            (path,),
        ).fetchone()

    def upsert_file(
        self,
        directory_id: int,
        path: str,
        mtime_ns: int,
        size_bytes: int,
    ) -> int:
        self._conn.execute(
            """
            INSERT INTO files(directory_id, path, mtime_ns, size_bytes, indexed_at)
            VALUES(?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(path) DO UPDATE SET
                directory_id = excluded.directory_id,
                mtime_ns = excluded.mtime_ns,
                size_bytes = excluded.size_bytes,
                indexed_at = CURRENT_TIMESTAMP
            """,
            (directory_id, path, mtime_ns, size_bytes),
        )
        row = self._conn.execute("SELECT id FROM files WHERE path = ?", (path,)).fetchone()
        assert row is not None
        return int(row["id"])

    def get_chunk_ids_for_file(self, file_id: int) -> list[int]:
        rows = self._conn.execute("SELECT id FROM chunks WHERE file_id = ?", (file_id,)).fetchall()
        return [int(row["id"]) for row in rows]

    def delete_chunks_for_file(self, file_id: int) -> list[int]:
        chunk_ids = self.get_chunk_ids_for_file(file_id)
        self._conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        return chunk_ids

    def delete_file(self, file_id: int) -> list[int]:
        chunk_ids = self.get_chunk_ids_for_file(file_id)
        # Explicitly delete chunks before the file so FTS5 triggers fire.
        self._conn.execute("DELETE FROM chunks WHERE file_id = ?", (file_id,))
        self._conn.execute("DELETE FROM files WHERE id = ?", (file_id,))
        return chunk_ids

    def insert_chunks(
        self,
        file_id: int,
        rows: Iterable[
            tuple[
                int,
                int,
                str,
                float | None,
                float | None,
                float | None,
                float | None,
                str | None,
            ]
        ],
    ) -> list[int]:
        chunk_ids: list[int] = []
        for page_number, chunk_number, text, x1, y1, x2, y2, segment_map in rows:
            cursor = self._conn.execute(
                """
                INSERT INTO chunks(file_id, page_number, chunk_number, text, bbox_x1, bbox_y1, bbox_x2, bbox_y2, segment_map)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (file_id, page_number, chunk_number, text, x1, y1, x2, y2, segment_map),
            )
            chunk_ids.append(int(cursor.lastrowid))
        return chunk_ids

    def delete_chunks_by_ids(self, chunk_ids: Iterable[int]) -> None:
        ids = list(chunk_ids)
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        self._conn.execute(f"DELETE FROM chunks WHERE id IN ({placeholders})", ids)

    def get_chunks_by_ids(self, chunk_ids: list[int], active_only: bool = True) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        clause = "AND d.active = 1" if active_only else ""
        rows = self._conn.execute(
            f"""
            SELECT
                c.id AS chunk_id,
                c.file_id,
                c.page_number,
                c.chunk_number,
                c.text,
                c.bbox_x1,
                c.bbox_y1,
                c.bbox_x2,
                c.bbox_y2,
                c.segment_map,
                f.path AS file_path,
                d.id AS directory_id,
                d.path AS directory_path,
                d.active AS directory_active
            FROM chunks c
            JOIN files f ON f.id = c.file_id
            JOIN directories d ON d.id = f.directory_id
            WHERE c.id IN ({placeholders}) {clause}
            """,
            chunk_ids,
        ).fetchall()
        return list(rows)

    def get_all_chunks_text(self) -> list[tuple[int, str]]:
        rows = self._conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
        return [(int(row["id"]), str(row["text"])) for row in rows]

    def fts5_search(self, query: str, limit: int) -> list[tuple[float, int]]:
        """Full-text search via FTS5. Returns ``(score, chunk_id)`` pairs.

        *query* should be a valid FTS5 match expression (supports AND, OR,
        NOT, quoted phrases, prefix*).  Scores are negated FTS5 ranks
        (higher = more relevant).
        """
        if not query.strip():
            return []
        rows = self._conn.execute(
            "SELECT rowid, rank FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (query, limit),
        ).fetchall()
        return [(-float(row["rank"]), int(row["rowid"])) for row in rows]

    def get_neighboring_chunks(
        self, file_id: int, page_number: int, chunk_number: int, window: int = 1
    ) -> list[sqlite3.Row]:
        rows = self._conn.execute(
            """
            SELECT id, page_number, chunk_number, text
            FROM chunks
            WHERE file_id = ?
              AND page_number = ?
              AND chunk_number BETWEEN ? AND ?
            ORDER BY chunk_number
            """,
            (file_id, page_number, chunk_number - window, chunk_number + window),
        ).fetchall()
        return list(rows)

    def total_chunk_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS count FROM chunks").fetchone()
        return int(row["count"]) if row else 0
