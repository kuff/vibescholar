from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from pypdf import PdfReader

from .db import IndexDatabase
from .embeddings import Embedder
from .vectors import FaissStore
from .text import chunk_text, clean_text

@dataclass
class IndexProgress:
    phase: str
    scanned_files: int = 0
    total_files: int = 0
    current_file: str = ""
    indexed_files: int = 0
    skipped_files: int = 0
    removed_files: int = 0
    chunks_added: int = 0
    errors: int = 0


ProgressCallback = Callable[[IndexProgress], None]


class IndexCancelled(Exception):
    """Raised when indexing is cancelled by user."""


@dataclass
class IndexStats:
    scanned_files: int = 0
    indexed_files: int = 0
    skipped_files: int = 0
    removed_files: int = 0
    chunks_added: int = 0
    errors: int = 0


class PdfIndexer:
    def __init__(
        self,
        db: IndexDatabase,
        embedder: Embedder,
        store: FaissStore,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        embed_batch_size: int = 64,
    ):
        self.db = db
        self.embedder = embedder
        self.store = store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_batch_size = embed_batch_size

    def index_directory(
        self,
        directory_id: int,
        directory_path: Path,
        force: bool = False,
        progress: ProgressCallback | None = None,
        cancel_event: threading.Event | None = None,
    ) -> IndexStats:
        stats = IndexStats()
        directory_path = directory_path.resolve()
        total_files = self._count_pdf_paths(directory_path)
        if progress:
            progress(IndexProgress(phase="start", total_files=total_files))
        known_files = {
            row["path"]: row
            for row in self.db.get_files_for_directory(directory_id)
        }
        seen_paths: set[str] = set()

        for pdf_path in self._iter_pdf_paths(directory_path):
            if cancel_event is not None and cancel_event.is_set():
                raise IndexCancelled()
            stats.scanned_files += 1
            full_path = str(pdf_path)
            seen_paths.add(full_path)

            if progress:
                progress(
                    IndexProgress(
                        phase="scan",
                        scanned_files=stats.scanned_files,
                        total_files=total_files,
                        current_file=full_path,
                        indexed_files=stats.indexed_files,
                        skipped_files=stats.skipped_files,
                        removed_files=stats.removed_files,
                        chunks_added=stats.chunks_added,
                        errors=stats.errors,
                    )
                )

            file_stat = pdf_path.stat()
            current = known_files.get(full_path)
            unchanged = (
                current is not None
                and int(current["mtime_ns"]) == int(file_stat.st_mtime_ns)
                and int(current["size_bytes"]) == int(file_stat.st_size)
            )
            if unchanged and not force:
                stats.skipped_files += 1
                if progress:
                    progress(
                        IndexProgress(
                            phase="skip",
                            scanned_files=stats.scanned_files,
                            total_files=total_files,
                            current_file=full_path,
                            indexed_files=stats.indexed_files,
                            skipped_files=stats.skipped_files,
                            removed_files=stats.removed_files,
                            chunks_added=stats.chunks_added,
                            errors=stats.errors,
                        )
                    )
                continue

            try:
                added = self._index_file(
                    directory_id=directory_id,
                    file_path=pdf_path,
                    mtime_ns=int(file_stat.st_mtime_ns),
                    size_bytes=int(file_stat.st_size),
                    cancel_event=cancel_event,
                )
                stats.chunks_added += added
                stats.indexed_files += 1
            except IndexCancelled:
                raise
            except Exception:
                stats.errors += 1
            finally:
                if progress:
                    progress(
                        IndexProgress(
                            phase="process",
                            scanned_files=stats.scanned_files,
                            total_files=total_files,
                            current_file=full_path,
                            indexed_files=stats.indexed_files,
                            skipped_files=stats.skipped_files,
                            removed_files=stats.removed_files,
                            chunks_added=stats.chunks_added,
                            errors=stats.errors,
                        )
                    )

        for old_path, row in known_files.items():
            if cancel_event is not None and cancel_event.is_set():
                raise IndexCancelled()
            if old_path in seen_paths:
                continue
            file_id = int(row["id"])
            removed_chunk_ids = self.db.delete_file(file_id)
            self.store.remove_ids(removed_chunk_ids)
            stats.removed_files += 1
            if progress:
                progress(
                    IndexProgress(
                        phase="remove",
                        scanned_files=stats.scanned_files,
                        total_files=total_files,
                        current_file=old_path,
                        indexed_files=stats.indexed_files,
                        skipped_files=stats.skipped_files,
                        removed_files=stats.removed_files,
                        chunks_added=stats.chunks_added,
                        errors=stats.errors,
                    )
                )

        self.db.touch_directory_indexed(directory_id)
        self.store.save()
        if progress:
            progress(
                IndexProgress(
                    phase="done",
                    scanned_files=stats.scanned_files,
                    total_files=total_files,
                    indexed_files=stats.indexed_files,
                    skipped_files=stats.skipped_files,
                    removed_files=stats.removed_files,
                    chunks_added=stats.chunks_added,
                    errors=stats.errors,
                )
            )
        return stats

    def _index_file(
        self,
        directory_id: int,
        file_path: Path,
        mtime_ns: int,
        size_bytes: int,
        cancel_event: threading.Event | None = None,
    ) -> int:
        file_path_str = str(file_path.resolve())
        chunks_added = 0
        added_vector_ids: list[int] = []
        removed_ids: list[int] = []

        try:
            with self.db.transaction():
                file_id = self.db.upsert_file(
                    directory_id=directory_id,
                    path=file_path_str,
                    mtime_ns=mtime_ns,
                    size_bytes=size_bytes,
                )

                removed_ids = self.db.delete_chunks_for_file(file_id)

                pending_rows: list[
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
                ] = []
                pending_texts: list[str] = []

                for page_number, chunk_number, text, x1, y1, x2, y2, segment_map in self._extract_chunks(file_path):
                    if cancel_event is not None and cancel_event.is_set():
                        raise IndexCancelled()
                    pending_rows.append((page_number, chunk_number, text, x1, y1, x2, y2, segment_map))
                    pending_texts.append(text)
                    if len(pending_rows) >= self.embed_batch_size:
                        chunk_ids = self._flush_chunks(file_id, pending_rows, pending_texts)
                        chunks_added += len(chunk_ids)
                        added_vector_ids.extend(chunk_ids)
                        pending_rows = []
                        pending_texts = []

                if pending_rows:
                    chunk_ids = self._flush_chunks(file_id, pending_rows, pending_texts)
                    chunks_added += len(chunk_ids)
                    added_vector_ids.extend(chunk_ids)
        except Exception:
            # Roll back any vectors that were added before a DB rollback.
            self.store.remove_ids(added_vector_ids)
            raise

        self.store.remove_ids(removed_ids)
        return chunks_added

    def _flush_chunks(
        self,
        file_id: int,
        rows: list[
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
        texts: list[str],
    ) -> list[int]:
        chunk_ids = self.db.insert_chunks(file_id, rows)
        try:
            vectors = self.embedder.embed_texts(texts)
            self.store.add_embeddings(vectors, chunk_ids)
        except Exception:
            self.db.delete_chunks_by_ids(chunk_ids)
            raise
        return chunk_ids

    def _extract_chunks(self, path: Path):
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")

        for page_index, page in enumerate(reader.pages, start=1):
            segments = self._collect_page_segments(page)
            if segments:
                for chunk_number, chunk in enumerate(self._chunk_page_segments(segments), start=1):
                    text, x1, y1, x2, y2, segment_map = chunk
                    yield (page_index, chunk_number, text, x1, y1, x2, y2, segment_map)
                continue

            # Fallback for pypdf versions/doc types where visitor callbacks aren't usable.
            try:
                raw_text = page.extract_text() or ""
            except Exception:
                raw_text = ""
            page_chunks = chunk_text(
                raw_text,
                chunk_size=self.chunk_size,
                overlap=self.chunk_overlap,
            )
            for chunk_number, chunk in enumerate(page_chunks, start=1):
                yield (page_index, chunk_number, chunk, None, None, None, None, None)

    @staticmethod
    def _collect_page_segments(page) -> list[tuple[str, float, float, float, float]]:
        segments: list[tuple[str, float, float, float, float]] = []

        def visitor_text(text, _cm, tm, _font_dict, font_size):
            cleaned = clean_text(str(text))
            if not cleaned:
                return
            x = float(tm[4])
            y = float(tm[5])
            size = float(font_size or 10.0)
            width = max(8.0, min(800.0, len(cleaned) * size * 0.42))
            height = max(8.0, size + 2.0)
            segments.append((cleaned, x, y, width, height))

        try:
            page.extract_text(visitor_text=visitor_text)
        except TypeError:
            # Older/alternate pypdf APIs may not support visitor_text.
            return []
        except Exception:
            return []
        segments.sort(key=lambda seg: (-round(seg[2], 1), seg[1]))
        return segments

    def _chunk_page_segments(
        self,
        segments: list[tuple[str, float, float, float, float]],
    ) -> list[tuple[str, float, float, float, float, str | None]]:
        chunks: list[tuple[str, float, float, float, float, str | None]] = []
        current: list[tuple[str, float, float, float, float]] = []
        char_count = 0

        for seg in segments:
            seg_len = len(seg[0])
            projected = char_count + (1 if char_count > 0 else 0) + seg_len
            if current and projected > self.chunk_size:
                chunks.append(self._finalize_chunk(current))
                current, char_count = self._tail_overlap_segments(current, self.chunk_overlap)

            current.append(seg)
            char_count += (1 if char_count > 0 else 0) + seg_len

        if current:
            chunks.append(self._finalize_chunk(current))
        return chunks

    @staticmethod
    def _finalize_chunk(
        segments: list[tuple[str, float, float, float, float]],
    ) -> tuple[str, float, float, float, float, str | None]:
        text = " ".join(seg[0] for seg in segments).strip()
        x1 = min(seg[1] for seg in segments)
        y1 = min(seg[2] for seg in segments)
        x2 = max(seg[1] + seg[3] for seg in segments)
        y2 = max(seg[2] + seg[4] for seg in segments)
        map_payload = []
        for seg_text, sx, sy, sw, sh in segments[:40]:
            map_payload.append(
                {
                    "t": seg_text[:200],
                    "x1": round(float(sx), 2),
                    "y1": round(float(sy), 2),
                    "x2": round(float(sx + sw), 2),
                    "y2": round(float(sy + sh), 2),
                }
            )
        segment_map = json.dumps(map_payload) if map_payload else None
        return (text, x1, y1, x2, y2, segment_map)

    @staticmethod
    def _tail_overlap_segments(
        segments: list[tuple[str, float, float, float, float]],
        overlap_chars: int,
    ) -> tuple[list[tuple[str, float, float, float, float]], int]:
        if overlap_chars <= 0:
            return ([], 0)
        collected: list[tuple[str, float, float, float, float]] = []
        chars = 0
        for seg in reversed(segments):
            seg_len = len(seg[0])
            extra = seg_len + (1 if chars > 0 else 0)
            if chars + extra > overlap_chars and collected:
                break
            collected.append(seg)
            chars += extra
            if chars >= overlap_chars:
                break
        collected.reverse()
        return (collected, chars)

    @staticmethod
    def _count_pdf_paths(root: Path) -> int:
        count = 0
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".pdf":
                count += 1
        return count

    @staticmethod
    def _iter_pdf_paths(root: Path):
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".pdf":
                yield path.resolve()
