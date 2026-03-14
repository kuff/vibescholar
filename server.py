"""VibeScholar MCP Server

Exposes a locally indexed PDF corpus to Claude Code via the Model Context
Protocol.  Provides semantic + keyword hybrid search with cross-encoder
reranking, page-level document reading, and corpus browsing.

Environment variables
---------------------
VIBESCHOLAR_DATA_DIR  Path to the data directory (default: ~/.vibescholar)
"""
from __future__ import annotations

import logging
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path, PureWindowsPath

# ── Logging (must go to stderr for stdio transport) ─────────────────
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(name)s: %(message)s",
)
logger = logging.getLogger("vibescholar-mcp")

# ── MCP server ──────────────────────────────────────────────────────
from mcp.server.fastmcp import FastMCP  # noqa: E402

mcp = FastMCP("vibescholar")


# ── Lazy backend ────────────────────────────────────────────────────
class _Backend:
    """Lazily-initialized wrapper around the VibeScholar search stack."""

    def __init__(self) -> None:
        data_dir_env = os.environ.get("VIBESCHOLAR_DATA_DIR")
        data_dir = Path(data_dir_env) if data_dir_env else None

        from vibescholar import config

        config.configure(data_dir=data_dir)
        config.ensure_data_dirs()

        from vibescholar.db import IndexDatabase
        from vibescholar.embeddings import Embedder
        from vibescholar.vectors import FaissStore
        from vibescholar.search import SearchService

        logger.info("Loading embedding model...")
        self.embedder = Embedder()
        self.db = IndexDatabase(config.DB_PATH)
        self.store = FaissStore(config.FAISS_INDEX_PATH, self.embedder.dimension)
        self.search_service = SearchService(self.db, self.embedder, self.store)

        self._file_map = self._build_file_map()
        logger.info("Backend ready (%d vectors)", self.store.ntotal)

    # ── helpers ──────────────────────────────────────────────────────

    def _build_file_map(self) -> dict[str, str]:
        """Map lowercase filename / stem -> absolute path for every indexed PDF."""
        mapping: dict[str, str] = {}
        for dir_row in self.db.list_directories():
            for file_row in self.db.get_files_for_directory(int(dir_row["id"])):
                full = str(file_row["path"])
                wp = PureWindowsPath(full)
                mapping[wp.name.lower()] = full
                mapping[wp.stem.lower()] = full
        return mapping

    @staticmethod
    def _to_wsl_path(path_str: str) -> Path:
        """Convert a Windows path to a WSL path if necessary."""
        if len(path_str) >= 3 and path_str[1] == ":" and path_str[2] == "\\":
            wp = PureWindowsPath(path_str)
            drive = wp.drive[0].lower()
            rest = wp.as_posix().split(":", 1)[1]
            return Path(f"/mnt/{drive}{rest}")
        return Path(path_str)

    def resolve_path(self, file_path: str) -> Path:
        """Resolve *file_path* to an existing PDF, accepting full paths, filenames,
        or stems (without extension)."""
        p = self._to_wsl_path(file_path)
        if p.exists():
            return p
        wp = PureWindowsPath(file_path)
        hit = self._file_map.get(wp.name.lower()) or self._file_map.get(
            wp.stem.lower()
        )
        if hit:
            return self._to_wsl_path(hit)
        raise FileNotFoundError(f"File not found in corpus: {file_path}")


_backend: _Backend | None = None


def _get_backend() -> _Backend:
    global _backend
    if _backend is None:
        _backend = _Backend()
    return _backend


def _clean(text: str) -> str:
    """Collapse whitespace."""
    return re.sub(r"\s+", " ", text).strip()


# ── Query cache ──────────────────────────────────────────────────────
_search_cache: OrderedDict[tuple[str, int, str], str] = OrderedDict()
_SEARCH_CACHE_MAX = 32


# ── Tools ───────────────────────────────────────────────────────────


@mcp.tool()
def search_papers(query: str, top_k: int = 10, directory: str = "") -> str:
    """Search indexed PDFs using hybrid semantic + keyword retrieval with cross-encoder reranking.

    Returns passages ranked by relevance, grouped by document. Use read_document
    on promising hits to get full page text.

    Supports search operators: AND, OR, NOT, "quoted phrases", prefix*.

    Args:
        query: Natural language search query.
        top_k: Maximum hits to return (default 10).
        directory: Filter to a specific directory by name substring (e.g. "CVPR").
                   Omit to search the entire corpus.
    """
    cache_key = (query, top_k, directory)
    if cache_key in _search_cache:
        _search_cache.move_to_end(cache_key)
        return _search_cache[cache_key]

    backend = _get_backend()

    dir_filter: set[int] | None = None
    if directory:
        all_dirs = backend.db.list_directories()
        dir_filter = {
            int(d["id"]) for d in all_dirs
            if directory.lower() in str(d["path"]).lower()
        }
        if not dir_filter:
            return f"No indexed directory matches '{directory}'. Use list_indexed to see available directories."

    results = backend.search_service.search(
        query=query, top_k=top_k, active_only=True, directory_ids=dir_filter
    )
    if not results:
        return "No results found in the corpus for this query."

    parts: list[str] = []
    for fr in results:
        name = PureWindowsPath(fr.file_path).name
        for hit in fr.hits:
            parts.append(
                f"[{name}  p.{hit.page_number}  score={hit.score:.3f}]\n"
                f"  path: {fr.file_path}\n"
                f"  {_clean(hit.snippet)}"
            )
    total_hits = sum(len(fr.hits) for fr in results)
    result = f"Found {total_hits} passages across {len(results)} documents.\n\n" + "\n\n".join(parts)

    _search_cache[cache_key] = result
    if len(_search_cache) > _SEARCH_CACHE_MAX:
        _search_cache.popitem(last=False)

    return result


@mcp.tool()
def read_document(file_path: str, pages: list[int] | None = None) -> str:
    """Extract text from a PDF in the corpus. Accepts full path, filename, or stem.

    Use after search_papers to read full page content of relevant hits.

    Args:
        file_path: Path, filename (e.g. "paper.pdf"), or stem (e.g. "paper").
        pages: 1-based page numbers to read. Omit to read all pages.
    """
    from pypdf import PdfReader

    backend = _get_backend()
    resolved = backend.resolve_path(file_path)

    reader = PdfReader(str(resolved))
    total = len(reader.pages)

    indices = (
        [p - 1 for p in pages if 1 <= p <= total] if pages else list(range(total))
    )

    texts: list[str] = []
    for idx in indices:
        raw = reader.pages[idx].extract_text() or ""
        cleaned = _clean(raw)
        if cleaned:
            texts.append(f"--- Page {idx + 1} ---\n{cleaned}")

    if not texts:
        return "No text could be extracted from the requested pages."

    header = f"Document: {resolved.name} ({total} pages total)\n\n"
    return header + "\n\n".join(texts)


@mcp.tool()
def list_indexed() -> str:
    """List all indexed directories and PDFs in the corpus with status and chunk counts.

    Use to discover available papers before searching.
    """
    backend = _get_backend()
    dirs = backend.db.list_directories()

    if not dirs:
        return "The corpus is empty — no directories have been indexed yet."

    parts: list[str] = []
    for d in dirs:
        status = "active" if d["active"] else "inactive"
        files = backend.db.get_files_for_directory(int(d["id"]))
        file_names = sorted(PureWindowsPath(f["path"]).name for f in files)

        section = (
            f"Directory: {d['path']}\n"
            f"  Status: {status} | Files: {d['file_count']} "
            f"| Chunks: {d['chunk_count']}\n"
            f"  Last indexed: {d['last_indexed_at'] or 'never'}"
        )
        if file_names:
            listing = "\n".join(f"    - {n}" for n in file_names)
            section += f"\n  Documents:\n{listing}"
        parts.append(section)

    total_chunks = backend.db.total_chunk_count()
    total_vectors = backend.store.ntotal
    parts.append(f"Totals: {total_chunks} chunks, {total_vectors} vectors")

    return "\n\n".join(parts)


# ── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
