"""VibeScholar MCP Server

Gives Claude access to two tiers of academic literature:

1. **Curated local corpus** — a vetted collection of top-tier venue papers
   (CVPR, NeurIPS, ICML, etc.) indexed for fast hybrid search.  Always
   searched first.

2. **Online (supplementary)** — Google Scholar search, citation tracking,
   author profiles, and on-demand PDF retrieval for context beyond the
   corpus.  Papers fetched online are NOT added to the local index.

Environment variables
---------------------
VIBESCHOLAR_DATA_DIR  Path to the data directory (default: ~/.vibescholar)
S2_API_KEY            Optional Semantic Scholar API key (higher rate limits).
VIBESCHOLAR_EMAIL     Email for the Unpaywall API.
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
    """Lazily-initialized wrapper around the VibeScholar search stack.

    Only the database is opened eagerly (cheap).  The embedding model, FAISS
    index, and search service are loaded on first use so that lightweight
    operations like ``list_indexed`` stay fast.
    """

    def __init__(self) -> None:
        data_dir_env = os.environ.get("VIBESCHOLAR_DATA_DIR")
        data_dir = Path(data_dir_env) if data_dir_env else None

        from vibescholar import config

        config.configure(data_dir=data_dir)
        config.ensure_data_dirs()

        from vibescholar.db import IndexDatabase

        self._config = config
        self.db = IndexDatabase(config.DB_PATH)
        self._file_map = self._build_file_map()

        # Heavy components are loaded lazily via properties below.
        self._embedder = None
        self._store = None
        self._search_service = None

    def _ensure_search_stack(self) -> None:
        """Load the embedding model, FAISS index, and search service (once)."""
        if self._embedder is not None:
            return

        from vibescholar.embeddings import Embedder
        from vibescholar.vectors import FaissStore
        from vibescholar.search import SearchService

        logger.info("Loading embedding model...")
        self._embedder = Embedder()
        self._store = FaissStore(self._config.FAISS_INDEX_PATH, self._embedder.dimension)
        self._search_service = SearchService(self.db, self._embedder, self._store)
        logger.info("Search stack ready (%d vectors)", self._store.ntotal)

    @property
    def embedder(self):
        self._ensure_search_stack()
        return self._embedder

    @property
    def store(self):
        self._ensure_search_stack()
        return self._store

    @property
    def search_service(self):
        self._ensure_search_stack()
        return self._search_service

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
    """Collapse whitespace.  Delegates to ``vibescholar.text.clean_text``."""
    from vibescholar.text import clean_text

    return clean_text(text)


# ── Query cache ──────────────────────────────────────────────────────
_search_cache: OrderedDict[tuple[str, int, str, str], str] = OrderedDict()
_SEARCH_CACHE_MAX = 32


# ── Tools ───────────────────────────────────────────────────────────


@mcp.tool()
def search_papers(
    query: str,
    top_k: int = 10,
    directory: str = "",
    detail: str = "detailed",
) -> str:
    """Search the curated local corpus of vetted, top-tier venue papers.

    This is the primary search tool.  The corpus contains hand-selected work
    from top venues (CVPR, NeurIPS, ICML, etc.) and should be searched first
    for any question about topics it covers.  Uses hybrid semantic + keyword
    retrieval with cross-encoder reranking for high-quality results.

    Use read_document on promising hits to get full page text.

    Query syntax (FTS5 operators — combine freely):
      - Natural language: "optimal transport for domain adaptation"
      - AND: "transformer AND attention" (both terms required)
      - OR: "GAN OR diffusion" (either term)
      - NOT: "segmentation NOT medical" (exclude term)
      - "quoted phrase": '"optimal transport"' (exact phrase match)
      - prefix*: "optim*" (matches optimize, optimization, optimal, etc.)
      - Combined: '"optimal transport" AND generative NOT video'

    Args:
        query: Natural language or structured query using operators above.
        top_k: Maximum hits to return (default 10).
        directory: Filter to a specific directory by name substring (e.g. "CVPR").
                   Omit to search the entire corpus.
        detail: "brief" for titles and scores only, "detailed" (default) includes
                text snippets. Use brief to scan many results without filling context.
    """
    cache_key = (query, top_k, directory, detail)
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

    brief = detail == "brief"
    parts: list[str] = []
    for fr in results:
        name = PureWindowsPath(fr.file_path).name
        for hit in fr.hits:
            if brief:
                parts.append(
                    f"[{name}  p.{hit.page_number}  score={hit.score:.3f}]"
                    f"  path: {fr.file_path}"
                )
            else:
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
    """Read full text from a paper in the curated local corpus.

    Accepts full path, filename, or stem.  Use after search_papers to read
    full page content of relevant hits from the vetted collection.

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
    """List all directories and papers in the curated local corpus.

    Shows indexed directories, file counts, chunk counts, and status.
    Use to discover what vetted material is available before searching.
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
    parts.append(f"Totals: {total_chunks} chunks")

    return "\n\n".join(parts)


# ── Online search caches ───────────────────────────────────────────

_online_search_cache: OrderedDict[tuple[str, int, str], str] = OrderedDict()
_ONLINE_SEARCH_CACHE_MAX = 32

# paper_id -> PaperResult  (avoids re-fetching between search_online and fetch_paper)
from vibescholar.online import PaperResult as _PaperResult  # noqa: E402

_paper_cache: OrderedDict[str, _PaperResult] = OrderedDict()
_PAPER_CACHE_MAX = 64


def _cache_paper(paper: _PaperResult) -> None:
    _paper_cache[paper.paper_id] = paper
    _paper_cache.move_to_end(paper.paper_id)
    while len(_paper_cache) > _PAPER_CACHE_MAX:
        _paper_cache.popitem(last=False)


def _format_authors(authors: list[str], brief: bool = False) -> str:
    if not authors:
        return "Unknown authors"
    if brief and len(authors) > 3:
        return "; ".join(authors[:3]) + " et al."
    if len(authors) > 6:
        return "; ".join(authors[:6]) + " et al."
    return "; ".join(authors)


# ── Online tools ───────────────────────────────────────────────────


def _format_online_results(
    papers: list[_PaperResult], detail: str, heading: str
) -> str:
    """Format a list of PaperResult objects into a human-readable string."""
    brief = detail == "brief"
    parts: list[str] = []
    for i, p in enumerate(papers, 1):
        year_str = str(p.year) if p.year else "n.d."
        authors_str = _format_authors(p.authors, brief=brief)

        if brief:
            venue_part = f" | {p.venue}" if p.venue else ""
            parts.append(
                f"{i}. [{p.paper_id}] \"{p.title}\" ({year_str})\n"
                f"   {authors_str}{venue_part} | Cited by: {p.citation_count}"
            )
        else:
            lines = [
                f"{i}. [{p.paper_id}] \"{p.title}\" ({year_str})",
                f"   {authors_str}",
            ]
            meta_parts = []
            if p.venue:
                meta_parts.append(p.venue)
            meta_parts.append(f"Cited by: {p.citation_count}")
            lines.append(f"   {' | '.join(meta_parts)}")

            if p.open_access_url:
                lines.append(f"   PDF: {p.open_access_url}")

            if p.abstract:
                lines.append(f"   Snippet: {p.abstract}")

            parts.append("\n".join(lines))

    return f"Found {len(papers)} {heading}.\n\n" + "\n\n".join(parts)


@mcp.tool()
async def search_online(
    query: str,
    limit: int = 10,
    detail: str = "detailed",
) -> str:
    """Search Google Scholar for papers beyond the curated local corpus.

    Use when the local corpus (search_papers) does not cover the topic, or
    when you need supplementary context, background, or recent work not yet
    in the vetted collection.  Returns metadata and snippets for discovery.

    Use fetch_paper on interesting results to retrieve and read the full text.

    Query syntax (Google Scholar operators — combine freely):
      - Natural language: "optimal transport for generative models"
      - "quoted phrase": '"neural radiance field"' (exact phrase match)
      - OR: "GAN OR diffusion" (either term; AND is implicit)
      - Exclude: "-survey" (exclude term)
      - Title only: "intitle:transformer" (term must appear in title)
      - Author: "author:hinton" (filter by author name)
      - Source: "source:NeurIPS" (filter by venue/journal)
      - Combined: '"optimal transport" author:cuturi -survey'
      - Year filtering is not supported via query syntax; use Scholar's UI.

    Args:
        query: Natural language or structured query using operators above.
        limit: Maximum number of results (default 10, max 20).
        detail: "brief" for titles/authors/year only, "detailed" (default)
                includes snippets and PDF links.
    """
    cache_key = (query, limit, detail)
    if cache_key in _online_search_cache:
        _online_search_cache.move_to_end(cache_key)
        return _online_search_cache[cache_key]

    from vibescholar.online import search_google_scholar

    papers = await search_google_scholar(query, limit)
    if not papers:
        return f"No results found on Google Scholar for: {query}"

    for p in papers:
        _cache_paper(p)

    result = _format_online_results(papers, detail, "papers on Google Scholar")

    _online_search_cache[cache_key] = result
    if len(_online_search_cache) > _ONLINE_SEARCH_CACHE_MAX:
        _online_search_cache.popitem(last=False)

    return result


@mcp.tool()
async def fetch_paper(paper_id: str, pages: list[int] | None = None) -> str:
    """Retrieve and read the full text of a paper found via online search.

    Use after search_online, cited_by_online, related_papers_online, or
    author_papers_online to get the actual content of a paper that is not
    in the local corpus.  Accepts Semantic Scholar IDs, DOIs (DOI:xxx), or
    ArXiv IDs (ArXiv:xxx).  Tries multiple sources to find an accessible PDF.

    The PDF is processed in memory and returned as context — it is NOT
    added to the curated local corpus.

    Args:
        paper_id: Paper identifier, e.g. "abc123def", "DOI:10.1234/example",
                  or "ArXiv:2401.12345".
        pages: 1-based page numbers to extract. Omit to read the entire paper.
    """
    from vibescholar.online import fetch_paper_pdf_text, get_paper_metadata

    # Resolve metadata (check cache first)
    paper = _paper_cache.get(paper_id)
    if paper is None:
        paper = await get_paper_metadata(paper_id)
    if paper is None:
        return f"Paper not found on Semantic Scholar: {paper_id}"

    _cache_paper(paper)

    # Attempt PDF download and extraction
    text, source = await fetch_paper_pdf_text(paper, pages)

    if text:
        year_str = str(paper.year) if paper.year else "n.d."
        header = (
            f"Paper: \"{paper.title}\" ({year_str})\n"
            f"Authors: {_format_authors(paper.authors)}\n"
            f"Source: {source}\n"
        )
        return header + "\n" + text

    # Graceful degradation: return whatever metadata we have
    lines = [f"Could not download PDF for \"{paper.title}\"."]

    meta = []
    if paper.year:
        meta.append(f"Year: {paper.year}")
    if paper.venue:
        meta.append(f"Venue: {paper.venue}")
    meta.append(f"Cited by: {paper.citation_count}")
    if paper.doi:
        meta.append(f"DOI: {paper.doi}")
    if paper.arxiv_id:
        meta.append(f"ArXiv: {paper.arxiv_id}")
    lines.append("\nAvailable metadata:\n  " + "\n  ".join(meta))

    if paper.abstract:
        lines.append(f"\nAbstract: {paper.abstract}")

    return "\n".join(lines)


@mcp.tool()
async def cited_by_online(title: str, limit: int = 10, detail: str = "detailed") -> str:
    """Find papers that cite a given work, via Google Scholar.

    Use to trace the impact of a paper or find follow-up work that builds
    on it.  Works for both papers in the local corpus and any other paper —
    just provide the title.

    Args:
        title: Title (or distinctive phrase) of the paper to find citations for.
        limit: Maximum number of citing papers to return (default 10, max 20).
        detail: "brief" for titles/authors/year only, "detailed" (default)
                includes snippets and PDF links.
    """
    from vibescholar.online import cited_by

    papers = await cited_by(title, limit)
    if not papers:
        return f"No citing papers found for: {title}"

    for p in papers:
        _cache_paper(p)

    return _format_online_results(papers, detail, f"papers citing \"{title}\"")


@mcp.tool()
async def related_papers_online(title: str, limit: int = 10, detail: str = "detailed") -> str:
    """Find papers related to a given work, via Google Scholar.

    Use to discover alternative approaches, concurrent work, or papers in
    the same research area.  Works for both local corpus papers and any
    other paper — just provide the title.

    Args:
        title: Title (or distinctive phrase) of the paper.
        limit: Maximum number of related papers to return (default 10, max 20).
        detail: "brief" for titles/authors/year only, "detailed" (default)
                includes snippets and PDF links.
    """
    from vibescholar.online import related_papers

    papers = await related_papers(title, limit)
    if not papers:
        return f"No related papers found for: {title}"

    for p in papers:
        _cache_paper(p)

    return _format_online_results(papers, detail, f"papers related to \"{title}\"")


@mcp.tool()
async def author_papers_online(author: str, limit: int = 20, detail: str = "detailed") -> str:
    """Find papers by a specific author via their Google Scholar profile.

    Use to explore a researcher's body of work — e.g. to find other relevant
    papers by an author encountered in the local corpus or online results.

    Args:
        author: Author name to search for (e.g. "Yann LeCun").
        limit: Maximum number of papers to return (default 20, max 100).
        detail: "brief" for titles/year only, "detailed" (default) includes
                venues and citation counts.
    """
    from vibescholar.online import author_papers

    papers = await author_papers(author, limit)
    if not papers:
        return f"No Google Scholar profile or papers found for: {author}"

    for p in papers:
        _cache_paper(p)

    return _format_online_results(papers, detail, f"papers by {author}")


@mcp.tool()
async def save_paper(paper: str, directory: str) -> str:
    """Save a PDF to a specified directory for offline reference.

    Works for both local corpus papers and online papers.  For local papers,
    pass a path, filename, or stem (same as read_document).  For online
    papers, pass a Semantic Scholar ID, DOI (DOI:xxx), or ArXiv ID (ArXiv:xxx).

    The tool tries to resolve the paper locally first.  If not found in the
    corpus, it treats the identifier as an online paper and downloads the PDF.

    Args:
        paper: Local file path/name/stem, or online paper identifier.
        directory: Destination directory path where the PDF will be saved.
    """
    import shutil

    dest_dir = Path(directory)
    if not dest_dir.is_dir():
        # Try WSL path conversion
        dest_dir = _Backend._to_wsl_path(directory)
        if not dest_dir.is_dir():
            return f"Directory does not exist: {directory}"

    # --- Try local corpus first ---
    backend = _get_backend()
    try:
        resolved = backend.resolve_path(paper)
        dest = dest_dir / resolved.name
        shutil.copy2(resolved, dest)
        return f"Saved local PDF to: {dest}"
    except FileNotFoundError:
        pass  # Not in corpus — try online

    # --- Online: resolve metadata ---
    from vibescholar.online import _download_pdf_bytes, get_paper_metadata

    paper_meta = _paper_cache.get(paper)
    if paper_meta is None:
        paper_meta = await get_paper_metadata(paper)
    if paper_meta is None:
        return f"Paper not found locally or on Semantic Scholar: {paper}"

    _cache_paper(paper_meta)

    # --- Download PDF ---
    pdf_bytes, source, tried = await _download_pdf_bytes(paper_meta)
    if pdf_bytes is None:
        reasons = "; ".join(f"{n}: {r}" for n, r in tried) if tried else "no sources available"
        return f"Could not download PDF for \"{paper_meta.title}\". Tried: {reasons}"

    # Build a safe filename from the title
    safe_title = re.sub(r"[^\w\s-]", "", paper_meta.title)
    safe_title = re.sub(r"\s+", "_", safe_title).strip("_")
    if not safe_title:
        safe_title = paper_meta.paper_id
    filename = f"{safe_title[:120]}.pdf"

    dest = dest_dir / filename
    dest.write_bytes(pdf_bytes)
    return (
        f"Saved \"{paper_meta.title}\" to: {dest}\n"
        f"Source: {source} | Size: {len(pdf_bytes):,} bytes"
    )


# ── Entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.transport != "stdio":
        mcp.settings.host = args.host
        mcp.settings.port = args.port
        # Allow connections from any host when serving over the network
        mcp.settings.transport_security.enable_dns_rebinding_protection = False
        # Stateless mode: no session affinity required (needed for Tailscale Funnel)
        mcp.settings.stateless_http = True

    mcp.run(transport=args.transport)
