"""Quick CLI to test search queries against the indexed corpus.

Usage:
    python query.py "your search query"
    python query.py "your search query" --top_k 5
    python query.py --stats
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path, PureWindowsPath

from vibescholar import config
from vibescholar.db import IndexDatabase
from vibescholar.embeddings import Embedder
from vibescholar.vectors import FaissStore
from vibescholar.search import SearchService


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the VibeScholar corpus")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--stats", action="store_true", help="Show corpus stats only")
    parser.add_argument("--no-rerank", action="store_true", help="Skip cross-encoder reranking")
    parser.add_argument("--data-dir", type=str, help="Custom data directory")
    args = parser.parse_args()

    config.configure(data_dir=Path(args.data_dir) if args.data_dir else None)
    config.ensure_data_dirs()

    db = IndexDatabase(config.DB_PATH)

    if args.stats:
        dirs = db.list_directories()
        chunks = db.total_chunk_count()
        print(f"Chunks: {chunks:,}")
        for d in dirs:
            status = "active" if d["active"] else "inactive"
            print(f"  [{status}] {d['path']}  ({d['file_count']} files, {d['chunk_count']} chunks)")

        # Check index type
        if config.FAISS_INDEX_PATH.exists():
            import faiss
            idx = faiss.read_index(str(config.FAISS_INDEX_PATH))
            size_mb = config.FAISS_INDEX_PATH.stat().st_size / 1024 / 1024
            inner = type(faiss.downcast_index(idx.index)).__name__ if hasattr(idx, "index") else "?"
            print(f"\nFAISS: {idx.ntotal:,} vectors, {size_mb:.1f} MB on disk")
            print(f"  Index type: {type(idx).__name__} -> {inner}")

        # Check FTS5
        fts = db._conn.execute("SELECT COUNT(*) AS c FROM chunks_fts").fetchone()
        print(f"FTS5:  {fts['c']:,} rows")
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    print("Loading models...", end=" ", flush=True)
    t0 = time.perf_counter()
    embedder = Embedder()
    store = FaissStore(config.FAISS_INDEX_PATH, embedder.dimension)
    search_service = SearchService(db, embedder, store)
    print(f"done ({time.perf_counter() - t0:.1f}s)")

    print(f"Searching for: {args.query!r}  (top_k={args.top_k})\n")
    t0 = time.perf_counter()
    results = search_service.search(
        query=args.query, top_k=args.top_k, active_only=True, rerank=not args.no_rerank,
    )
    elapsed = time.perf_counter() - t0

    if not results:
        print("No results found.")
        return

    total_hits = sum(len(fr.hits) for fr in results)
    print(f"Found {total_hits} passages across {len(results)} documents ({elapsed*1000:.0f}ms)\n")

    for fr in results:
        name = PureWindowsPath(fr.file_path).name
        for hit in fr.hits:
            snippet = " ".join(hit.snippet.split())
            print(f"[{name}  p.{hit.page_number}  score={hit.score:.3f}]")
            print(f"  {snippet[:200]}")
            print()


if __name__ == "__main__":
    main()
