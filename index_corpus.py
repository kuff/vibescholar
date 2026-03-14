"""Index all PDF directories in the cvprscrape corpus.

Usage (inside container on AI Cloud):
    python index_corpus.py /path/to/cvprscrape --data-dir /path/to/output
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Index PDF corpus for VibeScholar")
    parser.add_argument("corpus_dir", help="Root directory containing *_papers/ subdirs")
    parser.add_argument("--data-dir", required=True, help="Output directory for index files")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for embedding")
    parser.add_argument("--device-ids", type=str, default=None,
                        help="Comma-separated GPU device IDs (e.g. '0,1,2')")
    args = parser.parse_args()

    corpus = Path(args.corpus_dir)
    if not corpus.is_dir():
        print(f"Error: {corpus} is not a directory", file=sys.stderr)
        sys.exit(1)

    from vibescholar import config
    from vibescholar.db import IndexDatabase
    from vibescholar.embeddings import Embedder
    from vibescholar.vectors import FaissStore
    from vibescholar.indexer import PdfIndexer

    data_dir = Path(args.data_dir)
    config.configure(data_dir=data_dir)
    config.ensure_data_dirs()

    device_ids = [int(x) for x in args.device_ids.split(",")] if args.device_ids else None

    print(f"Data directory: {data_dir}")
    print(f"Loading embedding model (cuda={args.cuda}, device_ids={device_ids})...")
    t0 = time.time()
    embedder = Embedder(cuda=args.cuda, device_ids=device_ids)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    db = IndexDatabase(config.DB_PATH)
    store = FaissStore(config.FAISS_INDEX_PATH, embedder.dimension)

    indexer = PdfIndexer(db, embedder, store)

    # Find all *_papers directories
    paper_dirs = sorted(d for d in corpus.iterdir() if d.is_dir() and d.name.endswith("_papers"))
    print(f"\nFound {len(paper_dirs)} paper directories:")
    for d in paper_dirs:
        pdf_count = len(list(d.glob("*.pdf")))
        print(f"  {d.name}: {pdf_count} PDFs")

    total_t0 = time.time()
    for i, paper_dir in enumerate(paper_dirs, 1):
        print(f"\n[{i}/{len(paper_dirs)}] Indexing {paper_dir.name}...")
        dir_id = db.upsert_directory(str(paper_dir))

        t0 = time.time()
        stats = indexer.index_directory(dir_id, paper_dir)
        elapsed = time.time() - t0

        print(f"  Indexed: {stats.indexed_files} files, "
              f"Skipped: {stats.skipped_files}, "
              f"Errors: {stats.errors}, "
              f"Chunks: {stats.total_chunks}, "
              f"Time: {elapsed:.1f}s")

        # Save after each directory in case of interruption
        store.save()
        print(f"  FAISS saved ({store.ntotal} vectors total)")

    total_elapsed = time.time() - total_t0
    print(f"\nDone! Total time: {total_elapsed/60:.1f} minutes")
    print(f"Total chunks: {db.total_chunk_count()}")
    print(f"Total vectors: {store.ntotal}")
    print(f"DB size: {config.DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"FAISS size: {config.FAISS_INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
