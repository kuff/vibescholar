"""Parallel corpus indexer for VibeScholar.

Parallelizes PDF text extraction across CPU workers, then batch-embeds
on GPU.  Much faster than sequential indexing for large corpora.

Usage:
    python index_corpus.py /path/to/corpus --data-dir /path/to/output --cuda --workers 16
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


# ── Worker function (runs in subprocess, no shared state) ────────────

def _extract_chunks(pdf_path: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Extract text from a PDF and split into chunks. Returns (path, chunks).

    Each chunk is (page_number, chunk_number, text).  Runs in a worker
    process — imports are local to avoid pickling issues.
    """
    CHUNK_SIZE = 1200
    OVERLAP = 200

    try:
        import re
        from pypdf import PdfReader
        from vibescholar.text import clean_text, chunk_text

        reader = PdfReader(pdf_path)
        chunks: list[tuple[int, int, str]] = []
        for page_idx, page in enumerate(reader.pages):
            raw = page.extract_text() or ""
            # Strip control chars and surrogate Unicode (invalid UTF-8 from math symbols)
            raw = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", raw)
            raw = raw.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="replace")
            text = clean_text(raw)
            if not text or len(text) < 10:
                continue
            page_chunks = chunk_text(text, CHUNK_SIZE, OVERLAP)
            for chunk_idx, chunk in enumerate(page_chunks):
                if chunk.strip():
                    chunks.append((page_idx + 1, chunk_idx, chunk))
        return (pdf_path, chunks)
    except Exception:
        return (pdf_path, [])


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel corpus indexer for VibeScholar")
    parser.add_argument("corpus_dir", help="Root directory containing *_papers/ subdirs")
    parser.add_argument("--data-dir", required=True, help="Output directory for index files")
    parser.add_argument("--cuda", action="store_true", help="Use GPU for embedding")
    parser.add_argument("--device-ids", type=str, default=None,
                        help="Comma-separated GPU device IDs (e.g. '0,1,2')")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel PDF extraction workers")
    parser.add_argument("--embed-batch", type=int, default=4096,
                        help="Embedding batch size")
    args = parser.parse_args()

    corpus = Path(args.corpus_dir)
    if not corpus.is_dir():
        print(f"Error: {corpus} is not a directory", file=sys.stderr)
        sys.exit(1)

    from vibescholar import config
    from vibescholar.db import IndexDatabase
    from vibescholar.embeddings import Embedder
    from vibescholar.vectors import FaissStore

    data_dir = Path(args.data_dir)
    device_ids = [int(x) for x in args.device_ids.split(",")] if args.device_ids else None

    config.configure(data_dir=data_dir)
    config.ensure_data_dirs()

    print(f"Data directory: {data_dir}")
    print(f"Workers: {args.workers}, Embed batch: {args.embed_batch}")
    print(f"Loading embedding model (cuda={args.cuda}, device_ids={device_ids})...")
    t0 = time.time()
    embedder = Embedder(cuda=args.cuda, device_ids=device_ids)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    db = IndexDatabase(config.DB_PATH)
    store = FaissStore(config.FAISS_INDEX_PATH, embedder.dimension)

    # Find all *_papers directories
    paper_dirs = sorted(d for d in corpus.iterdir() if d.is_dir() and d.name.endswith("_papers"))
    print(f"\nFound {len(paper_dirs)} paper directories:")
    for d in paper_dirs:
        pdf_count = len(list(d.glob("*.pdf")))
        print(f"  {d.name}: {pdf_count} PDFs")

    total_t0 = time.time()

    for dir_idx, paper_dir in enumerate(paper_dirs, 1):
        print(f"\n[{dir_idx}/{len(paper_dirs)}] {paper_dir.name}")

        # ── Phase 1: Scan files, skip unchanged ──────────────────────
        dir_id = db.upsert_directory(str(paper_dir))
        pdfs = sorted(paper_dir.glob("*.pdf"))
        to_index: list[tuple[int, Path]] = []
        skipped = 0

        for pdf in pdfs:
            existing = db.get_file_by_path(str(pdf))
            if existing and int(existing["mtime_ns"]) == pdf.stat().st_mtime_ns:
                skipped += 1
                continue
            file_id = db.upsert_file(dir_id, str(pdf), pdf.stat().st_mtime_ns, pdf.stat().st_size)
            # Remove old chunks if re-indexing a changed file
            old_chunks = db.delete_chunks_for_file(file_id)
            if old_chunks:
                store.remove_ids(old_chunks)
            to_index.append((file_id, pdf))

        if not to_index:
            print(f"  Skipped all {skipped} files (unchanged)")
            db.touch_directory_indexed(dir_id)
            continue

        print(f"  {len(to_index)} to index, {skipped} skipped")

        # ── Phase 2: Extract chunks in parallel (CPU workers) ────────
        t0 = time.time()
        file_id_map: dict[str, int] = {str(pdf): fid for fid, pdf in to_index}
        pdf_paths = [str(pdf) for _, pdf in to_index]

        results: dict[str, list[tuple[int, int, str]]] = {}
        errors = 0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_extract_chunks, p): p for p in pdf_paths}
            for future in as_completed(futures):
                path, chunks = future.result()
                if chunks:
                    results[path] = chunks
                else:
                    errors += 1

        extract_time = time.time() - t0
        total_chunks = sum(len(c) for c in results.values())
        print(f"  Extracted {total_chunks} chunks from {len(results)} files "
              f"({errors} errors) in {extract_time:.1f}s")

        if not results:
            db.touch_directory_indexed(dir_id)
            continue

        # ── Phase 3: Batch embed all chunks on GPU ───────────────────
        t0 = time.time()
        all_texts = [str(text) for chunks in results.values() for _, _, text in chunks]
        # Verify all texts are non-empty strings (guards against pickle corruption)
        for i, t in enumerate(all_texts):
            if not isinstance(t, str) or not t:
                all_texts[i] = "empty"

        import numpy as np
        # Debug: check for non-string values
        bad = [(i, type(t), repr(t)[:80]) for i, t in enumerate(all_texts) if not isinstance(t, str)]
        if bad:
            print(f"  WARNING: {len(bad)} non-string texts found, first: {bad[0]}")
        print(f"  Embedding {len(all_texts)} chunks in batches of {args.embed_batch}...")

        all_vectors_list = []
        for batch_start in range(0, len(all_texts), args.embed_batch):
            batch = all_texts[batch_start:batch_start + args.embed_batch]
            all_vectors_list.append(embedder.embed_texts(batch))

        all_vectors = np.concatenate(all_vectors_list, axis=0)
        embed_time = time.time() - t0
        print(f"  Embedded {len(all_texts)} chunks in {embed_time:.1f}s")

        # ── Phase 4: Insert chunks + vectors into DB + FAISS ─────────
        t0 = time.time()
        vector_idx = 0
        for path, chunks in results.items():
            file_id = file_id_map[path]
            rows = [(pn, cn, text, None, None, None, None, None)
                    for pn, cn, text in chunks]
            chunk_ids = db.insert_chunks(file_id, rows)
            n = len(chunk_ids)
            store.add_embeddings(all_vectors[vector_idx:vector_idx + n], chunk_ids)
            vector_idx += n

        db.touch_directory_indexed(dir_id)
        insert_time = time.time() - t0
        print(f"  Inserted in {insert_time:.1f}s")

        # Save after each directory
        store.save()
        print(f"  Total: {store.ntotal} vectors, "
              f"times: extract={extract_time:.0f}s embed={embed_time:.0f}s insert={insert_time:.0f}s")

    total_elapsed = time.time() - total_t0
    print(f"\nDone! Total time: {total_elapsed / 60:.1f} minutes")
    print(f"Total chunks: {db.total_chunk_count()}")
    print(f"Total vectors: {store.ntotal}")
    print(f"DB size: {config.DB_PATH.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"FAISS size: {config.FAISS_INDEX_PATH.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
