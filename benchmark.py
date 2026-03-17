"""Benchmark search latency across the full VibeScholar pipeline.

Usage:
    python benchmark.py                    # default queries, top_k=5
    python benchmark.py --top_k 10         # custom top_k
    python benchmark.py --rounds 5         # repeat each query N times
    python benchmark.py --data-dir /path   # custom data directory
    python benchmark.py --no-rerank        # skip cross-encoder
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from vibescholar import config
from vibescholar.db import IndexDatabase
from vibescholar.embeddings import Embedder
from vibescholar.vectors import FaissStore
from vibescholar.search import SearchService
from vibescholar.reranker import CrossEncoderReranker
from vibescholar.text import clean_text

# Representative queries spanning different search behaviours
DEFAULT_QUERIES = [
    # Short keyword-style
    "attention mechanism",
    "object detection",
    "depth estimation",
    # Natural language
    "how does contrastive learning improve representation quality",
    "methods for handling occlusion in multi-object tracking",
    "self-supervised pretraining for medical image segmentation",
    # Specific / narrow
    "transformer encoder layer normalization",
    "YOLO anchor-free head design",
    # Broad / survey-style
    "survey of generative adversarial networks",
    "domain adaptation techniques for autonomous driving",
]


def _timed(fn, *args, **kwargs):
    """Run fn and return (result, elapsed_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, (time.perf_counter() - t0) * 1000


def benchmark_query(
    query: str,
    search_svc: SearchService,
    top_k: int,
    rerank: bool,
) -> dict:
    """Run a single query and return per-stage timings in ms."""
    store = search_svc.store
    embedder = search_svc.embedder
    db = search_svc.db

    request_k = min(max(top_k * 6, top_k), max(store.ntotal, top_k))
    rerank_pool_size_target = max(top_k * 3, 60)

    # 1. Embedding
    query_vector, t_embed = _timed(embedder.embed_texts, [query])
    query_vector = query_vector[0]

    # 2. FAISS search
    (scores, chunk_ids), t_faiss = _timed(store.search, query_vector, request_k)
    faiss_ranked = [
        (float(s), int(cid)) for s, cid in zip(scores, chunk_ids) if int(cid) >= 0
    ]

    # 3. FTS5 search
    fts5_ranked, t_fts5 = _timed(search_svc._fts5_search, query, request_k)

    # 4. RRF fusion
    def _fuse():
        if fts5_ranked:
            return SearchService._reciprocal_rank_fusion([faiss_ranked, fts5_ranked])
        return faiss_ranked

    fused, t_rrf = _timed(_fuse)

    pool_size = min(len(fused), rerank_pool_size_target)
    candidate_ids = [cid for _, cid in fused[:pool_size]]

    # 5. DB fetch (get_chunks_by_ids)
    rows, t_db = _timed(db.get_chunks_by_ids, candidate_ids, active_only=True)
    row_map = {int(row["chunk_id"]): row for row in rows}
    candidate_ids = [cid for cid in candidate_ids if cid in row_map]

    # 6. Cross-encoder reranking
    t_rerank = 0.0
    if rerank and candidate_ids:
        _, t_rerank = _timed(
            search_svc._cross_encoder_rerank,
            query, candidate_ids, row_map, len(candidate_ids),
        )

    # 7. Snippet generation (for all candidates)
    query_tokens = SearchService._tokens(query)

    def _snippets():
        for cid in candidate_ids[:top_k]:
            row = row_map.get(cid)
            if row:
                SearchService._ranked_preview(str(row["text"]), query)

    _, t_snippet = _timed(_snippets)

    t_total = t_embed + t_faiss + t_fts5 + t_rrf + t_db + t_rerank + t_snippet

    return {
        "query": query,
        "embed_ms": t_embed,
        "faiss_ms": t_faiss,
        "fts5_ms": t_fts5,
        "rrf_ms": t_rrf,
        "db_ms": t_db,
        "rerank_ms": t_rerank,
        "snippet_ms": t_snippet,
        "total_ms": t_total,
        "faiss_hits": len(faiss_ranked),
        "fts5_hits": len(fts5_ranked),
        "candidates": len(candidate_ids),
    }


def fmt(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.2f}s"
    return f"{ms:.1f}ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VibeScholar search")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3, help="Repeat each query N times")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--queries", nargs="*", help="Custom queries (overrides defaults)")
    args = parser.parse_args()

    config.configure(data_dir=Path(args.data_dir) if args.data_dir else None)
    config.ensure_data_dirs()

    queries = args.queries if args.queries else DEFAULT_QUERIES
    rerank = not args.no_rerank

    # --- Load models (timed) ---
    print("Loading models...", end=" ", flush=True)
    t0 = time.perf_counter()
    db = IndexDatabase(config.DB_PATH)
    embedder = Embedder()
    store = FaissStore(config.FAISS_INDEX_PATH, embedder.dimension)
    search_svc = SearchService(db, embedder, store)
    # Warm up reranker
    if rerank:
        search_svc._reranker = CrossEncoderReranker()
    t_load = time.perf_counter() - t0
    print(f"done ({t_load:.1f}s)")
    print(f"  Corpus: {store.ntotal:,} vectors, {db.total_chunk_count():,} chunks")
    print(f"  Settings: top_k={args.top_k}, rounds={args.rounds}, rerank={rerank}")
    print()

    # --- Warmup round (not counted) ---
    print("Warmup round...", end=" ", flush=True)
    for q in queries:
        benchmark_query(q, search_svc, args.top_k, rerank)
    print("done\n")

    # --- Benchmark rounds ---
    all_results: list[dict] = []
    stages = ["embed_ms", "faiss_ms", "fts5_ms", "rrf_ms", "db_ms", "rerank_ms", "snippet_ms", "total_ms"]

    for round_num in range(1, args.rounds + 1):
        print(f"--- Round {round_num}/{args.rounds} ---")
        for q in queries:
            result = benchmark_query(q, search_svc, args.top_k, rerank)
            all_results.append(result)
            print(f"  {fmt(result['total_ms']):>8s}  {q[:60]}")
        print()

    # --- Per-query summary ---
    print("=" * 90)
    print("PER-QUERY AVERAGES (across rounds)")
    print("=" * 90)
    header = f"{'Query':<50s} {'Embed':>7s} {'FAISS':>7s} {'FTS5':>7s} {'RRF':>7s} {'DB':>7s} {'Rerank':>7s} {'Snip':>7s} {'TOTAL':>8s}"
    print(header)
    print("-" * len(header))

    for q in queries:
        q_results = [r for r in all_results if r["query"] == q]
        row = f"{q[:50]:<50s}"
        for stage in stages:
            vals = [r[stage] for r in q_results]
            row += f" {fmt(statistics.mean(vals)):>7s}"
        print(row)

    # --- Aggregate summary ---
    print()
    print("=" * 90)
    print("AGGREGATE STATISTICS")
    print("=" * 90)

    print(f"{'Stage':<12s} {'Mean':>8s} {'Median':>8s} {'P95':>8s} {'Min':>8s} {'Max':>8s} {'% Total':>8s}")
    print("-" * 70)

    total_means = {}
    for stage in stages:
        vals = [r[stage] for r in all_results]
        mean = statistics.mean(vals)
        total_means[stage] = mean
        med = statistics.median(vals)
        sorted_vals = sorted(vals)
        p95 = sorted_vals[int(len(sorted_vals) * 0.95)]
        mn = min(vals)
        mx = max(vals)
        print(f"{stage:<12s} {fmt(mean):>8s} {fmt(med):>8s} {fmt(p95):>8s} {fmt(mn):>8s} {fmt(mx):>8s}")

    # Reprint with percentages
    total_mean = total_means["total_ms"]
    print()
    print("Stage breakdown (% of total):")
    for stage in stages[:-1]:  # skip total_ms
        pct = (total_means[stage] / total_mean * 100) if total_mean > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {stage:<12s} {pct:5.1f}%  {bar}")

    print(f"\n  Total queries: {len(all_results)}")
    print(f"  Mean search time: {fmt(total_mean)}")
    print(f"  Throughput: {1000/total_mean:.1f} queries/sec" if total_mean > 0 else "")


if __name__ == "__main__":
    main()
