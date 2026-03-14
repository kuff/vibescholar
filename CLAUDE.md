# VibeScholar — Development Guide

## Quick Reference

- **Run tests:** `python -m pytest tests/ -v`
- **Run fast unit tests only:** `python -m pytest tests/ -v --ignore=tests/test_index_and_retrieval.py` (~1s)
- **Test a query:** `python query.py "your query" --data-dir /tmp/test`
- **Package:** `vibescholar/` — the library. `server.py` — the MCP entry point.

## Architecture

Hybrid search pipeline: FAISS (semantic) + FTS5 (keyword) → Reciprocal Rank Fusion → cross-encoder reranking.

- **FTS5 over BM25:** FTS5 is built into SQLite, zero startup cost, always in sync via triggers. BM25 was removed because it loaded all chunks into memory.
- **HNSW over flat index:** `IndexHNSWFlat` gives sub-linear search at scale. Soft deletion (HNSW can't hard-remove vectors) tracked in `_deleted_ids` set.
- **RRF fusion (k=60):** Combines FAISS and FTS5 rankings without score calibration.

## Key Locations

| What | Where |
|------|-------|
| MCP tools (search_papers, read_document, list_indexed) | `server.py` |
| FTS5 query parser (AND/OR/NOT/"phrases"/prefix*) | `vibescholar/search.py:_fts5_query()` |
| FTS5 virtual table + sync triggers | `vibescholar/db.py:_migrate()` |
| HNSW soft deletion logic | `vibescholar/vectors.py` |
| PDF chunking + extraction | `vibescholar/indexer.py` |
| Search pipeline orchestration | `vibescholar/search.py:SearchService.search()` |
| Query-level LRU cache (32 entries) | `server.py:_search_cache` |

## Important Implementation Details

- **FTS5 cascade deletes:** SQLite cascade deletes do NOT fire triggers. `delete_directory()` and `delete_file()` explicitly delete chunks first so FTS5 stays in sync.
- **Soft deletion persistence:** Deleted vector IDs are saved to `.deleted.npy` alongside the FAISS index. Search over-fetches and filters.
- **FTS5 query fallback:** If the parsed query causes an `OperationalError`, the search falls back to plain tokenized terms.

## Test Data

10 academic PDFs in `tests/test_papers/` (~36 MB). The `indexed_backend` fixture in `conftest.py` builds the full index once per module (takes ~2 min). Use `empty_db` fixture for isolated mutation tests.
