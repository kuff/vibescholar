# VibeScholar

MCP server for semantic search over local academic PDF corpora. Exposes a hybrid retrieval pipeline to Claude Code via the Model Context Protocol.

## Architecture

```
Query ──→ FAISS (semantic, HNSW) ──┐
                                    ├→ RRF fusion → Cross-encoder rerank → Results
Query ──→ FTS5  (keyword)  ────────┘
```

**Stack:** FastEmbed (BAAI/bge-small-en-v1.5, 384-dim ONNX) · FAISS HNSW · SQLite FTS5 · FlashRank cross-encoder · FastMCP

## Setup

```bash
pip install -e .
```

### Index PDFs

```python
from vibescholar import config
from vibescholar.db import IndexDatabase
from vibescholar.embeddings import Embedder
from vibescholar.vectors import FaissStore
from vibescholar.indexer import PdfIndexer

config.configure()
config.ensure_data_dirs()

embedder = Embedder()
db = IndexDatabase(config.DB_PATH)
store = FaissStore(config.FAISS_INDEX_PATH, embedder.dimension)
indexer = PdfIndexer(db, embedder, store)

dir_id = db.upsert_directory("/path/to/pdfs")
indexer.index_directory(dir_id, Path("/path/to/pdfs"))
store.save()
```

### Configure as MCP server

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "vibescholar": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

## MCP Tools

### `search_papers(query, top_k=10, directory="")`

Search indexed PDFs using hybrid semantic + keyword retrieval with cross-encoder reranking.

Supports search operators: `AND`, `OR`, `NOT`, `"quoted phrases"`, `prefix*`.

```
search_papers("neural network AND optimization")
search_papers('"optimal transport"', top_k=5)
search_papers("deep learning", directory="CVPR")
```

### `read_document(file_path, pages=None)`

Extract text from a PDF in the corpus. Accepts full path, filename, or stem.

```
read_document("paper.pdf", pages=[1, 2, 3])
read_document("Meta_Optimal_Transport")
```

### `list_indexed()`

List all indexed directories and PDFs with status and chunk counts.

## CLI Testing

```bash
python query.py "your search query"
python query.py "your query" --top_k 10 --no-rerank
python query.py --stats
python query.py --stats --data-dir /path/to/custom/data
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VIBESCHOLAR_DATA_DIR` | `~/.vibescholar` | Data directory for DB, vectors, model cache |

Data directory layout:

```
~/.vibescholar/
  index.sqlite3      # SQLite database with FTS5
  vectors.faiss       # FAISS HNSW index
  vectors.deleted.npy # Soft-deleted vector IDs (if any)
  model_cache/        # FastEmbed + FlashRank model files
```

## Running Tests

```bash
python -m pytest tests/ -v                    # all 81 tests
python -m pytest tests/ -v -k "not indexed"   # fast unit tests only (~1s)
```

## Project Structure

```
vibescholar/
  config.py      # Path configuration & environment variables
  db.py          # SQLite schema, FTS5 full-text search, CRUD operations
  embeddings.py  # FastEmbed wrapper (BAAI/bge-small-en-v1.5)
  vectors.py     # FAISS HNSW vector store with soft deletion
  search.py      # Hybrid search pipeline (FAISS + FTS5 + RRF + reranking)
  reranker.py    # FlashRank cross-encoder wrapper
  indexer.py     # PDF text extraction, chunking, and embedding
  text.py        # Text utilities (clean_text, chunk_text)
server.py        # MCP server entry point (3 tools)
query.py         # CLI testing utility
tests/           # 81 tests across 6 test files
```
