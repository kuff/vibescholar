# VibeScholar

MCP server that gives Claude access to a curated local corpus of top-tier venue papers and the wider academic literature via Google Scholar.

## Architecture

Two retrieval paths: a **curated local corpus** of vetted, top-tier research (CVPR, NeurIPS, ICML, etc.) and **online search** for supplementary context from the wider literature.

```
Local:    Query → FAISS (semantic, HNSW) ─┐
                                           ├→ RRF fusion → Cross-encoder rerank → Results
          Query → FTS5  (keyword)  ────────┘

Online:   Query → Google Scholar (headless Chromium) → PDF cascade → In-memory text extraction
```

**Stack:** FastEmbed (BAAI/bge-small-en-v1.5, 384-dim ONNX) · FAISS HNSW · SQLite FTS5 · FlashRank cross-encoder · Google Scholar (Playwright) · Semantic Scholar · Unpaywall · FastMCP

## Setup

Requires Python 3.11+.

```bash
pip install -e .
playwright install chromium
```

For running tests:

```bash
pip install -e ".[test]"
```

### Index PDFs

Use the CLI indexer for a directory of PDFs:

```bash
python index_corpus.py /path/to/pdfs
```

Options:

```
--data-dir DIR    Store the index in DIR instead of ~/.vibescholar
--cuda            Use GPU for embedding (requires onnxruntime-gpu)
--workers N       Parallel PDF extraction workers (default: CPU count)
```

Or use the Python API:

```python
from pathlib import Path
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

**Local (stdio transport, for Claude Code):**

Add to your Claude Code MCP settings:

```json
{
  "mcpServers": {
    "vibescholar": {
      "command": "python",
      "args": ["/path/to/server.py"],
      "env": {
        "VIBESCHOLAR_DATA_DIR": "/path/to/data"
      }
    }
  }
}
```

**Remote (HTTP transport, for access from other machines):**

```bash
python server.py --transport streamable-http --host 127.0.0.1 --port 8765
```

Then expose via a reverse proxy or Tailscale Funnel. The server runs in stateless mode when using HTTP transport so it works behind proxies without session affinity.

```json
{
  "mcpServers": {
    "vibescholar": {
      "type": "url",
      "url": "https://your-host/mcp"
    }
  }
}
```

## MCP Tools

### Curated local corpus

The local corpus is your vetted collection of top-tier venue work. Always search here first.

#### `search_papers(query, top_k=10, directory="", detail="detailed")`

Primary search tool. Hybrid semantic + keyword retrieval with cross-encoder reranking over the curated corpus.

Supports search operators: `AND`, `OR`, `NOT`, `"quoted phrases"`, `prefix*`.

```
search_papers("neural network AND optimization")
search_papers('"optimal transport"', top_k=5, detail="brief")
search_papers("deep learning", directory="CVPR")
```

#### `read_document(file_path, pages=None)`

Read full text from a paper in the curated corpus. Accepts full path, filename, or stem.

```
read_document("paper.pdf", pages=[1, 2, 3])
read_document("Meta_Optimal_Transport")
```

#### `list_indexed()`

List all directories and papers in the curated corpus with status and chunk counts.

### Online (supplementary)

Use online tools when the local corpus does not cover the topic, or when you need supplementary context, background, citations, or recent work not yet in the vetted collection.

#### `search_online(query, limit=10, detail="detailed")`

Search Google Scholar for papers beyond the curated corpus.

```
search_online("attention mechanism transformer", limit=5)
search_online("graph neural networks", detail="brief")
```

#### `fetch_paper(paper_id, pages=None)`

Retrieve and read the full text of a paper found via online search. Accepts Semantic Scholar IDs (from `search_online` results), DOIs (`DOI:10.1234/example`), or ArXiv IDs (`ArXiv:2401.12345`).

The PDF is fetched into memory and returned as context — it is **not** added to the local corpus. If no PDF can be obtained, returns the available metadata and abstract as fallback.

```
fetch_paper("204e3073870fae3d05bcbc2f6a8e263d9b72e776")
fetch_paper("DOI:10.48550/arXiv.1706.03762", pages=[1, 2])
fetch_paper("ArXiv:1706.03762")
```

#### `cited_by_online(title, limit=10, detail="detailed")`

Find papers that cite a given work. Works for both local corpus papers and any other paper — just provide the title.

```
cited_by_online("Attention Is All You Need", limit=5)
```

#### `related_papers_online(title, limit=10, detail="detailed")`

Find papers related to a given work via Google Scholar's "Related articles" feature.

```
related_papers_online("Generative Adversarial Networks")
```

#### `author_papers_online(author, limit=20, detail="detailed")`

List a researcher's publications from their Google Scholar profile.

```
author_papers_online("Yann LeCun", limit=10)
```

#### `save_paper(paper, directory)`

Save a PDF to a specified directory for offline reference. Works for both local corpus papers (by path/filename/stem) and online papers (by Semantic Scholar ID, DOI, or ArXiv ID). Tries to resolve locally first, then falls back to online download.

```
save_paper("attention", "/home/user/references")
save_paper("DOI:10.48550/arXiv.1706.03762", "/home/user/references")
```

### Context management

All search and discovery tools accept a `detail` parameter:

- `"brief"` — compact output for scanning many results (titles, scores, authors only)
- `"detailed"` — full output with snippets (local) or abstracts (online)

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VIBESCHOLAR_DATA_DIR` | `~/.vibescholar` | Data directory for DB, vectors, model cache |
| `S2_API_KEY` | *(none)* | Semantic Scholar API key for higher rate limits ([request one](https://www.semanticscholar.org/product/api#api-key-form)) |
| `VIBESCHOLAR_EMAIL` | `vibescholar@users.noreply.github.com` | Email sent with Unpaywall API requests |

Data directory layout:

```
~/.vibescholar/
  index.sqlite3      # SQLite database with FTS5
  vectors.faiss       # FAISS HNSW index
  vectors.deleted.npy # Soft-deleted vector IDs (if any)
  model_cache/        # FastEmbed + FlashRank model files
```

## CLI Testing

```bash
python query.py "your search query"
python query.py "your query" --top_k 10 --no-rerank
python query.py --stats

# Live test of online search (requires Playwright + internet)
python test_online_live.py "optimal transport"
python test_online_live.py "diffusion models" --fetch
python test_online_live.py "attention" --save /tmp/papers
```

## Benchmarking

```bash
python benchmark.py                    # default queries, top_k=5, 3 rounds
python benchmark.py --top_k 10         # custom top_k
python benchmark.py --rounds 5         # repeat each query N times
python benchmark.py --no-rerank        # skip cross-encoder reranking
python benchmark.py --queries "query1" "query2"  # custom queries
```

Reports per-stage latency breakdown (embedding, FAISS, FTS5, RRF, DB fetch, reranking, snippets) with mean/median/P95/min/max statistics.

## Running Tests

```bash
# Fast unit tests (~6s, 216 tests)
python -m pytest tests/ -v --ignore=tests/test_index_and_retrieval.py

# Full suite including integration tests (~2 min)
python -m pytest tests/ -v
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
  online.py      # Google Scholar search, S2/Unpaywall clients, PDF cascade
server.py        # MCP server entry point (9 tools)
index_corpus.py  # CLI corpus indexer with parallel extraction
query.py         # CLI search testing utility
benchmark.py     # Search latency benchmarking
tests/           # 216 tests across 10 test files
```

## License

MIT
