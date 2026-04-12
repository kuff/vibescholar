# VibeScholar

MCP server that gives Claude access to academic literature — Google Scholar search with Semantic Scholar enrichment, citation tracking, author profiles, and an optional local PDF corpus.

## Architecture

Two retrieval paths: **online search** (primary) for the wider academic literature, and an optional **local corpus** for personally-indexed PDFs.

```
Online:   Query → Google Scholar (headless Chromium) → S2 enrichment → Results
                                                                     ↘ PDF cascade → In-memory text extraction

Local:    Query → FAISS (semantic, HNSW) ─┐
                                          ├→ RRF fusion → Cross-encoder rerank → Results
          Query → FTS5  (keyword)  ───────┘
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

### Online (primary)

The primary research tools. Google Scholar results are transparently enriched with Semantic Scholar metadata (stable IDs, full abstracts, DOIs) so papers remain accessible across sessions.

#### `search_online(query, limit=10, detail="detailed", year_min=None, year_max=None, sort="relevance", offset=0)`

Primary search tool for literature discovery.

Supports Google Scholar operators: `"quoted phrases"`, `OR`, `-exclude`, `intitle:`, `author:`, `source:`.

```
search_online("neural radiance field")
search_online("diffusion models", year_min=2023, sort="date")
search_online("attention mechanism", limit=10, offset=10)   # page 2
search_online("author:hinton deep learning", detail="brief")
```

#### `fetch_paper(paper_id, pages=None)`

Retrieve and read the full text of a paper found via online search. Accepts Semantic Scholar IDs (from `search_online` results), DOIs (`DOI:10.1234/example`), or ArXiv IDs (`ArXiv:2401.12345`).

The PDF is fetched into memory and returned as context — it is **not** added to the local corpus. If no PDF can be obtained, returns the available metadata and abstract as fallback.

```
fetch_paper("204e3073870fae3d05bcbc2f6a8e263d9b72e776")
fetch_paper("DOI:10.48550/arXiv.1706.03762", pages=[1, 2])
fetch_paper("ArXiv:1706.03762")
```

#### `cited_by_online(title, limit=10, detail="detailed", paper_id="")`

Find papers that cite a given work. Provide the title; optionally pass `paper_id` from a prior `search_online` result to skip the initial title lookup.

```
cited_by_online("Attention Is All You Need", limit=5)
cited_by_online("Attention Is All You Need", paper_id="204e3073...")
```

#### `related_papers_online(title, limit=10, detail="detailed", paper_id="")`

Find related papers via Google Scholar's "Related articles" feature. Same `paper_id` shortcut as `cited_by_online`.

```
related_papers_online("Generative Adversarial Networks")
```

#### `author_papers_online(author, limit=20, detail="detailed")`

List a researcher's publications from their Google Scholar profile.

```
author_papers_online("Yann LeCun", limit=10)
```

#### `save_paper(paper, directory)`

Save a PDF to a specified directory for offline reference. Works for both local corpus papers (by filename/stem) and online papers (by Semantic Scholar ID, DOI, or ArXiv ID). Tries to resolve locally first, then falls back to online download.

```
save_paper("DOI:10.48550/arXiv.1706.03762", "/home/user/references")
```

### Local corpus (secondary)

Optional tools for searching a personally-indexed collection of PDFs. Use `index_papers` to add folders, then `search_local` to search within them.

#### `search_local(query, top_k=10, directory="", detail="detailed")`

Search your locally indexed PDF library. Hybrid semantic + keyword retrieval with cross-encoder reranking.

Supports FTS5 operators: `AND`, `OR`, `NOT`, `"quoted phrases"`, `prefix*`.

```
search_local("neural network AND optimization")
search_local('"optimal transport"', top_k=5, detail="brief")
search_local("deep learning", directory="CVPR")
```

#### `read_document(file_path, pages=None)`

Read full text from a paper in the local corpus. Accepts filename or stem.

```
read_document("paper.pdf", pages=[1, 2, 3])
read_document("Meta_Optimal_Transport")
```

#### `list_indexed()`

List all indexed directories with file counts, chunk counts, and status.

#### `index_papers(folder, force=False)`

Index a folder of PDFs into the local corpus. Scans for PDFs and makes them searchable via `search_local`. Already-indexed files are skipped unless `force=True`.

```
index_papers("/path/to/papers")
index_papers("/path/to/papers", force=True)
```

### Context management

All search and discovery tools accept a `detail` parameter:

- `"brief"` — compact output for scanning many results (titles, scores, authors only)
- `"detailed"` ��� full output with snippets (local) or abstracts (online)

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

## Running Tests

```bash
# Fast unit tests (~6s)
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
  online.py      # Google Scholar, S2 enrichment, PDF cascade
server.py        # MCP server entry point (10 tools)
```

## License

MIT
