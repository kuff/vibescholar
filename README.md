# VibeScholar

MCP server that gives Claude (or any MCP client) direct access to the academic literature. Searches Google Scholar via headless Chromium for full coverage of the academic web, enriches results with Semantic Scholar metadata for stable IDs and abstracts, fetches and reads full-text PDFs on demand, and optionally searches a locally indexed PDF corpus with hybrid semantic + keyword retrieval.

## What does this look like?

You ask Claude a research question; it calls VibeScholar's tools to find and read papers. Example session (illustrative):

```
You: Find recent papers on neural scaling laws and summarize the key results.

 ── Claude calls search_online("neural scaling laws", year_min=2023, sort="date") ──

  Found 10 papers on Google Scholar.

  1. [a1b2c3d4e5f6] "Scaling Data-Constrained Language Models" (2024)
     Muennighoff; Rush; Barak; Scao; Piktus et al.
     NeurIPS | Cited by: 189
     PDF: https://arxiv.org/pdf/2305.16264.pdf
     Snippet: We investigate scaling language models in data-constrained
     regimes ...

  2. [f7e8d9c0b1a2] "Observational Scaling Laws" (2024)
     Ruan; Maddison
     NeurIPS | Cited by: 74
     PDF: https://arxiv.org/pdf/2405.10938.pdf
     Snippet: ...

 ── Claude calls fetch_paper("a1b2c3d4e5f6") ──

  Paper: "Scaling Data-Constrained Language Models" (2024)
  Authors: Muennighoff; Rush; Barak; Scao; Piktus et al.
  Source: ArXiv open access PDF
  Pages: 1-42

  --- Page 1 ---
  Abstract. ...

Claude reads the full text and synthesizes an answer for you.
```

All 10 tools work this way — Claude decides when to call them based on your research question.

## Setup

Requires Python 3.11+.

```bash
pip install -e .
playwright install chromium
```

### Configure as MCP server

**Local (stdio transport, for Claude Code):**

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

Then expose via a reverse proxy or Tailscale Funnel. The server runs in stateless mode when using HTTP transport so it works behind proxies without session affinity. See [REMOTE-SETUP.md](REMOTE-SETUP.md) for a full walkthrough.

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

## Tools

### Online (primary)

Google Scholar results are enriched with Semantic Scholar metadata (stable IDs, full abstracts, DOIs) so papers remain accessible across sessions.

| Tool | What it does |
|------|-------------|
| `search_online` | Literature search with year filters, pagination, sort by relevance or date. Supports Scholar operators: `"quoted phrases"`, `OR`, `-exclude`, `intitle:`, `author:`, `source:`. |
| `fetch_paper` | Fetch and read the full text of a paper. Accepts S2 IDs, DOIs (`DOI:10.1234/...`), or ArXiv IDs (`ArXiv:2401.12345`). PDF is read in memory, not saved. |
| `cited_by_online` | Papers that cite a given work. |
| `related_papers_online` | Related papers via Scholar's "Related articles". |
| `author_papers_online` | A researcher's publications from their Scholar profile. |
| `save_paper` | Download a PDF to a local directory for offline reference. |

### Local corpus (optional)

For searching a personally-indexed collection of PDFs. Index a folder once, then search it with hybrid semantic + keyword retrieval and cross-encoder reranking.

| Tool | What it does |
|------|-------------|
| `search_local` | Hybrid search over indexed PDFs. Supports FTS5 operators: `AND`, `OR`, `NOT`, `"phrases"`, `prefix*`. |
| `read_document` | Read full text from a locally indexed paper. |
| `index_papers` | Index a folder of PDFs into the local corpus. |
| `list_indexed` | Show indexed directories with file and chunk counts. |

All search tools accept a `detail` parameter: `"brief"` for compact output (titles, authors, scores) or `"detailed"` for full output with abstracts or text snippets.

## Architecture

```
Online:   Query → Google Scholar (headless Chromium) → S2 enrichment → Results
                                                                     ↘ PDF cascade → In-memory text extraction

Local:    Query → FAISS (semantic, HNSW) ─┐
                                          ├→ RRF fusion → Cross-encoder rerank → Results
          Query → FTS5  (keyword)  ───────┘
```

**Stack:** FastEmbed (BAAI/bge-small-en-v1.5, 384-dim ONNX) · FAISS HNSW · SQLite FTS5 · FlashRank cross-encoder · Google Scholar (Playwright) · Semantic Scholar · Unpaywall · FastMCP

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

## Testing

```bash
pip install -e ".[test]"

# Fast unit tests (~6s)
python -m pytest tests/ -v --ignore=tests/test_index_and_retrieval.py

# Full suite including integration tests (~2 min)
python -m pytest tests/ -v
```

## License

MIT
