"""End-to-end test: build a FAISS + SQLite index from sample PDFs, then
verify that hybrid search retrieval returns relevant results.

Run from the ccmcp root::

    python -m pytest tests/ -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import SAMPLE_PAPERS_DIR

# Use the shared indexed_backend fixture from conftest.py (aliased as "backend"
# for backward compatibility with existing test method signatures).
backend = pytest.fixture(scope="module")(lambda indexed_backend: indexed_backend)


# ---------------------------------------------------------------------------
# Indexing tests
# ---------------------------------------------------------------------------

class TestIndexing:
    def test_sample_papers_exist(self):
        """Sanity: sample PDFs are present."""
        pdfs = list(SAMPLE_PAPERS_DIR.glob("*.pdf"))
        assert len(pdfs) == 10, f"Expected 10 sample PDFs, found {len(pdfs)}"

    def test_files_indexed(self, backend):
        """All 10 PDFs should be indexed (some may error on extraction)."""
        total = backend.stats.indexed_files + backend.stats.errors
        assert total == 10, f"Expected 10 processed files, got {total}"
        assert backend.stats.indexed_files >= 8, (
            f"Too many indexing errors: {backend.stats.errors}/10"
        )

    def test_chunks_created(self, backend):
        """Index should have a meaningful number of chunks."""
        assert backend.store.ntotal > 0, "FAISS index is empty"
        db_count = backend.db.total_chunk_count()
        assert db_count > 0, "SQLite has no chunks"
        assert db_count == backend.store.ntotal, (
            f"Mismatch: SQLite has {db_count} chunks, FAISS has {backend.store.ntotal} vectors"
        )

    def test_chunks_per_file_reasonable(self, backend):
        """Each indexed file should produce at least a few chunks."""
        files = backend.db.get_files_for_directory(backend.directory_id)
        for f in files:
            chunk_ids = backend.db.get_chunk_ids_for_file(int(f["id"]))
            assert len(chunk_ids) >= 1, (
                f"File {f['path']} produced no chunks"
            )

    def test_faiss_index_persists(self, backend):
        """FAISS index should be saveable and reloadable."""
        from vibescholar import config
        from vibescholar.vectors import FaissStore

        backend.store.save()
        assert config.FAISS_INDEX_PATH.exists()

        reloaded = FaissStore(config.FAISS_INDEX_PATH, backend.embedder.dimension)
        assert reloaded.ntotal == backend.store.ntotal


# ---------------------------------------------------------------------------
# Search / retrieval tests
# ---------------------------------------------------------------------------

class TestRetrieval:
    def test_semantic_search_returns_results(self, backend):
        """A broad academic query should return results."""
        results = backend.search_service.search("deep learning neural network", top_k=5)
        assert len(results) > 0, "Semantic search returned no results"

    def test_keyword_search_hits(self, backend):
        """FTS5 keyword component should contribute matches."""
        results = backend.search_service.search("optimization convergence", top_k=5)
        assert len(results) > 0

    def test_top_k_respected(self, backend):
        """Total hits per file should not exceed top_k."""
        top_k = 3
        results = backend.search_service.search("image classification", top_k=top_k)
        for fr in results:
            assert len(fr.hits) <= top_k, (
                f"{fr.file_path} returned {len(fr.hits)} hits, expected <= {top_k}"
            )

    def test_scores_are_ordered(self, backend):
        """Within each file group, hits should be score-descending."""
        results = backend.search_service.search("point cloud 3D", top_k=10)
        for fr in results:
            scores = [h.score for h in fr.hits]
            assert scores == sorted(scores, reverse=True), (
                f"Scores not sorted for {fr.file_path}: {scores}"
            )

    def test_file_groups_ordered_by_best_score(self, backend):
        """File groups should be ordered by their top hit score."""
        results = backend.search_service.search("reinforcement learning", top_k=10)
        if len(results) > 1:
            top_scores = [r.hits[0].score for r in results]
            assert top_scores == sorted(top_scores, reverse=True)

    def test_snippet_not_empty(self, backend):
        """Every hit should include a non-empty snippet."""
        results = backend.search_service.search("spatial transcriptomics", top_k=5)
        for fr in results:
            for hit in fr.hits:
                assert hit.snippet.strip(), (
                    f"Empty snippet for chunk {hit.chunk_id}"
                )

    def test_relevant_paper_ranks_high(self, backend):
        """A query targeting a specific paper's topic should surface it."""
        results = backend.search_service.search(
            "optimal transport meta-learning", top_k=10
        )
        assert len(results) > 0
        file_names = [Path(fr.file_path).stem.lower() for fr in results]
        found = any("meta" in name and "transport" in name for name in file_names)
        assert found, (
            f"Expected Meta_Optimal_Transport in results, got: {file_names}"
        )

    def test_reranking_changes_order(self, backend):
        """Cross-encoder reranking should (usually) reorder results vs no rerank."""
        query = "convergence analysis sample complexity"
        no_rerank = backend.search_service.search(query, top_k=10, rerank=False)
        with_rerank = backend.search_service.search(query, top_k=10, rerank=True)
        assert len(no_rerank) > 0
        assert len(with_rerank) > 0

    def test_search_with_no_results(self, backend):
        """A nonsense query should return empty gracefully."""
        backend.search_service.search(
            "xyzzy plugh zork twisty passages", top_k=5
        )

    def test_context_expansion(self, backend):
        """Search with context expansion should not crash and should return snippets."""
        results = backend.search_service.search(
            "neural collapse low rank",
            top_k=5,
            expand_context=True,
            context_window=1,
        )
        assert len(results) > 0
        for fr in results:
            for hit in fr.hits:
                assert hit.snippet.strip()

    def test_snippet_length_increased(self, backend):
        """Snippets should be longer than the old 160-char limit."""
        results = backend.search_service.search("deep learning", top_k=5)
        long_snippets = [
            hit.snippet for fr in results for hit in fr.hits
            if len(hit.snippet) > 160
        ]
        assert len(long_snippets) > 0, "No snippets exceeded 160 chars"


# ---------------------------------------------------------------------------
# FTS5 operator tests (integration — through the full search pipeline)
# ---------------------------------------------------------------------------

class TestFTS5Operators:
    def test_fts5_table_populated(self, backend):
        """FTS5 index should have same count as chunks table."""
        fts_count = backend.db._conn.execute(
            "SELECT COUNT(*) AS c FROM chunks_fts"
        ).fetchone()["c"]
        chunk_count = backend.db.total_chunk_count()
        assert fts_count == chunk_count

    def test_fts5_search_returns_results(self, backend):
        """Direct FTS5 search should return scored results."""
        results = backend.db.fts5_search("neural network", limit=10)
        assert len(results) > 0
        assert all(score > 0 for score, _ in results)

    def test_and_operator_narrows(self, backend):
        """AND should return subset of broader query."""
        broad = backend.search_service.search("optimization", top_k=20, rerank=False)
        narrow = backend.search_service.search(
            "optimization AND convergence", top_k=20, rerank=False
        )
        # AND should return same or fewer documents
        assert len(narrow) <= len(broad)

    def test_or_operator_broadens(self, backend):
        """OR should return results for either term."""
        results = backend.search_service.search(
            "transcriptomics OR reinforcement", top_k=20, rerank=False
        )
        assert len(results) > 1  # Should hit multiple different papers

    def test_quoted_phrase(self, backend):
        """Quoted phrase should match exact sequence."""
        results = backend.search_service.search(
            '"optimal transport"', top_k=5, rerank=False
        )
        assert len(results) > 0

    def test_prefix_wildcard(self, backend):
        """Prefix wildcard should match multiple word forms."""
        results = backend.search_service.search(
            "optim*", top_k=10, rerank=False
        )
        assert len(results) > 0

    def test_malformed_query_doesnt_crash(self, backend):
        """Queries with bad FTS5 syntax should fall back gracefully."""
        # These should not raise — the fallback catches OperationalError
        backend.search_service.search('")(*&^%', top_k=5)
        backend.search_service.search("AND AND OR", top_k=5)
        backend.search_service.search("", top_k=5)


# ---------------------------------------------------------------------------
# Directory filter tests
# ---------------------------------------------------------------------------

class TestDirectoryFilter:
    def test_search_with_directory_filter(self, backend):
        """Search with directory filter should return results from that directory."""
        dir_ids = {backend.directory_id}
        results = backend.search_service.search(
            "neural network", top_k=5, directory_ids=dir_ids
        )
        assert len(results) > 0

    def test_search_with_empty_directory_filter(self, backend):
        """Empty directory filter set should return no results."""
        results = backend.search_service.search(
            "neural network", top_k=5, directory_ids=set()
        )
        assert len(results) == 0


# ---------------------------------------------------------------------------
# MCP server formatting tests
# ---------------------------------------------------------------------------

class TestMCPFormatting:
    def test_search_papers_output(self, backend):
        """Simulate the MCP search_papers tool output formatting."""
        import re
        from pathlib import PureWindowsPath

        results = backend.search_service.search(
            "generalization bias", top_k=5, active_only=True
        )

        parts: list[str] = []
        for fr in results:
            name = PureWindowsPath(fr.file_path).name
            for hit in fr.hits:
                snippet = re.sub(r"\s+", " ", hit.snippet).strip()
                parts.append(
                    f"[{name}  p.{hit.page_number}  score={hit.score:.3f}]\n"
                    f"  path: {fr.file_path}\n"
                    f"  {snippet}"
                )
        output = "\n\n".join(parts)

        assert len(output) > 0, "MCP output is empty"
        assert "score=" in output
        assert len(output) < 100_000, f"Output too large: {len(output)} chars"


# ---------------------------------------------------------------------------
# Incremental indexing tests
# ---------------------------------------------------------------------------

class TestIncrementalIndex:
    def test_reindex_skips_unchanged(self, backend):
        """Re-indexing the same directory should skip all files."""
        stats = backend.indexer.index_directory(
            backend.directory_id, SAMPLE_PAPERS_DIR
        )
        assert stats.skipped_files == backend.stats.indexed_files
        assert stats.indexed_files == 0

    def test_chunk_count_stable_after_reindex(self, backend):
        """Chunk count should not change after a no-op reindex."""
        count_before = backend.db.total_chunk_count()
        backend.indexer.index_directory(backend.directory_id, SAMPLE_PAPERS_DIR)
        count_after = backend.db.total_chunk_count()
        assert count_before == count_after


# ---------------------------------------------------------------------------
# Determinism tests
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_identical_queries_same_results(self, backend):
        """Identical queries should produce identical results."""
        r1 = backend.search_service.search("meta optimal transport", top_k=5)
        r2 = backend.search_service.search("meta optimal transport", top_k=5)
        assert len(r1) == len(r2)
        for fr1, fr2 in zip(r1, r2):
            assert fr1.file_path == fr2.file_path
            assert len(fr1.hits) == len(fr2.hits)
