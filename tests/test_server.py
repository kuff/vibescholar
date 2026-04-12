"""Tests for server.py MCP tool functions and helpers.

Uses a lightweight mock backend with a real IndexDatabase (tmp_path) but
mocked SearchService — no embedding model or FAISS index loaded.
"""

from __future__ import annotations

from pathlib import Path, PureWindowsPath
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vibescholar.search import FileSearchResult, SearchHit

TEST_PAPERS_DIR = Path(__file__).parent / "test_papers"


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_server_caches():
    """Clear all module-level caches between tests."""
    import server

    server._search_cache.clear()
    server._online_search_cache.clear()
    server._paper_cache.clear()
    yield
    server._search_cache.clear()
    server._online_search_cache.clear()
    server._paper_cache.clear()


@pytest.fixture
def mock_backend(tmp_path, monkeypatch):
    """Lightweight _Backend with real DB, fake search service."""
    from vibescholar import config

    config.configure(data_dir=tmp_path)
    config.ensure_data_dirs()

    from vibescholar.db import IndexDatabase

    db = IndexDatabase(config.DB_PATH)

    # Insert test data
    dir1_id = db.upsert_directory(r"C:\papers\CVPR")
    dir2_id = db.upsert_directory(r"C:\papers\NeurIPS")

    file1_id = db.upsert_file(dir1_id, r"C:\papers\CVPR\attention.pdf", mtime_ns=0, size_bytes=100)
    file2_id = db.upsert_file(dir1_id, r"C:\papers\CVPR\resnet.pdf", mtime_ns=0, size_bytes=200)
    file3_id = db.upsert_file(dir2_id, r"C:\papers\NeurIPS\gpt.pdf", mtime_ns=0, size_bytes=300)

    db.insert_chunks(file1_id, [
        (1, 0, "attention is all you need transformer", None, None, None, None, None),
        (2, 0, "self attention mechanism multi-head", None, None, None, None, None),
    ])
    db.insert_chunks(file2_id, [
        (1, 0, "deep residual learning for image recognition", None, None, None, None, None),
    ])
    db.insert_chunks(file3_id, [
        (1, 0, "generative pretrained transformer language model", None, None, None, None, None),
    ])

    # Build file_map
    file_map: dict[str, str] = {}
    for dir_row in db.list_directories():
        for file_row in db.get_files_for_directory(int(dir_row["id"])):
            full = str(file_row["path"])
            wp = PureWindowsPath(full)
            file_map[wp.name.lower()] = full
            file_map[wp.stem.lower()] = full

    # Build a duck-typed backend
    import server

    class FakeBackend:
        pass

    backend = FakeBackend()
    backend.db = db
    backend._file_map = file_map
    backend.search_service = MagicMock()

    # Bind resolve_path from real _Backend
    backend.resolve_path = lambda fp: server._Backend.resolve_path(backend, fp)
    # Make _to_wsl_path accessible
    backend._to_wsl_path = server._Backend._to_wsl_path

    monkeypatch.setattr(server, "_backend", backend)
    monkeypatch.setattr(server, "_get_backend", lambda: backend)

    # Store IDs for test assertions
    backend._dir1_id = dir1_id
    backend._dir2_id = dir2_id

    return backend


def _make_search_result(
    file_path: str = r"C:\papers\CVPR\attention.pdf",
    page: int = 1,
    score: float = 0.85,
    snippet: str = "attention is all you need",
) -> FileSearchResult:
    hit = SearchHit(
        chunk_id=1,
        file_path=file_path,
        page_number=page,
        score=score,
        snippet=snippet,
        bbox_x1=None, bbox_y1=None, bbox_x2=None, bbox_y2=None,
    )
    return FileSearchResult(file_path=file_path, hits=[hit])


def _make_paper_result(**overrides):
    from vibescholar.online import PaperResult

    defaults = dict(
        paper_id="abc123",
        title="Test Paper",
        authors=["Alice Smith", "Bob Jones"],
        year=2024,
        venue="NeurIPS",
        citation_count=42,
        abstract="This is a test abstract.",
        doi="10.1234/test",
        arxiv_id="2401.12345",
        open_access_url="https://example.com/paper.pdf",
        external_ids={"DOI": "10.1234/test"},
    )
    defaults.update(overrides)
    return PaperResult(**defaults)


# ── TestToWslPath ─────────────────────────────────────────────────


class TestToWslPath:
    def test_windows_c_drive(self):
        from server import _Backend

        result = _Backend._to_wsl_path(r"C:\Users\peter\paper.pdf")
        assert result == Path("/mnt/c/Users/peter/paper.pdf")

    def test_windows_d_drive(self):
        from server import _Backend

        result = _Backend._to_wsl_path(r"D:\data\file.pdf")
        assert result == Path("/mnt/d/data/file.pdf")

    def test_unix_passthrough(self):
        from server import _Backend

        result = _Backend._to_wsl_path("/mnt/c/file.pdf")
        assert result == Path("/mnt/c/file.pdf")

    def test_relative_passthrough(self):
        from server import _Backend

        result = _Backend._to_wsl_path("paper.pdf")
        assert result == Path("paper.pdf")

    def test_short_string(self):
        from server import _Backend

        result = _Backend._to_wsl_path("C")
        assert result == Path("C")


# ── TestResolvePath ───────────────────────────────────────────────


class TestResolvePath:
    def test_arbitrary_path_rejected(self, mock_backend, tmp_path):
        """Arbitrary filesystem paths outside the corpus are rejected."""
        real_file = tmp_path / "real.pdf"
        real_file.write_bytes(b"%PDF-1.4")
        with pytest.raises(FileNotFoundError, match="File not found in corpus"):
            mock_backend.resolve_path(str(real_file))

    def test_file_map_by_name(self, mock_backend):
        result = mock_backend.resolve_path("attention.pdf")
        assert "attention.pdf" in str(result)

    def test_file_map_by_stem(self, mock_backend):
        result = mock_backend.resolve_path("attention")
        assert "attention" in str(result)

    def test_not_found(self, mock_backend):
        with pytest.raises(FileNotFoundError, match="File not found in corpus"):
            mock_backend.resolve_path("nonexistent_paper.pdf")


# ── TestSearchLocal ───────────────────────────────────────────────


class TestSearchLocal:
    def test_basic_returns_formatted(self, mock_backend):
        from server import search_local

        mock_backend.search_service.search.return_value = [
            _make_search_result()
        ]
        result = search_local("attention")
        assert "Found 1 passages across 1 documents" in result
        assert "attention.pdf" in result
        assert "score=0.850" in result

    def test_cache_hit(self, mock_backend):
        from server import search_local

        mock_backend.search_service.search.return_value = [_make_search_result()]
        search_local("attention")
        search_local("attention")
        assert mock_backend.search_service.search.call_count == 1

    def test_cache_key_includes_detail(self, mock_backend):
        from server import search_local

        mock_backend.search_service.search.return_value = [_make_search_result()]
        search_local("attention", detail="brief")
        search_local("attention", detail="detailed")
        assert mock_backend.search_service.search.call_count == 2

    def test_brief_omits_snippets(self, mock_backend):
        from server import search_local

        mock_backend.search_service.search.return_value = [
            _make_search_result(snippet="this snippet should not appear in brief mode")
        ]
        result = search_local("attention", detail="brief")
        assert "this snippet should not appear" not in result
        # But filename and score are still present
        assert "attention.pdf" in result
        assert "score=" in result

    def test_detailed_includes_snippets(self, mock_backend):
        from server import search_local

        mock_backend.search_service.search.return_value = [
            _make_search_result(snippet="attention mechanism is powerful")
        ]
        result = search_local("attention", detail="detailed")
        assert "attention mechanism is powerful" in result

    def test_directory_filter_no_match(self, mock_backend):
        from server import search_local

        result = search_local("attention", directory="ICLR")
        assert "No indexed directory matches" in result
        mock_backend.search_service.search.assert_not_called()

    def test_directory_filter_match(self, mock_backend):
        from server import search_local

        mock_backend.search_service.search.return_value = [_make_search_result()]
        search_local("attention", directory="CVPR")
        call_kwargs = mock_backend.search_service.search.call_args
        dir_ids = call_kwargs.kwargs.get("directory_ids") or call_kwargs[1].get("directory_ids")
        assert mock_backend._dir1_id in dir_ids

    def test_no_results(self, mock_backend):
        from server import search_local

        mock_backend.search_service.search.return_value = []
        result = search_local("xyznonexistent")
        assert "No results found" in result


# ── TestReadDocument ──────────────────────────────────────────────


class TestReadDocument:
    def test_read_all_pages(self, mock_backend, monkeypatch):
        from server import read_document

        pdf_path = TEST_PAPERS_DIR / "Meta_Optimal_Transport.pdf"
        monkeypatch.setattr(mock_backend, "resolve_path", lambda fp: pdf_path)
        result = read_document("Meta_Optimal_Transport")
        assert result.startswith("Document: Meta_Optimal_Transport.pdf")
        assert "--- Page 1 ---" in result
        assert "--- Page 2 ---" in result

    def test_specific_pages(self, mock_backend, monkeypatch):
        from server import read_document

        pdf_path = TEST_PAPERS_DIR / "Meta_Optimal_Transport.pdf"
        monkeypatch.setattr(mock_backend, "resolve_path", lambda fp: pdf_path)
        result = read_document("Meta_Optimal_Transport", pages=[1, 2])
        assert "--- Page 1 ---" in result
        assert "--- Page 2 ---" in result
        assert "--- Page 3 ---" not in result

    def test_out_of_range_skipped(self, mock_backend, monkeypatch):
        from server import read_document

        pdf_path = TEST_PAPERS_DIR / "Meta_Optimal_Transport.pdf"
        monkeypatch.setattr(mock_backend, "resolve_path", lambda fp: pdf_path)
        result = read_document("Meta_Optimal_Transport", pages=[1, 9999])
        assert "--- Page 1 ---" in result
        # No crash, page 9999 silently skipped

    def test_file_not_found(self, mock_backend):
        from server import read_document

        with pytest.raises(FileNotFoundError):
            read_document("nonexistent_paper.pdf")

    def test_empty_extraction(self, mock_backend, monkeypatch):
        from server import read_document
        from unittest.mock import MagicMock as MM

        # Mock PdfReader to return pages with empty text
        mock_reader = MM()
        mock_page = MM()
        mock_page.extract_text.return_value = ""
        mock_reader.pages = [mock_page]

        monkeypatch.setattr(mock_backend, "resolve_path", lambda fp: Path("/fake/path.pdf"))
        monkeypatch.setattr("server.PdfReader", lambda path: mock_reader, raising=False)

        # Need to patch the import inside the function
        import server
        original = server.read_document.__wrapped__ if hasattr(server.read_document, '__wrapped__') else None

        # Patch pypdf.PdfReader at the import location
        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = read_document("path.pdf")
        assert "No text could be extracted" in result


# ── TestListIndexed ───────────────────────────────────────────────


class TestListIndexed:
    def test_empty_corpus(self, tmp_path, monkeypatch):
        """Empty DB should return 'corpus is empty' message."""
        from vibescholar import config

        config.configure(data_dir=tmp_path / "empty")
        config.ensure_data_dirs()

        from vibescholar.db import IndexDatabase

        empty_db = IndexDatabase(config.DB_PATH)

        import server

        class EmptyBackend:
            pass

        backend = EmptyBackend()
        backend.db = empty_db
        monkeypatch.setattr(server, "_backend", backend)
        monkeypatch.setattr(server, "_get_backend", lambda: backend)

        result = server.list_indexed()
        assert "corpus is empty" in result

    def test_directories_listed(self, mock_backend):
        from server import list_indexed

        result = list_indexed()
        assert "CVPR" in result
        assert "NeurIPS" in result
        assert "attention.pdf" in result
        assert "gpt.pdf" in result
        assert "Totals:" in result

    def test_inactive_shown(self, mock_backend):
        from server import list_indexed

        mock_backend.db.set_directory_active(mock_backend._dir2_id, False)
        result = list_indexed()
        assert "inactive" in result


# ── TestSearchOnline ──────────────────────────────────────────────


async def _passthrough_enrich(papers):
    """Enrich stub: return papers unchanged (no S2 API calls)."""
    return papers


class TestSearchOnline:
    @pytest.mark.asyncio
    async def test_basic_formatted(self, monkeypatch):
        import server

        papers = [_make_paper_result(), _make_paper_result(paper_id="def456", title="Paper Two")]

        async def mock_search(query, limit, **kwargs):
            return papers

        with patch("vibescholar.online.search_google_scholar", mock_search), \
             patch("vibescholar.online.enrich_with_s2", _passthrough_enrich):
            result = await server.search_online("attention", limit=10)

        assert "Found 2 papers" in result
        assert "Test Paper" in result
        assert "Paper Two" in result

    @pytest.mark.asyncio
    async def test_cache_hit(self, monkeypatch):
        import server

        call_count = {"n": 0}
        papers = [_make_paper_result()]

        async def mock_search(query, limit, **kwargs):
            call_count["n"] += 1
            return papers

        with patch("vibescholar.online.search_google_scholar", mock_search), \
             patch("vibescholar.online.enrich_with_s2", _passthrough_enrich):
            await server.search_online("attention", limit=10)
            await server.search_online("attention", limit=10)

        assert call_count["n"] == 1  # second call served from cache

    @pytest.mark.asyncio
    async def test_brief_omits_abstracts(self, monkeypatch):
        import server

        papers = [_make_paper_result(abstract="This detailed abstract should not appear")]

        async def mock_search(query, limit, **kwargs):
            return papers

        with patch("vibescholar.online.search_google_scholar", mock_search), \
             patch("vibescholar.online.enrich_with_s2", _passthrough_enrich):
            result = await server.search_online("test", detail="brief")

        assert "This detailed abstract" not in result
        assert "Test Paper" in result

    @pytest.mark.asyncio
    async def test_paper_cache_populated(self, monkeypatch):
        import server

        papers = [_make_paper_result(paper_id="cached123")]

        async def mock_search(query, limit, **kwargs):
            return papers

        with patch("vibescholar.online.search_google_scholar", mock_search), \
             patch("vibescholar.online.enrich_with_s2", _passthrough_enrich):
            await server.search_online("test")

        assert "cached123" in server._paper_cache

    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch):
        import server

        async def mock_search(query, limit, **kwargs):
            return []

        with patch("vibescholar.online.search_google_scholar", mock_search), \
             patch("vibescholar.online.enrich_with_s2", _passthrough_enrich):
            result = await server.search_online("xyznonexistent")

        assert "No results found" in result

    @pytest.mark.asyncio
    async def test_enrichment_called_and_stable_id_cached(self, monkeypatch):
        """enrich_with_s2 is called; enriched stable S2 IDs land in _paper_cache."""
        import server

        scholar_paper = _make_paper_result(paper_id="gs_0", title="Attention Is All You Need")
        enriched_paper = _make_paper_result(paper_id="s2stable123", title="Attention Is All You Need")

        async def mock_search(query, limit, **kwargs):
            return [scholar_paper]

        async def mock_enrich(papers):
            return [enriched_paper]

        with patch("vibescholar.online.search_google_scholar", mock_search), \
             patch("vibescholar.online.enrich_with_s2", mock_enrich):
            await server.search_online("attention")

        assert "s2stable123" in server._paper_cache
        assert "gs_0" not in server._paper_cache

    @pytest.mark.asyncio
    async def test_year_range_passed_through(self, monkeypatch):
        import server

        captured_kwargs: dict = {}
        papers = [_make_paper_result()]

        async def mock_search(query, limit, **kwargs):
            captured_kwargs.update(kwargs)
            return papers

        with patch("vibescholar.online.search_google_scholar", mock_search), \
             patch("vibescholar.online.enrich_with_s2", _passthrough_enrich):
            await server.search_online("test", year_min=2020, year_max=2024)

        assert captured_kwargs["year_min"] == 2020
        assert captured_kwargs["year_max"] == 2024

    @pytest.mark.asyncio
    async def test_sort_date_passed_through(self, monkeypatch):
        import server

        captured_kwargs: dict = {}
        papers = [_make_paper_result()]

        async def mock_search(query, limit, **kwargs):
            captured_kwargs.update(kwargs)
            return papers

        with patch("vibescholar.online.search_google_scholar", mock_search), \
             patch("vibescholar.online.enrich_with_s2", _passthrough_enrich):
            await server.search_online("test", sort="date")

        assert captured_kwargs["sort_by_date"] is True

    @pytest.mark.asyncio
    async def test_offset_passed_through(self, monkeypatch):
        import server

        captured_kwargs: dict = {}
        papers = [_make_paper_result()]

        async def mock_search(query, limit, **kwargs):
            captured_kwargs.update(kwargs)
            return papers

        with patch("vibescholar.online.search_google_scholar", mock_search), \
             patch("vibescholar.online.enrich_with_s2", _passthrough_enrich):
            await server.search_online("test", offset=20)

        assert captured_kwargs["offset"] == 20

    @pytest.mark.asyncio
    async def test_cache_key_includes_new_params(self, monkeypatch):
        """Different year/sort/offset produce separate cache entries."""
        import server

        call_count = {"n": 0}
        papers = [_make_paper_result()]

        async def mock_search(query, limit, **kwargs):
            call_count["n"] += 1
            return papers

        with patch("vibescholar.online.search_google_scholar", mock_search), \
             patch("vibescholar.online.enrich_with_s2", _passthrough_enrich):
            await server.search_online("test", year_min=2020)
            await server.search_online("test", year_min=2022)

        assert call_count["n"] == 2


# ── TestFetchPaper ────────────────────────────────────────────────


class TestFetchPaper:
    @pytest.mark.asyncio
    async def test_from_cache_success(self, monkeypatch):
        import server

        paper = _make_paper_result(paper_id="cached_paper")
        server._paper_cache["cached_paper"] = paper

        async def mock_fetch(p, pages=None):
            return "(2 pages)\n\n--- Page 1 ---\nFull text here", "ArXiv"

        with patch("vibescholar.online.fetch_paper_pdf_text", mock_fetch), \
             patch("vibescholar.online.get_paper_metadata"):
            result = await server.fetch_paper("cached_paper")

        assert 'Paper: "Test Paper"' in result
        assert "Source: ArXiv" in result
        assert "Full text here" in result

    @pytest.mark.asyncio
    async def test_cache_miss_fetches_metadata(self, monkeypatch):
        import server

        paper = _make_paper_result(paper_id="new_paper")
        meta_calls: list[str] = []

        async def mock_get_meta(pid):
            meta_calls.append(pid)
            return paper

        async def mock_fetch(p, pages=None):
            return "(1 pages)\n\n--- Page 1 ---\nText", "S2 OA"

        with patch("vibescholar.online.get_paper_metadata", mock_get_meta), \
             patch("vibescholar.online.fetch_paper_pdf_text", mock_fetch):
            result = await server.fetch_paper("new_paper")

        assert meta_calls == ["new_paper"]
        assert "Test Paper" in result

    @pytest.mark.asyncio
    async def test_not_found_on_s2(self, monkeypatch):
        import server

        async def mock_get_meta(pid):
            return None

        with patch("vibescholar.online.get_paper_metadata", mock_get_meta):
            result = await server.fetch_paper("nonexistent_id")

        assert "Paper not found on Semantic Scholar" in result

    @pytest.mark.asyncio
    async def test_pdf_fails_degradation(self, monkeypatch):
        import server

        paper = _make_paper_result(paper_id="no_pdf")
        server._paper_cache["no_pdf"] = paper

        async def mock_fetch(p, pages=None):
            return "", None  # PDF cascade failed

        with patch("vibescholar.online.fetch_paper_pdf_text", mock_fetch), \
             patch("vibescholar.online.get_paper_metadata"):
            result = await server.fetch_paper("no_pdf")

        assert "Could not download PDF" in result
        assert "This is a test abstract" in result
        assert "Cited by: 42" in result

    @pytest.mark.asyncio
    async def test_pages_passed_through(self, monkeypatch):
        import server

        paper = _make_paper_result(paper_id="pages_test")
        server._paper_cache["pages_test"] = paper
        captured_pages = {}

        async def mock_fetch(p, pages=None):
            captured_pages["pages"] = pages
            return "(1 pages)\n\n--- Page 1 ---\nText", "ArXiv"

        with patch("vibescholar.online.fetch_paper_pdf_text", mock_fetch), \
             patch("vibescholar.online.get_paper_metadata"):
            await server.fetch_paper("pages_test", pages=[1, 3, 5])

        assert captured_pages["pages"] == [1, 3, 5]


# ── TestSavePaper ─────────────────────────────────────────────────


class TestSavePaper:
    @pytest.mark.asyncio
    async def test_save_local_paper(self, mock_backend, tmp_path):
        """Saving a paper that exists in the local corpus copies the file."""
        import server
        import shutil

        # Create a real PDF file that resolve_path can find
        src_pdf = tmp_path / "source.pdf"
        src_pdf.write_bytes(b"%PDF-1.4 fake content")
        mock_backend.resolve_path = lambda fp: src_pdf

        dest_dir = tmp_path / "saved"
        dest_dir.mkdir()

        result = await server.save_paper("source", str(dest_dir))
        assert "Saved local PDF to:" in result
        assert (dest_dir / "source.pdf").exists()
        assert (dest_dir / "source.pdf").read_bytes() == b"%PDF-1.4 fake content"

    @pytest.mark.asyncio
    async def test_save_online_paper(self, mock_backend, tmp_path):
        """When not found locally, downloads from online and saves."""
        import server

        # resolve_path will fail → triggers online path
        mock_backend.resolve_path = lambda fp: (_ for _ in ()).throw(
            FileNotFoundError("not in corpus")
        )

        paper = _make_paper_result(paper_id="online123", title="Great Paper 2024")
        server._paper_cache["online123"] = paper
        pdf_content = b"%PDF-1.4 downloaded content"

        async def mock_download(p):
            return pdf_content, "ArXiv", []

        dest_dir = tmp_path / "downloads"
        dest_dir.mkdir()

        with patch("vibescholar.online._download_pdf_bytes", mock_download), \
             patch("vibescholar.online.get_paper_metadata"):
            result = await server.save_paper("online123", str(dest_dir))

        assert "Saved" in result
        assert "Great Paper 2024" in result
        assert "ArXiv" in result
        # File should exist with sanitized name
        saved_files = list(dest_dir.glob("*.pdf"))
        assert len(saved_files) == 1
        assert saved_files[0].read_bytes() == pdf_content

    @pytest.mark.asyncio
    async def test_save_directory_not_found(self, mock_backend):
        import server

        result = await server.save_paper("anything", "/nonexistent/directory/path")
        assert "Directory does not exist" in result

    @pytest.mark.asyncio
    async def test_save_online_pdf_download_fails(self, mock_backend, tmp_path):
        """When PDF can't be downloaded, returns error with reasons."""
        import server

        mock_backend.resolve_path = lambda fp: (_ for _ in ()).throw(
            FileNotFoundError("not in corpus")
        )

        paper = _make_paper_result(paper_id="nopdf")
        server._paper_cache["nopdf"] = paper

        async def mock_download(p):
            return None, None, [("ArXiv", "404"), ("Unpaywall", "no OA")]

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        with patch("vibescholar.online._download_pdf_bytes", mock_download), \
             patch("vibescholar.online.get_paper_metadata"):
            result = await server.save_paper("nopdf", str(dest_dir))

        assert "Could not download PDF" in result
        assert "ArXiv: 404" in result

    @pytest.mark.asyncio
    async def test_save_not_found_anywhere(self, mock_backend, tmp_path):
        """Paper not in corpus and not on Semantic Scholar."""
        import server

        mock_backend.resolve_path = lambda fp: (_ for _ in ()).throw(
            FileNotFoundError("not in corpus")
        )

        async def mock_get_meta(pid):
            return None

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        with patch("vibescholar.online.get_paper_metadata", mock_get_meta):
            result = await server.save_paper("unknown", str(dest_dir))

        assert "not found locally or on Semantic Scholar" in result

    @pytest.mark.asyncio
    async def test_save_filename_sanitized(self, mock_backend, tmp_path):
        """Special characters in title are sanitized for the filename."""
        import server

        mock_backend.resolve_path = lambda fp: (_ for _ in ()).throw(
            FileNotFoundError("not in corpus")
        )

        paper = _make_paper_result(
            paper_id="special",
            title='A "Great" Paper: With/Special <Characters> & More!',
        )
        server._paper_cache["special"] = paper

        async def mock_download(p):
            return b"%PDF-1.4", "ArXiv", []

        dest_dir = tmp_path / "dest"
        dest_dir.mkdir()

        with patch("vibescholar.online._download_pdf_bytes", mock_download), \
             patch("vibescholar.online.get_paper_metadata"):
            result = await server.save_paper("special", str(dest_dir))

        saved_files = list(dest_dir.glob("*.pdf"))
        assert len(saved_files) == 1
        filename = saved_files[0].name
        # No special characters in filename
        assert '"' not in filename
        assert '/' not in filename
        assert '<' not in filename
        assert '>' not in filename


# ── TestIndexPapers ───────────────────────────────────────────────


def _make_index_stats(**overrides):
    from vibescholar.indexer import IndexStats
    defaults = dict(
        scanned_files=5, indexed_files=3, skipped_files=2,
        removed_files=0, chunks_added=42, errors=0,
    )
    defaults.update(overrides)
    return IndexStats(**defaults)


class TestIndexPapers:
    @pytest.fixture(autouse=True)
    def _attach_indexer(self, mock_backend):
        """Add a mock indexer and _build_file_map to the fake backend."""
        mock_backend.indexer = MagicMock()
        mock_backend.indexer.index_directory.return_value = _make_index_stats()
        mock_backend._build_file_map = lambda: mock_backend._file_map

    @pytest.mark.asyncio
    async def test_folder_not_exist(self, mock_backend, tmp_path):
        import server

        result = await server.index_papers(str(tmp_path / "nonexistent"))
        assert "Folder does not exist" in result

    @pytest.mark.asyncio
    async def test_folder_is_file(self, mock_backend, tmp_path):
        import server

        a_file = tmp_path / "file.pdf"
        a_file.write_bytes(b"%PDF-1.4")
        result = await server.index_papers(str(a_file))
        assert "Not a directory" in result

    @pytest.mark.asyncio
    async def test_successful_indexing_summary(self, mock_backend, tmp_path):
        import server

        folder = tmp_path / "papers"
        folder.mkdir()
        result = await server.index_papers(str(folder))

        assert "Files scanned:  5" in result
        assert "Files indexed:  3" in result
        assert "Files skipped:  2" in result
        assert "Chunks added:   42" in result
        assert "Time elapsed:" in result

    @pytest.mark.asyncio
    async def test_search_cache_invalidated(self, mock_backend, tmp_path):
        import server

        server._search_cache[("test", 10, "", "detailed")] = "stale cached result"
        folder = tmp_path / "papers"
        folder.mkdir()
        await server.index_papers(str(folder))
        assert len(server._search_cache) == 0

    @pytest.mark.asyncio
    async def test_force_flag_passed_through(self, mock_backend, tmp_path):
        import server

        folder = tmp_path / "papers"
        folder.mkdir()
        await server.index_papers(str(folder), force=True)

        call_kwargs = mock_backend.indexer.index_directory.call_args
        assert call_kwargs.kwargs.get("force") is True

    @pytest.mark.asyncio
    async def test_indexer_exception_returns_error_string(self, mock_backend, tmp_path):
        import server

        mock_backend.indexer.index_directory.side_effect = RuntimeError("disk full")
        folder = tmp_path / "papers"
        folder.mkdir()
        result = await server.index_papers(str(folder))
        assert "Indexing failed" in result
        assert "disk full" in result

    @pytest.mark.asyncio
    async def test_upsert_directory_called(self, mock_backend, tmp_path):
        """DB upsert_directory is called with the resolved folder path."""
        import server

        folder = tmp_path / "papers"
        folder.mkdir()

        upserted: list[str] = []
        original_upsert = mock_backend.db.upsert_directory

        def recording_upsert(path):
            upserted.append(path)
            return original_upsert(path)

        mock_backend.db.upsert_directory = recording_upsert
        await server.index_papers(str(folder))
        assert len(upserted) == 1
        assert str(folder) in upserted[0]
