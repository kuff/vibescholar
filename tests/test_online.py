"""Tests for online paper search, PDF cascade, and detail formatting."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import httpx
import pytest

import vibescholar.online as _online_mod
from vibescholar.online import (
    PaperResult,
    _email,
    _extract_text_from_pdf_bytes,
    _parse_paper,
    _resolve_pdf_sources,
    _s2_headers,
    _try_download_pdf,
    _unpaywall_pdf_url,
    fetch_paper_pdf_text,
    get_paper_metadata,
    search_semantic_scholar,
)

# ── Fixtures & helpers ────────────────────────────────────────────

TEST_PAPERS_DIR = Path(__file__).parent / "test_papers"


@pytest.fixture(autouse=True)
def _no_retry_delays(monkeypatch):
    """Disable S2 retry sleeps in all tests."""
    monkeypatch.setattr(_online_mod, "_S2_RETRY_DELAYS", (0, 0, 0))


def _make_paper(**overrides) -> PaperResult:
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
        external_ids={"DOI": "10.1234/test", "ArXiv": "2401.12345"},
    )
    defaults.update(overrides)
    return PaperResult(**defaults)


def _s2_search_response(papers: list[dict]) -> dict:
    return {"total": len(papers), "offset": 0, "data": papers}


def _s2_paper_dict(**overrides) -> dict:
    defaults = {
        "paperId": "abc123def",
        "title": "Attention Is All You Need",
        "authors": [
            {"authorId": "1", "name": "Ashish Vaswani"},
            {"authorId": "2", "name": "Noam Shazeer"},
        ],
        "year": 2017,
        "venue": "NeurIPS",
        "citationCount": 120000,
        "abstract": "The dominant sequence transduction models...",
        "openAccessPdf": {"url": "https://example.com/paper.pdf", "status": "GOLD"},
        "externalIds": {
            "DOI": "10.48550/arXiv.1706.03762",
            "ArXiv": "1706.03762",
            "CorpusId": "13756489",
        },
    }
    defaults.update(overrides)
    return defaults


# ── TestParseAndDataclasses ───────────────────────────────────────


class TestParsePaper:
    def test_full_data(self):
        data = _s2_paper_dict()
        result = _parse_paper(data)
        assert result.paper_id == "abc123def"
        assert result.title == "Attention Is All You Need"
        assert result.authors == ["Ashish Vaswani", "Noam Shazeer"]
        assert result.year == 2017
        assert result.venue == "NeurIPS"
        assert result.citation_count == 120000
        assert result.doi == "10.48550/arXiv.1706.03762"
        assert result.arxiv_id == "1706.03762"
        assert result.open_access_url == "https://example.com/paper.pdf"

    def test_missing_fields(self):
        data = {"paperId": "xyz", "title": "Sparse"}
        result = _parse_paper(data)
        assert result.paper_id == "xyz"
        assert result.title == "Sparse"
        assert result.authors == []
        assert result.year is None
        assert result.venue == ""
        assert result.citation_count == 0
        assert result.abstract == ""
        assert result.doi is None
        assert result.arxiv_id is None
        assert result.open_access_url is None

    def test_null_open_access_pdf(self):
        data = _s2_paper_dict(openAccessPdf=None)
        result = _parse_paper(data)
        assert result.open_access_url is None


# ── TestSemanticScholarSearch ─────────────────────────────────────


class TestSemanticScholarSearch:
    @pytest.mark.asyncio
    async def test_basic_search(self, monkeypatch):
        response_body = _s2_search_response([_s2_paper_dict(), _s2_paper_dict(paperId="def456")])

        async def mock_get(self, url, **kwargs):
            return httpx.Response(200, json=response_body, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        results = await search_semantic_scholar("attention transformer")
        assert len(results) == 2
        assert results[0].paper_id == "abc123def"
        assert results[1].paper_id == "def456"

    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch):
        async def mock_get(self, url, **kwargs):
            return httpx.Response(200, json={"total": 0, "data": []}, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        results = await search_semantic_scholar("xyznonexistent")
        assert results == []

    @pytest.mark.asyncio
    async def test_rate_limit_429(self, monkeypatch):
        async def mock_get(self, url, **kwargs):
            return httpx.Response(429, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        results = await search_semantic_scholar("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_server_error_500(self, monkeypatch):
        async def mock_get(self, url, **kwargs):
            return httpx.Response(500, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        results = await search_semantic_scholar("test")
        assert results == []


# ── TestPaperMetadata ─────────────────────────────────────────────


class TestPaperMetadata:
    @pytest.mark.asyncio
    async def test_found(self, monkeypatch):
        async def mock_get(self, url, **kwargs):
            return httpx.Response(200, json=_s2_paper_dict(), request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        result = await get_paper_metadata("abc123def")
        assert result is not None
        assert result.title == "Attention Is All You Need"

    @pytest.mark.asyncio
    async def test_not_found(self, monkeypatch):
        async def mock_get(self, url, **kwargs):
            return httpx.Response(404, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        result = await get_paper_metadata("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_doi_prefix(self, monkeypatch):
        """S2 API accepts DOI:xxx identifiers; verify we pass them through."""
        captured_urls = []

        async def mock_get(self, url, **kwargs):
            captured_urls.append(str(url))
            return httpx.Response(200, json=_s2_paper_dict(), request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        await get_paper_metadata("DOI:10.1234/test")
        assert any("DOI:10.1234/test" in u for u in captured_urls)


# ── TestPdfResolutionCascade ──────────────────────────────────────


class TestPdfResolutionCascade:
    def test_all_sources_present(self):
        paper = _make_paper()
        sources = _resolve_pdf_sources(paper)
        names = [name for name, _ in sources]
        assert names == [
            "Semantic Scholar Open Access",
            "ArXiv",
            "Unpaywall",
            "Publisher direct",
        ]

    def test_no_doi_no_arxiv(self):
        paper = _make_paper(doi=None, arxiv_id=None, open_access_url=None)
        sources = _resolve_pdf_sources(paper)
        assert sources == []

    def test_only_arxiv(self):
        paper = _make_paper(doi=None, open_access_url=None)
        sources = _resolve_pdf_sources(paper)
        assert len(sources) == 1
        assert sources[0][0] == "ArXiv"
        assert "2401.12345" in sources[0][1]

    def test_only_doi(self):
        paper = _make_paper(arxiv_id=None, open_access_url=None)
        sources = _resolve_pdf_sources(paper)
        names = [n for n, _ in sources]
        assert "Unpaywall" in names
        assert "Publisher direct" in names

    @pytest.mark.asyncio
    async def test_cascade_first_fails_second_succeeds(self, monkeypatch):
        """Simulate: OA URL fails, ArXiv succeeds."""
        pdf_bytes = (TEST_PAPERS_DIR / "Meta_Optimal_Transport.pdf").read_bytes()
        call_count = {"n": 0}

        async def mock_get(self, url, **kwargs):
            call_count["n"] += 1
            url_str = str(url)
            if "example.com" in url_str:
                # First source fails
                return httpx.Response(404, request=httpx.Request("GET", url))
            if "arxiv.org" in url_str:
                # Second source succeeds
                return httpx.Response(
                    200,
                    content=pdf_bytes,
                    headers={"content-type": "application/pdf"},
                    request=httpx.Request("GET", url),
                )
            return httpx.Response(404, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        paper = _make_paper(doi=None)  # skip Unpaywall/publisher to simplify
        text, source = await fetch_paper_pdf_text(paper)
        assert source == "ArXiv"
        assert len(text) > 100

    @pytest.mark.asyncio
    async def test_cascade_all_fail(self, monkeypatch):
        async def mock_get(self, url, **kwargs):
            return httpx.Response(404, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        paper = _make_paper(doi=None)  # only OA + ArXiv
        text, source = await fetch_paper_pdf_text(paper)
        assert text == ""
        assert source is None


# ── TestPdfTextExtraction ─────────────────────────────────────────


class TestPdfTextExtraction:
    def test_real_pdf(self):
        pdf_path = TEST_PAPERS_DIR / "Meta_Optimal_Transport.pdf"
        pdf_bytes = pdf_path.read_bytes()
        text = _extract_text_from_pdf_bytes(pdf_bytes)
        assert "Page 1" in text
        assert len(text) > 500

    def test_page_filter(self):
        pdf_path = TEST_PAPERS_DIR / "Meta_Optimal_Transport.pdf"
        pdf_bytes = pdf_path.read_bytes()
        text = _extract_text_from_pdf_bytes(pdf_bytes, pages=[1])
        assert "Page 1" in text
        assert "Page 2" not in text

    def test_corrupted_bytes(self):
        text = _extract_text_from_pdf_bytes(b"not a pdf")
        assert text == ""

    def test_empty_bytes(self):
        text = _extract_text_from_pdf_bytes(b"")
        assert text == ""

    def test_page_out_of_range(self):
        pdf_path = TEST_PAPERS_DIR / "Meta_Optimal_Transport.pdf"
        pdf_bytes = pdf_path.read_bytes()
        text = _extract_text_from_pdf_bytes(pdf_bytes, pages=[9999])
        assert text == ""


# ── TestDetailFormatting ──────────────────────────────────────────


class TestDetailFormatting:
    """Test the formatting helpers used by server.py tools."""

    def test_format_authors_brief(self):
        from server import _format_authors

        authors = ["Alice", "Bob", "Charlie", "Diana"]
        assert _format_authors(authors, brief=True) == "Alice; Bob; Charlie et al."

    def test_format_authors_detailed(self):
        from server import _format_authors

        authors = ["Alice", "Bob"]
        assert _format_authors(authors) == "Alice; Bob"

    def test_format_authors_empty(self):
        from server import _format_authors

        assert _format_authors([]) == "Unknown authors"

    def test_format_authors_many_detailed(self):
        from server import _format_authors

        authors = [f"Author{i}" for i in range(10)]
        result = _format_authors(authors)
        assert "et al." in result
        assert result.count(";") == 5  # 6 authors shown, 5 semicolons


# ── TestCaching ───────────────────────────────────────────────────


class TestCaching:
    def test_paper_cache_eviction(self):
        from server import _cache_paper, _paper_cache, _PAPER_CACHE_MAX

        _paper_cache.clear()
        for i in range(_PAPER_CACHE_MAX + 10):
            _cache_paper(_make_paper(paper_id=f"paper_{i}"))
        assert len(_paper_cache) == _PAPER_CACHE_MAX
        # Earliest papers should be evicted
        assert "paper_0" not in _paper_cache
        assert f"paper_{_PAPER_CACHE_MAX + 9}" in _paper_cache

    def test_paper_cache_move_to_end(self):
        from server import _cache_paper, _paper_cache

        _paper_cache.clear()
        _cache_paper(_make_paper(paper_id="old"))
        _cache_paper(_make_paper(paper_id="new"))
        # Re-cache "old" — should move to end
        _cache_paper(_make_paper(paper_id="old"))
        keys = list(_paper_cache.keys())
        assert keys[-1] == "old"


# ── TestTryDownloadPdf ────────────────────────────────────────────


class TestTryDownloadPdf:
    @pytest.mark.asyncio
    async def test_pdf_content_type_success(self, monkeypatch):
        pdf_bytes = b"%PDF-1.4 fake pdf content"

        async def mock_get(self, url, **kwargs):
            return httpx.Response(
                200,
                content=pdf_bytes,
                headers={"content-type": "application/pdf"},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        result = await _try_download_pdf("https://example.com/paper.pdf")
        assert result == pdf_bytes

    @pytest.mark.asyncio
    async def test_html_content_type_rejected(self, monkeypatch):
        async def mock_get(self, url, **kwargs):
            return httpx.Response(
                200,
                content=b"<html>Not a PDF</html>",
                headers={"content-type": "text/html"},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        result = await _try_download_pdf("https://example.com/paper")
        assert result is None

    @pytest.mark.asyncio
    async def test_octet_stream_accepted(self, monkeypatch):
        pdf_bytes = b"%PDF-1.4 content"

        async def mock_get(self, url, **kwargs):
            return httpx.Response(
                200,
                content=pdf_bytes,
                headers={"content-type": "application/octet-stream"},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        result = await _try_download_pdf("https://example.com/paper.pdf")
        assert result == pdf_bytes

    @pytest.mark.asyncio
    async def test_oversized_rejected(self, monkeypatch):
        from vibescholar.online import _MAX_PDF_BYTES

        big_content = b"x" * (_MAX_PDF_BYTES + 1)

        async def mock_get(self, url, **kwargs):
            return httpx.Response(
                200,
                content=big_content,
                headers={"content-type": "application/pdf"},
                request=httpx.Request("GET", url),
            )

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        result = await _try_download_pdf("https://example.com/huge.pdf")
        assert result is None

    @pytest.mark.asyncio
    async def test_http_error(self, monkeypatch):
        async def mock_get(self, url, **kwargs):
            return httpx.Response(500, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        result = await _try_download_pdf("https://example.com/paper.pdf")
        assert result is None


# ── TestUnpaywallPdfUrl ───────────────────────────────────────────


class TestUnpaywallPdfUrl:
    @pytest.mark.asyncio
    async def test_url_for_pdf_found(self, monkeypatch):
        body = {
            "best_oa_location": {
                "url_for_pdf": "https://repo.example.com/paper.pdf",
                "url": "https://repo.example.com/landing",
            }
        }

        async def mock_get(self, url, **kwargs):
            return httpx.Response(200, json=body, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        result = await _unpaywall_pdf_url("10.1234/test")
        assert result == "https://repo.example.com/paper.pdf"

    @pytest.mark.asyncio
    async def test_url_fallback(self, monkeypatch):
        """When url_for_pdf is None, falls back to url."""
        body = {
            "best_oa_location": {
                "url_for_pdf": None,
                "url": "https://repo.example.com/landing",
            }
        }

        async def mock_get(self, url, **kwargs):
            return httpx.Response(200, json=body, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        result = await _unpaywall_pdf_url("10.1234/test")
        assert result == "https://repo.example.com/landing"

    @pytest.mark.asyncio
    async def test_not_found(self, monkeypatch):
        async def mock_get(self, url, **kwargs):
            return httpx.Response(404, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        result = await _unpaywall_pdf_url("10.1234/missing")
        assert result is None


# ── TestConfigHelpers ─────────────────────────────────────────────


class TestConfigHelpers:
    def test_s2_headers_without_key(self, monkeypatch):
        monkeypatch.delenv("S2_API_KEY", raising=False)
        headers = _s2_headers()
        assert "User-Agent" in headers
        assert "x-api-key" not in headers

    def test_s2_headers_with_key(self, monkeypatch):
        monkeypatch.setenv("S2_API_KEY", "test-key-123")
        headers = _s2_headers()
        assert headers["x-api-key"] == "test-key-123"

    def test_email_default(self, monkeypatch):
        monkeypatch.delenv("VIBESCHOLAR_EMAIL", raising=False)
        assert _email() == "vibescholar@users.noreply.github.com"

    def test_email_custom(self, monkeypatch):
        monkeypatch.setenv("VIBESCHOLAR_EMAIL", "me@example.com")
        assert _email() == "me@example.com"
