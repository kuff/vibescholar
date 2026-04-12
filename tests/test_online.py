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
    _title_similarity,
    _try_download_pdf,
    _unpaywall_pdf_url,
    enrich_with_s2,
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


# ── TestTitleSimilarity ───────────────────────────────────────────


class TestTitleSimilarity:
    def test_identical(self):
        assert _title_similarity("Attention Is All You Need", "Attention Is All You Need") == 1.0

    def test_case_insensitive(self):
        assert _title_similarity("Attention Is All You Need", "attention is all you need") == 1.0

    def test_punctuation_stripped(self):
        assert _title_similarity("Foo: A Survey", "Foo A Survey") > 0.85

    def test_high_similarity(self):
        assert _title_similarity("Attention Is All You Need", "Attention is All You Need!") >= 0.85

    def test_below_threshold(self):
        assert _title_similarity("Neural Style Transfer", "Generative Adversarial Networks") < 0.85

    def test_completely_different(self):
        assert _title_similarity("BERT Pre-training", "ResNet Deep Residual Learning") < 0.85

    def test_empty_first(self):
        assert _title_similarity("", "Some Title") == 0.0

    def test_empty_second(self):
        assert _title_similarity("Some Title", "") == 0.0

    def test_single_char_tokens_dropped(self):
        # "a" and "b" are length-1 and dropped; effective tokens may differ
        result = _title_similarity("Transformer", "Transformer")
        assert result == 1.0


# ── TestEnrichWithS2 ──────────────────────────────────────────────


def _make_gs_paper(**overrides) -> PaperResult:
    """Make a Scholar-style PaperResult with a gs_* ID."""
    defaults = dict(
        paper_id="gs_0",
        title="Attention Is All You Need",
        authors=["A Vaswani"],
        year=2017,
        venue="NeurIPS",
        citation_count=50000,
        abstract="Short snippet from Scholar.",
        doi=None,
        arxiv_id=None,
        open_access_url="https://arxiv.org/pdf/1706.03762",
        external_ids={},
    )
    defaults.update(overrides)
    return PaperResult(**defaults)


def _s2_enrich_response(paper_dict: dict) -> httpx.Response:
    """Build a mock S2 search response containing a single paper."""
    body = {"total": 1, "offset": 0, "data": [paper_dict]}
    return httpx.Response(200, json=body, request=httpx.Request("GET", "https://s2"))


class TestEnrichWithS2:
    @pytest.mark.asyncio
    async def test_match_replaces_gs_id(self, monkeypatch):
        """Matched Scholar paper gets S2 stable ID, full abstract, DOI."""
        s2_data = _s2_paper_dict()  # title: "Attention Is All You Need"

        async def mock_get(self, url, **kwargs):
            return _s2_enrich_response(s2_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        papers = await enrich_with_s2([_make_gs_paper()])
        assert len(papers) == 1
        assert papers[0].paper_id == "abc123def"
        assert papers[0].doi == "10.48550/arXiv.1706.03762"
        assert papers[0].arxiv_id == "1706.03762"
        assert "sequence transduction" in papers[0].abstract

    @pytest.mark.asyncio
    async def test_no_match_preserves_gs_paper(self, monkeypatch):
        """Low title similarity keeps gs_* ID unchanged."""
        s2_data = _s2_paper_dict(title="Deep Residual Learning for Image Recognition")

        async def mock_get(self, url, **kwargs):
            return _s2_enrich_response(s2_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        original = _make_gs_paper()
        papers = await enrich_with_s2([original])
        assert papers[0].paper_id == "gs_0"
        assert papers[0].abstract == "Short snippet from Scholar."

    @pytest.mark.asyncio
    async def test_s2_failure_preserves_gs_paper(self, monkeypatch):
        """S2 returning None leaves the original paper untouched."""
        async def mock_get(self, url, **kwargs):
            return httpx.Response(500, request=httpx.Request("GET", url))

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        original = _make_gs_paper()
        papers = await enrich_with_s2([original])
        assert papers[0].paper_id == "gs_0"

    @pytest.mark.asyncio
    async def test_scholar_pdf_promoted_when_s2_has_no_oa(self, monkeypatch):
        """Scholar's PDF becomes open_access_url when S2 has no OA URL."""
        s2_data = _s2_paper_dict(openAccessPdf=None)

        async def mock_get(self, url, **kwargs):
            return _s2_enrich_response(s2_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        gs = _make_gs_paper(open_access_url="https://arxiv.org/pdf/1706.03762")
        papers = await enrich_with_s2([gs])
        assert papers[0].open_access_url == "https://arxiv.org/pdf/1706.03762"
        assert "scholar_pdf_url" not in papers[0].external_ids

    @pytest.mark.asyncio
    async def test_scholar_pdf_stashed_when_s2_has_oa(self, monkeypatch):
        """Scholar's PDF URL is stashed in external_ids when S2 has its own OA URL."""
        s2_data = _s2_paper_dict()  # has openAccessPdf

        async def mock_get(self, url, **kwargs):
            return _s2_enrich_response(s2_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        scholar_pdf = "https://arxiv.org/pdf/1706.03762"
        gs = _make_gs_paper(open_access_url=scholar_pdf)
        papers = await enrich_with_s2([gs])
        # S2's OA URL takes primary slot
        assert papers[0].open_access_url == "https://example.com/paper.pdf"
        # Scholar's URL preserved as fallback
        assert papers[0].external_ids.get("scholar_pdf_url") == scholar_pdf

    @pytest.mark.asyncio
    async def test_same_urls_not_duplicated(self, monkeypatch):
        """No scholar_pdf_url stashed when both URLs are identical."""
        same_url = "https://example.com/paper.pdf"
        s2_data = _s2_paper_dict(openAccessPdf={"url": same_url})

        async def mock_get(self, url, **kwargs):
            return _s2_enrich_response(s2_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        gs = _make_gs_paper(open_access_url=same_url)
        papers = await enrich_with_s2([gs])
        assert papers[0].open_access_url == same_url
        assert "scholar_pdf_url" not in papers[0].external_ids

    @pytest.mark.asyncio
    async def test_empty_input(self, monkeypatch):
        papers = await enrich_with_s2([])
        assert papers == []

    @pytest.mark.asyncio
    async def test_scholar_navigation_urls_preserved(self, monkeypatch):
        """cited_by_url and related_url survive enrichment into external_ids."""
        s2_data = _s2_paper_dict()

        async def mock_get(self, url, **kwargs):
            return _s2_enrich_response(s2_data)

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        gs = _make_gs_paper(
            open_access_url=None,
            external_ids={
                "cited_by_url": "/scholar?cites=12345&hl=en",
                "related_url": "/scholar?q=related:abcde:scholar.google.com/&hl=en",
            },
        )
        papers = await enrich_with_s2([gs])
        assert papers[0].external_ids.get("cited_by_url") == "/scholar?cites=12345&hl=en"
        assert "related:" in papers[0].external_ids.get("related_url", "")
        # S2 external IDs are also present
        assert "DOI" in papers[0].external_ids

    @pytest.mark.asyncio
    async def test_partial_match(self, monkeypatch):
        """Mixed batch: matched paper gets S2 ID, unmatched keeps gs_*."""
        call_count = {"n": 0}

        async def mock_get(self, url, **kwargs):
            call_count["n"] += 1
            # First call matches, second call returns a different title
            if call_count["n"] == 1:
                return _s2_enrich_response(_s2_paper_dict())
            return _s2_enrich_response(
                _s2_paper_dict(paperId="other", title="Completely Different Paper Title Here")
            )

        monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)
        gs0 = _make_gs_paper(paper_id="gs_0")
        gs1 = _make_gs_paper(paper_id="gs_1", title="Something Else Entirely Unrelated Work")
        papers = await enrich_with_s2([gs0, gs1])
        assert papers[0].paper_id == "abc123def"  # matched
        assert papers[1].paper_id == "gs_1"        # unmatched


# ── TestPdfResolutionCascadeScholar ───────────────────────────────


class TestPdfResolutionCascadeScholar:
    """Additional _resolve_pdf_sources tests for the scholar_pdf_url field."""

    def test_scholar_pdf_url_in_external_ids(self):
        paper = _make_paper(
            open_access_url=None,
            doi=None,
            arxiv_id=None,
            external_ids={"scholar_pdf_url": "https://scholar.example.com/paper.pdf"},
        )
        sources = _resolve_pdf_sources(paper)
        names = [n for n, _ in sources]
        assert "Scholar PDF" in names

    def test_scholar_pdf_not_duplicated_when_promoted(self):
        """If scholar_pdf_url equals open_access_url, it is not added twice."""
        url = "https://example.com/paper.pdf"
        paper = _make_paper(
            open_access_url=url,
            doi=None,
            arxiv_id=None,
            external_ids={"scholar_pdf_url": url},
        )
        sources = _resolve_pdf_sources(paper)
        urls = [u for _, u in sources]
        assert urls.count(url) == 1

    def test_scholar_pdf_ordering(self):
        """Scholar PDF appears after S2 OA, ArXiv, Unpaywall, Publisher direct."""
        paper = _make_paper(
            external_ids={
                "DOI": "10.1234/test",
                "ArXiv": "2401.12345",
                "scholar_pdf_url": "https://scholar.example.com/paper.pdf",
            },
        )
        sources = _resolve_pdf_sources(paper)
        names = [n for n, _ in sources]
        assert names[-1] == "Scholar PDF"
