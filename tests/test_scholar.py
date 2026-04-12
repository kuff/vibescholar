"""Tests for Google Scholar search, citation/related/author lookup, and parsing.

All browser interactions are mocked via monkeypatching _fetch_scholar_page.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from vibescholar.online import (
    PaperResult,
    _parse_scholar_results,
    _raw_to_paper_results,
    _parse_author_profile,
    cited_by,
    related_papers,
    author_papers,
    search_google_scholar,
)


# ── Fixtures & sample HTML ────────────────────────────────────────


SCHOLAR_RESULT_HTML = """
<div class="gs_r gs_or gs_scl">
  <div class="gs_ggs gs_fl"><div><a href="https://arxiv.org/pdf/1706.03762">PDF</a></div></div>
  <div class="gs_ri">
    <h3 class="gs_rt"><a href="https://example.com/paper">Attention Is All You Need</a></h3>
    <div class="gs_a">A Vaswani, N Shazeer, N Parmar - Advances in neural information processing systems, 2017 - Springer</div>
    <div class="gs_rs">The dominant sequence transduction models are based on complex recurrent neural networks.</div>
    <div class="gs_fl gs_flb">
      <a href="/scholar?cites=12345&hl=en">Cited by 120000</a>
      <a href="/scholar?q=related:abcde:scholar.google.com/&hl=en">Related articles</a>
    </div>
  </div>
</div>
<div class="gs_r gs_or gs_scl">
  <div class="gs_ri">
    <h3 class="gs_rt"><a href="https://example.com/paper2">BERT: Pre-training of Deep Bidirectional Transformers</a></h3>
    <div class="gs_a">J Devlin, MW Chang - arXiv preprint arXiv:1810.04805, 2018</div>
    <div class="gs_rs">We introduce a new language representation model.</div>
    <div class="gs_fl gs_flb">
      <a href="/scholar?cites=67890&hl=en">Cited by 95000</a>
      <a href="/scholar?q=related:fghij:scholar.google.com/&hl=en">Related articles</a>
    </div>
  </div>
</div>
"""

SCHOLAR_RESULT_NO_CITATIONS_HTML = """
<div class="gs_r gs_or gs_scl">
  <div class="gs_ri">
    <h3 class="gs_rt"><a href="https://example.com/new">A Very New Paper</a></h3>
    <div class="gs_a">X Author - arXiv preprint, 2026</div>
    <div class="gs_rs">This paper is brand new.</div>
    <div class="gs_fl gs_flb">
    </div>
  </div>
</div>
"""

SCHOLAR_CAPTCHA_HTML = """
<html><body>Our systems have detected unusual traffic from your computer network.
<div id="captcha">CAPTCHA</div></body></html>
"""

AUTHOR_PROFILE_HTML = """
<table id="gsc_a_t">
  <tr class="gsc_a_tr">
    <td><a class="gsc_a_at">Attention Is All You Need</a>
      <div class="gs_gray">A Vaswani, N Shazeer - NeurIPS, 2017</div>
    </td>
    <td><a class="gsc_a_ac gs_ibl">120000</a></td>
    <td><span class="gsc_a_h gsc_a_hc gs_ibl"><span class="gs_oph">2017</span></span></td>
  </tr>
  <tr class="gsc_a_tr">
    <td><a class="gsc_a_at">Scaling Laws for Neural Language Models</a>
      <div class="gs_gray">J Kaplan - arXiv preprint, 2020</div>
    </td>
    <td><a class="gsc_a_ac gs_ibl">5000</a></td>
    <td><span class="gsc_a_h gsc_a_hc gs_ibl"><span class="gs_oph">2020</span></span></td>
  </tr>
  <tr class="gsc_a_tr">
    <td><a class="gsc_a_at">Empty Citation Paper</a>
      <div class="gs_gray">Z Researcher - Workshop, 2025</div>
    </td>
    <td><a class="gsc_a_ac gs_ibl"></a></td>
    <td><span class="gsc_a_h gsc_a_hc gs_ibl"><span class="gs_oph">2025</span></span></td>
  </tr>
</table>
"""

AUTHOR_SEARCH_HTML = """
<div class="gsc_1usr">
  <a href="/citations?user=ABCDEF123&hl=en">
    <span class="gs_hlt">Yann LeCun</span>
  </a>
</div>
"""


# ── TestParseScholarResults ───────────────────────────────────────


class TestParseScholarResults:
    def test_basic_parsing(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        assert len(raw) == 2

    def test_title_extracted(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        assert raw[0]["title"] == "Attention Is All You Need"
        assert raw[1]["title"] == "BERT: Pre-training of Deep Bidirectional Transformers"

    def test_pdf_url_extracted(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        assert raw[0]["pdf_url"] == "https://arxiv.org/pdf/1706.03762"
        assert raw[1]["pdf_url"] is None  # second result has no PDF

    def test_citation_count(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        assert raw[0]["cite_count"] == 120000
        assert raw[1]["cite_count"] == 95000

    def test_cited_by_url(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        assert "cites=12345" in raw[0]["cited_by_url"]
        assert "cites=67890" in raw[1]["cited_by_url"]

    def test_related_url(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        assert "related:" in raw[0]["related_url"]

    def test_authors_raw(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        assert "Vaswani" in raw[0]["authors_raw"]

    def test_snippet(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        assert "transduction" in raw[0]["snippet"]

    def test_no_citations(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_NO_CITATIONS_HTML)
        assert len(raw) == 1
        assert raw[0]["cite_count"] == 0
        assert raw[0]["cited_by_url"] is None
        assert raw[0]["related_url"] is None

    def test_captcha_returns_empty(self):
        raw = _parse_scholar_results(SCHOLAR_CAPTCHA_HTML)
        assert raw == []

    def test_empty_html(self):
        raw = _parse_scholar_results("")
        assert raw == []

    def test_html_without_results(self):
        raw = _parse_scholar_results("<html><body><p>Nothing here</p></body></html>")
        assert raw == []


# ── TestRawToPaperResults ─────────────────────────────────────────


class TestRawToPaperResults:
    def test_conversion(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        papers = _raw_to_paper_results(raw)
        assert len(papers) == 2
        assert isinstance(papers[0], PaperResult)

    def test_authors_parsed(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        papers = _raw_to_paper_results(raw)
        assert "A Vaswani" in papers[0].authors

    def test_year_extracted(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        papers = _raw_to_paper_results(raw)
        assert papers[0].year == 2017
        assert papers[1].year == 2018

    def test_venue_extracted(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        papers = _raw_to_paper_results(raw)
        assert "neural information" in papers[0].venue.lower()

    def test_citation_count_preserved(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        papers = _raw_to_paper_results(raw)
        assert papers[0].citation_count == 120000

    def test_open_access_url(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        papers = _raw_to_paper_results(raw)
        assert papers[0].open_access_url == "https://arxiv.org/pdf/1706.03762"
        assert papers[1].open_access_url is None

    def test_external_ids_contain_urls(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        papers = _raw_to_paper_results(raw)
        assert "cited_by_url" in papers[0].external_ids
        assert "related_url" in papers[0].external_ids

    def test_paper_ids_sequential(self):
        raw = _parse_scholar_results(SCHOLAR_RESULT_HTML)
        papers = _raw_to_paper_results(raw)
        assert papers[0].paper_id == "gs_0"
        assert papers[1].paper_id == "gs_1"

    def test_empty_input(self):
        assert _raw_to_paper_results([]) == []


# ── TestParseAuthorProfile ────────────────────────────────────────


class TestParseAuthorProfile:
    def test_basic_parsing(self):
        papers = _parse_author_profile(AUTHOR_PROFILE_HTML)
        assert len(papers) == 3

    def test_title(self):
        papers = _parse_author_profile(AUTHOR_PROFILE_HTML)
        assert papers[0].title == "Attention Is All You Need"
        assert papers[1].title == "Scaling Laws for Neural Language Models"

    def test_citation_count(self):
        papers = _parse_author_profile(AUTHOR_PROFILE_HTML)
        assert papers[0].citation_count == 120000
        assert papers[1].citation_count == 5000

    def test_empty_citation(self):
        papers = _parse_author_profile(AUTHOR_PROFILE_HTML)
        assert papers[2].citation_count == 0

    def test_year(self):
        papers = _parse_author_profile(AUTHOR_PROFILE_HTML)
        assert papers[0].year == 2017
        assert papers[1].year == 2020

    def test_paper_ids(self):
        papers = _parse_author_profile(AUTHOR_PROFILE_HTML)
        assert papers[0].paper_id == "gs_author_0"

    def test_empty_html(self):
        assert _parse_author_profile("") == []

    def test_no_rows(self):
        assert _parse_author_profile("<table></table>") == []


# ── TestCitedBy (async, mocked browser) ───────────────────────────


class TestCitedBy:
    @pytest.mark.asyncio
    async def test_finds_citing_papers(self):
        call_urls: list[str] = []

        async def mock_fetch(url):
            call_urls.append(url)
            if "cites=" in url:
                return SCHOLAR_RESULT_HTML  # citing papers
            return SCHOLAR_RESULT_HTML  # initial search (has cited_by_url)

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await cited_by("Attention Is All You Need")

        assert len(papers) == 2
        assert len(call_urls) == 2
        assert "cites=" in call_urls[1]

    @pytest.mark.asyncio
    async def test_no_cited_by_url(self):
        async def mock_fetch(url):
            return SCHOLAR_RESULT_NO_CITATIONS_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await cited_by("A Very New Paper")

        assert papers == []

    @pytest.mark.asyncio
    async def test_empty_search(self):
        async def mock_fetch(url):
            return ""

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await cited_by("nonexistent")

        assert papers == []

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        async def mock_fetch(url):
            return SCHOLAR_RESULT_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await cited_by("Attention", limit=1)

        assert len(papers) <= 1

    @pytest.mark.asyncio
    async def test_direct_url_skips_search(self):
        """Passing cited_by_url directly skips the initial title search."""
        call_urls: list[str] = []

        async def mock_fetch(url):
            call_urls.append(url)
            return SCHOLAR_RESULT_HTML

        direct = "/scholar?cites=12345&hl=en"
        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await cited_by("Anything", cited_by_url=direct)

        assert len(call_urls) == 1  # only one call, no initial search
        assert "cites=12345" in call_urls[0]
        assert len(papers) == 2


# ── TestRelatedPapers (async, mocked browser) ────────────────────


class TestRelatedPapers:
    @pytest.mark.asyncio
    async def test_finds_related(self):
        call_urls: list[str] = []

        async def mock_fetch(url):
            call_urls.append(url)
            return SCHOLAR_RESULT_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await related_papers("Attention Is All You Need")

        assert len(papers) == 2
        assert len(call_urls) == 2
        assert "related:" in call_urls[1]

    @pytest.mark.asyncio
    async def test_no_related_url(self):
        async def mock_fetch(url):
            return SCHOLAR_RESULT_NO_CITATIONS_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await related_papers("A Very New Paper")

        assert papers == []

    @pytest.mark.asyncio
    async def test_empty_search(self):
        async def mock_fetch(url):
            return ""

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await related_papers("nonexistent")

        assert papers == []

    @pytest.mark.asyncio
    async def test_direct_url_skips_search(self):
        """Passing related_url directly skips the initial title search."""
        call_urls: list[str] = []

        async def mock_fetch(url):
            call_urls.append(url)
            return SCHOLAR_RESULT_HTML

        direct = "/scholar?q=related:abcde:scholar.google.com/&hl=en"
        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await related_papers("Anything", related_url=direct)

        assert len(call_urls) == 1  # only one call, no initial search
        assert "related:" in call_urls[0]
        assert len(papers) == 2


# ── TestAuthorPapers (async, mocked browser) ──────────────────────


class TestAuthorPapers:
    @pytest.mark.asyncio
    async def test_finds_author_papers(self):
        call_urls: list[str] = []

        async def mock_fetch(url):
            call_urls.append(url)
            if "search_authors" in url:
                return AUTHOR_SEARCH_HTML
            return AUTHOR_PROFILE_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await author_papers("Yann LeCun")

        assert len(papers) == 3
        assert papers[0].title == "Attention Is All You Need"
        assert len(call_urls) == 2
        assert "user=ABCDEF123" in call_urls[1]

    @pytest.mark.asyncio
    async def test_no_profile_found(self):
        async def mock_fetch(url):
            return "<html><body>No results</body></html>"

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await author_papers("Nonexistent Person")

        assert papers == []

    @pytest.mark.asyncio
    async def test_empty_response(self):
        async def mock_fetch(url):
            return ""

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await author_papers("Anyone")

        assert papers == []

    @pytest.mark.asyncio
    async def test_limit_respected(self):
        async def mock_fetch(url):
            if "search_authors" in url:
                return AUTHOR_SEARCH_HTML
            return AUTHOR_PROFILE_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await author_papers("Yann LeCun", limit=1)

        assert len(papers) <= 1


# ── TestSearchGoogleScholar (async, mocked browser) ───────────────


class TestSearchGoogleScholar:
    @pytest.mark.asyncio
    async def test_returns_papers(self):
        async def mock_fetch(url):
            return SCHOLAR_RESULT_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await search_google_scholar("attention")

        assert len(papers) == 2
        assert papers[0].title == "Attention Is All You Need"

    @pytest.mark.asyncio
    async def test_empty_on_failure(self):
        async def mock_fetch(url):
            return ""

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await search_google_scholar("anything")

        assert papers == []

    @pytest.mark.asyncio
    async def test_limit(self):
        async def mock_fetch(url):
            return SCHOLAR_RESULT_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            papers = await search_google_scholar("attention", limit=1)

        assert len(papers) == 1

    @pytest.mark.asyncio
    async def test_year_min_in_url(self):
        captured: list[str] = []

        async def mock_fetch(url):
            captured.append(url)
            return SCHOLAR_RESULT_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            await search_google_scholar("attention", year_min=2020)

        assert "as_ylo=2020" in captured[0]

    @pytest.mark.asyncio
    async def test_year_max_in_url(self):
        captured: list[str] = []

        async def mock_fetch(url):
            captured.append(url)
            return SCHOLAR_RESULT_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            await search_google_scholar("attention", year_max=2023)

        assert "as_yhi=2023" in captured[0]

    @pytest.mark.asyncio
    async def test_year_range_in_url(self):
        captured: list[str] = []

        async def mock_fetch(url):
            captured.append(url)
            return SCHOLAR_RESULT_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            await search_google_scholar("attention", year_min=2020, year_max=2023)

        assert "as_ylo=2020" in captured[0]
        assert "as_yhi=2023" in captured[0]

    @pytest.mark.asyncio
    async def test_sort_by_date_in_url(self):
        captured: list[str] = []

        async def mock_fetch(url):
            captured.append(url)
            return SCHOLAR_RESULT_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            await search_google_scholar("attention", sort_by_date=True)

        assert "scisbd=1" in captured[0]

    @pytest.mark.asyncio
    async def test_sort_by_relevance_no_scisbd(self):
        captured: list[str] = []

        async def mock_fetch(url):
            captured.append(url)
            return SCHOLAR_RESULT_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            await search_google_scholar("attention", sort_by_date=False)

        assert "scisbd" not in captured[0]

    @pytest.mark.asyncio
    async def test_offset_in_url(self):
        captured: list[str] = []

        async def mock_fetch(url):
            captured.append(url)
            return SCHOLAR_RESULT_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            await search_google_scholar("attention", offset=20)

        assert "start=20" in captured[0]

    @pytest.mark.asyncio
    async def test_zero_offset_not_in_url(self):
        captured: list[str] = []

        async def mock_fetch(url):
            captured.append(url)
            return SCHOLAR_RESULT_HTML

        with patch("vibescholar.online._fetch_scholar_page", mock_fetch):
            await search_google_scholar("attention", offset=0)

        assert "start=" not in captured[0]


# ── TestFormatOnlineResults (server.py helper) ────────────────────


class TestFormatOnlineResults:
    def _make_paper(self, **overrides):
        defaults = dict(
            paper_id="gs_0", title="Test Paper", authors=["Alice", "Bob"],
            year=2024, venue="NeurIPS", citation_count=42,
            abstract="A test snippet.", doi=None, arxiv_id=None,
            open_access_url="https://example.com/paper.pdf", external_ids={},
        )
        defaults.update(overrides)
        return PaperResult(**defaults)

    def test_detailed_format(self):
        from server import _format_online_results
        papers = [self._make_paper()]
        result = _format_online_results(papers, "detailed", "papers on Scholar")
        assert "Found 1 papers on Scholar" in result
        assert "Test Paper" in result
        assert "Alice" in result
        assert "NeurIPS" in result
        assert "Cited by: 42" in result
        assert "PDF: https://example.com/paper.pdf" in result
        assert "Snippet: A test snippet." in result

    def test_brief_format(self):
        from server import _format_online_results
        papers = [self._make_paper()]
        result = _format_online_results(papers, "brief", "papers")
        assert "A test snippet" not in result
        assert "PDF:" not in result
        assert "Test Paper" in result
        assert "Cited by: 42" in result

    def test_no_venue(self):
        from server import _format_online_results
        papers = [self._make_paper(venue="")]
        result = _format_online_results(papers, "detailed", "papers")
        assert "Cited by: 42" in result  # still shows citations

    def test_no_pdf(self):
        from server import _format_online_results
        papers = [self._make_paper(open_access_url=None)]
        result = _format_online_results(papers, "detailed", "papers")
        assert "PDF:" not in result

    def test_multiple_papers(self):
        from server import _format_online_results
        papers = [self._make_paper(paper_id="gs_0"), self._make_paper(paper_id="gs_1", title="Paper Two")]
        result = _format_online_results(papers, "detailed", "papers")
        assert "Found 2 papers" in result
        assert "1." in result
        assert "2." in result


# ── TestMCPTools (server.py tools, mocked) ────────────────────────


class TestMCPToolsCitedBy:
    @pytest.mark.asyncio
    async def test_returns_formatted(self):
        import server

        papers = [PaperResult(
            paper_id="gs_0", title="Citing Paper", authors=["Eve"],
            year=2023, venue="ICML", citation_count=10, abstract="Cites the original.",
            doi=None, arxiv_id=None, open_access_url=None, external_ids={},
        )]

        async def mock_cited_by(title, limit, **kwargs):
            return papers

        with patch("vibescholar.online.cited_by", mock_cited_by):
            result = await server.cited_by_online("Attention")

        assert "1 papers citing" in result
        assert "Citing Paper" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        import server

        async def mock_cited_by(title, limit, **kwargs):
            return []

        with patch("vibescholar.online.cited_by", mock_cited_by):
            result = await server.cited_by_online("Unknown Paper")

        assert "No citing papers found" in result

    @pytest.mark.asyncio
    async def test_paper_id_passes_cached_url(self):
        """When paper_id is provided, cited_by_url from cache is forwarded."""
        import server

        # Cache a paper with a cited_by_url in external_ids
        cached = PaperResult(
            paper_id="s2_abc", title="Cached Paper", authors=["Alice"],
            year=2020, venue="NeurIPS", citation_count=100, abstract="",
            doi=None, arxiv_id=None, open_access_url=None,
            external_ids={"cited_by_url": "/scholar?cites=99999&hl=en"},
        )
        server._paper_cache["s2_abc"] = cached

        captured_kwargs: dict = {}

        async def mock_cited_by(title, limit, **kwargs):
            captured_kwargs.update(kwargs)
            return [PaperResult(
                paper_id="gs_0", title="Citer", authors=["Bob"],
                year=2023, venue="ICML", citation_count=5, abstract="",
                doi=None, arxiv_id=None, open_access_url=None, external_ids={},
            )]

        with patch("vibescholar.online.cited_by", mock_cited_by):
            result = await server.cited_by_online("Cached Paper", paper_id="s2_abc")

        assert captured_kwargs.get("cited_by_url") == "/scholar?cites=99999&hl=en"
        assert "Citer" in result

    @pytest.mark.asyncio
    async def test_paper_id_no_cached_url_falls_back(self):
        """When paper_id is given but no cited_by_url in cache, falls back to title."""
        import server

        cached = PaperResult(
            paper_id="s2_nocite", title="No Cites", authors=["Alice"],
            year=2020, venue="", citation_count=0, abstract="",
            doi=None, arxiv_id=None, open_access_url=None, external_ids={},
        )
        server._paper_cache["s2_nocite"] = cached

        captured_kwargs: dict = {}

        async def mock_cited_by(title, limit, **kwargs):
            captured_kwargs.update(kwargs)
            return []

        with patch("vibescholar.online.cited_by", mock_cited_by):
            await server.cited_by_online("No Cites", paper_id="s2_nocite")

        assert captured_kwargs.get("cited_by_url") is None


class TestMCPToolsRelated:
    @pytest.mark.asyncio
    async def test_returns_formatted(self):
        import server

        papers = [PaperResult(
            paper_id="gs_0", title="Related Work", authors=["Frank"],
            year=2022, venue="CVPR", citation_count=5, abstract="Related content.",
            doi=None, arxiv_id=None, open_access_url=None, external_ids={},
        )]

        async def mock_related(title, limit, **kwargs):
            return papers

        with patch("vibescholar.online.related_papers", mock_related):
            result = await server.related_papers_online("Attention")

        assert "1 papers related to" in result
        assert "Related Work" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        import server

        async def mock_related(title, limit, **kwargs):
            return []

        with patch("vibescholar.online.related_papers", mock_related):
            result = await server.related_papers_online("Unknown")

        assert "No related papers found" in result

    @pytest.mark.asyncio
    async def test_paper_id_passes_cached_url(self):
        """When paper_id is provided, related_url from cache is forwarded."""
        import server

        cached = PaperResult(
            paper_id="s2_rel", title="Cached Paper", authors=["Alice"],
            year=2020, venue="NeurIPS", citation_count=100, abstract="",
            doi=None, arxiv_id=None, open_access_url=None,
            external_ids={"related_url": "/scholar?q=related:xyz:scholar.google.com/&hl=en"},
        )
        server._paper_cache["s2_rel"] = cached

        captured_kwargs: dict = {}

        async def mock_related(title, limit, **kwargs):
            captured_kwargs.update(kwargs)
            return [PaperResult(
                paper_id="gs_0", title="Similar Work", authors=["Bob"],
                year=2023, venue="ICML", citation_count=5, abstract="",
                doi=None, arxiv_id=None, open_access_url=None, external_ids={},
            )]

        with patch("vibescholar.online.related_papers", mock_related):
            result = await server.related_papers_online("Cached Paper", paper_id="s2_rel")

        assert "related:" in captured_kwargs.get("related_url", "")
        assert "Similar Work" in result


class TestMCPToolsAuthor:
    @pytest.mark.asyncio
    async def test_returns_formatted(self):
        import server

        papers = [PaperResult(
            paper_id="gs_author_0", title="Author Paper", authors=["LeCun"],
            year=2020, venue="Nature", citation_count=999, abstract="",
            doi=None, arxiv_id=None, open_access_url=None, external_ids={},
        )]

        async def mock_author(name, limit):
            return papers

        with patch("vibescholar.online.author_papers", mock_author):
            result = await server.author_papers_online("LeCun")

        assert "1 papers by LeCun" in result
        assert "Author Paper" in result

    @pytest.mark.asyncio
    async def test_no_results(self):
        import server

        async def mock_author(name, limit):
            return []

        with patch("vibescholar.online.author_papers", mock_author):
            result = await server.author_papers_online("Nobody")

        assert "No Google Scholar profile" in result
