"""Online academic paper search and retrieval.

Provides search via Google Scholar (Playwright-driven headless Chromium),
metadata enrichment via Semantic Scholar, open-access PDF discovery via
Unpaywall, and in-memory PDF text extraction.  Papers are fetched on demand
and returned as context — they are **not** indexed into the local corpus.

Environment variables
---------------------
S2_API_KEY            Optional Semantic Scholar API key (higher rate limits).
VIBESCHOLAR_EMAIL     Email for the Unpaywall API (default: vibescholar@users.noreply.github.com).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import re as _re
from dataclasses import dataclass, field
from urllib.parse import quote_plus

import httpx

from .text import clean_text

logger = logging.getLogger("vibescholar-mcp.online")

# ── Configuration ──────────────────────────────────────────────────

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_S2_FIELDS = (
    "paperId,title,authors,year,venue,citationCount,"
    "abstract,openAccessPdf,externalIds"
)
_UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
_CORE_BASE = "https://api.core.ac.uk/v3"
_USER_AGENT = "VibeScholar/0.1 (academic research tool)"
_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50 MB
_PDF_TIMEOUT = 30.0
_API_TIMEOUT = 15.0


def _email() -> str:
    return os.environ.get("VIBESCHOLAR_EMAIL", "vibescholar@users.noreply.github.com")


def _s2_headers() -> dict[str, str]:
    headers: dict[str, str] = {"User-Agent": _USER_AGENT}
    key = os.environ.get("S2_API_KEY")
    if key:
        headers["x-api-key"] = key
    return headers


# ── HTTP client (module-level, lazy) ──────────────────────────────

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return a shared async HTTP client, creating it on first call."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=httpx.Timeout(_API_TIMEOUT, connect=10.0),
            headers={"User-Agent": _USER_AGENT},
        )
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None


# ── Data structures ───────────────────────────────────────────────


@dataclass
class PaperResult:
    """Metadata for a single academic paper from Semantic Scholar."""

    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str
    citation_count: int
    abstract: str  # empty string when unavailable
    doi: str | None
    arxiv_id: str | None
    open_access_url: str | None
    external_ids: dict[str, str] = field(default_factory=dict)


def _parse_paper(data: dict) -> PaperResult:
    """Parse a Semantic Scholar API response dict into a PaperResult."""
    authors = [a.get("name", "") for a in (data.get("authors") or [])]
    ext = data.get("externalIds") or {}
    oa = data.get("openAccessPdf") or {}
    return PaperResult(
        paper_id=data.get("paperId", ""),
        title=data.get("title", ""),
        authors=authors,
        year=data.get("year"),
        venue=data.get("venue", "") or "",
        citation_count=data.get("citationCount", 0) or 0,
        abstract=data.get("abstract", "") or "",
        doi=ext.get("DOI"),
        arxiv_id=ext.get("ArXiv"),
        open_access_url=oa.get("url"),
        external_ids={k: str(v) for k, v in ext.items()},
    )


def _title_similarity(a: str, b: str) -> float:
    """Return a Jaccard token-overlap similarity in [0, 1] between two titles.

    Both strings are lowercased and stripped of non-alphanumeric characters.
    Tokens of length <= 1 are dropped.  Returns 0.0 if either string is empty.
    """
    def _tokens(s: str) -> set[str]:
        return {t for t in _re.findall(r"[^\W_]+", s.lower(), _re.UNICODE) if len(t) > 1}

    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ── S2 enrichment of Scholar results ─────────────────────────────────

_S2_ENRICH_CONCURRENCY = 5   # max parallel S2 lookups per Scholar search
_S2_ENRICH_THRESHOLD = 0.85  # minimum Jaccard similarity to accept a match


async def enrich_with_s2(papers: list[PaperResult]) -> list[PaperResult]:
    """Enrich Google Scholar results with Semantic Scholar metadata.

    For each Scholar paper, fires a parallel S2 title-search call.  On a
    confident title match (Jaccard ≥ 0.85) the ephemeral ``gs_*`` paper ID
    is replaced with the stable S2 ID, and the abstract, DOI, ArXiv ID, and
    open-access URL are filled in from S2.  Papers with no S2 match are
    returned unchanged.

    The Scholar sidebar PDF link is never silently dropped:
    - If S2 has no OA URL, Scholar's PDF is promoted to ``open_access_url``.
    - If S2 has an OA URL, Scholar's PDF (if different) is stashed in
      ``external_ids["scholar_pdf_url"]`` for use by the PDF cascade.
    """
    sem = asyncio.Semaphore(_S2_ENRICH_CONCURRENCY)

    async def _enrich_one(paper: PaperResult) -> PaperResult:
        if not paper.title or not paper.title.strip():
            return paper
        async with sem:
            try:
                resp = await _s2_get(
                    "/paper/search",
                    params={"query": paper.title, "limit": 1, "fields": _S2_FIELDS},
                )
            except Exception:
                return paper

            if resp is None or resp.status_code != 200:
                return paper

            try:
                body = resp.json()
            except Exception:
                return paper

            hits = body.get("data") or []
            if not hits:
                return paper

            s2 = _parse_paper(hits[0])
            if _title_similarity(paper.title, s2.title) < _S2_ENRICH_THRESHOLD:
                return paper

            # Merge: S2 data wins; preserve Scholar PDF link
            scholar_pdf = paper.open_access_url
            s2_oa = s2.open_access_url

            merged_ext = dict(s2.external_ids)
            # Preserve Scholar navigation URLs (cited_by, related)
            for key in ("cited_by_url", "related_url"):
                if key in paper.external_ids:
                    merged_ext[key] = paper.external_ids[key]
            if s2_oa:
                final_oa = s2_oa
                if scholar_pdf and scholar_pdf != s2_oa:
                    merged_ext["scholar_pdf_url"] = scholar_pdf
            elif scholar_pdf:
                final_oa = scholar_pdf  # promote Scholar PDF as primary OA URL
            else:
                final_oa = None

            return PaperResult(
                paper_id=s2.paper_id,
                title=s2.title,
                authors=s2.authors if s2.authors else paper.authors,
                year=s2.year if s2.year is not None else paper.year,
                venue=s2.venue if s2.venue else paper.venue,
                citation_count=s2.citation_count or paper.citation_count,
                abstract=s2.abstract if s2.abstract else paper.abstract,
                doi=s2.doi,
                arxiv_id=s2.arxiv_id,
                open_access_url=final_oa,
                external_ids=merged_ext,
            )

    enriched = await asyncio.gather(*[_enrich_one(p) for p in papers])
    return list(enriched)


# ── Google Scholar search (Playwright) ─────────────────────────────

_SCHOLAR_BASE = "https://scholar.google.com/scholar"


def _parse_scholar_results(html: str) -> list[dict]:
    """Parse Google Scholar results page HTML into raw dicts.

    Each dict contains: title, authors_raw, snippet, url, pdf_url,
    cite_count, cited_by_url, related_url.
    """
    from html.parser import HTMLParser

    results: list[dict] = []

    class _ScholarParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self._in_result = False
            self._result_div_depth = 0
            self._current: dict = {}
            self._capture: str | None = None
            self._capture_depth = 0
            self._buf: list[str] = []
            # Track the link we're currently inside (for matching text to href)
            self._current_href: str | None = None

        def handle_starttag(self, tag, attrs):
            ad = dict(attrs)
            cls = ad.get("class", "")

            # Each result block
            if tag == "div" and "gs_r" in cls and "gs_or" in cls:
                self._in_result = True
                self._result_div_depth = 1
                self._current = {
                    "title": "", "authors_raw": "", "snippet": "",
                    "url": None, "pdf_url": None, "cite_count": 0,
                    "cited_by_url": None, "related_url": None,
                }
                return

            if not self._in_result:
                return

            # Track div nesting inside the result block
            if tag == "div":
                self._result_div_depth += 1

            # Title link
            if tag == "h3" and "gs_rt" in cls:
                self._capture = "title"
                self._capture_depth = 0
                self._buf = []
            if self._capture == "title" and tag == "a" and self._current.get("url") is None:
                self._current["url"] = ad.get("href")

            # PDF link (green sidebar)
            if tag == "div" and "gs_ggs" in cls:
                self._capture = "pdf_scan"
            if self._capture == "pdf_scan" and tag == "a":
                href = ad.get("href", "")
                if href:
                    self._current["pdf_url"] = href
                    self._capture = None

            # Author / venue / year line
            if tag == "div" and "gs_a" in cls:
                self._capture = "authors"
                self._buf = []

            # Snippet
            if tag == "div" and "gs_rs" in cls:
                self._capture = "snippet"
                self._buf = []

            # Track current link href for citation/related detection
            if tag == "a":
                self._current_href = ad.get("href")

            if self._capture in ("title", "authors", "snippet"):
                self._capture_depth += 1

        def handle_endtag(self, tag):
            if self._capture in ("title", "authors", "snippet"):
                self._capture_depth -= 1
                if self._capture_depth <= 0:
                    text = clean_text("".join(self._buf))
                    if self._capture == "title":
                        text = _re.sub(r"^\[.*?\]\s*", "", text)
                        self._current["title"] = text
                    elif self._capture == "authors":
                        self._current["authors_raw"] = text
                    elif self._capture == "snippet":
                        self._current["snippet"] = text
                    self._capture = None

            if tag == "a":
                self._current_href = None

            # Track result block depth — close only when we return to depth 0
            if tag == "div" and self._in_result:
                self._result_div_depth -= 1
                if self._result_div_depth <= 0 and self._current.get("title"):
                    results.append(self._current.copy())
                    self._in_result = False
                    self._current = {}

        def handle_data(self, data):
            if self._capture in ("title", "authors", "snippet"):
                self._buf.append(data)

            if not self._in_result:
                return

            # Detect "Cited by N" and capture the link
            m = _re.search(
                r"(?:Cited by|Citeret af|Cit[ée] par|Citado por|Zitiert von:?)\s*(\d+)",
                data,
            )
            if m:
                self._current["cite_count"] = int(m.group(1))
                if self._current_href and "cites=" in (self._current_href or ""):
                    self._current["cited_by_url"] = self._current_href

            # Detect "Related articles" link
            if _re.search(r"(?:Related articles|Relaterede)", data) and self._current_href:
                if "related:" in (self._current_href or ""):
                    self._current["related_url"] = self._current_href

    parser = _ScholarParser()
    parser.feed(html)
    return results


def _raw_to_paper_results(raw_results: list[dict]) -> list[PaperResult]:
    """Convert raw parsed dicts to PaperResult objects."""
    paper_results: list[PaperResult] = []
    for i, r in enumerate(raw_results):
        # Parse "Author1, Author2 - Venue, Year - Publisher" line
        authors_raw = r.get("authors_raw", "")
        parts = authors_raw.split(" - ")
        authors = [a.strip() for a in parts[0].split(",")] if parts else []
        venue = ""
        year = None
        if len(parts) >= 2:
            venue_year = parts[1]
            year_match = _re.search(r"\b(19|20)\d{2}\b", venue_year)
            if year_match:
                year = int(year_match.group())
                venue = venue_year.replace(year_match.group(), "").strip(", ")
            else:
                venue = venue_year.strip()

        paper_results.append(PaperResult(
            paper_id=f"gs_{i}",
            title=r.get("title", ""),
            authors=authors,
            year=year,
            venue=venue,
            citation_count=r.get("cite_count", 0),
            abstract=r.get("snippet", ""),
            doi=None,
            arxiv_id=None,
            open_access_url=r.get("pdf_url"),
            external_ids={
                k: v for k, v in [
                    ("cited_by_url", r.get("cited_by_url")),
                    ("related_url", r.get("related_url")),
                ] if v
            },
        ))
    return paper_results


_SCHOLAR_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

_SCHOLAR_PROFILES_BASE = "https://scholar.google.com/citations"


async def _fetch_scholar_page(url: str) -> str:
    """Fetch a Google Scholar page using headless Chromium with stealth.

    Returns the page HTML, or empty string on failure.
    """
    from playwright.async_api import async_playwright
    from playwright_stealth import Stealth

    try:
        stealth = Stealth()
        async with async_playwright() as pw:
            stealth.hook_playwright_context(pw)
            browser = await pw.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            context = await browser.new_context(
                user_agent=_SCHOLAR_USER_AGENT,
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
            )
            page = await context.new_page()
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            html = await page.content()
            await browser.close()
    except Exception as exc:
        logger.warning("Google Scholar browser error: %s", exc)
        return ""

    return html


async def search_google_scholar(
    query: str,
    limit: int = 10,
    year_min: int | None = None,
    year_max: int | None = None,
    sort_by_date: bool = False,
    offset: int = 0,
) -> list[PaperResult]:
    """Search Google Scholar using a headless Chromium instance with stealth.

    Parameters
    ----------
    query : str
        Search query (supports Scholar operators).
    limit : int
        Maximum results to return (capped at 20 per page).
    year_min : int | None
        Earliest publication year (Scholar ``as_ylo`` param).
    year_max : int | None
        Latest publication year (Scholar ``as_yhi`` param).
    sort_by_date : bool
        If *True*, sort results by date instead of relevance.
    offset : int
        0-based result offset for pagination (Scholar ``start`` param).
    """
    url = f"{_SCHOLAR_BASE}?q={quote_plus(query)}&num={min(limit, 20)}&hl=en"
    if year_min is not None:
        url += f"&as_ylo={year_min}"
    if year_max is not None:
        url += f"&as_yhi={year_max}"
    if sort_by_date:
        url += "&scisbd=1"
    if offset > 0:
        url += f"&start={offset}"
    html = await _fetch_scholar_page(url)
    if not html:
        return []
    return _raw_to_paper_results(_parse_scholar_results(html))[:limit]


async def cited_by(
    title: str, limit: int = 10, cited_by_url: str | None = None,
) -> list[PaperResult]:
    """Find papers that cite a given work.

    If *cited_by_url* is provided (e.g. from a previous search result's
    ``external_ids``), it is followed directly.  Otherwise, Scholar is
    searched for *title* first to locate the "Cited by" link.
    """
    if cited_by_url is None:
        # Step 1: find the paper and its cited_by_url
        search_url = f"{_SCHOLAR_BASE}?q={quote_plus(title)}&num=1&hl=en"
        html = await _fetch_scholar_page(search_url)
        if not html:
            return []

        raw = _parse_scholar_results(html)
        if not raw or not raw[0].get("cited_by_url"):
            return []

        cited_by_url = raw[0]["cited_by_url"]

    # Step 2: follow the cited-by link
    if not cited_by_url.startswith("http"):
        cited_by_url = f"https://scholar.google.com{cited_by_url}"
    # Append limit and language
    sep = "&" if "?" in cited_by_url else "?"
    cited_by_url += f"{sep}num={min(limit, 20)}&hl=en"

    cited_html = await _fetch_scholar_page(cited_by_url)
    if not cited_html:
        return []

    return _raw_to_paper_results(_parse_scholar_results(cited_html))[:limit]


async def related_papers(
    title: str, limit: int = 10, related_url: str | None = None,
) -> list[PaperResult]:
    """Find papers related to a given work via Google Scholar.

    If *related_url* is provided (e.g. from a previous search result's
    ``external_ids``), it is followed directly.  Otherwise, Scholar is
    searched for *title* first to locate the "Related articles" link.
    """
    if related_url is None:
        search_url = f"{_SCHOLAR_BASE}?q={quote_plus(title)}&num=1&hl=en"
        html = await _fetch_scholar_page(search_url)
        if not html:
            return []

        raw = _parse_scholar_results(html)
        if not raw or not raw[0].get("related_url"):
            return []

        related_url = raw[0]["related_url"]

    if not related_url.startswith("http"):
        related_url = f"https://scholar.google.com{related_url}"
    sep = "&" if "?" in related_url else "?"
    related_url += f"{sep}num={min(limit, 20)}&hl=en"

    related_html = await _fetch_scholar_page(related_url)
    if not related_html:
        return []

    return _raw_to_paper_results(_parse_scholar_results(related_html))[:limit]


def _parse_author_profile(html: str) -> list[PaperResult]:
    """Parse a Google Scholar author profile page into PaperResult objects."""
    from html.parser import HTMLParser

    results: list[dict] = []

    class _ProfileParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self._in_row = False
            self._current: dict = {}
            self._capture: str | None = None
            self._buf: list[str] = []

        def handle_starttag(self, tag, attrs):
            ad = dict(attrs)
            cls = ad.get("class", "")

            # Each paper row
            if tag == "tr" and "gsc_a_tr" in cls:
                self._in_row = True
                self._current = {"title": "", "meta": "", "year": None, "cite_count": 0}
                return

            if not self._in_row:
                return

            # Title cell
            if tag == "a" and "gsc_a_at" in cls:
                self._capture = "title"
                self._buf = []

            # Author/venue cell
            if tag == "div" and "gs_gray" in cls:
                self._capture = "meta"
                self._buf = []

            # Citation count cell
            if tag == "a" and "gsc_a_ac" in cls:
                self._capture = "cites"
                self._buf = []

            # Year cell
            if tag == "span" and "gsc_a_h" in cls:
                self._capture = "year"
                self._buf = []

        def handle_endtag(self, tag):
            if self._capture == "title" and tag == "a":
                self._current["title"] = clean_text("".join(self._buf))
                self._capture = None
            elif self._capture == "meta" and tag == "div":
                self._current["meta"] = clean_text("".join(self._buf))
                self._capture = None
            elif self._capture == "cites" and tag == "a":
                txt = clean_text("".join(self._buf))
                if txt.isdigit():
                    self._current["cite_count"] = int(txt)
                self._capture = None
            elif self._capture == "year" and tag == "span":
                txt = clean_text("".join(self._buf))
                m = _re.search(r"\d{4}", txt)
                if m:
                    self._current["year"] = int(m.group())
                self._capture = None

            if tag == "tr" and self._in_row:
                if self._current.get("title"):
                    results.append(self._current.copy())
                self._in_row = False
                self._current = {}

        def handle_data(self, data):
            if self._capture is not None:
                self._buf.append(data)

    parser = _ProfileParser()
    parser.feed(html)

    paper_results: list[PaperResult] = []
    for i, r in enumerate(results):
        # meta is typically "Author1, Author2 - Venue, Year" or just "Venue, Year"
        meta = r.get("meta", "")
        parts = meta.split(" - ") if " - " in meta else [meta]
        authors_str = parts[0] if len(parts) >= 2 else ""
        venue = parts[1] if len(parts) >= 2 else parts[0]
        authors = [a.strip() for a in authors_str.split(",")] if authors_str else []
        # Strip year from venue if present
        venue = _re.sub(r",?\s*\d{4}\s*$", "", venue).strip(", ")

        paper_results.append(PaperResult(
            paper_id=f"gs_author_{i}",
            title=r.get("title", ""),
            authors=authors,
            year=r.get("year"),
            venue=venue,
            citation_count=r.get("cite_count", 0),
            abstract="",
            doi=None,
            arxiv_id=None,
            open_access_url=None,
            external_ids={},
        ))
    return paper_results


async def author_papers(author_name: str, limit: int = 20) -> list[PaperResult]:
    """Find papers by a specific author via their Google Scholar profile.

    Searches for the author on Scholar, follows their profile link if found,
    and parses the publications list.
    """
    # Search for the author's profile
    profile_search_url = (
        f"{_SCHOLAR_PROFILES_BASE}?view_op=search_authors"
        f"&mauthors={quote_plus(author_name)}&hl=en"
    )
    html = await _fetch_scholar_page(profile_search_url)
    if not html:
        return []

    # Extract the first profile link
    m = _re.search(r'href="(/citations\?user=[^"&]+)', html)
    if not m:
        logger.info("No Scholar profile found for: %s", author_name)
        return []

    profile_url = f"https://scholar.google.com{m.group(1)}&hl=en&sortby=pubdate&pagesize={min(limit, 100)}"
    profile_html = await _fetch_scholar_page(profile_url)
    if not profile_html:
        return []

    return _parse_author_profile(profile_html)[:limit]


# ── Semantic Scholar API ──────────────────────────────────────────

_S2_MAX_RETRIES = 3
_S2_RETRY_DELAYS = (1.0, 3.0, 10.0)


async def _s2_get(path: str, params: dict) -> httpx.Response | None:
    """GET from the S2 API with retry on 429 rate-limit responses."""
    import asyncio

    client = get_http_client()
    url = f"{_S2_BASE}{path}"
    for attempt in range(_S2_MAX_RETRIES):
        try:
            resp = await client.get(
                url, params=params, headers=_s2_headers(), timeout=_API_TIMEOUT,
            )
            if resp.status_code != 429:
                return resp
            delay = _S2_RETRY_DELAYS[min(attempt, len(_S2_RETRY_DELAYS) - 1)]
            logger.info("S2 rate-limited (429), retrying in %.0fs...", delay)
            await asyncio.sleep(delay)
        except httpx.HTTPError as exc:
            logger.warning("S2 request error: %s", exc)
            return None
    logger.warning("S2 rate limit persisted after %d retries", _S2_MAX_RETRIES)
    return None


async def search_semantic_scholar(
    query: str, limit: int = 10
) -> list[PaperResult]:
    """Search Semantic Scholar and return parsed results."""
    resp = await _s2_get(
        "/paper/search",
        params={"query": query, "limit": min(limit, 100), "fields": _S2_FIELDS},
    )
    if resp is None:
        return []
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("S2 search HTTP %s: %s", exc.response.status_code, exc)
        return []

    body = resp.json()
    return [_parse_paper(d) for d in (body.get("data") or [])]


async def get_paper_metadata(paper_id: str) -> PaperResult | None:
    """Fetch metadata for a single paper by S2 ID, DOI:xxx, ArXiv:xxx, etc."""
    resp = await _s2_get(
        f"/paper/{paper_id}",
        params={"fields": _S2_FIELDS},
    )
    if resp is None:
        return None
    if resp.status_code == 404:
        return None
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.warning("S2 paper HTTP %s: %s", exc.response.status_code, exc)
        return None

    return _parse_paper(resp.json())


# ── PDF resolution cascade ────────────────────────────────────────


def _resolve_pdf_sources(paper: PaperResult) -> list[tuple[str, str]]:
    """Return an ordered list of (source_name, url_or_doi) to try for PDF download."""
    sources: list[tuple[str, str]] = []
    if paper.open_access_url:
        sources.append(("Semantic Scholar Open Access", paper.open_access_url))
    if paper.arxiv_id:
        sources.append(("ArXiv", f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf"))
    if paper.doi:
        sources.append(("Unpaywall", paper.doi))  # needs API call first
        sources.append(("Publisher direct", f"https://doi.org/{paper.doi}"))
    # Scholar sidebar PDF — stashed during S2 enrichment when S2 also has an OA URL
    scholar_pdf = paper.external_ids.get("scholar_pdf_url")
    if scholar_pdf and scholar_pdf != paper.open_access_url:
        sources.append(("Scholar PDF", scholar_pdf))
    return sources


async def _unpaywall_pdf_url(doi: str) -> str | None:
    """Query Unpaywall for an open-access PDF URL.

    Checks ``best_oa_location`` first, then falls back to scanning all
    ``oa_locations`` in preference order so that author-hosted PDFs surfaced
    only in secondary locations are not silently missed.
    """
    client = get_http_client()
    try:
        resp = await client.get(
            f"{_UNPAYWALL_BASE}/{doi}",
            params={"email": _email()},
            timeout=_API_TIMEOUT,
        )
        if resp.status_code != 200:
            return None
        body = resp.json()

        # Collect all candidate locations: best first, then the rest
        best = body.get("best_oa_location") or {}
        all_locations: list[dict] = [best] if best else []
        for loc in body.get("oa_locations") or []:
            if loc not in all_locations:
                all_locations.append(loc)

        for loc in all_locations:
            url = loc.get("url_for_pdf") or loc.get("url")
            if url:
                return url
        return None
    except httpx.HTTPError as exc:
        logger.warning("Unpaywall error: %s", exc)
        return None


async def _core_pdf_url(doi: str | None, title: str | None) -> str | None:
    """Query the CORE API for an open-access PDF URL.

    CORE (core.ac.uk) aggregates open-access papers from thousands of
    repositories and often surfaces author-hosted preprints that are not
    indexed by Unpaywall.  Searches by DOI first; falls back to title.

    """

    client = get_http_client()
    headers = {"User-Agent": _USER_AGENT}
    core_api_key = os.environ.get("CORE_API_KEY")
    if core_api_key:
        headers["Authorization"] = f"Bearer {core_api_key}"

    queries: list[str] = []
    if doi:
        queries.append(f'doi:"{doi}"')
    if title:
        queries.append(f'title:"{title}"')

    for q in queries:
        try:
            resp = await client.get(
                f"{_CORE_BASE}/search/works",
                params={"q": q, "limit": 3},
                headers=headers,
                timeout=_API_TIMEOUT,
            )
            if resp.status_code != 200:
                continue
            data = resp.json()
            for result in data.get("results") or []:
                # Verify title match when searching by title
                if title and not doi:
                    result_title = result.get("title") or ""
                    if _title_similarity(title, result_title) < _S2_ENRICH_THRESHOLD:
                        continue
                # Prefer direct download URL
                dl_url = result.get("downloadUrl")
                if dl_url:
                    return dl_url
                # Fall back to any PDF link in the links array
                for link in result.get("links") or []:
                    if isinstance(link, dict) and link.get("type") == "download":
                        return link.get("url")
        except httpx.HTTPError as exc:
            logger.debug("CORE API error: %s", exc)
            continue

    return None


async def _try_download_pdf(url: str) -> bytes | None:
    """Download a PDF from *url*.  Returns bytes or None on failure."""
    client = get_http_client()
    try:
        resp = await client.get(url, timeout=_PDF_TIMEOUT)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.debug("PDF download failed (%s): %s", url, exc)
        return None

    content_type = resp.headers.get("content-type", "").lower()
    if "pdf" not in content_type and "octet-stream" not in content_type:
        logger.debug("Not a PDF (content-type: %s): %s", content_type, url)
        return None

    if len(resp.content) > _MAX_PDF_BYTES:
        logger.warning("PDF too large (%d bytes): %s", len(resp.content), url)
        return None

    return resp.content


def _pdf_title_matches(pdf_bytes: bytes, expected_title: str, threshold: float = 0.5) -> bool:
    """Return True if *expected_title* appears in the first two pages of the PDF.

    Uses a lower Jaccard threshold (default 0.5) than S2 enrichment to allow
    for minor OCR differences or truncated titles in the PDF header.
    Returns True when *expected_title* is empty (no check possible).
    """
    if not expected_title:
        return True
    text = _extract_text_from_pdf_bytes(pdf_bytes, pages=[1, 2])
    if not text:
        return True  # can't verify — assume OK rather than discarding
    return _title_similarity(expected_title, text) >= threshold


# ── In-memory PDF text extraction ─────────────────────────────────


def _extract_text_from_pdf_bytes(
    pdf_bytes: bytes, pages: list[int] | None = None
) -> str:
    """Extract cleaned text from in-memory PDF bytes.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file content.
    pages : list[int] | None
        1-based page numbers to extract.  ``None`` extracts all pages.

    Returns
    -------
    str
        Formatted text with ``--- Page N ---`` headers, or empty string on failure.
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        logger.warning("Failed to parse PDF: %s", exc)
        return ""

    total = len(reader.pages)
    indices = (
        [p - 1 for p in pages if 1 <= p <= total] if pages else list(range(total))
    )

    texts: list[str] = []
    for idx in indices:
        try:
            raw = reader.pages[idx].extract_text() or ""
        except Exception:
            continue
        cleaned = clean_text(raw)
        if cleaned:
            texts.append(f"--- Page {idx + 1} ---\n{cleaned}")

    if not texts:
        return ""
    return f"({total} pages)\n\n" + "\n\n".join(texts)


# ── High-level fetch ──────────────────────────────────────────────


async def _scholar_pdf_url_for_title(title: str) -> str | None:
    """Search Google Scholar for *title* and return a PDF sidebar URL if found.

    Only returns a URL when the top result has a Jaccard title similarity of
    at least ``_S2_ENRICH_THRESHOLD`` (0.85) to avoid downloading the wrong paper.
    """
    if not title:
        return None
    url = f"{_SCHOLAR_BASE}?q={quote_plus(title)}&num=3&hl=en"
    html = await _fetch_scholar_page(url)
    if not html:
        return None
    for result in _parse_scholar_results(html)[:3]:
        if _title_similarity(title, result.get("title", "")) >= _S2_ENRICH_THRESHOLD:
            pdf_url = result.get("pdf_url")
            if pdf_url:
                return pdf_url
    return None


async def _download_pdf_bytes(
    paper: PaperResult,
) -> tuple[bytes | None, str | None, list[tuple[str, str]]]:
    """Run the PDF resolution cascade and return raw bytes.

    Returns ``(pdf_bytes, source_name, tried)`` where *tried* records
    each source attempted and why it failed (empty on success).
    """
    sources = _resolve_pdf_sources(paper)
    tried: list[tuple[str, str]] = []

    for source_name, url_or_doi in sources:
        if source_name == "Unpaywall":
            resolved = await _unpaywall_pdf_url(url_or_doi)
            if not resolved:
                tried.append((source_name, "no OA location"))
                continue
            url = resolved
        else:
            url = url_or_doi

        pdf_bytes = await _try_download_pdf(url)
        if pdf_bytes is None:
            tried.append((source_name, "download failed or not PDF"))
            continue

        return pdf_bytes, source_name, tried

    # Second resort: CORE aggregator — often has author-hosted preprints not in Unpaywall
    core_url = await _core_pdf_url(paper.doi, paper.title)
    if core_url:
        pdf_bytes = await _try_download_pdf(core_url)
        if pdf_bytes is not None:
            if _pdf_title_matches(pdf_bytes, paper.title):
                return pdf_bytes, "CORE", tried
            logger.warning("CORE PDF title mismatch for '%s': %s", paper.title, core_url[:80])
            tried.append(("CORE", f"title mismatch: {core_url[:80]}"))
        else:
            tried.append(("CORE", f"download failed: {core_url[:80]}"))
    else:
        tried.append(("CORE", "no PDF found (CORE_API_KEY not set or no result)"))

    # Last resort: search Google Scholar for a sidebar PDF link
    scholar_pdf = await _scholar_pdf_url_for_title(paper.title)
    if scholar_pdf:
        pdf_bytes = await _try_download_pdf(scholar_pdf)
        if pdf_bytes is not None:
            return pdf_bytes, "Google Scholar", tried
        tried.append(("Google Scholar", "download failed or not PDF"))
    else:
        tried.append(("Google Scholar", "no PDF link found"))

    summary = "; ".join(f"{name}: {reason}" for name, reason in tried)
    logger.info("No PDF obtained for '%s'. Tried: %s", paper.title, summary)
    return None, None, tried


async def fetch_paper_pdf_text(
    paper: PaperResult, pages: list[int] | None = None
) -> tuple[str, str | None]:
    """Attempt to download and extract text from a paper's PDF.

    Returns
    -------
    tuple[str, str | None]
        ``(extracted_text, source_name)`` on success, or
        ``("", None)`` when no PDF could be obtained.  The caller should
        fall back to returning metadata/abstract in that case.
    """
    pdf_bytes, source_name, tried = await _download_pdf_bytes(paper)

    if pdf_bytes is not None:
        text = _extract_text_from_pdf_bytes(pdf_bytes, pages)
        if text:
            return text, source_name

    return "", None
