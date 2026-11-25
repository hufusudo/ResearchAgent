"""OpenAlex API client and abstract inverted-index reconstruction."""

from __future__ import annotations

import re
import time
from typing import Any

import httpx

from core.schemas import Paper

OPENALEX_WORKS_URL = "https://api.openalex.org/works"

_ARXIV_DOI_RE = re.compile(
    r"10\.48550/arxiv\.([a-z-]+/\d{7}|\d{4}\.\d{4,5}(?:v\d+)?)",
    re.IGNORECASE,
)
_ARXIV_URL_RE = re.compile(
    r"arxiv\.org/(?:abs|pdf)/([a-z-]+/\d{7}|\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?",
    re.IGNORECASE,
)


def _extract_arxiv_id(text: str | None) -> str | None:
    """Extract an arXiv identifier from an arXiv DOI or landing/PDF URL."""
    if not text:
        return None
    cleaned = str(text).strip()
    if not cleaned:
        return None
    for pattern in (_ARXIV_DOI_RE, _ARXIV_URL_RE):
        match = pattern.search(cleaned)
        if match:
            return match.group(1)
    return None


def _normalize_pdf_url(raw_url: str | None) -> str | None:
    """Normalize arXiv abs/pdf URLs into direct PDF URLs."""
    if not raw_url:
        return None
    cleaned = str(raw_url).strip()
    if not cleaned:
        return None
    arxiv_id = _extract_arxiv_id(cleaned)
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return cleaned


def _resolve_openalex_pdf_url(work: dict[str, Any], doi: str) -> str | None:
    """Resolve the best direct PDF URL from OpenAlex locations, arXiv DOIs, and OA links."""
    best_oa = work.get("best_oa_location") or {}
    primary_loc = work.get("primary_location") or {}
    oa_info = work.get("open_access") or {}

    direct_candidates: list[str | None] = [
        best_oa.get("pdf_url"),
        primary_loc.get("pdf_url"),
    ]
    for loc in work.get("locations") or []:
        if isinstance(loc, dict):
            direct_candidates.append(loc.get("pdf_url"))

    for candidate in direct_candidates:
        normalized = _normalize_pdf_url(candidate)
        if normalized:
            return normalized

    ids_obj = work.get("ids") or {}
    arxiv_sources: list[str | None] = [
        doi,
        oa_info.get("oa_url"),
        best_oa.get("landing_page_url"),
        primary_loc.get("landing_page_url"),
        ids_obj.get("arxiv") if isinstance(ids_obj, dict) else None,
    ]
    for loc in work.get("locations") or []:
        if isinstance(loc, dict):
            arxiv_sources.append(loc.get("landing_page_url"))

    for candidate in arxiv_sources:
        arxiv_id = _extract_arxiv_id(candidate)
        if arxiv_id:
            return f"https://arxiv.org/pdf/{arxiv_id}.pdf"

    return _normalize_pdf_url(oa_info.get("oa_url"))


def reconstruct_openalex_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct plain abstract text from OpenAlex's abstract_inverted_index."""
    if not inverted_index:
        return ""
    positioned_words: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if not positions:
            continue
        for pos in positions:
            positioned_words.append((int(pos), word))
    if not positioned_words:
        return ""
    positioned_words.sort(key=lambda pair: pair[0])
    return " ".join(word for _, word in positioned_words).strip()


def parse_openalex_works(payload: dict[str, Any]) -> list[Paper]:
    """Parse an OpenAlex /works JSON payload into normalized Paper objects."""
    results = payload.get("results") or []
    papers: list[Paper] = []

    for work in results:
        raw_id = str(work.get("id") or "").strip()
        title = " ".join(str(work.get("display_name") or work.get("title") or "").split())
        if not raw_id or not title:
            continue

        ext_id = raw_id.rstrip("/").split("/")[-1]
        year = int(work.get("publication_year") or 0)
        cited_by = work.get("cited_by_count")
        citation_count = int(cited_by) if cited_by is not None else None

        authors: list[str] = []
        for authorship in work.get("authorships") or []:
            author_obj = authorship.get("author") or {}
            name = str(author_obj.get("display_name") or "").strip()
            if name:
                authors.append(name)

        primary_loc = work.get("primary_location") or {}
        source_obj = primary_loc.get("source") or {}
        venue = str(source_obj.get("display_name") or "").strip()

        doi = str(work.get("doi") or "").strip()
        url = doi or raw_id
        pdf_url = _resolve_openalex_pdf_url(work, doi)

        abstract = reconstruct_openalex_abstract(work.get("abstract_inverted_index"))
        if not abstract:
            abstract = f"{title} ({venue or 'OpenAlex'}, {year or 'n.d.'})."

        categories: list[str] = []
        for topic in (work.get("topics") or work.get("concepts") or [])[:4]:
            cname = str(topic.get("display_name") or "").strip()
            if cname:
                categories.append(cname)

        papers.append(
            Paper(
                external_id=ext_id,
                source="openalex",
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                url=url,
                pdf_url=pdf_url,
                categories=categories,
                citation_count=citation_count,
                venue=venue,
                full_text_available=bool(pdf_url),
            )
        )

    return papers


class OpenAlexClient:
    """Synchronous HTTP client for querying the OpenAlex scholarly catalog."""

    def __init__(
        self,
        email: str = "",
        timeout: float = 30.0,
        max_retries: int = 2,
        base_delay: float = 1.0,
        user_agent: str = "ReacherAgent/0.2 (academic-survey-agent)",
    ) -> None:
        self.email = email.strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.headers = {"User-Agent": user_agent}

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        """Search OpenAlex works by relevance."""
        cleaned = " ".join(query.strip().split())
        if not cleaned:
            return []

        params: dict[str, Any] = {
            "search": cleaned,
            "per-page": max(1, min(max_results, 50)),
            "sort": "relevance_score:desc",
        }
        if self.email:
            params["mailto"] = self.email

        last_error: Exception | None = None
        with httpx.Client(
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True,
        ) as client:
            for attempt in range(self.max_retries):
                try:
                    response = client.get(OPENALEX_WORKS_URL, params=params)
                    if response.status_code in {429, 500, 502, 503, 504}:
                        last_error = RuntimeError(
                            f"OpenAlex API returned HTTP {response.status_code}"
                        )
                        if attempt < self.max_retries - 1:
                            time.sleep(self.base_delay * (2**attempt))
                            continue
                        response.raise_for_status()
                    response.raise_for_status()
                    return parse_openalex_works(response.json())
                except httpx.HTTPError as exc:
                    last_error = exc
                    if attempt < self.max_retries - 1:
                        time.sleep(self.base_delay * (2**attempt))
                        continue
                    raise RuntimeError(f"OpenAlex search failed for '{cleaned}': {exc}") from exc

        raise RuntimeError(f"OpenAlex search failed for '{cleaned}': {last_error}")
