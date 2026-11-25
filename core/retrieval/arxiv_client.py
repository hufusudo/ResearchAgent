"""arXiv Atom API client, parser, PDF downloader, and paper deduplication."""

from __future__ import annotations

import re
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

from core.schemas import Paper

ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_API_URL = "https://export.arxiv.org/api/query"


def _clean_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return " ".join(text.split())


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _extract_arxiv_id(raw_id_url: str) -> str:
    raw = raw_id_url.strip().rstrip("/").split("/")[-1]
    return re.sub(r"v\d+$", "", raw)


def parse_arxiv_atom(xml_text: str) -> list[Paper]:
    """Parse an arXiv Atom XML feed into normalized Paper models."""
    if not xml_text.strip():
        return []

    root = ET.fromstring(xml_text)
    papers: list[Paper] = []

    for entry in root.findall(f"{ATOM_NS}entry"):
        raw_id = _clean_whitespace(entry.findtext(f"{ATOM_NS}id"))
        title = _clean_whitespace(entry.findtext(f"{ATOM_NS}title"))
        abstract = _clean_whitespace(entry.findtext(f"{ATOM_NS}summary"))
        published = _clean_whitespace(entry.findtext(f"{ATOM_NS}published"))

        if not raw_id or not title:
            continue
        # arXiv returns a synthetic error entry when a query is malformed.
        if raw_id.startswith("http://arxiv.org/api/errors") or title.lower() == "error":
            continue

        year_match = re.search(r"(\d{4})", published)
        year = int(year_match.group(1)) if year_match else 0

        authors = [
            _clean_whitespace(author.findtext(f"{ATOM_NS}name"))
            for author in entry.findall(f"{ATOM_NS}author")
        ]
        authors = [a for a in authors if a]

        pdf_url: str | None = None
        html_url = raw_id
        for link in entry.findall(f"{ATOM_NS}link"):
            rel = link.attrib.get("rel", "")
            href = link.attrib.get("href", "").strip()
            link_title = link.attrib.get("title", "").lower()
            link_type = link.attrib.get("type", "").lower()
            if not href:
                continue
            if link_title == "pdf" or link_type == "application/pdf":
                pdf_url = href
            elif rel == "alternate" and href:
                html_url = href

        ext_id = _extract_arxiv_id(raw_id)
        if not pdf_url and ext_id:
            pdf_url = f"https://arxiv.org/pdf/{ext_id}.pdf"

        categories = [
            cat.attrib["term"].strip()
            for cat in entry.findall(f"{ATOM_NS}category")
            if cat.attrib.get("term")
        ]

        papers.append(
            Paper(
                external_id=ext_id,
                source="arxiv",
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                url=html_url,
                pdf_url=pdf_url,
                categories=categories,
                full_text_available=bool(pdf_url),
            )
        )

    return papers


def _looks_like_direct_pdf(url: str | None) -> bool:
    if not url:
        return False
    lowered = url.strip().lower()
    return lowered.endswith(".pdf") or ".pdf?" in lowered or "arxiv.org/pdf/" in lowered


def _enrich_paper(existing: Paper, incoming: Paper) -> None:
    """Merge richer metadata from duplicate hits across academic sources."""
    if incoming.citation_count is not None:
        if existing.citation_count is None or incoming.citation_count > existing.citation_count:
            existing.citation_count = incoming.citation_count
    if not existing.venue and incoming.venue:
        existing.venue = incoming.venue
    if incoming.pdf_url and (
        not existing.pdf_url
        or (_looks_like_direct_pdf(incoming.pdf_url) and not _looks_like_direct_pdf(existing.pdf_url))
    ):
        existing.pdf_url = incoming.pdf_url
        existing.full_text_available = True
    if not existing.authors and incoming.authors:
        existing.authors = list(incoming.authors)
    if len(existing.abstract) < 80 and len(incoming.abstract) > len(existing.abstract) + 40:
        existing.abstract = incoming.abstract


def deduplicate_and_number_papers(papers: list[Paper]) -> list[Paper]:
    """Deduplicate papers by external_id and normalized title, enriching metadata, and number P01..PNN."""
    by_ext: dict[str, Paper] = {}
    by_title: dict[str, Paper] = {}
    unique: list[Paper] = []

    for paper in papers:
        ext_key = f"{paper.source}:{paper.external_id.strip().lower()}" if paper.external_id else ""
        title_key = _normalize_title(paper.title)

        existing = (by_ext.get(ext_key) if ext_key else None) or (
            by_title.get(title_key) if title_key else None
        )
        if existing is not None:
            _enrich_paper(existing, paper)
            continue

        cloned = paper.model_copy(deep=True)
        if ext_key:
            by_ext[ext_key] = cloned
        if title_key:
            by_title[title_key] = cloned
        unique.append(cloned)

    for idx, paper in enumerate(unique, start=1):
        paper.paper_id = f"P{idx:02d}"

    return unique


class ArxivClient:
    """Synchronous HTTP client for querying arXiv metadata and downloading PDFs."""

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 2,
        base_delay: float = 1.0,
        min_search_interval: float = 1.0,
        rate_limit_cooldown: float = 120.0,
        user_agent: str = "ReacherAgent/0.1 (academic-survey-prototype)",
    ) -> None:
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self.base_delay = base_delay
        self.min_search_interval = min_search_interval
        self.rate_limit_cooldown = rate_limit_cooldown
        self.headers = {"User-Agent": user_agent}
        self._search_lock = threading.Lock()
        self._last_search_ts = 0.0
        self._cooldown_until = 0.0

    @staticmethod
    def _format_query(query: str) -> str:
        cleaned = " ".join(query.strip().split())
        if not cleaned:
            return "all:machine learning"
        if re.search(r"\b(?:all|ti|abs|au|cat):", cleaned):
            return cleaned
        return f"all:{cleaned}"

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        """Search arXiv Atom API with thread-safe pacing, 429 cooldown, and retry handling."""
        params = {
            "search_query": self._format_query(query),
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        last_error: Exception | None = None

        with self._search_lock:
            now = time.monotonic()
            if now < self._cooldown_until:
                raise RuntimeError("arXiv API in 429 rate-limit cooldown; skipping.")
            wait = self.min_search_interval - (now - self._last_search_ts)
            if wait > 0:
                time.sleep(wait)
            with httpx.Client(
                timeout=self.timeout,
                headers=self.headers,
                follow_redirects=True,
            ) as client:
                for attempt in range(self.max_retries):
                    try:
                        response = client.get(ARXIV_API_URL, params=params)
                        self._last_search_ts = time.monotonic()
                        if response.status_code == 429:
                            last_error = RuntimeError("arXiv API returned HTTP 429")
                            if attempt < self.max_retries - 1:
                                time.sleep(self.base_delay * (2**attempt))
                                continue
                            self._cooldown_until = time.monotonic() + self.rate_limit_cooldown
                            raise last_error
                        if response.status_code in {500, 502, 503, 504}:
                            last_error = RuntimeError(
                                f"arXiv API returned HTTP {response.status_code}"
                            )
                            if attempt < self.max_retries - 1:
                                time.sleep(self.base_delay * (2**attempt))
                                continue
                            response.raise_for_status()
                        response.raise_for_status()
                        return parse_arxiv_atom(response.text)
                    except httpx.HTTPError as exc:
                        self._last_search_ts = time.monotonic()
                        last_error = exc
                        if attempt < self.max_retries - 1:
                            time.sleep(self.base_delay * (2**attempt))
                            continue
                        raise RuntimeError(f"arXiv search failed for '{query}': {exc}") from exc

        raise RuntimeError(f"arXiv search failed for '{query}': {last_error}")

    def download_pdf(self, pdf_url: str, dest_path: Path) -> Path | None:
        """Download an academic PDF; returns None on any network or content error."""
        if not pdf_url:
            return None

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with httpx.Client(
                timeout=max(self.timeout, 45.0),
                headers=self.headers,
                follow_redirects=True,
            ) as client:
                for attempt in range(self.max_retries):
                    response = client.get(pdf_url)
                    if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries - 1:
                        time.sleep(self.base_delay * (2**attempt))
                        continue
                    if response.status_code != 200:
                        return None
                    content = response.content
                    if not content or not content[:1024].lstrip().startswith(b"%PDF"):
                        return None
                    dest_path.write_bytes(content)
                    return dest_path
        except Exception:
            return None

        return None
