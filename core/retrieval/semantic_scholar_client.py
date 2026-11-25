"""Semantic Scholar Academic Graph search client."""

from __future__ import annotations

import time
from typing import Any

import httpx

from core.schemas import Paper

S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_FIELDS = "paperId,title,authors,year,abstract,citationCount,venue,url,openAccessPdf,externalIds"


def parse_semantic_scholar_response(payload: dict[str, Any]) -> list[Paper]:
    """Parse Semantic Scholar /paper/search JSON into normalized Paper models."""
    rows = payload.get("data") or []
    papers: list[Paper] = []

    for row in rows:
        paper_id = str(row.get("paperId") or "").strip()
        title = " ".join(str(row.get("title") or "").split())
        if not paper_id or not title:
            continue

        year = int(row.get("year") or 0)
        abstract = " ".join(str(row.get("abstract") or "").split())
        venue = str(row.get("venue") or "").strip()
        if not abstract:
            abstract = f"{title} ({venue or 'Semantic Scholar'}, {year or 'n.d.'})."

        authors = [
            str(a.get("name") or "").strip()
            for a in (row.get("authors") or [])
            if str(a.get("name") or "").strip()
        ]
        cit = row.get("citationCount")
        citation_count = int(cit) if cit is not None else None

        oa_pdf = row.get("openAccessPdf") or {}
        pdf_url = str(oa_pdf.get("url") or "").strip() or None
        url = str(row.get("url") or "").strip() or (
            f"https://www.semanticscholar.org/paper/{paper_id}"
        )

        papers.append(
            Paper(
                external_id=paper_id,
                source="semantic_scholar",
                title=title,
                authors=authors,
                year=year,
                abstract=abstract,
                url=url,
                pdf_url=pdf_url,
                citation_count=citation_count,
                venue=venue,
                full_text_available=bool(pdf_url),
            )
        )

    return papers


class SemanticScholarClient:
    """Synchronous HTTP client for Semantic Scholar paper search."""

    def __init__(
        self,
        api_key: str = "",
        timeout: float = 30.0,
        max_retries: int = 2,
        base_delay: float = 1.5,
    ) -> None:
        self.api_key = api_key.strip()
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.headers = {"User-Agent": "ReacherAgent/0.2"}
        if self.api_key:
            self.headers["x-api-key"] = self.api_key

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        """Query Semantic Scholar Graph API for papers matching query."""
        cleaned = " ".join(query.strip().split())
        if not cleaned:
            return []

        params = {
            "query": cleaned,
            "limit": max(1, min(max_results, 50)),
            "fields": S2_FIELDS,
        }
        last_error: Exception | None = None
        with httpx.Client(
            timeout=self.timeout,
            headers=self.headers,
            follow_redirects=True,
        ) as client:
            for attempt in range(self.max_retries):
                try:
                    response = client.get(S2_SEARCH_URL, params=params)
                    if response.status_code in {429, 500, 502, 503, 504}:
                        last_error = RuntimeError(
                            f"Semantic Scholar API returned HTTP {response.status_code}"
                        )
                        if attempt < self.max_retries - 1:
                            time.sleep(self.base_delay * (2**attempt))
                            continue
                        response.raise_for_status()
                    response.raise_for_status()
                    return parse_semantic_scholar_response(response.json())
                except httpx.HTTPError as exc:
                    last_error = exc
                    if attempt < self.max_retries - 1:
                        time.sleep(self.base_delay * (2**attempt))
                        continue
                    raise RuntimeError(
                        f"Semantic Scholar search failed for '{cleaned}': {exc}"
                    ) from exc

        raise RuntimeError(f"Semantic Scholar search failed for '{cleaned}': {last_error}")
