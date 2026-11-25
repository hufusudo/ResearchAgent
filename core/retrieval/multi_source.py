"""Multi-source academic retriever combining arXiv, OpenAlex, and Semantic Scholar."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.retrieval.arxiv_client import ArxivClient
from core.retrieval.openalex_client import OpenAlexClient
from core.retrieval.semantic_scholar_client import SemanticScholarClient
from core.schemas import Paper


class MultiSourceRetriever:
    """Aggregates academic searches across enabled sources with per-source fault isolation."""

    def __init__(
        self,
        sources: list[str] | None = None,
        arxiv_client: ArxivClient | None = None,
        openalex_client: OpenAlexClient | None = None,
        semantic_scholar_client: SemanticScholarClient | None = None,
    ) -> None:
        valid = {"arxiv", "openalex", "semantic_scholar"}
        cleaned = [s.strip().lower() for s in (sources or ["arxiv", "openalex"]) if s.strip().lower() in valid]
        self.sources = cleaned or ["arxiv"]
        self.arxiv_client = arxiv_client or ArxivClient()
        self.openalex_client = openalex_client or OpenAlexClient()
        self.semantic_scholar_client = semantic_scholar_client or SemanticScholarClient()

    def _search_single_source(self, source: str, query: str, max_results: int) -> list[Paper]:
        if source == "arxiv":
            return self.arxiv_client.search(query=query, max_results=max_results)
        if source == "openalex":
            return self.openalex_client.search(query=query, max_results=max_results)
        if source == "semantic_scholar":
            return self.semantic_scholar_client.search(query=query, max_results=max_results)
        return []

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        """Search all enabled academic sources concurrently and combine raw hits."""
        hits: list[Paper] = []
        source_errors: list[str] = []

        if len(self.sources) <= 1:
            for source in self.sources:
                try:
                    hits.extend(self._search_single_source(source, query, max_results))
                except Exception as exc:
                    source_errors.append(f"{source}: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=len(self.sources)) as executor:
                futures = [
                    (source, executor.submit(self._search_single_source, source, query, max_results))
                    for source in self.sources
                ]
                for source, fut in futures:
                    try:
                        hits.extend(fut.result())
                    except Exception as exc:
                        source_errors.append(f"{source}: {exc}")

        if not hits and len(source_errors) == len(self.sources):
            raise RuntimeError("; ".join(source_errors))

        return hits

    def download_pdf(self, pdf_url: str, dest_path: Path) -> Path | None:
        """Download any open-access PDF URL with %PDF validation."""
        return self.arxiv_client.download_pdf(pdf_url, dest_path)
