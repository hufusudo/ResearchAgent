"""Unit tests for OpenAlex, Semantic Scholar, cross-source enrichment, and MultiSourceRetriever."""

from __future__ import annotations

from core.retrieval.arxiv_client import deduplicate_and_number_papers
from core.retrieval.multi_source import MultiSourceRetriever
from core.retrieval.openalex_client import (
    parse_openalex_works,
    reconstruct_openalex_abstract,
)
from core.retrieval.semantic_scholar_client import parse_semantic_scholar_response
from core.schemas import Paper


def test_reconstruct_openalex_abstract_and_parse_works() -> None:
    inv = {
        "Hybrid": [0],
        "retrieval": [1],
        "improves": [2],
        "citation": [3],
        "grounding.": [4],
    }
    assert reconstruct_openalex_abstract(inv) == "Hybrid retrieval improves citation grounding."

    payload = {
        "results": [
            {
                "id": "https://openalex.org/W4390001234",
                "display_name": "Agentic Literature Grounding",
                "publication_year": 2025,
                "cited_by_count": 42,
                "doi": "https://doi.org/10.1234/example",
                "authorships": [
                    {"author": {"display_name": "Alice Chen"}},
                    {"author": {"display_name": "Bob Patel"}},
                ],
                "primary_location": {"source": {"display_name": "ACL"}},
                "best_oa_location": {"pdf_url": "https://arxiv.org/pdf/2501.00001.pdf"},
                "abstract_inverted_index": inv,
                "topics": [{"display_name": "Information Retrieval"}],
            }
        ]
    }
    papers = parse_openalex_works(payload)
    assert len(papers) == 1
    p = papers[0]
    assert p.source == "openalex"
    assert p.external_id == "W4390001234"
    assert p.citation_count == 42
    assert p.venue == "ACL"
    assert p.pdf_url == "https://arxiv.org/pdf/2501.00001.pdf"
    assert p.abstract == "Hybrid retrieval improves citation grounding."
    assert p.authors == ["Alice Chen", "Bob Patel"]


def test_parse_semantic_scholar_response() -> None:
    payload = {
        "data": [
            {
                "paperId": "s2abc123",
                "title": "Self-Reflective RAG Benchmarks",
                "year": 2024,
                "abstract": "Evaluates bounded reflection in academic QA.",
                "citationCount": 19,
                "venue": "EMNLP",
                "url": "https://www.semanticscholar.org/paper/s2abc123",
                "openAccessPdf": {"url": "https://example.org/paper.pdf"},
                "authors": [{"name": "Jane Doe"}],
            }
        ]
    }
    papers = parse_semantic_scholar_response(payload)
    assert len(papers) == 1
    assert papers[0].source == "semantic_scholar"
    assert papers[0].citation_count == 19
    assert papers[0].venue == "EMNLP"
    assert papers[0].pdf_url == "https://example.org/paper.pdf"


def test_deduplicate_enriches_metadata_across_sources() -> None:
    arxiv_hit = Paper(
        external_id="2402.03367",
        source="arxiv",
        title="Hybrid Dense and BM25 Retrieval",
        authors=["David Kim"],
        year=2024,
        abstract="Detailed arXiv abstract describing dense and BM25 hybrid search with RRF.",
        url="https://arxiv.org/abs/2402.03367",
        pdf_url="https://arxiv.org/pdf/2402.03367.pdf",
    )
    openalex_hit = Paper(
        external_id="W999",
        source="openalex",
        title="Hybrid Dense and BM25 Retrieval!",
        authors=["David Kim"],
        year=2024,
        abstract="Short",
        url="https://openalex.org/W999",
        citation_count=37,
        venue="SIGIR",
    )

    merged = deduplicate_and_number_papers([arxiv_hit, openalex_hit])
    assert len(merged) == 1
    assert merged[0].paper_id == "P01"
    assert merged[0].citation_count == 37
    assert merged[0].venue == "SIGIR"
    assert "Detailed arXiv abstract" in merged[0].abstract


def test_multi_source_retriever_tolerates_partial_source_failure() -> None:
    class _FailingArxiv:
        def search(self, query: str, max_results: int = 10):
            raise RuntimeError("arXiv temporary 503")

        def download_pdf(self, pdf_url: str, dest_path):
            return None

    class _OkOpenAlex:
        def search(self, query: str, max_results: int = 10):
            return [
                Paper(
                    external_id="W111",
                    source="openalex",
                    title="OpenAlex Paper",
                    year=2025,
                    abstract="OpenAlex abstract.",
                    url="https://openalex.org/W111",
                )
            ]

    retriever = MultiSourceRetriever(
        sources=["arxiv", "openalex"],
        arxiv_client=_FailingArxiv(),  # type: ignore[arg-type]
        openalex_client=_OkOpenAlex(),  # type: ignore[arg-type]
    )
    hits = retriever.search("agentic rag", max_results=5)
    assert len(hits) == 1
    assert hits[0].source == "openalex"


def test_openalex_resolves_arxiv_doi_and_abs_url_to_direct_pdf() -> None:
    payload = {
        "results": [
            {
                "id": "https://openalex.org/W4400000001",
                "display_name": "Agentic AutoSurvey via arXiv DOI",
                "publication_year": 2025,
                "doi": "https://doi.org/10.48550/arxiv.2509.18661",
                "best_oa_location": {},
                "open_access": {},
                "abstract_inverted_index": {"Agentic": [0], "survey.": [1]},
            },
            {
                "id": "https://openalex.org/W4400000002",
                "display_name": "ArXiv Landing Page OA URL",
                "publication_year": 2024,
                "doi": "",
                "best_oa_location": {},
                "open_access": {"oa_url": "https://arxiv.org/abs/2401.12345v2"},
                "abstract_inverted_index": {"Landing": [0], "page.": [1]},
            },
        ]
    }
    papers = parse_openalex_works(payload)
    assert len(papers) == 2
    assert papers[0].pdf_url == "https://arxiv.org/pdf/2509.18661.pdf"
    assert papers[0].full_text_available is True
    assert papers[1].pdf_url == "https://arxiv.org/pdf/2401.12345v2.pdf"
    assert papers[1].full_text_available is True


def test_deduplicate_prefers_direct_pdf_url_over_landing_page() -> None:
    openalex_hit = Paper(
        external_id="W555",
        source="openalex",
        title="Direct PDF Preference Study",
        year=2025,
        abstract="Detailed study of direct PDF resolution across scholarly sources.",
        url="https://doi.org/10.48550/arxiv.2505.00001",
        pdf_url="https://example.org/landing-page",
    )
    arxiv_hit = Paper(
        external_id="2505.00001",
        source="arxiv",
        title="Direct PDF Preference Study",
        year=2025,
        abstract="Detailed study of direct PDF resolution across scholarly sources.",
        url="https://arxiv.org/abs/2505.00001",
        pdf_url="https://arxiv.org/pdf/2505.00001.pdf",
    )

    merged = deduplicate_and_number_papers([openalex_hit, arxiv_hit])
    assert len(merged) == 1
    assert merged[0].pdf_url == "https://arxiv.org/pdf/2505.00001.pdf"
