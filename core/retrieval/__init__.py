"""Retrieval, PDF parsing, and Hybrid RAG components."""

from core.retrieval.arxiv_client import (
    ArxivClient,
    deduplicate_and_number_papers,
    parse_arxiv_atom,
)
from core.retrieval.hybrid_rag import HybridRAGIndex, tokenize
from core.retrieval.multi_source import MultiSourceRetriever
from core.retrieval.openalex_client import (
    OpenAlexClient,
    parse_openalex_works,
    reconstruct_openalex_abstract,
)
from core.retrieval.pdf_parser import (
    chunk_abstract_only,
    chunk_text_blocks,
    detect_section_heading,
    load_local_pdfs,
    parse_pdf_to_chunks,
    rebind_chunk_paper_id,
)
from core.retrieval.semantic_scholar_client import (
    SemanticScholarClient,
    parse_semantic_scholar_response,
)

__all__ = [
    "ArxivClient",
    "HybridRAGIndex",
    "MultiSourceRetriever",
    "OpenAlexClient",
    "SemanticScholarClient",
    "chunk_abstract_only",
    "chunk_text_blocks",
    "deduplicate_and_number_papers",
    "detect_section_heading",
    "load_local_pdfs",
    "parse_arxiv_atom",
    "parse_openalex_works",
    "parse_pdf_to_chunks",
    "parse_semantic_scholar_response",
    "rebind_chunk_paper_id",
    "reconstruct_openalex_abstract",
    "tokenize",
]
