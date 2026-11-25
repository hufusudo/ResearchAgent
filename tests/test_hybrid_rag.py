"""Unit tests for embeddings and HybridRAGIndex (dense, BM25, and RRF)."""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from core.llm.embeddings import LocalHashEmbedder
from core.retrieval.hybrid_rag import HybridRAGIndex
from core.schemas import TextChunk


def _sample_chunks() -> list[TextChunk]:
    return [
        TextChunk(
            chunk_id="P01-C01",
            paper_id="P01",
            title="GraphRAG-XL for Scientific Literature",
            year=2025,
            url="https://arxiv.org/abs/2501.00001",
            section="Methodology",
            page=3,
            text=(
                "GraphRAG-XL combines stateful LangGraph orchestration with hybrid retrieval "
                "evaluated on HotpotQA-2025 multi-hop benchmarks."
            ),
        ),
        TextChunk(
            chunk_id="P02-C01",
            paper_id="P02",
            title="Stateful Agentic Workflows",
            year=2024,
            url="https://arxiv.org/abs/2404.00002",
            section="Introduction",
            page=1,
            text=(
                "Stateful orchestration and semantic dense vector retrieval allow research agents "
                "to synthesize literature systematically across sub-questions."
            ),
        ),
        TextChunk(
            chunk_id="P03-C01",
            paper_id="P03",
            title="Tokenization Notes on HotpotQA-2025",
            year=2024,
            url="https://arxiv.org/abs/2403.00003",
            section="Appendix",
            page=9,
            text="A brief formatting table listing HotpotQA-2025 split sizes without agentic methods.",
        ),
        TextChunk(
            chunk_id="P04-C01",
            paper_id="P04",
            title="Ray Tracing for Volumetrics",
            year=2023,
            url="https://arxiv.org/abs/2301.00004",
            section="Rendering",
            page=4,
            text="Monte Carlo light transport and neural radiance caching improve 3D cloud rendering.",
        ),
    ]


def test_local_hash_embedder_is_deterministic_and_normalized() -> None:
    embedder = LocalHashEmbedder(dim=128)
    v1 = embedder.embed_query("Hybrid retrieval with BM25 and dense vectors")
    v2 = embedder.embed_query("Hybrid retrieval with BM25 and dense vectors")
    assert v1 == v2
    assert len(v1) == 128
    assert np.isclose(np.linalg.norm(v1), 1.0, atol=1e-5)


def test_hybrid_rag_dense_bm25_and_rrf_modes() -> None:
    embedder = LocalHashEmbedder(dim=384)
    index = HybridRAGIndex(
        collection_name=f"test_rag_{uuid4().hex[:8]}",
        embedder=embedder,
        persist_dir=None,
    )
    index.add_chunks(_sample_chunks())
    assert index.chunk_count == 4

    bm25_hits = index.search("GraphRAG-XL HotpotQA-2025", top_k=2, mode="bm25")
    assert len(bm25_hits) == 2
    assert bm25_hits[0].chunk.chunk_id == "P01-C01"
    assert bm25_hits[0].bm25_score > 0.0

    dense_hits = index.search("stateful orchestration literature synthesis", top_k=2, mode="dense")
    assert len(dense_hits) == 2
    assert dense_hits[0].chunk.paper_id in {"P01", "P02"}
    assert dense_hits[0].dense_score > 0.0

    hybrid_hits = index.search(
        "GraphRAG-XL stateful orchestration HotpotQA-2025",
        top_k=3,
        mode="hybrid",
    )
    assert len(hybrid_hits) == 3
    assert hybrid_hits[0].chunk.chunk_id == "P01-C01"
    assert hybrid_hits[0].rrf_score > 0.0
    assert hybrid_hits[0].dense_score > 0.0
    assert hybrid_hits[0].bm25_score > 0.0


def test_hybrid_rag_max_per_paper_diversity_cap() -> None:
    embedder = LocalHashEmbedder(dim=384)
    index = HybridRAGIndex(
        collection_name=f"test_div_{uuid4().hex[:8]}",
        embedder=embedder,
        persist_dir=None,
    )
    chunks = [
        TextChunk(
            chunk_id=f"P01-C{i:02d}",
            paper_id="P01",
            title="Long Monograph on Citation Grounding",
            year=2025,
            url="https://arxiv.org/abs/2501.00001",
            section="Method",
            page=i,
            text=f"Citation grounding hybrid RAG passage number {i} with BM25 lexical matching.",
        )
        for i in range(1, 6)
    ] + [
        TextChunk(
            chunk_id="P02-C01",
            paper_id="P02",
            title="Second Paper on Hybrid Retrieval",
            year=2024,
            url="https://arxiv.org/abs/2402.00002",
            section="Evaluation",
            page=2,
            text="Hybrid RAG citation grounding evaluation across multi-hop datasets.",
        ),
        TextChunk(
            chunk_id="P03-C01",
            paper_id="P03",
            title="Third Paper on Citation Attribution",
            year=2024,
            url="https://arxiv.org/abs/2403.00003",
            section="Results",
            page=4,
            text="BM25 and dense vector fusion improves citation attribution recall.",
        ),
    ]
    index.add_chunks(chunks)

    results = index.search(
        "hybrid RAG citation grounding BM25",
        top_k=4,
        mode="hybrid",
        max_per_paper=2,
    )
    assert len(results) == 4
    p01_count = sum(1 for r in results if r.chunk.paper_id == "P01")
    assert p01_count <= 2
    assert {r.chunk.paper_id for r in results} >= {"P01", "P02", "P03"}
