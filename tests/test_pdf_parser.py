"""Unit tests for PDF parsing, section tracking, chunking, and fallback behavior."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from core.retrieval.pdf_parser import (
    chunk_abstract_only,
    chunk_text_blocks,
    detect_section_heading,
    load_local_pdfs,
    parse_pdf_to_chunks,
    rebind_chunk_paper_id,
)
from core.schemas import Paper


def _create_sample_pdf(path: Path) -> None:
    doc = pymupdf.open()
    doc.set_metadata({"title": "Sample Agentic RAG Study", "author": "Jane Doe, John Roe"})

    page1 = doc.new_page()
    page1_text = (
        "1. Introduction\n"
        + "Autonomous literature review agents combine structured planning with grounded retrieval. " * 6
        + "\n\n2. Methodology\n"
        + "We integrate dense embeddings with BM25 lexical matching using reciprocal rank fusion. " * 6
    )
    page1.insert_textbox(pymupdf.Rect(50, 50, 545, 780), page1_text, fontsize=10)

    page2 = doc.new_page()
    page2_text = (
        "3. Conclusion\n"
        + "Experimental results on HotpotQA and QASPER confirm lower citation hallucination rates. " * 5
    )
    page2.insert_textbox(pymupdf.Rect(50, 50, 545, 780), page2_text, fontsize=10)

    doc.save(path)
    doc.close()


def test_detect_section_heading() -> None:
    assert detect_section_heading("1. Introduction") == "1. Introduction"
    assert detect_section_heading("Methodology") == "Methodology"
    assert detect_section_heading("2.1 Experimental Setup") == "2.1 Experimental Setup"
    assert detect_section_heading("This is a normal sentence ending with a period.") is None


def test_parse_pdf_to_chunks_preserves_pages_and_sections(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _create_sample_pdf(pdf_path)

    paper = Paper(
        paper_id="P01",
        external_id="2405.00001",
        source="arxiv",
        title="Sample Agentic RAG Study",
        year=2024,
        abstract="Abstract fallback text.",
        url="https://arxiv.org/abs/2405.00001",
    )
    chunks = parse_pdf_to_chunks(paper, pdf_path, chunk_size=350, chunk_overlap=60)

    assert len(chunks) >= 3
    assert chunks[0].chunk_id == "P01-C01"
    assert chunks[1].chunk_id == "P01-C02"
    assert all(c.paper_id == "P01" for c in chunks)
    assert {c.page for c in chunks} == {1, 2}
    sections = {c.section for c in chunks}
    assert any("Introduction" in s or "Methodology" in s for s in sections)
    assert any("Conclusion" in s for s in sections)


def test_corrupt_pdf_falls_back_cleanly(tmp_path: Path) -> None:
    bad_pdf = tmp_path / "corrupt.pdf"
    bad_pdf.write_bytes(b"not a valid pdf header")

    paper = Paper(
        paper_id="P02",
        external_id="2405.00002",
        source="arxiv",
        title="Corrupt PDF Paper",
        year=2024,
        abstract="Fallback abstract describing hybrid retrieval.",
        url="https://arxiv.org/abs/2405.00002",
    )
    assert parse_pdf_to_chunks(paper, bad_pdf) == []

    fallback_chunks = chunk_abstract_only(paper)
    assert len(fallback_chunks) == 1
    assert fallback_chunks[0].chunk_id == "P02-C01"
    assert fallback_chunks[0].section == "Abstract"
    assert "Fallback abstract describing hybrid retrieval." in fallback_chunks[0].text


def test_load_local_pdfs_and_rebind(tmp_path: Path) -> None:
    local_dir = tmp_path / "local_pdfs"
    local_dir.mkdir()
    _create_sample_pdf(local_dir / "agentic_rag.pdf")

    loaded = load_local_pdfs(local_dir, chunk_size=500, chunk_overlap=80)
    assert len(loaded) == 1
    paper, chunks = loaded[0]
    assert paper.source == "local_pdf"
    assert paper.title == "Sample Agentic RAG Study"
    assert len(chunks) >= 1

    rebound = rebind_chunk_paper_id(chunks, "P07")
    assert rebound[0].paper_id == "P07"
    assert rebound[0].chunk_id == "P07-C01"


def test_skip_trailing_references_and_max_chunks() -> None:
    paper = Paper(
        paper_id="P03",
        external_id="2501.00003",
        source="arxiv",
        title="Paper With References",
        year=2025,
        abstract="Abstract.",
        url="https://arxiv.org/abs/2501.00003",
    )
    pages = [
        (1, "1. Introduction\nMain contribution paragraph about agentic retrieval.\n"),
        (2, "2. Conclusion\nSummary of empirical findings.\n\n7 References cited\n[1] Smith et al. 2023.\n[2] Doe et al. 2024.\n"),
    ]
    chunks = chunk_text_blocks(paper, pages, chunk_size=500, chunk_overlap=50, skip_references=True)
    assert len(chunks) == 2
    assert all("Smith et al." not in c.text for c in chunks)

    capped = chunk_text_blocks(
        paper,
        [(1, "1. Introduction\n" + ("Word " * 200))],
        chunk_size=120,
        chunk_overlap=20,
        max_chunks=2,
    )
    assert len(capped) == 2
