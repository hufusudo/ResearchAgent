"""End-to-end Stage-1 LangGraph workflow tests using offline mocks and local PDFs."""

from __future__ import annotations

import json
from pathlib import Path

import pymupdf

from core.config import Settings
from core.graph.workflow import run_research
from core.schemas import TraceEvent


def _write_local_pdf(path: Path) -> None:
    doc = pymupdf.open()
    doc.set_metadata({"title": "Local Grounded RAG Benchmarks", "author": "Local Author"})
    page = doc.new_page()
    page.insert_textbox(
        pymupdf.Rect(50, 50, 545, 780),
        (
            "1. Introduction\n"
            "Local grounded RAG evaluation benchmarks measure citation accuracy and claim faithfulness.\n\n"
            "2. Experiments\n"
            "Combining dense embeddings with BM25 lexical search improves multi-hop recall."
        ),
        fontsize=10,
    )
    doc.save(path)
    doc.close()


def test_stage1_workflow_end_to_end_mock(tmp_path: Path) -> None:
    local_dir = tmp_path / "local_pdfs"
    local_dir.mkdir()
    _write_local_pdf(local_dir / "local_rag_paper.pdf")

    settings = Settings(
        llm_api_key="",
        embedding_provider="local_hash",
        max_papers_per_query=6,
        top_k_papers=4,
        max_iterations=2,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )

    events: list[TraceEvent] = []
    final_state, out_dir = run_research(
        topic="Agentic RAG for academic literature review",
        settings=settings,
        mock=True,
        download_pdfs=False,
        local_pdf_dir=local_dir,
        persist_rag=False,
        on_event=events.append,
    )

    assert [e.node for e in events] == [
        "planner",
        "retriever_screener",
        "ingestion",
        "extractor",
        "evaluator",
        "query_rewriter",
        "retriever_screener",
        "ingestion",
        "extractor",
        "evaluator",
        "synthesizer",
        "verifier",
    ]

    plan = final_state["plan"]
    assert len(plan.sub_questions) >= 3

    selected = final_state["selected_papers"]
    # Initial top_k_papers=4 plus up to 2 supplemental papers discovered during reflection.
    assert len(selected) >= 5
    assert [p.paper_id for p in selected] == [f"P{i:02d}" for i in range(1, len(selected) + 1)]
    assert final_state["iteration"] == 1
    assert len(final_state["evaluation_history"]) == 2
    assert final_state["latest_evaluation"].stop_reason == "evidence_sufficient"
    assert len(final_state["rewrite_history"]) == 1

    chunks = final_state["chunks"]
    assert len(chunks) >= len(selected)
    valid_chunk_ids = {c.chunk_id for c in chunks}

    evidence_map = final_state["evidence_by_subquestion"]
    assert len(evidence_map) == len(plan.sub_questions)
    total_evidence = sum(len(items) for items in evidence_map.values())
    assert total_evidence > 0
    for items in evidence_map.values():
        for item in items:
            assert item.source_chunk_ids
            assert all(cid in valid_chunk_ids for cid in item.source_chunk_ids)

    ver_report = final_state["verification_report"]
    assert ver_report is not None
    assert ver_report.reference_validity_rate == 1.0
    assert ver_report.invalid_citations == []

    draft = final_state["draft_markdown"]
    assert "| Paper |" in draft
    assert "[P01]" in draft
    assert "## References" in draft
    assert "## Citation Verification Summary" in draft
    for paper in selected:
        assert f"**[{paper.paper_id}]** {paper.title}" in draft

    report_path = out_dir / "report.md"
    state_path = out_dir / "state.json"
    assert report_path.exists()
    assert state_path.exists()

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["topic"] == "Agentic RAG for academic literature review"
    assert len(persisted["selected_papers"]) == len(selected)


def test_local_pdf_always_retained_even_with_many_online_candidates(tmp_path: Path) -> None:
    from core.schemas import Paper as SchemaPaper

    local_dir = tmp_path / "local_pdfs"
    local_dir.mkdir()
    _write_local_pdf(local_dir / "priority_local_paper.pdf")

    class _ManyOnlineClient:
        def search(self, query: str, max_results: int = 10) -> list[SchemaPaper]:
            return [
                SchemaPaper(
                    external_id=f"2501.{idx:05d}",
                    source="arxiv",
                    title=f"Online Survey Paper {idx}",
                    authors=["Author A"],
                    year=2025,
                    abstract=f"Abstract for online paper {idx} about hybrid RAG.",
                    url=f"https://arxiv.org/abs/2501.{idx:05d}",
                )
                for idx in range(1, 36)
            ]

        def download_pdf(self, pdf_url: str, dest_path: Path) -> Path | None:
            return None

    settings = Settings(
        llm_api_key="",
        embedding_provider="local_hash",
        max_papers_per_query=35,
        top_k_papers=5,
        max_iterations=0,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )

    final_state, _ = run_research(
        topic="Agentic RAG for academic literature review",
        settings=settings,
        mock=True,
        download_pdfs=False,
        local_pdf_dir=local_dir,
        persist_rag=False,
        arxiv_override=_ManyOnlineClient(),  # type: ignore[arg-type]
    )

    selected = final_state["selected_papers"]
    assert len(selected) == 5
    local_papers = [p for p in selected if p.source == "local_pdf"]
    assert len(local_papers) == 1
    assert local_papers[0].ingest_status == "full_text"
