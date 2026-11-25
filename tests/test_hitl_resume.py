"""Tests for Human-in-the-Loop (HITL) pause, plan edit, and SqliteSaver resume."""

from __future__ import annotations

from pathlib import Path

from core.config import Settings
from core.graph.workflow import resume_research_session, start_research_session


def test_hitl_pause_edit_and_resume_from_sqlite_checkpoint(tmp_path: Path) -> None:
    settings = Settings(
        llm_api_key="",
        embedding_provider="local_hash",
        max_papers_per_query=4,
        top_k_papers=4,
        max_iterations=1,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )

    paused_state, status, out_dir = start_research_session(
        topic="Agentic RAG literature synthesis",
        settings=settings,
        mock=True,
        download_pdfs=False,
        persist_rag=False,
        require_plan_approval=True,
    )

    assert status == "awaiting_approval"
    assert paused_state["plan"] is not None
    assert not paused_state.get("selected_papers")
    assert (out_dir / "state.json").exists()

    edited_plan = paused_state["plan"].model_copy(deep=True)
    edited_plan.refined_scope = "Edited HITL Scope: Grounded Agentic Literature Synthesis."
    edited_plan.sub_questions[0].search_queries.append("custom user approved query")

    final_state, final_dir = resume_research_session(
        run_id=paused_state["run_id"],
        approved_plan=edited_plan,
        settings=settings,
        mock=True,
        download_pdfs=False,
        persist_rag=False,
    )

    assert final_dir == out_dir
    assert final_state["plan"].refined_scope == "Edited HITL Scope: Grounded Agentic Literature Synthesis."
    assert "custom user approved query" in final_state["executed_queries"]
    assert len(final_state["selected_papers"]) >= 4
    assert "## References" in final_state["draft_markdown"]
    assert "## Citation Verification Summary" in final_state["draft_markdown"]
    assert (final_dir / "report.md").exists()
