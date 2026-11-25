"""Tests for bounded Evaluator reflection, query rewriting, and stopping guards."""

from __future__ import annotations

from pathlib import Path

from core.config import Settings
from core.graph.workflow import MockArxivClient, run_research
from core.llm.client import MockLLMClient
from core.schemas import (
    EvaluationResult,
    Paper,
    SubQuestionEvaluation,
    TraceEvent,
)


class _AlwaysInsufficientLLM(MockLLMClient):
    """Mock LLM that always asks for more retrieval to verify hard stopping guards."""

    def invoke_structured(self, system_prompt: str, user_prompt: str, schema, temperature: float = 0.1):
        if schema is EvaluationResult:
            return EvaluationResult(
                iteration=0,
                overall_sufficient=False,
                needs_more_retrieval=True,
                weak_subquestion_ids=["SQ2"],
                subquestion_evaluations=[
                    SubQuestionEvaluation(
                        sub_question_id="SQ1",
                        coverage_score=8.5,
                        sufficient=True,
                    ),
                    SubQuestionEvaluation(
                        sub_question_id="SQ2",
                        coverage_score=4.0,
                        sufficient=False,
                        gaps=["Still missing benchmark numbers."],
                    ),
                    SubQuestionEvaluation(
                        sub_question_id="SQ3",
                        coverage_score=8.5,
                        sufficient=True,
                    ),
                ],
                reasoning="Always flags SQ2 weak to test guards.",
            )
        return super().invoke_structured(system_prompt, user_prompt, schema, temperature=temperature)


class _DuplicateOnlyArxivClient(MockArxivClient):
    """Returns identical initial papers on every query so supplemental search finds 0 new papers."""

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        return [self._make_paper(item) for item in self._CATALOG[: min(4, max_results)]]


def test_reflection_loop_appends_papers_incrementally(tmp_path: Path) -> None:
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
    state, _ = run_research(
        topic="Agentic RAG grounding",
        settings=settings,
        mock=True,
        download_pdfs=False,
        persist_rag=False,
        on_event=events.append,
    )

    selected = state["selected_papers"]
    # Initial 4 papers (P01..P04) preserved, supplemental 2 papers appended as P05, P06.
    assert len(selected) == 6
    assert [p.paper_id for p in selected] == ["P01", "P02", "P03", "P04", "P05", "P06"]
    assert state["iteration"] == 1
    assert len(state["rewrite_history"]) == 1
    assert state["latest_evaluation"].stop_reason == "evidence_sufficient"


def test_max_iterations_hard_cap_prevents_infinite_loop(tmp_path: Path) -> None:
    settings = Settings(
        llm_api_key="",
        embedding_provider="local_hash",
        max_papers_per_query=6,
        top_k_papers=4,
        max_iterations=1,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )
    state, _ = run_research(
        topic="Agentic RAG grounding",
        settings=settings,
        mock=True,
        download_pdfs=False,
        persist_rag=False,
        llm_override=_AlwaysInsufficientLLM(),
    )

    assert state["iteration"] == 1
    assert state["latest_evaluation"].needs_more_retrieval is False
    assert state["latest_evaluation"].stop_reason == "max_iterations_reached"


def test_marginal_gain_circuit_breaker_stops_on_zero_new_papers(tmp_path: Path) -> None:
    settings = Settings(
        llm_api_key="",
        embedding_provider="local_hash",
        max_papers_per_query=4,
        top_k_papers=4,
        max_iterations=3,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )
    state, _ = run_research(
        topic="Agentic RAG grounding",
        settings=settings,
        mock=True,
        download_pdfs=False,
        persist_rag=False,
        llm_override=_AlwaysInsufficientLLM(),
        arxiv_override=_DuplicateOnlyArxivClient(),
    )

    # Round 1 returns 0 new unique papers -> circuit breaker stops immediately before max_iterations=3.
    assert state["iteration"] == 1
    assert state["last_new_paper_count"] == 0
    assert state["last_overlap_ratio"] == 1.0
    assert state["latest_evaluation"].needs_more_retrieval is False
    assert state["latest_evaluation"].stop_reason == "marginal_gain_circuit_breaker"
