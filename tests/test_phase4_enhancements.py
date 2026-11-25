"""Unit and integration tests for Phase 4 upgrades and eval framework."""

from __future__ import annotations

import ast
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx

from core.config import Settings
from core.graph.workflow import run_research
from core.llm.client import MockLLMClient
from core.llm.embeddings import OpenAIEmbedder
from core.retrieval.arxiv_client import ArxivClient
from core.schemas import ClaimVerificationBatch, ClaimVerificationItem
from eval.runner import run_benchmark_suite


class _FlawedThenFixableLLM(MockLLMClient):
    """Emits an invalid [P99] citation & phantom claim in initial synthesis, fixed on revision."""

    def invoke_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        if "citation revision editor" in system_prompt.lower():
            return super().invoke_text(system_prompt, user_prompt, temperature=temperature)
        base = super().invoke_text(system_prompt, user_prompt, temperature=temperature)
        return (
            base
            + "\n\nAn unverified phantom benchmark [P99] asserts magical 99.9% accuracy [P01]."
        )

    def invoke_structured(self, system_prompt: str, user_prompt: str, schema: Any, temperature: float = 0.1):
        if schema is ClaimVerificationBatch and "phantom" in user_prompt.lower():
            return ClaimVerificationBatch(
                items=[
                    ClaimVerificationItem(
                        claim="Phantom claim",
                        cited_paper_ids=["P01"],
                        matched_chunk_ids=["P01-C01"],
                        verdict="unsupported",
                        reason="Source excerpts do not mention phantom 99.9% accuracy.",
                    )
                    for _ in range(len([b for b in user_prompt.split("Claim ") if b.strip() and b[0].isdigit()]))
                ]
            )
        return super().invoke_structured(system_prompt, user_prompt, schema, temperature=temperature)


def test_verifier_guided_revision_loop_fixes_invalid_citation_and_caps_revisions(tmp_path: Path) -> None:
    settings = Settings(
        llm_api_key="",
        embedding_provider="local_hash",
        max_papers_per_query=4,
        top_k_papers=4,
        max_iterations=0,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )
    events = []
    state, _ = run_research(
        topic="Agentic RAG grounding",
        settings=settings,
        mock=True,
        download_pdfs=False,
        persist_rag=False,
        llm_override=_FlawedThenFixableLLM(trigger_reflection_on_first_round=False),
        max_verification_revisions=1,
        on_event=events.append,
    )

    assert state["revision_count"] == 1
    assert "reviser" in [e.node for e in events]
    assert state["pre_revision_verification_report"] is not None
    assert "P99" in state["pre_revision_verification_report"].invalid_citations
    final_vr = state["verification_report"]
    assert final_vr is not None
    assert final_vr.invalid_citations == []
    assert final_vr.reference_validity_rate == 1.0
    assert final_vr.claim_support_rate >= state["pre_revision_verification_report"].claim_support_rate
    assert state["draft_markdown"].count("## Citation Verification Summary") == 1

    # Verify max_verification_revisions=0 disables revision loop completely.
    events_no_rev = []
    state_no_rev, _ = run_research(
        topic="Agentic RAG grounding",
        settings=settings,
        mock=True,
        download_pdfs=False,
        persist_rag=False,
        llm_override=_FlawedThenFixableLLM(trigger_reflection_on_first_round=False),
        max_verification_revisions=0,
        on_event=events_no_rev.append,
    )
    assert state_no_rev["revision_count"] == 0
    assert "reviser" not in [e.node for e in events_no_rev]
    assert "P99" in state_no_rev["verification_report"].invalid_citations


def test_ablation_switches_and_usage_telemetry(tmp_path: Path) -> None:
    settings = Settings(
        llm_api_key="",
        embedding_provider="local_hash",
        max_papers_per_query=4,
        top_k_papers=4,
        max_iterations=0,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )
    llm = MockLLMClient(trigger_reflection_on_first_round=False)
    state, _ = run_research(
        topic="Hybrid RAG retrieval ablation",
        settings=settings,
        mock=True,
        download_pdfs=False,
        persist_rag=False,
        llm_override=llm,
        retrieval_mode="dense",
        max_per_paper=None,
        use_evidence_matrix=False,
    )
    assert all(len(v) == 0 for v in state["evidence_by_subquestion"].values())
    assert "## 3. Cross-Literature Comparison Matrix" in state["draft_markdown"]
    stats = llm.get_usage_stats()
    assert stats["calls"] >= 3
    assert stats["total_tokens"] > 0


def test_concurrent_openai_embedder_preserves_order_and_retries_429(monkeypatch: Any) -> None:
    attempts = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        batch = payload["input"]
        if batch[0] == "item-00" and attempts["count"] == 0:
            attempts["count"] += 1
            return httpx.Response(429, text="rate limited")
        data = [
            {"index": i, "embedding": [float(int(text.split("-")[1])), 1.0]}
            for i, text in enumerate(batch)
        ]
        return httpx.Response(200, json={"data": data})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def _mock_client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr("core.llm.embeddings.httpx.Client", _mock_client)

    embedder = OpenAIEmbedder(
        api_key="test-key",
        base_url="https://mock.local/v1",
        model="bge-m3",
        batch_size=3,
        max_workers=3,
        base_delay=0.01,
    )
    texts = [f"item-{i:02d}" for i in range(8)]
    vectors = embedder.embed_documents(texts)
    assert len(vectors) == 8
    assert [int(v[0]) for v in vectors] == list(range(8))
    stats = embedder.get_usage_stats()
    assert stats["calls"] == 3
    assert stats["texts_embedded"] == 8


def test_core_remains_strictly_decoupled_from_api_ui_and_eval() -> None:
    forbidden: list[str] = []
    for py_file in Path("core").rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("api", "ui", "eval")):
                        forbidden.append(f"{py_file}:{alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("api", "ui", "eval")):
                    forbidden.append(f"{py_file}:{node.module}")
    assert forbidden == []


def test_eval_benchmark_suite_mock_end_to_end(tmp_path: Path) -> None:
    out_dir = tmp_path / "eval_results"
    report_path = tmp_path / "report.md"
    settings = Settings(
        llm_api_key="",
        embedding_provider="local_hash",
        max_papers_per_query=5,
        top_k_papers=4,
        max_iterations=1,
        data_dir=str(tmp_path / "data"),
        output_dir=str(tmp_path / "outputs"),
    )

    summary = run_benchmark_suite(
        topic_ids=["T01", "T02"],
        mock=True,
        resume=False,
        top_k_papers=4,
        max_papers_per_query=5,
        max_iterations=1,
        download_pdfs=False,
        output_dir=out_dir,
        report_path=report_path,
        settings_override=settings,
    )

    assert summary["topic_count"] == 2
    for csv_name in (
        "system_comparison.csv",
        "ablation_retrieval.csv",
        "ablation_reflection.csv",
        "ablation_matrix.csv",
        "ablation_verifier_revision.csv",
    ):
        assert (out_dir / csv_name).exists()
    assert (out_dir / "summary.json").exists()
    assert report_path.exists()

    for svg_name in (
        "fig1_system_comparison.svg",
        "fig2_retrieval_ablation.svg",
        "fig3_reflection_ablation.svg",
        "fig4_matrix_and_revision.svg",
        "fig5_latency_and_cost.svg",
    ):
        svg_path = out_dir / "charts" / svg_name
        assert svg_path.exists()
        root = ET.fromstring(svg_path.read_text(encoding="utf-8"))
        assert root.tag.endswith("svg")

    sys_agg = summary["system_comparison_agg"]
    assert sys_agg["direct_llm"]["reference_validity_rate_mean"] < 1.0
    assert sys_agg["reacher_agent"]["reference_validity_rate_mean"] == 1.0
    assert 1.0 <= sys_agg["reacher_agent"]["rubric_overall_mean"] <= 5.0


def test_arxiv_client_429_cooldown_fails_fast(monkeypatch: Any) -> None:
    import time as pytime

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(429, text="rate limited")

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def _mock_client(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr("core.retrieval.arxiv_client.httpx.Client", _mock_client)

    client = ArxivClient(
        max_retries=2,
        base_delay=0.01,
        min_search_interval=0.0,
        rate_limit_cooldown=60.0,
    )
    import pytest

    with pytest.raises(RuntimeError, match="429"):
        client.search("test query 1", max_results=3)
    assert calls["count"] == 2

    t0 = pytime.monotonic()
    with pytest.raises(RuntimeError, match="cooldown"):
        client.search("test query 2", max_results=3)
    elapsed = pytime.monotonic() - t0
    assert elapsed < 0.1
    assert calls["count"] == 2

