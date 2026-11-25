"""System-level baseline runners: Direct LLM, Naive RAG, and ReacherAgent Full."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config import Settings
from core.graph.workflow import MockArxivClient, run_research
from core.llm.client import create_llm_client
from core.llm.embeddings import create_embedder
from core.retrieval.arxiv_client import ArxivClient, deduplicate_and_number_papers
from core.retrieval.hybrid_rag import HybridRAGIndex
from core.retrieval.multi_source import MultiSourceRetriever
from core.retrieval.openalex_client import OpenAlexClient
from core.retrieval.pdf_parser import chunk_abstract_only, parse_pdf_to_chunks
from core.schemas import Paper, ResearchPlan, ResearchState, TextChunk
from core.verifier.citation_verifier import (
    format_verification_summary_markdown,
    verify_citations,
)
from eval.benchmark_topics import BenchmarkTopic
from eval.metrics import (
    evaluate_subquestion_coverage,
    evaluate_synthesis_rubric,
    verify_external_references,
)


def _format_naive_references(papers: list[Paper]) -> str:
    if not papers:
        return "## References\n\n- No papers retrieved."
    lines = ["## References", ""]
    for p in papers:
        authors = ", ".join(p.authors[:4]) + (" et al." if len(p.authors) > 4 else "")
        authors = authors or "Unknown authors"
        lines.append(
            f"- **[{p.paper_id}]** {p.title} ({p.year or 'n.d.'}). {authors}. Source: {p.url}"
        )
    return "\n".join(lines)


def _extract_node_timings(state: ResearchState, total_elapsed: float) -> dict[str, float]:
    """Compute approximate per-node duration attribution from trace timestamps."""
    trace = state.get("trace") or []
    if not trace:
        return {}
    timestamps: list[datetime] = []
    for ev in trace:
        try:
            timestamps.append(datetime.fromisoformat(ev.timestamp))
        except Exception:
            return {}

    node_totals: dict[str, float] = {}
    # Trace events are emitted at node completion; attribute inter-event delta to the completing node.
    for idx in range(1, len(trace)):
        dt = max(0.0, (timestamps[idx] - timestamps[idx - 1]).total_seconds())
        node = trace[idx].node
        node_totals[node] = round(node_totals.get(node, 0.0) + dt, 3)

    accounted = sum(node_totals.values())
    first_node = trace[0].node
    first_est = max(0.0, total_elapsed - accounted)
    if first_est > 0:
        node_totals[first_node] = round(node_totals.get(first_node, 0.0) + first_est, 3)
    return node_totals


def run_direct_llm_baseline(
    topic: BenchmarkTopic,
    settings: Settings,
    canonical_plan: ResearchPlan,
    *,
    openalex_client: Any = None,
    arxiv_client: Any = None,
    mock: bool = False,
) -> dict[str, Any]:
    """Baseline A: Direct single-pass LLM survey without external retrieval."""
    llm = create_llm_client(settings, mock=mock)
    llm.reset_usage_stats()

    t0 = time.perf_counter()
    if mock:
        # Deliberately include one valid catalog paper and one hallucinated paper to test detection.
        draft = (
            f"# Literature Survey: {topic.title}\n\n"
            "## 1. Background and Scope\n"
            f"Direct parametric knowledge survey on {topic.title} [P01] [P02].\n\n"
            "## 2. Thematic Synthesis by Sub-Question\n"
            "### SQ1: Core Architectures\n"
            "Stateful orchestration improves workflow reliability [P01], while fabricated study [P02] "
            "claims 99.9% zero-shot perfection.\n\n"
            "## 3. Cross-Literature Comparison Matrix\n"
            "| Paper | Core Method | Dataset / Setting | Key Findings | Limitations |\n"
            "| :--- | :--- | :--- | :--- | :--- |\n"
            "| [P01] | Stateful multi-agent graph | Multi-hop QA | Improves coverage | Latency overhead |\n"
            "| [P02] | Imaginary universal solver | Synthetic | Perfect accuracy | Hallucinated ref |\n\n"
            "## 4. Research Gaps and Future Directions\n"
            "Key challenges include hallucination mitigation [P01] [P02].\n\n"
            "## References\n\n"
            "- **[P01]** Stateful Multi-Agent Literature Workflows with Directed Graphs (2024). Alice Chen, Bob Patel.\n"
            "- **[P02]** OmniSurvey: Zero-Hallucination Neural Literature Synthesizer at Scale (2025). John Doe.\n"
        )
    else:
        system_prompt = (
            "You are an academic survey author writing an English literature survey purely from your knowledge.\n"
            "Write a cohesive Markdown survey with:\n"
            "1. `# Literature Survey: <Topic>`\n"
            "2. `## 1. Background and Scope`\n"
            "3. `## 2. Thematic Synthesis by Sub-Question`\n"
            "4. `## 3. Cross-Literature Comparison Matrix` (Markdown table)\n"
            "5. `## 4. Research Gaps and Future Directions`\n"
            "6. `## References` listing 8 to 10 real academic papers formatted strictly as:\n"
            "   `- **[P01]** Full Paper Title (Year). Authors. Venue/Source.`\n"
            "Cite papers inline using `[P01]`, `[P02]`, etc."
        )
        user_prompt = (
            f"Research Topic: {topic.title}\n"
            f"English Domain Hint: {topic.english_query_hint}\n"
            f"Expected Aspects: {', '.join(topic.expected_aspects)}"
        )
        draft = llm.invoke_text(system_prompt=system_prompt, user_prompt=user_prompt, temperature=0.25)

    gen_elapsed = time.perf_counter() - t0
    llm_gen_usage = llm.get_usage_stats()

    judge_llm = create_llm_client(settings, mock=mock)
    ext_ver = verify_external_references(
        draft_markdown=draft,
        openalex_client=openalex_client,
        arxiv_client=arxiv_client,
        llm=judge_llm,
        mock=mock,
    )
    cov = evaluate_subquestion_coverage(draft, canonical_plan, judge_llm, mock=mock)
    rubric = evaluate_synthesis_rubric(draft, topic.title, judge_llm, mock=mock)

    return {
        "system": "direct_llm",
        "topic_id": topic.topic_id,
        "elapsed_seconds": round(gen_elapsed, 3),
        "paper_count": ext_ver.total_listed_references,
        "verified_paper_count": ext_ver.verified_references,
        "full_text_count": 0,
        "chunk_count": 0,
        "reference_validity_rate": ext_ver.effective_reference_validity_rate,
        "external_reference_validity_rate": ext_ver.external_reference_validity_rate,
        "claim_support_rate": ext_ver.effective_claim_support_rate,
        "raw_claim_support_rate": ext_ver.claim_support_rate,
        "subquestion_coverage_rate": cov.subquestion_coverage_rate,
        "avg_coverage_score": cov.avg_coverage_score,
        "rubric": rubric.model_dump(mode="json"),
        "llm_usage": llm_gen_usage,
        "embedding_usage": {"calls": 0, "texts_embedded": 0, "elapsed_seconds": 0.0},
        "external_verification": ext_ver.model_dump(mode="json"),
        "draft_chars": len(draft),
        "draft_markdown": draft,
    }


def run_naive_rag_baseline(
    topic: BenchmarkTopic,
    settings: Settings,
    canonical_plan: ResearchPlan,
    *,
    top_k_papers: int = 6,
    download_pdfs: bool = True,
    mock: bool = False,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Baseline B: Single-shot single-query Dense-only RAG without planning, matrix, or reflection."""
    llm = create_llm_client(settings, mock=mock)
    embedder = create_embedder(settings, force_local=mock)
    llm.reset_usage_stats()
    embedder.reset_usage_stats()

    if mock:
        search_client: Any = MockArxivClient()
    else:
        search_client = MultiSourceRetriever(
            sources=settings.academic_sources_list,
            arxiv_client=ArxivClient(),
            openalex_client=OpenAlexClient(email=settings.openalex_email),
        )

    t0 = time.perf_counter()
    raw_hits = search_client.search(
        query=topic.english_query_hint,
        max_results=max(top_k_papers * 2, 10),
    )
    unique_papers = deduplicate_and_number_papers(raw_hits)[:top_k_papers]
    selected_papers = [
        p.model_copy(update={"paper_id": f"P{idx:02d}", "ingest_status": "pending"})
        for idx, p in enumerate(unique_papers, start=1)
    ]

    pdf_dir = (work_dir or settings.data_path / "eval_naive_pdfs") / topic.topic_id

    def _ingest_one(paper: Paper) -> tuple[Paper, list[TextChunk]]:
        chunks: list[TextChunk] = []
        status = "abstract_only"
        full_ok = False
        if download_pdfs and paper.pdf_url:
            dest = pdf_dir / f"{paper.paper_id}.pdf"
            dl = search_client.download_pdf(paper.pdf_url, dest)
            if dl is not None:
                parsed = parse_pdf_to_chunks(paper=paper, pdf_path=dl)
                if parsed:
                    chunks = parsed
                    status = "full_text"
                    full_ok = True
        if not chunks:
            chunks = chunk_abstract_only(paper)
        return (
            paper.model_copy(update={"ingest_status": status, "full_text_available": full_ok}),
            chunks,
        )

    if selected_papers:
        if len(selected_papers) == 1 or mock:
            ingested_pairs = [_ingest_one(p) for p in selected_papers]
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(selected_papers))) as executor:
                ingested_pairs = list(executor.map(_ingest_one, selected_papers))
    else:
        ingested_pairs = []

    final_papers = [pair[0] for pair in ingested_pairs]
    all_chunks = [c for _, c_list in ingested_pairs for c in c_list]

    rag = HybridRAGIndex(
        collection_name=f"naive_{topic.topic_id}_{int(time.time())}",
        embedder=embedder,
        persist_dir=None,
    )
    rag.add_chunks(all_chunks)
    # Dense-only, no max_per_paper diversity cap (true Naive RAG baseline).
    retrieved = rag.search(
        query=topic.english_query_hint,
        top_k=12,
        mode="dense",
        max_per_paper=None,
    )

    papers_block = "\n".join(
        f"- [{p.paper_id}] {p.title} ({p.year}): {p.abstract[:250]}"
        for p in final_papers
    ) or "No papers found."
    chunks_block = "\n\n".join(
        f"[{r.chunk.chunk_id}] ({r.chunk.paper_id}, p.{r.chunk.page}): {r.chunk.text[:800]}"
        for r in retrieved
    ) or "No excerpts."

    naive_system = (
        "You are an academic writer producing a literature survey directly from the retrieved passages.\n"
        "Include sections:\n"
        "1. `# Literature Survey: <Topic>`\n"
        "2. `## 1. Background and Scope`\n"
        "3. `## 2. Thematic Synthesis`\n"
        "4. `## 3. Cross-Literature Comparison Matrix` (Markdown table)\n"
        "5. `## 4. Research Gaps and Future Directions`\n"
        "Cite inline using ONLY provided IDs like `[P01]`, `[P02]`. Do NOT write a References section."
    )
    naive_user = (
        f"Research Topic: {topic.title}\n\n"
        f"Retrieved Papers:\n{papers_block}\n\n"
        f"Retrieved Passages:\n{chunks_block}"
    )
    body = llm.invoke_text(system_prompt=naive_system, user_prompt=naive_user, temperature=0.25)
    cleaned_body = body.split("\n## References")[0].rstrip()
    draft = f"{cleaned_body}\n\n{_format_naive_references(final_papers)}\n"

    elapsed = time.perf_counter() - t0
    llm_usage = llm.get_usage_stats()
    emb_usage = embedder.get_usage_stats()

    judge_llm = create_llm_client(settings, mock=mock)
    ver_report = verify_citations(
        draft_markdown=draft,
        selected_papers=final_papers,
        chunks=all_chunks,
        llm=judge_llm,
        max_claims=16,
    )
    draft_with_ver = f"{draft.rstrip()}\n\n{format_verification_summary_markdown(ver_report)}\n"
    cov = evaluate_subquestion_coverage(draft_with_ver, canonical_plan, judge_llm, mock=mock)
    rubric = evaluate_synthesis_rubric(draft_with_ver, topic.title, judge_llm, mock=mock)

    full_text_count = sum(1 for p in final_papers if p.ingest_status == "full_text")
    return {
        "system": "naive_rag",
        "topic_id": topic.topic_id,
        "elapsed_seconds": round(elapsed, 3),
        "paper_count": len(final_papers),
        "full_text_count": full_text_count,
        "chunk_count": len(all_chunks),
        "reference_validity_rate": ver_report.reference_validity_rate,
        "claim_support_rate": ver_report.claim_support_rate,
        "checked_claims": ver_report.checked_claims,
        "supported_count": ver_report.supported_count,
        "partially_supported_count": ver_report.partially_supported_count,
        "unsupported_count": ver_report.unsupported_count,
        "invalid_citations": ver_report.invalid_citations,
        "subquestion_coverage_rate": cov.subquestion_coverage_rate,
        "avg_coverage_score": cov.avg_coverage_score,
        "rubric": rubric.model_dump(mode="json"),
        "llm_usage": llm_usage,
        "embedding_usage": emb_usage,
        "draft_chars": len(draft_with_ver),
        "draft_markdown": draft_with_ver,
    }


def run_reacher_full(
    topic: BenchmarkTopic,
    settings: Settings,
    *,
    top_k_papers: int = 6,
    max_papers_per_query: int = 5,
    max_iterations: int = 2,
    download_pdfs: bool = True,
    mock: bool = False,
) -> tuple[dict[str, Any], ResearchState, dict[str, Any]]:
    """System C: Full ReacherAgent 9-node workflow capturing Iter-0 snapshot and usage telemetry."""
    llm = create_llm_client(settings, mock=mock)
    embedder = create_embedder(settings, force_local=mock)
    llm.reset_usage_stats()
    embedder.reset_usage_stats()

    iter0_box: dict[str, Any] = {}

    def _capture_iter0(snap: dict[str, Any]) -> None:
        iter0_box.update(snap)

    t0 = time.perf_counter()
    final_state, out_dir = run_research(
        topic=topic.title,
        settings=settings,
        mock=mock,
        top_k_papers=top_k_papers,
        max_papers_per_query=max_papers_per_query,
        max_iterations=max_iterations,
        download_pdfs=download_pdfs,
        persist_rag=False,
        llm_override=llm,
        embedder_override=embedder,
        retrieval_mode="hybrid",
        max_per_paper=3,
        use_evidence_matrix=True,
        max_verification_revisions=1,
        on_iter0_snapshot=_capture_iter0,
    )
    elapsed = time.perf_counter() - t0
    llm_usage = llm.get_usage_stats()
    emb_usage = embedder.get_usage_stats()

    plan: ResearchPlan = final_state["plan"]
    selected_papers = final_state.get("selected_papers") or []
    chunks = final_state.get("chunks") or []
    draft = final_state.get("draft_markdown", "")
    ver_report = final_state.get("verification_report")
    pre_ver = final_state.get("pre_revision_verification_report") or ver_report
    eval_hist = final_state.get("evaluation_history") or []
    ev_map = final_state.get("evidence_by_subquestion") or {}

    judge_llm = create_llm_client(settings, mock=mock)
    cov = evaluate_subquestion_coverage(draft, plan, judge_llm, mock=mock)
    rubric = evaluate_synthesis_rubric(draft, topic.title, judge_llm, mock=mock)

    full_text_count = sum(1 for p in selected_papers if p.ingest_status == "full_text")
    node_timings = _extract_node_timings(final_state, elapsed)

    metrics = {
        "system": "reacher_agent",
        "topic_id": topic.topic_id,
        "run_id": final_state.get("run_id", ""),
        "output_dir": str(out_dir),
        "elapsed_seconds": round(elapsed, 3),
        "node_timings_seconds": node_timings,
        "paper_count": len(selected_papers),
        "candidate_count": len(final_state.get("candidate_papers") or []),
        "full_text_count": full_text_count,
        "chunk_count": len(chunks),
        "evidence_count": sum(len(v) for v in ev_map.values()),
        "reflection_iterations": final_state.get("iteration", 0),
        "stop_reason": (
            final_state["latest_evaluation"].stop_reason
            if final_state.get("latest_evaluation")
            else ""
        ),
        "revision_count": final_state.get("revision_count", 0),
        "reference_validity_rate": ver_report.reference_validity_rate if ver_report else 0.0,
        "claim_support_rate": ver_report.claim_support_rate if ver_report else 0.0,
        "pre_revision_validity_rate": pre_ver.reference_validity_rate if pre_ver else 0.0,
        "pre_revision_claim_support_rate": pre_ver.claim_support_rate if pre_ver else 0.0,
        "pre_revision_unsupported": pre_ver.unsupported_count if pre_ver else 0,
        "checked_claims": ver_report.checked_claims if ver_report else 0,
        "supported_count": ver_report.supported_count if ver_report else 0,
        "partially_supported_count": ver_report.partially_supported_count if ver_report else 0,
        "unsupported_count": ver_report.unsupported_count if ver_report else 0,
        "invalid_citations": ver_report.invalid_citations if ver_report else [],
        "subquestion_coverage_rate": cov.subquestion_coverage_rate,
        "avg_coverage_score": cov.avg_coverage_score,
        "rubric": rubric.model_dump(mode="json"),
        "llm_usage": llm_usage,
        "embedding_usage": emb_usage,
        "eval_history_count": len(eval_hist),
        "draft_chars": len(draft),
        "draft_markdown": draft,
    }
    return metrics, final_state, iter0_box
