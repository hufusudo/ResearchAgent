"""LangGraph Stage-2 workflow definition with reflection loop, verifier, and HITL."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import uuid4

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from core.config import Settings, get_settings
from core.graph.nodes import (
    ArxivClientProtocol,
    WorkflowDependencies,
    make_evaluator_node,
    make_extractor_node,
    make_ingestion_node,
    make_planner_node,
    make_query_rewriter_node,
    make_retriever_screener_node,
    make_reviser_node,
    make_synthesizer_node,
    make_verifier_node,
)
from core.llm.client import create_llm_client
from core.llm.embeddings import create_embedder
from core.retrieval.arxiv_client import ArxivClient
from core.retrieval.hybrid_rag import HybridRAGIndex
from core.retrieval.multi_source import MultiSourceRetriever
from core.retrieval.openalex_client import OpenAlexClient
from core.retrieval.semantic_scholar_client import SemanticScholarClient
from core.schemas import Paper, ResearchPlan, ResearchState, TraceEvent


class MockArxivClient:
    """Deterministic offline academic retrieval mock supporting initial & reflection rounds."""

    _CATALOG: list[dict[str, Any]] = [
        {
            "external_id": "2404.10981",
            "title": "Stateful Multi-Agent Literature Workflows with Directed Graphs",
            "authors": ["Alice Chen", "Bob Patel"],
            "year": 2024,
            "abstract": (
                "We study stateful multi-agent orchestration using cyclic state graphs for "
                "autonomous academic research. Task decomposition into complementary sub-questions "
                "improves coverage over single-shot prompting on multi-hop synthesis benchmarks."
            ),
            "categories": ["cs.CL", "cs.AI"],
        },
        {
            "external_id": "2402.03367",
            "title": "Hybrid Dense and BM25 Retrieval with Reciprocal Rank Fusion for Scientific QA",
            "authors": ["David Kim", "Elena Gomez"],
            "year": 2024,
            "abstract": (
                "Combining dense semantic embeddings with BM25 lexical matching via Reciprocal "
                "Rank Fusion (RRF) consistently outperforms single-retriever RAG baselines on "
                "domain-specific academic terminology, dataset names, and benchmark acronyms."
            ),
            "categories": ["cs.IR", "cs.CL"],
        },
        {
            "external_id": "2310.11511",
            "title": "Self-Reflective Retrieval-Augmented Generation with Evidence Critique",
            "authors": ["Akari Sato", "Liam O'Connor"],
            "year": 2023,
            "abstract": (
                "We introduce an iterative critique mechanism that evaluates retrieval sufficiency "
                "per sub-question, triggering bounded query reformulation when initial passages "
                "fail to support comparative claims."
            ),
            "categories": ["cs.CL"],
        },
        {
            "external_id": "2405.14831",
            "title": "Citation Grounding and Provenance Verification in Automated Literature Surveys",
            "authors": ["Maya Lin", "Noah Rossi"],
            "year": 2024,
            "abstract": (
                "Automated survey generators often fabricate references or misattribute claims. "
                "Binding every extracted evidence item to immutable chunk identifiers reduces "
                "unsupported citations while preserving fine-grained source traceability."
            ),
            "categories": ["cs.CL", "cs.DL"],
        },
        {
            "external_id": "2406.07712",
            "title": "Structured Evidence Matrices for Cross-Paper Comparative Synthesis",
            "authors": ["Sophia Wang", "Lucas Weber"],
            "year": 2024,
            "abstract": (
                "Rather than summarizing papers sequentially, extracting normalized schemas of "
                "methods, datasets, metrics, and limitations prior to drafting yields clearer "
                "cross-literature comparisons on computer science survey benchmarks."
            ),
            "categories": ["cs.CL"],
        },
        {
            "external_id": "2501.04210",
            "title": "Benchmarking Faithfulness and Coverage in Agentic Literature Review Systems",
            "authors": ["Ethan Zhao", "Priya Nair"],
            "year": 2025,
            "abstract": (
                "We evaluate agentic literature review pipelines against direct LLM and naive RAG "
                "baselines across citation accuracy, claim faithfulness, and sub-topic coverage."
            ),
            "categories": ["cs.AI", "cs.IR"],
        },
        {
            "external_id": "2502.09104",
            "title": "BM25 and Dense Vector Reranking for Multi-Hop Citation Attribution",
            "authors": ["Clara Moreau", "Kenji Takahashi"],
            "year": 2025,
            "abstract": (
                "Supplemental empirical evaluation of hybrid BM25 and dense vector reranking "
                "demonstrates 18 percent higher citation attribution recall on multi-hop academic "
                "surveys compared with dense-only retrieval."
            ),
            "categories": ["cs.IR", "cs.CL"],
        },
        {
            "external_id": "2503.11220",
            "title": "Marginal-Gain Stopping Criteria in Iterative Agentic Literature Search",
            "authors": ["Hannah Becker", "Omar Hassan"],
            "year": 2025,
            "abstract": (
                "We analyze stopping rules for reflective literature search agents, showing that "
                "combining overlap-ratio circuit breakers with bounded iteration caps prevents "
                "redundant retrieval loops without sacrificing citation grounding."
            ),
            "categories": ["cs.AI", "cs.DL"],
        },
    ]

    def _make_paper(self, item: dict[str, Any]) -> Paper:
        return Paper(
            external_id=item["external_id"],
            source="arxiv",
            title=item["title"],
            authors=item["authors"],
            year=item["year"],
            abstract=item["abstract"],
            url=f"https://arxiv.org/abs/{item['external_id']}",
            pdf_url=f"https://arxiv.org/pdf/{item['external_id']}.pdf",
            categories=item["categories"],
            full_text_available=True,
        )

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        q = query.lower()
        if any(marker in q for marker in ("sq1", "sq2", "sq3", "comparative", "supplemental")):
            items = [self._CATALOG[6], self._CATALOG[7], self._CATALOG[0]]
            return [self._make_paper(item) for item in items[:max_results]]
        return [self._make_paper(item) for item in self._CATALOG[: min(6, max_results)]]

    def download_pdf(self, pdf_url: str, dest_path: Path) -> Path | None:
        return None


def _route_after_evaluator(state: ResearchState) -> str:
    latest = state.get("latest_evaluation")
    if latest is not None and latest.needs_more_retrieval:
        return "query_rewriter"
    return "synthesizer"


def _make_route_after_verifier(deps: WorkflowDependencies) -> Callable[[ResearchState], str]:
    def _route_after_verifier(state: ResearchState) -> str:
        revision_count = state.get("revision_count", 0)
        max_revisions = state.get("max_verification_revisions", deps.max_verification_revisions)
        if revision_count >= max_revisions:
            return "end"
        report = state.get("verification_report")
        if report is None:
            return "end"
        needs_fix = (
            bool(report.invalid_citations)
            or report.unsupported_count > 0
            or (
                report.claim_support_rate < deps.verifier_support_threshold
                and report.partially_supported_count > 0
            )
        )
        return "reviser" if needs_fix else "end"

    return _route_after_verifier


def build_graph(
    deps: WorkflowDependencies,
    checkpointer: Any = None,
    interrupt_before: list[str] | None = None,
):
    """Compile the Stage-2/4 LangGraph pipeline with reflection loop, verifier revision, and optional HITL."""
    builder = StateGraph(ResearchState)
    builder.add_node("planner", make_planner_node(deps))
    builder.add_node("retriever_screener", make_retriever_screener_node(deps))
    builder.add_node("ingestion", make_ingestion_node(deps))
    builder.add_node("extractor", make_extractor_node(deps))
    builder.add_node("evaluator", make_evaluator_node(deps))
    builder.add_node("query_rewriter", make_query_rewriter_node(deps))
    builder.add_node("synthesizer", make_synthesizer_node(deps))
    builder.add_node("verifier", make_verifier_node(deps))
    builder.add_node("reviser", make_reviser_node(deps))

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "retriever_screener")
    builder.add_edge("retriever_screener", "ingestion")
    builder.add_edge("ingestion", "extractor")
    builder.add_edge("extractor", "evaluator")
    builder.add_conditional_edges(
        "evaluator",
        _route_after_evaluator,
        {
            "query_rewriter": "query_rewriter",
            "synthesizer": "synthesizer",
        },
    )
    builder.add_edge("query_rewriter", "retriever_screener")
    builder.add_edge("synthesizer", "verifier")
    builder.add_conditional_edges(
        "verifier",
        _make_route_after_verifier(deps),
        {
            "reviser": "reviser",
            "end": END,
        },
    )
    builder.add_edge("reviser", "verifier")

    return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _build_deps(
    run_id: str,
    cfg: Settings,
    mock: bool,
    download_pdfs: bool,
    local_pdf_dir: Path | None,
    persist_rag: bool,
    on_event: Callable[[TraceEvent], None] | None,
    llm_override: Any,
    arxiv_override: ArxivClientProtocol | None,
    sources: list[str] | None = None,
    embedder_override: Any = None,
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    max_per_paper: int | None = 3,
    use_evidence_matrix: bool = True,
    max_verification_revisions: int = 1,
    verifier_support_threshold: float = 0.85,
    max_concurrency: int = 4,
    on_iter0_snapshot: Callable[[dict[str, Any]], None] | None = None,
) -> WorkflowDependencies:
    llm = llm_override or create_llm_client(cfg, mock=mock)
    embedder = embedder_override or create_embedder(cfg, force_local=mock)
    if arxiv_override is not None:
        arxiv_client: ArxivClientProtocol = arxiv_override
    elif mock:
        arxiv_client = MockArxivClient()
    else:
        arxiv_client = MultiSourceRetriever(
            sources=sources or cfg.academic_sources_list,
            arxiv_client=ArxivClient(),
            openalex_client=OpenAlexClient(email=cfg.openalex_email),
            semantic_scholar_client=SemanticScholarClient(api_key=cfg.semantic_scholar_api_key),
        )
    persist_dir = (cfg.data_path / "chroma") if persist_rag else None
    rag_index = HybridRAGIndex(
        collection_name=f"run_{run_id}",
        embedder=embedder,
        persist_dir=persist_dir,
    )
    return WorkflowDependencies(
        settings=cfg,
        llm=llm,
        embedder=embedder,
        arxiv_client=arxiv_client,
        rag_index=rag_index,
        download_pdfs=download_pdfs,
        local_pdf_dir=local_pdf_dir,
        on_event=on_event,
        retrieval_mode=retrieval_mode,
        max_per_paper=max_per_paper,
        use_evidence_matrix=use_evidence_matrix,
        max_verification_revisions=max_verification_revisions,
        verifier_support_threshold=verifier_support_threshold,
        max_concurrency=max_concurrency,
        on_iter0_snapshot=on_iter0_snapshot,
    )


def _apply_settings_overrides(
    settings: Settings | None,
    top_k_papers: int | None = None,
    max_papers_per_query: int | None = None,
    max_iterations: int | None = None,
    llm_model: str | None = None,
) -> Settings:
    cfg = settings or get_settings()
    overrides: dict[str, Any] = {}
    if top_k_papers is not None:
        overrides["top_k_papers"] = top_k_papers
    if max_papers_per_query is not None:
        overrides["max_papers_per_query"] = max_papers_per_query
    if max_iterations is not None:
        overrides["max_iterations"] = max_iterations
    if llm_model is not None and llm_model.strip():
        overrides["llm_model"] = llm_model.strip()
    return cfg.model_copy(update=overrides) if overrides else cfg


def _persist_outputs(run_output_dir: Path, state: ResearchState) -> None:
    run_output_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_output_dir / "report.md"
    state_path = run_output_dir / "state.json"
    if state.get("draft_markdown"):
        report_path.write_text(state["draft_markdown"], encoding="utf-8")
    state_path.write_text(
        json.dumps(_to_jsonable(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def start_research_session(
    topic: str,
    settings: Settings | None = None,
    mock: bool = False,
    top_k_papers: int | None = None,
    max_papers_per_query: int | None = None,
    max_iterations: int | None = None,
    download_pdfs: bool = True,
    local_pdf_dir: Path | None = None,
    persist_rag: bool = True,
    require_plan_approval: bool = False,
    run_id: str | None = None,
    sources: list[str] | None = None,
    llm_model: str | None = None,
    on_event: Callable[[TraceEvent], None] | None = None,
    llm_override: Any = None,
    arxiv_override: ArxivClientProtocol | None = None,
    embedder_override: Any = None,
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    max_per_paper: int | None = 3,
    use_evidence_matrix: bool = True,
    max_verification_revisions: int = 1,
    verifier_support_threshold: float = 0.85,
    max_concurrency: int = 4,
    on_iter0_snapshot: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[ResearchState, str, Path]:
    """Start a literature survey session, optionally pausing after planning for HITL approval."""
    cfg = _apply_settings_overrides(
        settings,
        top_k_papers=top_k_papers,
        max_papers_per_query=max_papers_per_query,
        max_iterations=max_iterations,
        llm_model=llm_model,
    )
    active_run_id = run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    run_output_dir = cfg.output_path / active_run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    deps = _build_deps(
        run_id=active_run_id,
        cfg=cfg,
        mock=mock,
        download_pdfs=download_pdfs,
        local_pdf_dir=local_pdf_dir,
        persist_rag=persist_rag,
        on_event=on_event,
        llm_override=llm_override,
        arxiv_override=arxiv_override,
        sources=sources,
        embedder_override=embedder_override,
        retrieval_mode=retrieval_mode,
        max_per_paper=max_per_paper,
        use_evidence_matrix=use_evidence_matrix,
        max_verification_revisions=max_verification_revisions,
        verifier_support_threshold=verifier_support_threshold,
        max_concurrency=max_concurrency,
        on_iter0_snapshot=on_iter0_snapshot,
    )

    initial_state: ResearchState = {
        "run_id": active_run_id,
        "topic": topic,
        "iteration": 0,
        "max_iterations": cfg.max_iterations,
        "revision_count": 0,
        "max_verification_revisions": max_verification_revisions,
        "trace": [],
        "errors": [],
    }

    checkpoint_db = cfg.data_path / "checkpoints.db"
    interrupt_nodes = ["retriever_screener"] if require_plan_approval else None
    config = {"configurable": {"thread_id": active_run_id}}

    with SqliteSaver.from_conn_string(str(checkpoint_db)) as checkpointer:
        graph = build_graph(deps, checkpointer=checkpointer, interrupt_before=interrupt_nodes)
        result_state: ResearchState = graph.invoke(initial_state, config=config)
        snapshot = graph.get_state(config)
        is_paused = bool(snapshot.next)

    status = "awaiting_approval" if is_paused else "completed"
    _persist_outputs(run_output_dir, result_state)
    return result_state, status, run_output_dir


def resume_research_session(
    run_id: str,
    approved_plan: ResearchPlan | None = None,
    settings: Settings | None = None,
    mock: bool = False,
    top_k_papers: int | None = None,
    max_papers_per_query: int | None = None,
    max_iterations: int | None = None,
    download_pdfs: bool = True,
    local_pdf_dir: Path | None = None,
    persist_rag: bool = True,
    sources: list[str] | None = None,
    llm_model: str | None = None,
    on_event: Callable[[TraceEvent], None] | None = None,
    llm_override: Any = None,
    arxiv_override: ArxivClientProtocol | None = None,
) -> tuple[ResearchState, Path]:
    """Resume a paused HITL research session from SQLite checkpoint."""
    cfg = _apply_settings_overrides(
        settings,
        top_k_papers=top_k_papers,
        max_papers_per_query=max_papers_per_query,
        max_iterations=max_iterations,
        llm_model=llm_model,
    )
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    run_output_dir = cfg.output_path / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    deps = _build_deps(
        run_id=run_id,
        cfg=cfg,
        mock=mock,
        download_pdfs=download_pdfs,
        local_pdf_dir=local_pdf_dir,
        persist_rag=persist_rag,
        on_event=on_event,
        llm_override=llm_override,
        arxiv_override=arxiv_override,
        sources=sources,
    )

    checkpoint_db = cfg.data_path / "checkpoints.db"
    config = {"configurable": {"thread_id": run_id}}

    with SqliteSaver.from_conn_string(str(checkpoint_db)) as checkpointer:
        graph = build_graph(deps, checkpointer=checkpointer)
        if approved_plan is not None:
            graph.update_state(config, {"plan": approved_plan})
        final_state: ResearchState = graph.invoke(None, config=config)

    _persist_outputs(run_output_dir, final_state)
    return final_state, run_output_dir


def run_research(
    topic: str,
    settings: Settings | None = None,
    mock: bool = False,
    top_k_papers: int | None = None,
    max_papers_per_query: int | None = None,
    max_iterations: int | None = None,
    download_pdfs: bool = True,
    local_pdf_dir: Path | None = None,
    persist_rag: bool = True,
    sources: list[str] | None = None,
    llm_model: str | None = None,
    on_event: Callable[[TraceEvent], None] | None = None,
    llm_override: Any = None,
    arxiv_override: ArxivClientProtocol | None = None,
    embedder_override: Any = None,
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    max_per_paper: int | None = 3,
    use_evidence_matrix: bool = True,
    max_verification_revisions: int = 1,
    verifier_support_threshold: float = 0.85,
    max_concurrency: int = 4,
    on_iter0_snapshot: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[ResearchState, Path]:
    """Execute a complete Stage-2/4 literature survey run and persist outputs."""
    final_state, _, run_output_dir = start_research_session(
        topic=topic,
        settings=settings,
        mock=mock,
        top_k_papers=top_k_papers,
        max_papers_per_query=max_papers_per_query,
        max_iterations=max_iterations,
        download_pdfs=download_pdfs,
        local_pdf_dir=local_pdf_dir,
        persist_rag=persist_rag,
        require_plan_approval=False,
        sources=sources,
        llm_model=llm_model,
        on_event=on_event,
        llm_override=llm_override,
        arxiv_override=arxiv_override,
        embedder_override=embedder_override,
        retrieval_mode=retrieval_mode,
        max_per_paper=max_per_paper,
        use_evidence_matrix=use_evidence_matrix,
        max_verification_revisions=max_verification_revisions,
        verifier_support_threshold=verifier_support_threshold,
        max_concurrency=max_concurrency,
        on_iter0_snapshot=on_iter0_snapshot,
    )
    return final_state, run_output_dir
