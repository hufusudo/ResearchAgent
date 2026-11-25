"""LangGraph nodes for Stage-2 literature survey workflow with reflection and verification."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, runtime_checkable

from core.config import Settings
from core.graph.prompts import (
    DIRECT_CHUNK_SYNTHESIZER_SYSTEM_PROMPT,
    EVALUATOR_SYSTEM_PROMPT,
    EXTRACTOR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    QUERY_REWRITER_SYSTEM_PROMPT,
    SCREENER_SYSTEM_PROMPT,
    SYNTHESIZER_SYSTEM_PROMPT,
    VERIFIER_REVISER_SYSTEM_PROMPT,
)
from core.llm.client import LLMClientProtocol
from core.llm.embeddings import EmbedderProtocol
from core.retrieval.arxiv_client import deduplicate_and_number_papers
from core.retrieval.hybrid_rag import HybridRAGIndex
from core.retrieval.pdf_parser import (
    chunk_abstract_only,
    load_local_pdfs,
    parse_pdf_to_chunks,
    rebind_chunk_paper_id,
)
from core.schemas import (
    EvaluationResult,
    EvidenceExtractionResult,
    EvidenceItem,
    Paper,
    QueryRewriteResult,
    ResearchPlan,
    ResearchState,
    RetrievedChunk,
    RewrittenQuery,
    ScreeningResult,
    SubQuestionEvaluation,
    TextChunk,
    TraceEvent,
)
from core.verifier.citation_verifier import (
    format_verification_summary_markdown,
    verify_citations,
)


@runtime_checkable
class ArxivClientProtocol(Protocol):
    """Protocol for academic search and PDF download clients."""

    def search(self, query: str, max_results: int = 10) -> list[Paper]:
        """Search papers matching a query."""
        ...

    def download_pdf(self, pdf_url: str, dest_path: Path) -> Path | None:
        """Download PDF to destination path, returning None on failure."""
        ...


@dataclass
class WorkflowDependencies:
    """Injected runtime dependencies for LangGraph workflow nodes."""

    settings: Settings
    llm: LLMClientProtocol
    embedder: EmbedderProtocol
    arxiv_client: ArxivClientProtocol
    rag_index: HybridRAGIndex
    download_pdfs: bool = True
    local_pdf_dir: Path | None = None
    on_event: Callable[[TraceEvent], None] | None = None
    local_chunks_cache: dict[str, list[TextChunk]] = field(default_factory=dict)
    retrieval_mode: Literal["hybrid", "dense", "bm25"] = "hybrid"
    max_per_paper: int | None = 3
    use_evidence_matrix: bool = True
    max_verification_revisions: int = 1
    verifier_support_threshold: float = 0.85
    max_concurrency: int = 4
    on_iter0_snapshot: Callable[[dict[str, Any]], None] | None = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _paper_identity(paper: Paper) -> tuple[str, str]:
    ext = f"{paper.source}:{paper.external_id.strip().lower()}" if paper.external_id else ""
    return ext, _normalize_title_key(paper.title)


def _record_trace(
    deps: WorkflowDependencies,
    trace: list[TraceEvent] | None,
    step: str,
    node: str,
    message: str,
    metadata: dict[str, Any] | None = None,
) -> list[TraceEvent]:
    updated = list(trace or [])
    event = TraceEvent(
        step=step,
        node=node,
        message=message,
        timestamp=_now_iso(),
        metadata=metadata or {},
    )
    updated.append(event)
    if deps.on_event is not None:
        deps.on_event(event)
    return updated


def _screen_papers(
    deps: WorkflowDependencies,
    plan: ResearchPlan,
    papers: list[Paper],
    errors: list[str],
) -> list[Paper]:
    if not papers:
        return []
    scored = [p.model_copy(deep=True) for p in papers]
    try:
        screen_batch = scored[:30]
        sq_summary = "\n".join(
            f"- [{sq.id}] {sq.title}: {sq.description}" for sq in plan.sub_questions
        )
        def _fmt_meta(p: Paper) -> str:
            extras = [str(p.year), p.source]
            if p.venue:
                extras.append(f"venue={p.venue}")
            if p.citation_count is not None:
                extras.append(f"citations={p.citation_count}")
            return ", ".join(extras)

        paper_summaries = "\n\n".join(
            f"[{p.paper_id}] ({_fmt_meta(p)}) {p.title}\nAbstract: {p.abstract[:650]}"
            for p in screen_batch
        )
        screen_prompt = (
            f"Research Topic: {plan.topic}\n"
            f"Refined Scope: {plan.refined_scope}\n\n"
            f"Sub-Questions:\n{sq_summary}\n\n"
            f"Candidate Papers:\n{paper_summaries}"
        )
        result = deps.llm.invoke_structured(
            system_prompt=SCREENER_SYSTEM_PROMPT,
            user_prompt=screen_prompt,
            schema=ScreeningResult,
        )
        score_map = {s.paper_id: s for s in result.scores}
        for paper in scored:
            if paper.paper_id in score_map:
                paper.relevance_score = float(score_map[paper.paper_id].relevance_score)
                paper.relevance_reason = score_map[paper.paper_id].relevance_reason
        scored.sort(key=lambda p: (p.relevance_score, p.year), reverse=True)
    except Exception as exc:
        errors.append(f"LLM screening fallback used: {exc}")
    return scored


def make_planner_node(deps: WorkflowDependencies):
    """Create the planner node that decomposes the research topic."""

    def planner_node(state: ResearchState) -> dict[str, Any]:
        topic = state["topic"]
        errors = list(state.get("errors") or [])
        plan = deps.llm.invoke_structured(
            system_prompt=PLANNER_SYSTEM_PROMPT,
            user_prompt=f"Topic: {topic}",
            schema=ResearchPlan,
        )
        if not plan.topic.strip():
            plan.topic = topic
        for idx, sq in enumerate(plan.sub_questions, start=1):
            if not sq.id.strip():
                sq.id = f"SQ{idx}"

        trace = _record_trace(
            deps,
            state.get("trace"),
            step="plan",
            node="planner",
            message=f"Planned {len(plan.sub_questions)} sub-questions for topic.",
            metadata={"sub_question_ids": [sq.id for sq in plan.sub_questions]},
        )
        return {
            "plan": plan,
            "iteration": 0,
            "max_iterations": state.get("max_iterations", deps.settings.max_iterations),
            "executed_queries": [],
            "pending_queries": [],
            "target_subquestion_ids": [],
            "last_new_paper_count": 0,
            "last_overlap_ratio": 0.0,
            "evaluation_history": [],
            "rewrite_history": [],
            "latest_evaluation": None,
            "verification_report": None,
            "pre_revision_verification_report": None,
            "revision_count": 0,
            "max_verification_revisions": state.get(
                "max_verification_revisions", deps.max_verification_revisions
            ),
            "trace": trace,
            "errors": errors,
        }

    return planner_node


def _run_concurrent_searches(
    deps: WorkflowDependencies,
    queries: list[str],
    *,
    error_prefix: str = "Search failed for query",
) -> tuple[list[Paper], list[str]]:
    """Run multiple search queries concurrently while preserving query hit order."""
    if not queries:
        return [], []
    max_results = deps.settings.max_papers_per_query
    hits: list[Paper] = []
    errors: list[str] = []

    if len(queries) == 1 or deps.max_concurrency <= 1:
        for q in queries:
            try:
                hits.extend(deps.arxiv_client.search(query=q, max_results=max_results))
            except Exception as exc:
                errors.append(f"{error_prefix} '{q}': {exc}")
        return hits, errors

    workers = min(3, deps.max_concurrency, len(queries))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            (q, executor.submit(deps.arxiv_client.search, query=q, max_results=max_results))
            for q in queries
        ]
        for q, fut in futures:
            try:
                hits.extend(fut.result())
            except Exception as exc:
                errors.append(f"{error_prefix} '{q}': {exc}")
    return hits, errors


def make_retriever_screener_node(deps: WorkflowDependencies):
    """Create the initial and supplemental retrieval + abstract screening node."""

    def retriever_screener_node(state: ResearchState) -> dict[str, Any]:
        plan = state["plan"]
        errors = list(state.get("errors") or [])
        iteration = state.get("iteration", 0)
        executed_queries = list(state.get("executed_queries") or [])
        executed_lower = {q.lower() for q in executed_queries}

        if iteration == 0:
            raw_papers: list[Paper] = []
            if deps.local_pdf_dir is not None:
                try:
                    local_items = load_local_pdfs(deps.local_pdf_dir)
                    for local_paper, local_chunks in local_items:
                        deps.local_chunks_cache[local_paper.external_id] = local_chunks
                        raw_papers.append(local_paper)
                except Exception as exc:
                    errors.append(f"Failed loading local PDFs from {deps.local_pdf_dir}: {exc}")

            batch_queries: list[str] = []
            for sq in plan.sub_questions:
                for raw_q in sq.search_queries or [sq.title]:
                    q = " ".join(raw_q.strip().split())
                    if q and q.lower() not in executed_lower:
                        executed_lower.add(q.lower())
                        executed_queries.append(q)
                        batch_queries.append(q)

            batch_hits, batch_errors = _run_concurrent_searches(
                deps, batch_queries, error_prefix="Search failed for query"
            )
            raw_papers.extend(batch_hits)
            errors.extend(batch_errors)

            candidates = deduplicate_and_number_papers(raw_papers)
            if not candidates:
                trace = _record_trace(
                    deps,
                    state.get("trace"),
                    step="retrieve_screen",
                    node="retriever_screener",
                    message="No candidate papers found across queries.",
                    metadata={"iteration": 0, "candidate_count": 0, "selected_count": 0},
                )
                return {
                    "candidate_papers": [],
                    "selected_papers": [],
                    "executed_queries": executed_queries,
                    "pending_queries": [],
                    "last_new_paper_count": 0,
                    "last_overlap_ratio": 0.0,
                    "trace": trace,
                    "errors": errors,
                }

            top_k = max(1, deps.settings.top_k_papers)
            scored_candidates = _screen_papers(deps, plan, candidates, errors)
            local_candidates: list[Paper] = []
            online_candidates: list[Paper] = []
            for paper in scored_candidates:
                if paper.source == "local_pdf":
                    if not paper.relevance_reason:
                        paper = paper.model_copy(
                            update={"relevance_reason": "User-supplied local PDF."}
                        )
                    local_candidates.append(paper)
                else:
                    online_candidates.append(paper)

            online_slots = max(0, top_k - len(local_candidates))
            chosen = local_candidates + online_candidates[:online_slots]
            selected_papers = [
                paper.model_copy(update={"paper_id": f"P{idx:02d}", "ingest_status": "pending"})
                for idx, paper in enumerate(chosen, start=1)
            ]

            trace = _record_trace(
                deps,
                state.get("trace"),
                step="retrieve_screen",
                node="retriever_screener",
                message=(
                    f"Retrieved {len(candidates)} unique candidates; "
                    f"selected top {len(selected_papers)} papers."
                ),
                metadata={
                    "iteration": 0,
                    "candidate_count": len(candidates),
                    "selected_count": len(selected_papers),
                    "queries_executed": len(batch_queries),
                },
            )
            return {
                "candidate_papers": candidates,
                "selected_papers": selected_papers,
                "executed_queries": executed_queries,
                "pending_queries": [],
                "last_new_paper_count": len(selected_papers),
                "last_overlap_ratio": 0.0,
                "trace": trace,
                "errors": errors,
            }

        # Supplemental retrieval round (iteration > 0)
        existing_candidates = list(state.get("candidate_papers") or [])
        existing_selected = list(state.get("selected_papers") or [])
        pending_queries = list(state.get("pending_queries") or [])

        known_ext_ids: set[str] = set()
        known_titles: set[str] = set()
        for known in [*existing_candidates, *existing_selected]:
            ext_k, title_k = _paper_identity(known)
            if ext_k:
                known_ext_ids.add(ext_k)
            if title_k:
                known_titles.add(title_k)

        cleaned_queries: list[str] = []
        for raw_q in pending_queries:
            q = " ".join(raw_q.strip().split())
            if not q:
                continue
            if q.lower() not in executed_lower:
                executed_lower.add(q.lower())
                executed_queries.append(q)
            cleaned_queries.append(q)

        queries_ran = len(cleaned_queries)
        raw_hits, search_errors = _run_concurrent_searches(
            deps,
            cleaned_queries,
            error_prefix="Supplemental search failed for query",
        )
        errors.extend(search_errors)

        duplicate_hits = 0
        new_raw: list[Paper] = []
        for hit in raw_hits:
            ext_k, title_k = _paper_identity(hit)
            if (ext_k and ext_k in known_ext_ids) or (title_k and title_k in known_titles):
                duplicate_hits += 1
            else:
                new_raw.append(hit)

        overlap_ratio = round(duplicate_hits / len(raw_hits), 4) if raw_hits else 1.0
        new_unique = deduplicate_and_number_papers(new_raw)

        if not new_unique:
            trace = _record_trace(
                deps,
                state.get("trace"),
                step="retrieve_screen",
                node="retriever_screener",
                message=(
                    f"Supplemental search (round {iteration}) found 0 new unique papers "
                    f"(overlap={overlap_ratio:.0%})."
                ),
                metadata={
                    "iteration": iteration,
                    "new_paper_count": 0,
                    "overlap_ratio": overlap_ratio,
                    "queries_executed": queries_ran,
                },
            )
            return {
                "executed_queries": executed_queries,
                "pending_queries": [],
                "last_new_paper_count": 0,
                "last_overlap_ratio": overlap_ratio,
                "trace": trace,
                "errors": errors,
            }

        cand_offset = len(existing_candidates)
        for idx, paper in enumerate(new_unique, start=1):
            paper.paper_id = f"P{cand_offset + idx:02d}"

        scored_new = _screen_papers(deps, plan, new_unique, errors)
        supplemental_k = min(3, max(1, deps.settings.top_k_papers // 2))
        sel_offset = len(existing_selected)
        appended_selected = [
            paper.model_copy(
                update={
                    "paper_id": f"P{sel_offset + idx:02d}",
                    "ingest_status": "pending",
                }
            )
            for idx, paper in enumerate(scored_new[:supplemental_k], start=1)
        ]

        updated_candidates = existing_candidates + new_unique
        updated_selected = existing_selected + appended_selected
        new_ids = ", ".join(p.paper_id for p in appended_selected)

        trace = _record_trace(
            deps,
            state.get("trace"),
            step="retrieve_screen",
            node="retriever_screener",
            message=(
                f"Supplemental search (round {iteration}) found {len(new_unique)} new candidates "
                f"(overlap={overlap_ratio:.0%}); appended {len(appended_selected)} papers ({new_ids})."
            ),
            metadata={
                "iteration": iteration,
                "new_paper_count": len(appended_selected),
                "overlap_ratio": overlap_ratio,
                "new_paper_ids": [p.paper_id for p in appended_selected],
                "queries_executed": queries_ran,
            },
        )
        return {
            "candidate_papers": updated_candidates,
            "selected_papers": updated_selected,
            "executed_queries": executed_queries,
            "pending_queries": [],
            "last_new_paper_count": len(appended_selected),
            "last_overlap_ratio": overlap_ratio,
            "trace": trace,
            "errors": errors,
        }

    return retriever_screener_node


def make_ingestion_node(deps: WorkflowDependencies):
    """Create the incremental concurrent PDF download, parsing, fallback, and Hybrid RAG indexing node."""

    def _ingest_single_paper(paper: Paper, pdf_dir: Path) -> tuple[Paper, list[TextChunk], list[str]]:
        paper_chunks: list[TextChunk] = []
        local_errors: list[str] = []
        ingest_status: Literal["pending", "full_text", "abstract_only", "failed"] = "abstract_only"
        full_text_available = False

        if paper.source == "local_pdf" and paper.external_id in deps.local_chunks_cache:
            cached = deps.local_chunks_cache[paper.external_id]
            paper_chunks = rebind_chunk_paper_id(cached, paper.paper_id)
            ingest_status = "full_text"
            full_text_available = True
        elif deps.download_pdfs and paper.pdf_url:
            dest_path = pdf_dir / f"{paper.paper_id}.pdf"
            downloaded = deps.arxiv_client.download_pdf(paper.pdf_url, dest_path)
            if downloaded is not None:
                parsed = parse_pdf_to_chunks(paper=paper, pdf_path=downloaded)
                if parsed:
                    paper_chunks = parsed
                    ingest_status = "full_text"
                    full_text_available = True
                else:
                    local_errors.append(
                        f"PDF parsing failed for {paper.paper_id} ({paper.title[:50]}); "
                        "fell back to abstract."
                    )
            else:
                local_errors.append(
                    f"PDF download unavailable for {paper.paper_id}; fell back to abstract."
                )

        if not paper_chunks:
            paper_chunks = chunk_abstract_only(paper)
            ingest_status = "abstract_only"
            full_text_available = False

        updated_paper = paper.model_copy(
            update={
                "full_text_available": full_text_available,
                "ingest_status": ingest_status,
            }
        )
        return updated_paper, paper_chunks, local_errors

    def ingestion_node(state: ResearchState) -> dict[str, Any]:
        selected = list(state.get("selected_papers") or [])
        existing_chunks = list(state.get("chunks") or [])
        run_id = state.get("run_id", "default")
        errors = list(state.get("errors") or [])
        pdf_dir = deps.settings.data_path / "pdfs" / run_id

        pending_indices = [idx for idx, p in enumerate(selected) if p.ingest_status == "pending"]
        newly_processed = len(pending_indices)
        results_by_idx: dict[int, tuple[Paper, list[TextChunk], list[str]]] = {}

        if pending_indices:
            if len(pending_indices) == 1 or deps.max_concurrency <= 1:
                for idx in pending_indices:
                    results_by_idx[idx] = _ingest_single_paper(selected[idx], pdf_dir)
            else:
                workers = min(deps.max_concurrency, len(pending_indices))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = {
                        idx: executor.submit(_ingest_single_paper, selected[idx], pdf_dir)
                        for idx in pending_indices
                    }
                    for idx, fut in futures.items():
                        results_by_idx[idx] = fut.result()

        updated_papers: list[Paper] = []
        new_chunks: list[TextChunk] = []
        full_text_count = 0
        abstract_only_count = 0

        for idx, paper in enumerate(selected):
            if idx not in results_by_idx:
                updated_papers.append(paper)
                continue
            up_paper, p_chunks, p_errors = results_by_idx[idx]
            updated_papers.append(up_paper)
            new_chunks.extend(p_chunks)
            errors.extend(p_errors)
            if up_paper.ingest_status == "full_text":
                full_text_count += 1
            else:
                abstract_only_count += 1

        if new_chunks:
            deps.rag_index.add_chunks(new_chunks)
        combined_chunks = existing_chunks + new_chunks

        trace = _record_trace(
            deps,
            state.get("trace"),
            step="ingest",
            node="ingestion",
            message=(
                f"Indexed {len(new_chunks)} new chunks across {newly_processed} papers "
                f"({full_text_count} full_text, {abstract_only_count} abstract_only; "
                f"total index: {len(combined_chunks)} chunks)."
            ),
            metadata={
                "new_chunk_count": len(new_chunks),
                "total_chunk_count": len(combined_chunks),
                "newly_processed_papers": newly_processed,
                "full_text_count": full_text_count,
                "abstract_only_count": abstract_only_count,
            },
        )
        return {
            "selected_papers": updated_papers,
            "chunks": combined_chunks,
            "trace": trace,
            "errors": errors,
        }

    return ingestion_node


def make_extractor_node(deps: WorkflowDependencies):
    """Create the Hybrid RAG retrieval (with per-paper diversity) and concurrent evidence extraction node."""

    def _process_subquestion(
        sq: Any,
        valid_paper_ids: set[str],
    ) -> tuple[str, list[RetrievedChunk], list[EvidenceItem], str | None]:
        recent_queries = sq.search_queries[-3:] if sq.search_queries else []
        query_text = f"{sq.title} {' '.join(recent_queries)}".strip()
        retrieved = deps.rag_index.search(
            query=query_text,
            top_k=8,
            mode=deps.retrieval_mode,
            max_per_paper=deps.max_per_paper,
        )

        if not retrieved or not deps.use_evidence_matrix:
            return sq.id, retrieved, [], None

        valid_chunk_ids = {r.chunk.chunk_id for r in retrieved}
        chunks_block = "\n\n".join(
            (
                f"[{r.chunk.chunk_id}] Paper={r.chunk.paper_id} | "
                f"Section={r.chunk.section} | Page={r.chunk.page} | Title={r.chunk.title}\n"
                f"{r.chunk.text}"
            )
            for r in retrieved
        )
        user_prompt = (
            f"Sub-Question ID: {sq.id}\n"
            f"Sub-Question Title: {sq.title}\n"
            f"Description: {sq.description}\n\n"
            f"Retrieved Chunks:\n{chunks_block}"
        )
        try:
            extracted = deps.llm.invoke_structured(
                system_prompt=EXTRACTOR_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema=EvidenceExtractionResult,
            )
            verified_items: list[EvidenceItem] = []
            for item in extracted.items:
                if item.paper_id not in valid_paper_ids:
                    continue
                filtered_cids = [cid for cid in item.source_chunk_ids if cid in valid_chunk_ids]
                if not filtered_cids:
                    continue
                verified_items.append(
                    item.model_copy(
                        update={
                            "sub_question_id": sq.id,
                            "source_chunk_ids": filtered_cids,
                        }
                    )
                )
            return sq.id, retrieved, verified_items, None
        except Exception as exc:
            return sq.id, retrieved, [], f"Evidence extraction failed for {sq.id}: {exc}"

    def extractor_node(state: ResearchState) -> dict[str, Any]:
        plan = state["plan"]
        selected_papers = state.get("selected_papers") or []
        valid_paper_ids = {p.paper_id for p in selected_papers}
        errors = list(state.get("errors") or [])
        iteration = state.get("iteration", 0)
        target_ids = set(state.get("target_subquestion_ids") or [])

        retrieved_by_sq = dict(state.get("retrieved_chunks_by_subquestion") or {})
        evidence_by_sq = dict(state.get("evidence_by_subquestion") or {})

        sqs_to_process = [
            sq
            for sq in plan.sub_questions
            if iteration == 0 or not target_ids or sq.id in target_ids or sq.id not in evidence_by_sq
        ]

        sq_outputs: list[tuple[str, list[RetrievedChunk], list[EvidenceItem], str | None]] = []
        if sqs_to_process:
            if len(sqs_to_process) == 1 or deps.max_concurrency <= 1:
                sq_outputs = [_process_subquestion(sq, valid_paper_ids) for sq in sqs_to_process]
            else:
                workers = min(deps.max_concurrency, len(sqs_to_process))
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(_process_subquestion, sq, valid_paper_ids)
                        for sq in sqs_to_process
                    ]
                    sq_outputs = [fut.result() for fut in futures]

        updated_items_count = 0
        for sq_id, retrieved, verified_items, err_msg in sq_outputs:
            retrieved_by_sq[sq_id] = retrieved
            if err_msg is not None:
                errors.append(err_msg)
                evidence_by_sq.setdefault(sq_id, [])
            else:
                evidence_by_sq[sq_id] = verified_items
                updated_items_count += len(verified_items)

        total_items = sum(len(items) for items in evidence_by_sq.values())
        trace = _record_trace(
            deps,
            state.get("trace"),
            step="extract",
            node="extractor",
            message=(
                f"Extracted {updated_items_count} evidence items across {len(sqs_to_process)} "
                f"sub-questions (total matrix: {total_items} items)."
            ),
            metadata={
                "iteration": iteration,
                "processed_subquestions": [sq.id for sq in sqs_to_process],
                "total_evidence_count": total_items,
            },
        )
        return {
            "retrieved_chunks_by_subquestion": retrieved_by_sq,
            "evidence_by_subquestion": evidence_by_sq,
            "target_subquestion_ids": [],
            "trace": trace,
            "errors": errors,
        }

    return extractor_node


def make_evaluator_node(deps: WorkflowDependencies):
    """Create the critic/evaluator node with deterministic iteration and overlap guards."""

    def evaluator_node(state: ResearchState) -> dict[str, Any]:
        plan = state["plan"]
        evidence_by_sq = state.get("evidence_by_subquestion") or {}
        retrieved_by_sq = state.get("retrieved_chunks_by_subquestion") or {}
        iteration = state.get("iteration", 0)
        max_iterations = state.get("max_iterations", deps.settings.max_iterations)
        last_new_paper_count = state.get("last_new_paper_count", 0)
        last_overlap_ratio = state.get("last_overlap_ratio", 0.0)
        errors = list(state.get("errors") or [])

        sq_stats: dict[str, tuple[int, int, int]] = {}
        stat_lines: list[str] = []
        for sq in plan.sub_questions:
            items = evidence_by_sq.get(sq.id, [])
            ev_count = len(items)
            uniq_papers = len({item.paper_id for item in items})
            ret_count = len(retrieved_by_sq.get(sq.id, []))
            sq_stats[sq.id] = (ev_count, uniq_papers, ret_count)

            sample_findings = "; ".join(
                f"[{it.paper_id}] {it.key_findings[:120]}" for it in items[:3]
            ) or "No evidence extracted."
            stat_lines.append(
                f"- [{sq.id}] {sq.title}: evidence={ev_count}, unique_papers={uniq_papers}, "
                f"retrieved_chunks={ret_count}\n  Findings: {sample_findings}"
            )

        eval_prompt = (
            f"Research Topic: {plan.topic}\n"
            f"Current Iteration: {iteration}\n"
            f"Max Iterations: {max_iterations}\n"
            f"Last Supplemental New Papers: {last_new_paper_count}\n"
            f"Last Overlap Ratio: {last_overlap_ratio}\n\n"
            f"Sub-Question Coverage:\n" + "\n".join(stat_lines)
        )

        try:
            eval_result = deps.llm.invoke_structured(
                system_prompt=EVALUATOR_SYSTEM_PROMPT,
                user_prompt=eval_prompt,
                schema=EvaluationResult,
            )
        except Exception as exc:
            errors.append(f"Evaluator LLM fallback used: {exc}")
            fallback_evals: list[SubQuestionEvaluation] = []
            weak_ids: list[str] = []
            for sq in plan.sub_questions:
                ev_count, uniq_papers, _ = sq_stats[sq.id]
                sufficient = ev_count >= 2 and uniq_papers >= 2
                if not sufficient:
                    weak_ids.append(sq.id)
                fallback_evals.append(
                    SubQuestionEvaluation(
                        sub_question_id=sq.id,
                        evidence_count=ev_count,
                        unique_paper_count=uniq_papers,
                        coverage_score=8.0 if sufficient else 4.5,
                        sufficient=sufficient,
                        gaps=[] if sufficient else ["Insufficient multi-paper evidence coverage."],
                    )
                )
            eval_result = EvaluationResult(
                iteration=iteration,
                overall_sufficient=not bool(weak_ids),
                needs_more_retrieval=bool(weak_ids),
                weak_subquestion_ids=weak_ids,
                subquestion_evaluations=fallback_evals,
                reasoning="Heuristic coverage fallback applied.",
            )

        # Synchronize exact deterministic counts onto subquestion_evaluations.
        valid_sq_ids = {sq.id for sq in plan.sub_questions}
        normalized_evals: list[SubQuestionEvaluation] = []
        eval_by_id = {ev.sub_question_id: ev for ev in eval_result.subquestion_evaluations}
        for sq in plan.sub_questions:
            ev_count, uniq_papers, _ = sq_stats[sq.id]
            existing_ev = eval_by_id.get(sq.id)
            if existing_ev is not None:
                normalized_evals.append(
                    existing_ev.model_copy(
                        update={
                            "evidence_count": ev_count,
                            "unique_paper_count": uniq_papers,
                        }
                    )
                )
            else:
                sufficient = ev_count >= 2 and uniq_papers >= 2
                normalized_evals.append(
                    SubQuestionEvaluation(
                        sub_question_id=sq.id,
                        evidence_count=ev_count,
                        unique_paper_count=uniq_papers,
                        coverage_score=8.0 if sufficient else 5.0,
                        sufficient=sufficient,
                        gaps=[] if sufficient else ["Additional evidence needed."],
                    )
                )

        weak_ids = [sq_id for sq_id in eval_result.weak_subquestion_ids if sq_id in valid_sq_ids]
        if not weak_ids:
            weak_ids = [ev.sub_question_id for ev in normalized_evals if not ev.sufficient]

        # Enforce hard stopping guards after LLM critique.
        if iteration >= max_iterations:
            needs_more = False
            stop_reason = "max_iterations_reached"
        elif iteration > 0 and (last_new_paper_count == 0 or last_overlap_ratio >= 0.85):
            needs_more = False
            stop_reason = "marginal_gain_circuit_breaker"
        elif not weak_ids or not eval_result.needs_more_retrieval:
            needs_more = False
            stop_reason = "evidence_sufficient"
        else:
            needs_more = True
            stop_reason = "insufficient_coverage"

        final_eval = eval_result.model_copy(
            update={
                "iteration": iteration,
                "overall_sufficient": not bool(weak_ids),
                "needs_more_retrieval": needs_more,
                "weak_subquestion_ids": weak_ids,
                "subquestion_evaluations": normalized_evals,
                "stop_reason": stop_reason,
            }
        )

        history = list(state.get("evaluation_history") or [])
        history.append(final_eval)

        action_msg = (
            f"Triggering supplemental retrieval for {', '.join(weak_ids)}"
            if needs_more
            else f"Proceeding to synthesis ({stop_reason})"
        )
        trace = _record_trace(
            deps,
            state.get("trace"),
            step="evaluate",
            node="evaluator",
            message=f"Iteration {iteration} evaluation: {action_msg}.",
            metadata={
                "iteration": iteration,
                "needs_more_retrieval": needs_more,
                "stop_reason": stop_reason,
                "weak_subquestion_ids": weak_ids,
            },
        )
        if iteration == 0 and deps.on_iter0_snapshot is not None:
            try:
                deps.on_iter0_snapshot(
                    {
                        "run_id": state.get("run_id", ""),
                        "topic": state.get("topic", ""),
                        "plan": plan.model_copy(deep=True),
                        "selected_papers": [p.model_copy(deep=True) for p in (state.get("selected_papers") or [])],
                        "chunks": [c.model_copy(deep=True) for c in (state.get("chunks") or [])],
                        "retrieved_chunks_by_subquestion": {
                            k: [r.model_copy(deep=True) for r in v]
                            for k, v in retrieved_by_sq.items()
                        },
                        "evidence_by_subquestion": {
                            k: [e.model_copy(deep=True) for e in v]
                            for k, v in evidence_by_sq.items()
                        },
                        "evaluation_iter0": final_eval.model_copy(deep=True),
                        "errors": list(errors),
                        "trace": [t.model_copy(deep=True) for t in trace],
                    }
                )
            except Exception:
                pass
        return {
            "latest_evaluation": final_eval,
            "evaluation_history": history,
            "trace": trace,
            "errors": errors,
        }

    return evaluator_node


def make_query_rewriter_node(deps: WorkflowDependencies):
    """Create the query rewriter node for bounded reflection iterations."""

    def query_rewriter_node(state: ResearchState) -> dict[str, Any]:
        plan = state["plan"].model_copy(deep=True)
        latest_eval = state.get("latest_evaluation")
        executed_queries = list(state.get("executed_queries") or [])
        executed_lower = {q.lower() for q in executed_queries}
        iteration = state.get("iteration", 0)
        errors = list(state.get("errors") or [])

        weak_ids = (
            list(latest_eval.weak_subquestion_ids)
            if latest_eval and latest_eval.weak_subquestion_ids
            else [plan.sub_questions[0].id]
        )
        sq_map = {sq.id: sq for sq in plan.sub_questions}
        gaps_by_sq = {
            ev.sub_question_id: ev.gaps
            for ev in (latest_eval.subquestion_evaluations if latest_eval else [])
        }

        weak_details = "\n".join(
            (
                f"- [{sq_id}] {sq_map[sq_id].title if sq_id in sq_map else sq_id}: "
                f"gaps={'; '.join(gaps_by_sq.get(sq_id, [])) or 'insufficient coverage'}"
            )
            for sq_id in weak_ids
        )
        rewrite_prompt = (
            f"Research Topic: {plan.topic}\n"
            f"Current Iteration: {iteration}\n"
            f"Weak Sub-Questions: {', '.join(weak_ids)}\n\n"
            f"Details:\n{weak_details}\n\n"
            f"Already Executed Queries (DO NOT REPEAT):\n"
            + "\n".join(f"- {q}" for q in executed_queries)
        )

        try:
            rewrite_res = deps.llm.invoke_structured(
                system_prompt=QUERY_REWRITER_SYSTEM_PROMPT,
                user_prompt=rewrite_prompt,
                schema=QueryRewriteResult,
            )
        except Exception as exc:
            errors.append(f"Query rewriter fallback used: {exc}")
            rewrite_res = QueryRewriteResult(
                iteration=iteration + 1,
                rewritten_queries=[
                    RewrittenQuery(
                        sub_question_id=sq_id,
                        query=f"{sq_map[sq_id].title if sq_id in sq_map else plan.topic} benchmark evaluation",
                        rationale="Fallback query reformulation for missing coverage.",
                    )
                    for sq_id in weak_ids
                ],
                reasoning="Fallback reformulation.",
            )

        pending_queries: list[str] = []
        filtered_rewritten: list[RewrittenQuery] = []
        seen_pending: set[str] = set()

        for rq in rewrite_res.rewritten_queries:
            cleaned = " ".join(rq.query.strip().split())
            if not cleaned or cleaned.lower() in executed_lower or cleaned.lower() in seen_pending:
                continue
            seen_pending.add(cleaned.lower())
            pending_queries.append(cleaned)
            filtered_rewritten.append(rq.model_copy(update={"query": cleaned}))
            if rq.sub_question_id in sq_map:
                sq_map[rq.sub_question_id].search_queries.append(cleaned)

        target_sq_ids = list(weak_ids)
        added_new_sq = None
        if rewrite_res.new_sub_question is not None and len(plan.sub_questions) < 6:
            candidate_sq = rewrite_res.new_sub_question.model_copy(deep=True)
            if not candidate_sq.id.strip() or candidate_sq.id in sq_map:
                candidate_sq.id = f"SQ{len(plan.sub_questions) + 1}"
            plan.sub_questions.append(candidate_sq)
            sq_map[candidate_sq.id] = candidate_sq
            target_sq_ids.append(candidate_sq.id)
            added_new_sq = candidate_sq
            for q in candidate_sq.search_queries:
                cleaned = " ".join(q.strip().split())
                if cleaned and cleaned.lower() not in executed_lower and cleaned.lower() not in seen_pending:
                    seen_pending.add(cleaned.lower())
                    pending_queries.append(cleaned)

        # Ensure at least one non-duplicate query if all LLM outputs collided.
        if not pending_queries and weak_ids:
            sq_ref = sq_map.get(weak_ids[0])
            fallback_q = (
                f"{sq_ref.title if sq_ref else plan.topic} comparative study {iteration + 1}"
            )
            if fallback_q.lower() not in executed_lower:
                pending_queries.append(fallback_q)
                filtered_rewritten.append(
                    RewrittenQuery(
                        sub_question_id=weak_ids[0],
                        query=fallback_q,
                        rationale="Non-duplicate fallback query.",
                    )
                )
                if sq_ref is not None:
                    sq_ref.search_queries.append(fallback_q)

        next_iteration = iteration + 1
        final_rewrite = rewrite_res.model_copy(
            update={
                "iteration": next_iteration,
                "rewritten_queries": filtered_rewritten,
                "new_sub_question": added_new_sq,
            }
        )
        rewrite_history = list(state.get("rewrite_history") or [])
        rewrite_history.append(final_rewrite)

        trace = _record_trace(
            deps,
            state.get("trace"),
            step="rewrite",
            node="query_rewriter",
            message=(
                f"Rewrote {len(pending_queries)} queries for round {next_iteration} "
                f"targeting {', '.join(target_sq_ids)}."
            ),
            metadata={
                "iteration": next_iteration,
                "pending_queries": pending_queries,
                "target_subquestion_ids": target_sq_ids,
            },
        )
        return {
            "plan": plan,
            "iteration": next_iteration,
            "pending_queries": pending_queries,
            "target_subquestion_ids": list(dict.fromkeys(target_sq_ids)),
            "rewrite_history": rewrite_history,
            "trace": trace,
            "errors": errors,
        }

    return query_rewriter_node


def _format_references_section(papers: list[Paper]) -> str:
    if not papers:
        return "## References\n\n- No papers were retrieved for this survey."

    lines = ["## References", ""]
    for paper in papers:
        authors_str = ", ".join(paper.authors[:4])
        if len(paper.authors) > 4:
            authors_str += " et al."
        if not authors_str:
            authors_str = "Unknown authors"
        year_str = str(paper.year) if paper.year else "n.d."
        meta_notes: list[str] = []
        if paper.venue:
            meta_notes.append(paper.venue)
        if paper.citation_count is not None:
            meta_notes.append(f"citations: {paper.citation_count}")
        meta_notes.append(f"ingest: {paper.ingest_status}")
        lines.append(
            f"- **[{paper.paper_id}]** {paper.title} ({year_str}). "
            f"{authors_str}. Source: {paper.url} _({', '.join(meta_notes)})_"
        )
    return "\n".join(lines)


def make_synthesizer_node(deps: WorkflowDependencies):
    """Create the survey synthesizer node with deterministic reference appending."""

    def synthesizer_node(state: ResearchState) -> dict[str, Any]:
        plan = state["plan"]
        selected_papers = state.get("selected_papers") or []
        evidence_by_sq = state.get("evidence_by_subquestion") or {}
        retrieved_by_sq = state.get("retrieved_chunks_by_subquestion") or {}

        papers_block = "\n".join(
            f"- [{p.paper_id}] {p.title} ({p.year}, score={p.relevance_score}): {p.abstract[:300]}"
            for p in selected_papers
        ) or "No papers selected."

        sq_blocks: list[str] = []
        for sq in plan.sub_questions:
            if deps.use_evidence_matrix:
                ev_items = evidence_by_sq.get(sq.id, [])
                ev_lines = "\n".join(
                    (
                        f"  * [{ev.paper_id}] Method: {ev.method} | Setting: {ev.dataset_or_setting} | "
                        f"Findings: {ev.key_findings} | Limitations: {ev.limitations} | "
                        f"Chunks: {', '.join(ev.source_chunk_ids)}"
                    )
                    for ev in ev_items
                ) or "  * No structured evidence extracted."

                top_chunks = retrieved_by_sq.get(sq.id, [])[:3]
                chunk_lines = "\n".join(
                    f"  * [{r.chunk.chunk_id}] ({r.chunk.paper_id}): {r.chunk.text[:320]}"
                    for r in top_chunks
                ) or "  * No retrieved excerpts."

                sq_blocks.append(
                    f"[{sq.id}] {sq.title}\n"
                    f"Scope: {sq.description}\n"
                    f"Structured Evidence:\n{ev_lines}\n"
                    f"Key Excerpts:\n{chunk_lines}"
                )
            else:
                top_chunks = retrieved_by_sq.get(sq.id, [])[:6]
                chunk_lines = "\n".join(
                    f"  * [{r.chunk.chunk_id}] ({r.chunk.paper_id}, {r.chunk.section}, p.{r.chunk.page}): {r.chunk.text[:800]}"
                    for r in top_chunks
                ) or "  * No retrieved excerpts."

                sq_blocks.append(
                    f"[{sq.id}] {sq.title}\n"
                    f"Scope: {sq.description}\n"
                    f"Retrieved Excerpts:\n{chunk_lines}"
                )

        section_heading = (
            "Sub-Question Evidence & Excerpts"
            if deps.use_evidence_matrix
            else "Sub-Question Retrieved Excerpts"
        )
        user_prompt = (
            f"Research Topic: {plan.topic}\n"
            f"Refined Scope: {plan.refined_scope}\n\n"
            f"Selected Papers:\n{papers_block}\n\n"
            f"{section_heading}:\n\n" + "\n\n".join(sq_blocks)
        )

        system_prompt = (
            SYNTHESIZER_SYSTEM_PROMPT
            if deps.use_evidence_matrix
            else DIRECT_CHUNK_SYNTHESIZER_SYSTEM_PROMPT
        )
        body = deps.llm.invoke_text(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.25,
        )
        cleaned_body = re.split(
            r"\n##\s+References\b",
            body,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].rstrip()
        full_markdown = f"{cleaned_body}\n\n{_format_references_section(selected_papers)}\n"

        trace = _record_trace(
            deps,
            state.get("trace"),
            step="synthesize",
            node="synthesizer",
            message=f"Generated literature survey draft ({len(full_markdown)} chars).",
            metadata={
                "draft_chars": len(full_markdown),
                "reference_count": len(selected_papers),
                "use_evidence_matrix": deps.use_evidence_matrix,
            },
        )
        return {
            "draft_markdown": full_markdown,
            "trace": trace,
        }

    return synthesizer_node


def make_verifier_node(deps: WorkflowDependencies):
    """Create the citation provenance verification node."""

    def verifier_node(state: ResearchState) -> dict[str, Any]:
        draft = state.get("draft_markdown", "")
        selected_papers = state.get("selected_papers") or []
        chunks = state.get("chunks") or []
        revision_count = state.get("revision_count", 0)

        clean_draft = re.split(
            r"\n##\s+Citation Verification Summary\b",
            draft,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].rstrip()

        report = verify_citations(
            draft_markdown=clean_draft,
            selected_papers=selected_papers,
            chunks=chunks,
            llm=deps.llm,
        )
        summary_block = format_verification_summary_markdown(report)
        updated_markdown = f"{clean_draft}\n\n{summary_block}\n"

        pre_rev = state.get("pre_revision_verification_report")
        if revision_count == 0 and pre_rev is None:
            pre_rev = report

        trace = _record_trace(
            deps,
            state.get("trace"),
            step="verify",
            node="verifier",
            message=(
                f"Citation audit: validity={report.reference_validity_rate:.0%}, "
                f"claim_support={report.claim_support_rate:.0%} "
                f"({report.supported_count}/{report.checked_claims} supported)."
            ),
            metadata={
                "reference_validity_rate": report.reference_validity_rate,
                "claim_support_rate": report.claim_support_rate,
                "invalid_citations": report.invalid_citations,
                "checked_claims": report.checked_claims,
                "revision_count": revision_count,
            },
        )
        return {
            "draft_markdown": updated_markdown,
            "verification_report": report,
            "pre_revision_verification_report": pre_rev,
            "trace": trace,
        }

    return verifier_node


def make_reviser_node(deps: WorkflowDependencies):
    """Create the verifier-guided lightweight citation revision node."""

    def reviser_node(state: ResearchState) -> dict[str, Any]:
        draft = state.get("draft_markdown", "")
        selected_papers = state.get("selected_papers") or []
        chunks = state.get("chunks") or []
        report = state.get("verification_report")
        errors = list(state.get("errors") or [])
        revision_count = state.get("revision_count", 0)

        original_body = re.split(
            r"\n##\s+(?:References|Citation Verification Summary)\b",
            draft,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].rstrip()

        valid_ids = [p.paper_id for p in selected_papers if p.paper_id]
        chunks_by_id = {c.chunk_id: c for c in chunks}
        chunks_by_paper: dict[str, list[TextChunk]] = {}
        for c in chunks:
            chunks_by_paper.setdefault(c.paper_id, []).append(c)

        flagged_blocks: list[str] = []
        if report is not None:
            for idx, item in enumerate(report.details, start=1):
                if item.verdict == "supported":
                    continue
                matched: list[TextChunk] = []
                seen_cids: set[str] = set()
                for cid in item.matched_chunk_ids:
                    if cid in chunks_by_id and cid not in seen_cids:
                        seen_cids.add(cid)
                        matched.append(chunks_by_id[cid])
                if not matched:
                    for pid in item.cited_paper_ids:
                        p_chunks = chunks_by_paper.get(pid, [])
                        if p_chunks and p_chunks[0].chunk_id not in seen_cids:
                            seen_cids.add(p_chunks[0].chunk_id)
                            matched.append(p_chunks[0])
                excerpt_text = "\n".join(
                    f"    [{c.chunk_id}] ({c.paper_id}): {c.text[:1200]}"
                    for c in matched[:4]
                ) or "    (No source chunk excerpts available)"
                flagged_blocks.append(
                    f"Issue {idx} [{item.verdict}]: {item.claim}\n"
                    f"  Cited Papers: {', '.join(f'[{pid}]' for pid in item.cited_paper_ids) or 'None'}\n"
                    f"  Verifier Reason: {item.reason}\n"
                    f"  Source Excerpts:\n{excerpt_text}"
                )

        invalid_str = (
            ", ".join(f"[{cid}]" for cid in (report.invalid_citations if report else []))
            or "None"
        )
        issues_block = "\n\n".join(flagged_blocks) or "Fix any invalid citations listed above."
        user_prompt = (
            f"Valid Paper IDs: {', '.join(f'[{pid}]' for pid in valid_ids)}\n"
            f"Invalid Citations to Remove: {invalid_str}\n\n"
            f"Current Survey Body:\n{original_body}\n\n"
            f"Flagged Issues:\n{issues_block}"
        )

        revised_body = original_body
        try:
            candidate = deps.llm.invoke_text(
                system_prompt=VERIFIER_REVISER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=0.15,
            )
            cleaned_candidate = re.split(
                r"\n##\s+(?:References|Citation Verification Summary)\b",
                candidate,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].rstrip()
            min_len = max(200, int(0.4 * len(original_body))) if len(original_body) >= 200 else 1
            if len(cleaned_candidate) >= min_len:
                revised_body = cleaned_candidate
            else:
                errors.append(
                    "Reviser output was unexpectedly short; retained original draft body."
                )
        except Exception as exc:
            errors.append(f"Verifier-guided revision failed; retained original draft: {exc}")

        updated_markdown = f"{revised_body}\n\n{_format_references_section(selected_papers)}\n"
        new_revision_count = revision_count + 1
        trace = _record_trace(
            deps,
            state.get("trace"),
            step="revise",
            node="reviser",
            message=(
                f"Applied lightweight citation revision #{new_revision_count} "
                f"for {len(flagged_blocks)} flagged claim(s) and invalid={invalid_str}."
            ),
            metadata={
                "revision_count": new_revision_count,
                "flagged_issue_count": len(flagged_blocks),
                "invalid_citations": list(report.invalid_citations) if report else [],
            },
        )
        return {
            "draft_markdown": updated_markdown,
            "revision_count": new_revision_count,
            "trace": trace,
            "errors": errors,
        }

    return reviser_node
