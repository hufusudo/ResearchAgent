"""Controlled module-level ablation runners for Phase 4 evaluation."""

from __future__ import annotations

import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.config import Settings
from core.graph.nodes import (
    WorkflowDependencies,
    make_reviser_node,
    make_synthesizer_node,
    make_verifier_node,
)
from core.graph.workflow import MockArxivClient
from core.llm.client import create_llm_client
from core.llm.embeddings import create_embedder
from core.retrieval.hybrid_rag import HybridRAGIndex
from core.schemas import (
    EvaluationResult,
    ResearchPlan,
    ResearchState,
    RetrievedChunk,
    TextChunk,
)
from core.verifier.citation_verifier import verify_citations
from eval.benchmark_topics import BenchmarkTopic
from eval.metrics import (
    ChunkRelevanceBatch,
    compute_retrieval_ablation_metrics,
    evaluate_subquestion_coverage,
    evaluate_synthesis_rubric,
)


def _judge_pooled_chunks(
    sq_title: str,
    sq_desc: str,
    pooled_chunks: dict[str, TextChunk],
    llm: Any,
    *,
    mock: bool = False,
) -> dict[str, int]:
    """Blind pooled relevance judgment (0/1/2) across chunks retrieved by any of the 4 modes."""
    if not pooled_chunks:
        return {}

    if mock:
        query_toks = {
            tok for tok in re.findall(r"[a-z0-9]+", f"{sq_title} {sq_desc}".lower()) if len(tok) >= 4
        }
        grades: dict[str, int] = {}
        for cid, chunk in pooled_chunks.items():
            c_toks = set(re.findall(r"[a-z0-9]+", f"{chunk.title} {chunk.text}".lower()))
            overlap = len(query_toks & c_toks)
            if overlap >= 3:
                grades[cid] = 2
            elif overlap >= 1:
                grades[cid] = 1
            else:
                grades[cid] = 0
        return grades

    items = list(pooled_chunks.items())
    rng = random.Random(42)
    rng.shuffle(items)

    chunks_block = "\n\n".join(
        f"[{cid}] ({c.title} | {c.section}): {c.text[:900]}"
        for cid, c in items[:32]
    )
    system_prompt = (
        "You are an academic retrieval relevance assessor.\n"
        "Grade each candidate passage for how well it helps answer the given research sub-question:\n"
        "- 2: Directly provides substantive methodology, empirical findings, or benchmark evidence for this sub-question\n"
        "- 1: Topically related background or partial mention\n"
        "- 0: Off-topic or noisy boilerplate\n"
        "Return one ChunkRelevanceItem per provided chunk_id."
    )
    user_prompt = (
        f"Sub-Question: {sq_title}\n"
        f"Scope: {sq_desc}\n\n"
        f"Candidate Passages:\n{chunks_block}"
    )
    try:
        batch = llm.invoke_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=ChunkRelevanceBatch,
            temperature=0.0,
        )
        grades = {it.chunk_id: int(it.grade) for it in batch.items if it.chunk_id in pooled_chunks}
        for cid in pooled_chunks:
            grades.setdefault(cid, 0)
        return grades
    except Exception:
        return {cid: 1 for cid in pooled_chunks}


def run_retrieval_ablation(
    topic: BenchmarkTopic,
    plan: ResearchPlan,
    chunks: list[TextChunk],
    embedder: Any,
    settings: Settings,
    *,
    top_k: int = 8,
    mock: bool = False,
) -> dict[str, Any]:
    """Ablation 1: Dense-only vs BM25-only vs Hybrid RRF vs Hybrid + max_per_paper=3."""
    if not chunks or not plan.sub_questions:
        return {"topic_id": topic.topic_id, "modes": {}, "per_subquestion": []}

    index = HybridRAGIndex(
        collection_name=f"ablate_ret_{topic.topic_id}_{int(time.time())}",
        embedder=embedder,
        persist_dir=None,
    )
    index.add_chunks(chunks)

    mode_configs: list[tuple[str, str, int | None]] = [
        ("dense_only", "dense", None),
        ("bm25_only", "bm25", None),
        ("hybrid_rrf", "hybrid", None),
        ("hybrid_diversity", "hybrid", 3),
    ]

    judge_llm = create_llm_client(settings, mock=mock)

    def _eval_sq(sq: Any) -> dict[str, Any]:
        recent_queries = sq.search_queries[-3:] if sq.search_queries else []
        query_text = f"{sq.title} {' '.join(recent_queries)}".strip()
        ranked_by_mode: dict[str, list[RetrievedChunk]] = {}
        pooled: dict[str, TextChunk] = {}

        for mode_label, rag_mode, max_pp in mode_configs:
            hits = index.search(
                query=query_text,
                top_k=top_k,
                mode=rag_mode,  # type: ignore[arg-type]
                max_per_paper=max_pp,
            )
            ranked_by_mode[mode_label] = hits
            for h in hits:
                pooled[h.chunk.chunk_id] = h.chunk

        grades = _judge_pooled_chunks(
            sq.title,
            sq.description,
            pooled,
            judge_llm,
            mock=mock,
        )
        sq_metrics = compute_retrieval_ablation_metrics(
            query=query_text,
            ranked_by_mode=ranked_by_mode,
            relevance_grades=grades,
            top_k=top_k,
        )
        return {
            "sub_question_id": sq.id,
            "query": query_text,
            "pooled_candidates": len(pooled),
            "metrics": sq_metrics,
        }

    sqs = plan.sub_questions
    if len(sqs) == 1 or mock:
        per_sq = [_eval_sq(sq) for sq in sqs]
    else:
        with ThreadPoolExecutor(max_workers=min(4, len(sqs))) as executor:
            per_sq = list(executor.map(_eval_sq, sqs))

    mode_averages: dict[str, dict[str, float]] = {}
    metric_keys = [
        "precision_at_k",
        "recall_at_k",
        "ndcg_at_k",
        "mrr",
        "unique_papers_at_k",
        "max_single_paper_share",
        "term_hit_rate",
    ]
    for mode_label, _, _ in mode_configs:
        agg: dict[str, float] = {}
        for mkey in metric_keys:
            vals = [row["metrics"][mode_label][mkey] for row in per_sq]
            agg[mkey] = round(sum(vals) / len(vals), 4) if vals else 0.0
        mode_averages[mode_label] = agg

    return {
        "topic_id": topic.topic_id,
        "subquestion_count": len(per_sq),
        "modes": mode_averages,
        "per_subquestion": per_sq,
    }


def _build_lightweight_deps(
    settings: Settings,
    llm: Any,
    embedder: Any,
    *,
    use_evidence_matrix: bool = True,
    max_verification_revisions: int = 1,
) -> WorkflowDependencies:
    dummy_rag = HybridRAGIndex(
        collection_name=f"ablate_noop_{int(time.time() * 1000)}",
        embedder=embedder,
        persist_dir=None,
    )
    return WorkflowDependencies(
        settings=settings,
        llm=llm,
        embedder=embedder,
        arxiv_client=MockArxivClient(),
        rag_index=dummy_rag,
        use_evidence_matrix=use_evidence_matrix,
        max_verification_revisions=max_verification_revisions,
    )


def _eval_coverage_stats(eval_res: EvaluationResult | None) -> tuple[float, int, float]:
    if eval_res is None or not eval_res.subquestion_evaluations:
        return 0.0, 0, 0.0
    sub_evals = eval_res.subquestion_evaluations
    avg_score = round(sum(e.coverage_score for e in sub_evals) / len(sub_evals), 4)
    weak_count = sum(1 for e in sub_evals if not e.sufficient)
    multi_paper_ratio = round(
        sum(1 for e in sub_evals if e.unique_paper_count >= 2) / len(sub_evals),
        4,
    )
    return avg_score, weak_count, multi_paper_ratio


def run_reflection_ablation(
    topic: BenchmarkTopic,
    iter0_snapshot: dict[str, Any],
    full_state: ResearchState,
    full_metrics: dict[str, Any],
    settings: Settings,
    *,
    mock: bool = False,
) -> dict[str, Any]:
    """Ablation 2: Paired comparison of max_iterations=0 (no reflection) vs max_iterations=2."""
    plan: ResearchPlan = iter0_snapshot["plan"]
    iter0_papers = iter0_snapshot["selected_papers"]
    iter0_chunks = iter0_snapshot["chunks"]
    iter0_ev = iter0_snapshot["evidence_by_subquestion"]
    iter0_ret = iter0_snapshot["retrieved_chunks_by_subquestion"]
    eval_iter0: EvaluationResult | None = iter0_snapshot.get("evaluation_iter0")
    avg_eval0, weak0, multi_ratio0 = _eval_coverage_stats(eval_iter0)
    eval_final: EvaluationResult | None = full_state.get("latest_evaluation")
    avg_eval_full, weak_full, multi_ratio_full = _eval_coverage_stats(eval_final)

    ev_count0 = sum(len(v) for v in iter0_ev.values())
    ev_count_full = full_metrics.get("evidence_count", 0)

    iter2_payload = {
        "paper_count": full_metrics.get("paper_count", 0),
        "chunk_count": full_metrics.get("chunk_count", 0),
        "evidence_count": ev_count_full,
        "reflection_iterations": full_metrics.get("reflection_iterations", 0),
        "stop_reason": full_metrics.get("stop_reason", ""),
        "evaluator_avg_coverage": avg_eval_full,
        "weak_subquestions": weak_full,
        "multi_paper_sq_ratio": multi_ratio_full,
        "reference_validity_rate": full_metrics.get("reference_validity_rate", 0.0),
        "claim_support_rate": full_metrics.get("claim_support_rate", 0.0),
        "draft_sq_coverage_rate": full_metrics.get("subquestion_coverage_rate", 0.0),
        "draft_avg_coverage_score": full_metrics.get("avg_coverage_score", 0.0),
        "rubric_overall": full_metrics.get("rubric", {}).get("overall_score", 0.0),
    }

    if full_state.get("iteration", 0) == 0:
        return {
            "topic_id": topic.topic_id,
            "iter0": {
                "paper_count": len(iter0_papers),
                "chunk_count": len(iter0_chunks),
                "evidence_count": ev_count0,
                "evaluator_avg_coverage": avg_eval0,
                "weak_subquestions": weak0,
                "multi_paper_sq_ratio": multi_ratio0,
                "reference_validity_rate": full_metrics.get("reference_validity_rate", 0.0),
                "claim_support_rate": full_metrics.get("claim_support_rate", 0.0),
                "draft_sq_coverage_rate": full_metrics.get("subquestion_coverage_rate", 0.0),
                "draft_avg_coverage_score": full_metrics.get("avg_coverage_score", 0.0),
                "rubric_overall": full_metrics.get("rubric", {}).get("overall_score", 0.0),
            },
            "iter2_full": iter2_payload,
        }

    llm = create_llm_client(settings, mock=mock)
    embedder = create_embedder(settings, force_local=True)
    deps = _build_lightweight_deps(
        settings,
        llm,
        embedder,
        use_evidence_matrix=True,
        max_verification_revisions=1,
    )

    synth_fn = make_synthesizer_node(deps)
    ver_fn = make_verifier_node(deps)
    rev_fn = make_reviser_node(deps)

    ablate_state: ResearchState = {
        "run_id": f"{full_state.get('run_id', 'run')}_iter0",
        "topic": topic.title,
        "plan": plan,
        "selected_papers": iter0_papers,
        "chunks": iter0_chunks,
        "retrieved_chunks_by_subquestion": iter0_ret,
        "evidence_by_subquestion": iter0_ev,
        "revision_count": 0,
        "max_verification_revisions": 1,
        "trace": [],
        "errors": [],
    }
    ablate_state.update(synth_fn(ablate_state))  # type: ignore[arg-type]
    ablate_state.update(ver_fn(ablate_state))  # type: ignore[arg-type]

    vr0 = ablate_state.get("verification_report")
    if vr0 is not None and (
        vr0.invalid_citations
        or vr0.unsupported_count > 0
        or (vr0.claim_support_rate < 0.85 and vr0.partially_supported_count > 0)
    ):
        ablate_state.update(rev_fn(ablate_state))  # type: ignore[arg-type]
        ablate_state.update(ver_fn(ablate_state))  # type: ignore[arg-type]

    vr_final0 = ablate_state.get("verification_report")
    draft0 = ablate_state.get("draft_markdown", "")
    judge_llm = create_llm_client(settings, mock=mock)
    cov0 = evaluate_subquestion_coverage(draft0, plan, judge_llm, mock=mock)
    rubric0 = evaluate_synthesis_rubric(draft0, topic.title, judge_llm, mock=mock)

    return {
        "topic_id": topic.topic_id,
        "iter0": {
            "paper_count": len(iter0_papers),
            "chunk_count": len(iter0_chunks),
            "evidence_count": ev_count0,
            "evaluator_avg_coverage": avg_eval0,
            "weak_subquestions": weak0,
            "multi_paper_sq_ratio": multi_ratio0,
            "reference_validity_rate": vr_final0.reference_validity_rate if vr_final0 else 0.0,
            "claim_support_rate": vr_final0.claim_support_rate if vr_final0 else 0.0,
            "draft_sq_coverage_rate": cov0.subquestion_coverage_rate,
            "draft_avg_coverage_score": cov0.avg_coverage_score,
            "rubric_overall": rubric0.overall_score,
        },
        "iter2_full": iter2_payload,
    }


def run_evidence_matrix_ablation(
    topic: BenchmarkTopic,
    full_state: ResearchState,
    full_metrics: dict[str, Any],
    settings: Settings,
    *,
    mock: bool = False,
) -> dict[str, Any]:
    """Ablation 3: Paired comparison of Direct Chunk Synthesis vs Evidence Matrix Synthesis."""
    plan: ResearchPlan = full_state["plan"]
    selected_papers = full_state.get("selected_papers") or []
    chunks = full_state.get("chunks") or []
    retrieved_by_sq = full_state.get("retrieved_chunks_by_subquestion") or {}

    llm = create_llm_client(settings, mock=mock)
    embedder = create_embedder(settings, force_local=True)
    deps = _build_lightweight_deps(
        settings,
        llm,
        embedder,
        use_evidence_matrix=False,
        max_verification_revisions=0,
    )
    synth_fn = make_synthesizer_node(deps)

    ablate_state: ResearchState = {
        "run_id": f"{full_state.get('run_id', 'run')}_nomatrix",
        "topic": topic.title,
        "plan": plan,
        "selected_papers": selected_papers,
        "chunks": chunks,
        "retrieved_chunks_by_subquestion": retrieved_by_sq,
        "evidence_by_subquestion": {},
        "trace": [],
        "errors": [],
    }
    synth_out = synth_fn(ablate_state)
    direct_draft = synth_out.get("draft_markdown", "")

    judge_llm = create_llm_client(settings, mock=mock)
    ver_direct = verify_citations(
        draft_markdown=direct_draft,
        selected_papers=selected_papers,
        chunks=chunks,
        llm=judge_llm,
        max_claims=16,
    )
    cov_direct = evaluate_subquestion_coverage(direct_draft, plan, judge_llm, mock=mock)
    rubric_direct = evaluate_synthesis_rubric(direct_draft, topic.title, judge_llm, mock=mock)

    return {
        "topic_id": topic.topic_id,
        "direct_chunks": {
            "reference_validity_rate": ver_direct.reference_validity_rate,
            "claim_support_rate": ver_direct.claim_support_rate,
            "subquestion_coverage_rate": cov_direct.subquestion_coverage_rate,
            "rubric": rubric_direct.model_dump(mode="json"),
        },
        "evidence_matrix": {
            "reference_validity_rate": full_metrics.get("reference_validity_rate", 0.0),
            "claim_support_rate": full_metrics.get("claim_support_rate", 0.0),
            "subquestion_coverage_rate": full_metrics.get("subquestion_coverage_rate", 0.0),
            "rubric": full_metrics.get("rubric", {}),
        },
    }


def extract_verifier_revision_metrics(
    topic: BenchmarkTopic,
    full_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Ablation 4: Extract pre-revision vs post-revision citation grounding uplift."""
    before_support = float(full_metrics.get("pre_revision_claim_support_rate", 0.0))
    after_support = float(full_metrics.get("claim_support_rate", 0.0))
    before_valid = float(full_metrics.get("pre_revision_validity_rate", 0.0))
    after_valid = float(full_metrics.get("reference_validity_rate", 0.0))
    return {
        "topic_id": topic.topic_id,
        "revision_triggered": bool(full_metrics.get("revision_count", 0) > 0),
        "revision_count": int(full_metrics.get("revision_count", 0)),
        "before_validity_rate": before_valid,
        "after_validity_rate": after_valid,
        "before_claim_support_rate": before_support,
        "after_claim_support_rate": after_support,
        "claim_support_delta": round(after_support - before_support, 4),
        "before_unsupported_count": int(full_metrics.get("pre_revision_unsupported", 0)),
        "after_unsupported_count": int(full_metrics.get("unsupported_count", 0)),
    }
