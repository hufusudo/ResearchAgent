"""Phase 3 multi-topic case study runner, citation re-auditor, and summary builder."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from api.settings_store import RuntimeSettingsStore
from core.config import Settings, get_settings
from core.graph.workflow import start_research_session
from core.llm.client import create_llm_client
from core.schemas import CitationVerificationReport, Paper, TextChunk, TraceEvent
from core.verifier.citation_verifier import (
    format_verification_summary_markdown,
    verify_citations,
)

SUMMARY_PATH = Path("docs/case_studies_summary.json")


@dataclass(frozen=True)
class CaseSpec:
    """Configuration for a reproducible Phase 3 live case study."""

    key: str
    label: str
    topic: str
    sources: tuple[str, ...] = ("arxiv", "openalex")
    top_k_papers: int = 6
    max_papers_per_query: int = 6
    max_iterations: int = 2
    download_pdfs: bool = True
    local_pdf_dir: Path | None = None


CASE_SPECS: list[CaseSpec] = [
    CaseSpec(
        key="case2_hybrid_rag",
        label="Case 2 · Hybrid RAG & Knowledge Graphs",
        topic="Hybrid Retrieval-Augmented Generation with Knowledge Graphs and Reranking",
        sources=("arxiv", "openalex"),
        top_k_papers=6,
        max_papers_per_query=6,
        max_iterations=2,
        download_pdfs=True,
        local_pdf_dir=None,
    ),
    CaseSpec(
        key="case3_test_time_scaling",
        label="Case 3 · 中文主题跨语言综述（Test-Time Scaling & Self-Reflection）",
        topic="大语言模型测试时扩展与自反思推理机制综述",
        sources=("arxiv", "openalex"),
        top_k_papers=6,
        max_papers_per_query=6,
        max_iterations=2,
        download_pdfs=True,
        local_pdf_dir=None,
    ),
    CaseSpec(
        key="case4_local_pdf_hybrid",
        label="Case 4 · 本地 PDF + 在线多源混合调研",
        topic="Citation Faithfulness and Hallucination Mitigation in Retrieval-Augmented Scientific QA",
        sources=("arxiv", "openalex"),
        top_k_papers=6,
        max_papers_per_query=6,
        max_iterations=2,
        download_pdfs=True,
        local_pdf_dir=Path("data/demo_local_pdfs"),
    ),
]


def _load_effective_settings() -> Settings:
    base = get_settings()
    store = RuntimeSettingsStore(base)
    return store.get_effective_settings()


def _load_summary_doc() -> dict[str, Any]:
    if SUMMARY_PATH.exists():
        try:
            return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"updated_at": "", "reaudits": {}, "cases": {}}


def _save_summary_doc(doc: dict[str, Any]) -> None:
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    SUMMARY_PATH.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _extract_run_metrics(
    run_id: str,
    state_data: dict[str, Any],
    *,
    case_key: str | None = None,
    label: str | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    plan = state_data.get("plan") or {}
    sub_questions = plan.get("sub_questions") or []
    candidates = state_data.get("candidate_papers") or []
    selected = state_data.get("selected_papers") or []
    chunks = state_data.get("chunks") or []
    ev_map = state_data.get("evidence_by_subquestion") or {}
    eval_hist = state_data.get("evaluation_history") or []
    rewrite_hist = state_data.get("rewrite_history") or []
    latest_eval = state_data.get("latest_evaluation") or {}
    vr = state_data.get("verification_report") or {}
    draft = state_data.get("draft_markdown") or ""

    ingest_counts = {"full_text": 0, "abstract_only": 0, "pending": 0}
    source_counts: dict[str, int] = {}
    for paper in selected:
        st = paper.get("ingest_status") or "abstract_only"
        ingest_counts[st] = ingest_counts.get(st, 0) + 1
        src = paper.get("source") or "unknown"
        source_counts[src] = source_counts.get(src, 0) + 1

    ev_total = sum(len(v) for v in ev_map.values())
    ev_per_sq = {sq_id: len(items) for sq_id, items in ev_map.items()}

    return {
        "run_id": run_id,
        "case_key": case_key,
        "label": label,
        "topic": state_data.get("topic", ""),
        "refined_scope": plan.get("refined_scope", ""),
        "sub_question_count": len(sub_questions),
        "sub_questions": [
            {"id": sq.get("id"), "title": sq.get("title")} for sq in sub_questions
        ],
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "ingest_counts": ingest_counts,
        "source_counts": source_counts,
        "chunk_count": len(chunks),
        "evidence_total": ev_total,
        "evidence_per_subquestion": ev_per_sq,
        "executed_query_count": len(state_data.get("executed_queries") or []),
        "iteration": int(state_data.get("iteration") or 0),
        "max_iterations": int(state_data.get("max_iterations") or 0),
        "evaluation_rounds": len(eval_hist),
        "rewrite_rounds": len(rewrite_hist),
        "stop_reason": latest_eval.get("stop_reason"),
        "last_new_paper_count": state_data.get("last_new_paper_count"),
        "last_overlap_ratio": state_data.get("last_overlap_ratio"),
        "verification": {
            "total_cited_claims": vr.get("total_cited_claims"),
            "checked_claims": vr.get("checked_claims"),
            "supported_count": vr.get("supported_count"),
            "partially_supported_count": vr.get("partially_supported_count"),
            "unsupported_count": vr.get("unsupported_count"),
            "invalid_citations": vr.get("invalid_citations") or [],
            "uncited_paper_ids": vr.get("uncited_paper_ids") or [],
            "reference_validity_rate": vr.get("reference_validity_rate"),
            "claim_support_rate": vr.get("claim_support_rate"),
        },
        "draft_chars": len(draft),
        "error_count": len(state_data.get("errors") or []),
        "errors": state_data.get("errors") or [],
        "elapsed_seconds": round(elapsed_seconds, 2) if elapsed_seconds is not None else None,
        "papers": [
            {
                "paper_id": p.get("paper_id"),
                "source": p.get("source"),
                "year": p.get("year"),
                "venue": p.get("venue"),
                "citation_count": p.get("citation_count"),
                "ingest_status": p.get("ingest_status"),
                "title": p.get("title"),
                "url": p.get("url"),
            }
            for p in selected
        ],
    }


def _strip_verification_summary(markdown: str) -> str:
    return re.split(
        r"\n##\s+Citation Verification Summary\b",
        markdown,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].rstrip()


def _report_to_dict(report: CitationVerificationReport) -> dict[str, Any]:
    return {
        "total_cited_claims": report.total_cited_claims,
        "checked_claims": report.checked_claims,
        "supported_count": report.supported_count,
        "partially_supported_count": report.partially_supported_count,
        "unsupported_count": report.unsupported_count,
        "invalid_citations": list(report.invalid_citations),
        "uncited_paper_ids": list(report.uncited_paper_ids),
        "reference_validity_rate": report.reference_validity_rate,
        "claim_support_rate": report.claim_support_rate,
    }


def reaudit_run(run_id: str, settings: Settings, write: bool = False) -> dict[str, Any]:
    """Re-run Citation Verifier on an existing outputs/<run_id> run and compare metrics."""
    run_dir = settings.output_path / run_id
    state_path = run_dir / "state.json"
    report_path = run_dir / "report.md"
    if not state_path.exists():
        raise FileNotFoundError(f"State file not found: {state_path}")

    state_data = json.loads(state_path.read_text(encoding="utf-8"))
    before_vr = state_data.get("verification_report") or {}
    before_snapshot = {
        "total_cited_claims": before_vr.get("total_cited_claims"),
        "checked_claims": before_vr.get("checked_claims"),
        "supported_count": before_vr.get("supported_count"),
        "partially_supported_count": before_vr.get("partially_supported_count"),
        "unsupported_count": before_vr.get("unsupported_count"),
        "invalid_citations": before_vr.get("invalid_citations") or [],
        "uncited_paper_ids": before_vr.get("uncited_paper_ids") or [],
        "reference_validity_rate": before_vr.get("reference_validity_rate"),
        "claim_support_rate": before_vr.get("claim_support_rate"),
    }

    papers = [Paper.model_validate(p) for p in state_data.get("selected_papers") or []]
    chunks = [TextChunk.model_validate(c) for c in state_data.get("chunks") or []]
    draft = state_data.get("draft_markdown") or (
        report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    )

    llm = create_llm_client(settings, mock=False)
    t0 = time.perf_counter()
    new_report = verify_citations(
        draft_markdown=draft,
        selected_papers=papers,
        chunks=chunks,
        llm=llm,
    )
    elapsed = time.perf_counter() - t0
    after_snapshot = _report_to_dict(new_report)

    print(f"[REAUDIT] run_id={run_id} ({elapsed:.1f}s)", flush=True)
    print(
        f"  Before: validity={before_snapshot.get('reference_validity_rate')}, "
        f"support={before_snapshot.get('claim_support_rate')} "
        f"({before_snapshot.get('supported_count')}/{before_snapshot.get('checked_claims')} supported, "
        f"{before_snapshot.get('partially_supported_count')} partial, "
        f"{before_snapshot.get('unsupported_count')} unsupported)",
        flush=True,
    )
    print(
        f"  After:  validity={after_snapshot['reference_validity_rate']}, "
        f"support={after_snapshot['claim_support_rate']} "
        f"({after_snapshot['supported_count']}/{after_snapshot['checked_claims']} supported, "
        f"{after_snapshot['partially_supported_count']} partial, "
        f"{after_snapshot['unsupported_count']} unsupported)",
        flush=True,
    )

    if write:
        base_body = _strip_verification_summary(draft)
        summary_block = format_verification_summary_markdown(new_report)
        updated_markdown = f"{base_body}\n\n{summary_block}\n"
        state_data["verification_report"] = new_report.model_dump(mode="json")
        state_data["draft_markdown"] = updated_markdown
        state_path.write_text(
            json.dumps(state_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_path.write_text(updated_markdown, encoding="utf-8")
        print(f"  Updated {state_path} and {report_path}", flush=True)

    summary_doc = _load_summary_doc()
    summary_doc.setdefault("reaudits", {})[run_id] = {
        "reaudited_at": datetime.now(timezone.utc).isoformat(),
        "written": write,
        "elapsed_seconds": round(elapsed, 2),
        "before": before_snapshot,
        "after": after_snapshot,
    }
    summary_doc.setdefault("cases", {})[run_id] = _extract_run_metrics(
        run_id,
        state_data,
        case_key="case1_multi_agent_survey",
        label="Case 1 · 多智能体文献综述与事实核验（阶段二基线 + Verifier 重核验）",
    )
    _save_summary_doc(summary_doc)
    return summary_doc["reaudits"][run_id]


def run_case(spec: CaseSpec, settings: Settings) -> dict[str, Any]:
    """Execute one live Phase 3 case study and persist its summary metrics."""
    print(f"\n=== Starting {spec.label} ===", flush=True)
    print(f"Topic: {spec.topic}", flush=True)
    print(
        f"Config: sources={list(spec.sources)}, top_k={spec.top_k_papers}, "
        f"max_per_query={spec.max_papers_per_query}, max_iter={spec.max_iterations}, "
        f"local_pdf_dir={spec.local_pdf_dir}",
        flush=True,
    )

    def on_event(event: TraceEvent) -> None:
        print(f"  [{event.node}] {event.message}", flush=True)

    t0 = time.perf_counter()
    state, status, out_dir = start_research_session(
        topic=spec.topic,
        settings=settings,
        mock=False,
        top_k_papers=spec.top_k_papers,
        max_papers_per_query=spec.max_papers_per_query,
        max_iterations=spec.max_iterations,
        download_pdfs=spec.download_pdfs,
        local_pdf_dir=spec.local_pdf_dir,
        persist_rag=True,
        require_plan_approval=False,
        sources=list(spec.sources),
        on_event=on_event,
    )
    elapsed = time.perf_counter() - t0
    run_id = str(state.get("run_id") or out_dir.name)
    state_data = json.loads((out_dir / "state.json").read_text(encoding="utf-8"))
    metrics = _extract_run_metrics(
        run_id,
        state_data,
        case_key=spec.key,
        label=spec.label,
        elapsed_seconds=elapsed,
    )

    summary_doc = _load_summary_doc()
    summary_doc.setdefault("cases", {})[run_id] = metrics
    _save_summary_doc(summary_doc)

    vr = metrics["verification"]
    print(
        f"=== Finished {spec.key} -> run_id={run_id} ({elapsed:.1f}s, status={status}) | "
        f"papers={metrics['selected_count']} ({metrics['ingest_counts']}), "
        f"chunks={metrics['chunk_count']}, ev={metrics['evidence_total']}, "
        f"iter={metrics['iteration']} ({metrics['stop_reason']}), "
        f"validity={vr['reference_validity_rate']}, support={vr['claim_support_rate']} ===",
        flush=True,
    )
    return metrics


def summarize_all_outputs(settings: Settings) -> dict[str, Any]:
    """Scan all persisted outputs/<run_id>/state.json runs and refresh summary JSON."""
    summary_doc = _load_summary_doc()
    existing_cases = summary_doc.get("cases") or {}
    out_root = settings.output_path
    if out_root.exists():
        for child in sorted(out_root.iterdir()):
            state_path = child / "state.json"
            if not child.is_dir() or not state_path.exists():
                continue
            run_id = child.name
            prev = existing_cases.get(run_id) or {}
            state_data = json.loads(state_path.read_text(encoding="utf-8"))
            existing_cases[run_id] = _extract_run_metrics(
                run_id,
                state_data,
                case_key=prev.get("case_key"),
                label=prev.get("label"),
                elapsed_seconds=prev.get("elapsed_seconds"),
            )
    summary_doc["cases"] = existing_cases
    _save_summary_doc(summary_doc)
    print(f"Summarized {len(existing_cases)} runs into {SUMMARY_PATH}", flush=True)
    return summary_doc


def main() -> None:
    parser = argparse.ArgumentParser(description="ReacherAgent Phase 3 case study runner")
    parser.add_argument(
        "--cases",
        default="all",
        help="Comma-separated case keys (e.g. case2_hybrid_rag,case3_test_time_scaling) or 'all'.",
    )
    parser.add_argument(
        "--reaudit",
        default="",
        help="Re-run Citation Verifier on an existing outputs/<run_id> directory.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist re-audit updates back into state.json and report.md.",
    )
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Only scan outputs/*/state.json and refresh docs/case_studies_summary.json.",
    )
    args = parser.parse_args()

    settings = _load_effective_settings()
    if args.summarize_only:
        summarize_all_outputs(settings)
        return

    if args.reaudit:
        reaudit_run(args.reaudit.strip(), settings, write=args.write)
        return

    if args.cases.strip().lower() == "all":
        selected_specs = CASE_SPECS
    else:
        wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
        selected_specs = [spec for spec in CASE_SPECS if spec.key in wanted]
        missing = wanted - {s.key for s in selected_specs}
        if missing:
            raise ValueError(f"Unknown case keys: {sorted(missing)}")

    for spec in selected_specs:
        run_case(spec, settings)


if __name__ == "__main__":
    main()
