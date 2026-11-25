"""CLI orchestrator for Phase 4 baseline comparisons and module ablations."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from api.settings_store import RuntimeSettingsStore
from core.config import Settings, get_settings
from core.llm.embeddings import create_embedder
from core.retrieval.arxiv_client import ArxivClient
from core.retrieval.openalex_client import OpenAlexClient
from core.schemas import ResearchPlan, ResearchState
from eval.ablations import (
    extract_verifier_revision_metrics,
    run_evidence_matrix_ablation,
    run_reflection_ablation,
    run_retrieval_ablation,
)
from eval.baselines import (
    run_direct_llm_baseline,
    run_naive_rag_baseline,
    run_reacher_full,
)
from eval.benchmark_topics import BenchmarkTopic, get_topics_by_ids
from eval.charts import generate_all_charts
from eval.report_builder import build_all_artifacts


def _load_effective_settings(mock: bool = False) -> Settings:
    base = get_settings()
    if mock:
        return base.model_copy(
            update={
                "llm_api_key": "",
                "embedding_provider": "local_hash",
            }
        )
    try:
        return RuntimeSettingsStore(base).get_effective_settings()
    except Exception:
        return base


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return obj.model_dump(mode="json")
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _strip_large_drafts(section_data: dict[str, Any]) -> dict[str, Any]:
    """Keep summary metrics in raw/<topic>.json while storing full draft separately if needed."""
    cleaned = dict(section_data)
    cleaned.pop("draft_markdown", None)
    return cleaned


def _load_step_cache(cache_path: Path) -> dict[str, Any]:
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def _save_step_cache(cache_path: Path, cache: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(cache, f)
    tmp.replace(cache_path)


def run_single_topic(
    topic: BenchmarkTopic,
    settings: Settings,
    output_dir: Path,
    *,
    top_k_papers: int = 6,
    max_papers_per_query: int = 5,
    max_iterations: int = 2,
    download_pdfs: bool = True,
    mock: bool = False,
    resume: bool = False,
) -> dict[str, Any]:
    """Execute all systems & ablations for one benchmark topic with atomic checkpointing."""
    raw_path = output_dir / "raw" / f"{topic.topic_id}.json"
    cache_path = output_dir / "cache" / f"{topic.topic_id}_steps.pkl"
    drafts_dir = output_dir / "drafts" / topic.topic_id
    drafts_dir.mkdir(parents=True, exist_ok=True)

    if resume and raw_path.exists():
        existing = json.loads(raw_path.read_text(encoding="utf-8"))
        if existing.get("completed"):
            print(f"[{topic.topic_id}] Cached & complete, skipping (--resume).")
            return existing

    step_cache = _load_step_cache(cache_path) if resume else {}
    topic_settings = settings.model_copy(
        update={"data_dir": str(output_dir / "work" / topic.topic_id)}
    )
    print(f"\n=== [{topic.topic_id}] ({topic.language}) {topic.title} ===")

    # 1. Run ReacherAgent Full first to produce canonical ResearchPlan & corpus + iter0 snapshot
    if "reacher" in step_cache:
        reacher_metrics, full_state, iter0_snapshot = step_cache["reacher"]
        print(f"[{topic.topic_id}] 1/6 Loaded ReacherAgent (Full) from step cache.")
    else:
        print(f"[{topic.topic_id}] 1/6 Running ReacherAgent (Full)...")
        reacher_metrics, full_state, iter0_snapshot = run_reacher_full(
            topic,
            topic_settings,
            top_k_papers=top_k_papers,
            max_papers_per_query=max_papers_per_query,
            max_iterations=max_iterations,
            download_pdfs=download_pdfs,
            mock=mock,
        )
        step_cache["reacher"] = (reacher_metrics, full_state, iter0_snapshot)
        _save_step_cache(cache_path, step_cache)
    (drafts_dir / "reacher_agent.md").write_text(
        reacher_metrics.get("draft_markdown", ""), encoding="utf-8"
    )
    canonical_plan: ResearchPlan = full_state["plan"]
    print(
        f"[{topic.topic_id}]   -> elapsed={reacher_metrics['elapsed_seconds']:.1f}s, "
        f"papers={reacher_metrics['paper_count']} (full={reacher_metrics['full_text_count']}), "
        f"validity={reacher_metrics['reference_validity_rate']:.1%}, "
        f"support={reacher_metrics['claim_support_rate']:.1%}, "
        f"rubric={reacher_metrics['rubric']['overall_score']:.2f}"
    )

    # 2. Direct LLM Baseline
    if "direct_llm" in step_cache:
        direct_metrics = step_cache["direct_llm"]
        print(f"[{topic.topic_id}] 2/6 Loaded Direct LLM Baseline from step cache.")
    else:
        t_step = time.perf_counter()
        print(f"[{topic.topic_id}] 2/6 Running Direct LLM Baseline + external reference verification...")
        openalex_client = None if mock else OpenAlexClient(email=topic_settings.openalex_email)
        arxiv_client = None if mock else ArxivClient()
        direct_metrics = run_direct_llm_baseline(
            topic,
            topic_settings,
            canonical_plan,
            openalex_client=openalex_client,
            arxiv_client=arxiv_client,
            mock=mock,
        )
        step_cache["direct_llm"] = direct_metrics
        _save_step_cache(cache_path, step_cache)
        print(f"[{topic.topic_id}]   -> ({time.perf_counter()-t_step:.1f}s)")
    (drafts_dir / "direct_llm.md").write_text(
        direct_metrics.get("draft_markdown", ""), encoding="utf-8"
    )
    print(
        f"[{topic.topic_id}]   -> Direct LLM: validity={direct_metrics['reference_validity_rate']:.1%}, "
        f"support={direct_metrics['claim_support_rate']:.1%}, "
        f"rubric={direct_metrics['rubric']['overall_score']:.2f}"
    )

    # 3. Naive RAG Baseline
    if "naive_rag" in step_cache:
        naive_metrics = step_cache["naive_rag"]
        print(f"[{topic.topic_id}] 3/6 Loaded Naive RAG Baseline from step cache.")
    else:
        t_step = time.perf_counter()
        print(f"[{topic.topic_id}] 3/6 Running Naive RAG Baseline...")
        naive_metrics = run_naive_rag_baseline(
            topic,
            topic_settings,
            canonical_plan,
            top_k_papers=top_k_papers,
            download_pdfs=download_pdfs,
            mock=mock,
            work_dir=output_dir / "naive_pdfs",
        )
        step_cache["naive_rag"] = naive_metrics
        _save_step_cache(cache_path, step_cache)
        print(f"[{topic.topic_id}]   -> ({time.perf_counter()-t_step:.1f}s)")
    (drafts_dir / "naive_rag.md").write_text(
        naive_metrics.get("draft_markdown", ""), encoding="utf-8"
    )
    print(
        f"[{topic.topic_id}]   -> Naive RAG: validity={naive_metrics['reference_validity_rate']:.1%}, "
        f"support={naive_metrics['claim_support_rate']:.1%}, "
        f"rubric={naive_metrics['rubric']['overall_score']:.2f}"
    )

    # 4. Ablation 1: 4-way Retrieval Strategy
    if "ablation_retrieval" in step_cache:
        ab_retrieval = step_cache["ablation_retrieval"]
        print(f"[{topic.topic_id}] 4/6 Loaded Retrieval Ablation from step cache.")
    else:
        t_step = time.perf_counter()
        print(f"[{topic.topic_id}] 4/6 Running Retrieval Strategy Ablation (4-way)...")
        embedder = create_embedder(topic_settings, force_local=mock)
        ab_retrieval = run_retrieval_ablation(
            topic,
            canonical_plan,
            full_state.get("chunks") or [],
            embedder,
            topic_settings,
            top_k=8,
            mock=mock,
        )
        step_cache["ablation_retrieval"] = ab_retrieval
        _save_step_cache(cache_path, step_cache)
        print(f"[{topic.topic_id}]   -> ({time.perf_counter()-t_step:.1f}s) Retrieval ablation done.")

    # 5. Ablation 2: Reflection Loop (max_iterations=0 vs 2)
    if "ablation_reflection" in step_cache:
        ab_reflection = step_cache["ablation_reflection"]
        print(f"[{topic.topic_id}] 5/6 Loaded Reflection Ablation from step cache.")
    else:
        t_step = time.perf_counter()
        print(f"[{topic.topic_id}] 5/6 Running Reflection Loop Ablation (iter=0 vs iter=2)...")
        ab_reflection = run_reflection_ablation(
            topic,
            iter0_snapshot,
            full_state,
            reacher_metrics,
            topic_settings,
            mock=mock,
        )
        step_cache["ablation_reflection"] = ab_reflection
        _save_step_cache(cache_path, step_cache)
        print(f"[{topic.topic_id}]   -> ({time.perf_counter()-t_step:.1f}s) Reflection ablation done.")

    # 6. Ablation 3 & 4: Evidence Matrix vs Direct Chunks + Verifier Revision Uplift
    if "ablation_matrix" in step_cache:
        ab_matrix = step_cache["ablation_matrix"]
        print(f"[{topic.topic_id}] 6/6 Loaded Matrix Ablation from step cache.")
    else:
        t_step = time.perf_counter()
        print(f"[{topic.topic_id}] 6/6 Running Evidence Matrix Ablation & Verifier Revision Audit...")
        ab_matrix = run_evidence_matrix_ablation(
            topic,
            full_state,
            reacher_metrics,
            topic_settings,
            mock=mock,
        )
        step_cache["ablation_matrix"] = ab_matrix
        _save_step_cache(cache_path, step_cache)
        print(f"[{topic.topic_id}]   -> ({time.perf_counter()-t_step:.1f}s) Matrix & revision audit done.")
    ab_revision = extract_verifier_revision_metrics(topic, reacher_metrics)

    record = {
        "topic_id": topic.topic_id,
        "language": topic.language,
        "domain": topic.domain,
        "topic_title": topic.title,
        "completed": True,
        "direct_llm": _strip_large_drafts(direct_metrics),
        "naive_rag": _strip_large_drafts(naive_metrics),
        "reacher_agent": _strip_large_drafts(reacher_metrics),
        "ablation_retrieval": ab_retrieval,
        "ablation_reflection": ab_reflection,
        "ablation_matrix": ab_matrix,
        "ablation_verifier_revision": ab_revision,
    }
    _atomic_write_json(raw_path, record)
    print(f"[{topic.topic_id}] Saved {raw_path}")
    return record


def run_benchmark_suite(
    *,
    topic_ids: list[str] | None = None,
    mock: bool = False,
    resume: bool = False,
    top_k_papers: int = 6,
    max_papers_per_query: int = 5,
    max_iterations: int = 2,
    download_pdfs: bool = True,
    workers: int = 1,
    output_dir: Path = Path("eval/results"),
    report_path: Path = Path("docs/phase4_evaluation_report.md"),
    settings_override: Settings | None = None,
) -> dict[str, Any]:
    """Run selected benchmark topics and generate all CSV, JSON, SVG, and Markdown artifacts."""
    settings = settings_override or _load_effective_settings(mock=mock)
    topics = get_topics_by_ids(topic_ids)
    failures: list[tuple[str, str]] = []

    if workers <= 1 or len(topics) <= 1:
        for topic in topics:
            try:
                run_single_topic(
                    topic,
                    settings,
                    output_dir,
                    top_k_papers=top_k_papers,
                    max_papers_per_query=max_papers_per_query,
                    max_iterations=max_iterations,
                    download_pdfs=download_pdfs,
                    mock=mock,
                    resume=resume,
                )
                build_all_artifacts(output_dir=output_dir, report_path=report_path)
            except Exception as exc:
                print(f"[{topic.topic_id}] ERROR: {exc}")
                failures.append((topic.topic_id, str(exc)))
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(topics))) as executor:
            futs = {
                executor.submit(
                    run_single_topic,
                    topic,
                    settings,
                    output_dir,
                    top_k_papers=top_k_papers,
                    max_papers_per_query=max_papers_per_query,
                    max_iterations=max_iterations,
                    download_pdfs=download_pdfs,
                    mock=mock,
                    resume=resume,
                ): topic
                for topic in topics
            }
            for fut in as_completed(futs):
                topic = futs[fut]
                try:
                    fut.result()
                    build_all_artifacts(output_dir=output_dir, report_path=report_path)
                except Exception as exc:
                    print(f"[{topic.topic_id}] ERROR: {exc}")
                    failures.append((topic.topic_id, str(exc)))

    summary = build_all_artifacts(output_dir=output_dir, report_path=report_path)
    if failures:
        print(f"\n[WARN] {len(failures)} topic(s) failed (re-run with --resume): {failures}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ReacherAgent Phase 4 Quantitative Evaluation & Ablation Runner"
    )
    parser.add_argument(
        "--topics",
        type=str,
        default="",
        help="Comma-separated topic IDs (e.g. T01,T02).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 10 bilingual benchmark topics (T01..T10).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run offline deterministic dry run with MockLLMClient and LocalHashEmbedder.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip topics that already have completed raw/<topic_id>.json.",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Selected papers cap per run.")
    parser.add_argument("--max-per-query", type=int, default=5, help="Max results per source query.")
    parser.add_argument("--max-iterations", type=int, default=2, help="Max reflection iterations.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent topics to run.")
    parser.add_argument("--no-pdf-download", action="store_true", help="Disable PDF full-text downloads.")
    parser.add_argument("--output-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--report-path", type=Path, default=Path("docs/phase4_evaluation_report.md"))
    parser.add_argument(
        "--rebuild-only",
        action="store_true",
        help="Only rebuild CSVs, SVG charts, and Markdown report from existing raw/*.json.",
    )
    parser.add_argument(
        "--charts-only",
        action="store_true",
        help="Only regenerate Chinese SVG charts from existing summary.json without overwriting Markdown report.",
    )
    args = parser.parse_args()

    if args.charts_only:
        summary_path = args.output_dir / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        paths = generate_all_charts(summary, args.output_dir / "charts")
        print(f"Regenerated {len(paths)} Chinese SVG charts -> {args.output_dir / 'charts'}")
        return

    if args.rebuild_only:
        summary = build_all_artifacts(output_dir=args.output_dir, report_path=args.report_path)
        print(f"Rebuilt artifacts for {summary['topic_count']} topics -> {args.report_path}")
        return

    topic_ids: list[str] | None = None
    if args.topics.strip():
        topic_ids = [t.strip() for t in args.topics.split(",") if t.strip()]
    elif not args.all:
        topic_ids = ["T01"]

    summary = run_benchmark_suite(
        topic_ids=topic_ids,
        mock=args.mock,
        resume=args.resume,
        top_k_papers=args.top_k,
        max_papers_per_query=args.max_per_query,
        max_iterations=args.max_iterations,
        download_pdfs=not args.no_pdf_download,
        workers=args.workers,
        output_dir=args.output_dir,
        report_path=args.report_path,
    )
    print(
        f"\n[DONE] Evaluated {summary['topic_count']} topic(s). "
        f"Artifacts in {args.output_dir} and {args.report_path}."
    )


if __name__ == "__main__":
    main()
