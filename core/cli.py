"""Typer CLI entrypoint for ReacherAgent."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from core.config import get_settings
from core.graph.workflow import resume_research_session, start_research_session
from core.schemas import ResearchPlan, TraceEvent

app = typer.Typer(
    name="reacher",
    help="ReacherAgent CLI - Autonomous literature survey agent.",
    add_completion=False,
)
console = Console()


def _display_plan(plan: ResearchPlan) -> None:
    console.print(f"\n[bold cyan]Proposed Scope:[/bold cyan] {plan.refined_scope}")
    plan_table = Table(title="Proposed Research Plan")
    plan_table.add_column("ID", style="cyan", no_wrap=True)
    plan_table.add_column("Sub-Question", style="white")
    plan_table.add_column("Search Queries", style="green")
    for sq in plan.sub_questions:
        plan_table.add_row(sq.id, sq.title, "\n".join(sq.search_queries))
    console.print(plan_table)


@app.command("check-config")
def check_config() -> None:
    """Display active non-secret configuration loaded from environment and .env."""
    settings = get_settings()
    table = Table(title="ReacherAgent Configuration")
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    api_status = "configured" if settings.llm_api_key.strip() else "NOT SET"
    emb_key_status = "configured" if settings.effective_embedding_api_key else "NOT SET"

    table.add_row("LLM_BASE_URL", settings.llm_base_url)
    table.add_row("LLM_MODEL", settings.llm_model)
    table.add_row("LLM_API_KEY", api_status)
    table.add_row("EMBEDDING_PROVIDER", settings.embedding_provider)
    table.add_row("EMBEDDING_BASE_URL", settings.effective_embedding_base_url)
    table.add_row("EMBEDDING_MODEL", settings.embedding_model)
    table.add_row("EMBEDDING_API_KEY", emb_key_status)
    table.add_row("MAX_PAPERS_PER_QUERY", str(settings.max_papers_per_query))
    table.add_row("TOP_K_PAPERS", str(settings.top_k_papers))
    table.add_row("MAX_ITERATIONS", str(settings.max_iterations))
    table.add_row("DATA_DIR", str(settings.data_path))
    table.add_row("OUTPUT_DIR", str(settings.output_path))

    console.print(table)


@app.command("run")
def run_survey(
    topic: str = typer.Argument(..., help="Research topic to investigate."),
    top_k: Optional[int] = typer.Option(
        None, "--top-k", min=1, help="Maximum number of selected papers."
    ),
    max_per_query: Optional[int] = typer.Option(
        None, "--max-per-query", min=1, help="Maximum arXiv hits per search query."
    ),
    max_iterations: Optional[int] = typer.Option(
        None, "--max-iterations", min=0, help="Maximum reflection retrieval rounds."
    ),
    local_pdfs: Optional[Path] = typer.Option(
        None, "--local-pdfs", help="Optional folder of local PDF papers."
    ),
    no_pdf_download: bool = typer.Option(
        False, "--no-pdf-download", help="Skip PDF download and index abstracts only."
    ),
    hitl: bool = typer.Option(
        False, "--hitl", help="Pause after planning to review/confirm sub-questions before retrieval."
    ),
    mock: bool = typer.Option(
        False, "--mock", help="Run offline with deterministic mock LLM and arXiv data."
    ),
) -> None:
    """Run an end-to-end Stage-2 literature survey for a research topic."""
    settings = get_settings()
    mode_label = "MOCK (offline)" if mock else f"LIVE ({settings.llm_model})"
    console.print(f"[bold cyan]ReacherAgent[/bold cyan] | Mode: {mode_label}")
    console.print(f"[bold]Topic:[/bold] {topic}\n")

    def on_event(event: TraceEvent) -> None:
        console.print(f"[bold blue]\\[{event.node}][/bold blue] {event.message}")

    try:
        state, status, output_dir = start_research_session(
            topic=topic,
            settings=settings,
            mock=mock,
            top_k_papers=top_k,
            max_papers_per_query=max_per_query,
            max_iterations=max_iterations,
            download_pdfs=not no_pdf_download,
            local_pdf_dir=local_pdfs,
            require_plan_approval=hitl,
            on_event=on_event,
        )

        if status == "awaiting_approval":
            plan = state["plan"]
            _display_plan(plan)
            extra_query = typer.prompt(
                "Optional: add an extra English search query to SQ1 (press Enter to skip)",
                default="",
            ).strip()
            if extra_query and plan.sub_questions:
                plan = plan.model_copy(deep=True)
                plan.sub_questions[0].search_queries.append(extra_query)

            if not typer.confirm("Proceed with this research plan?", default=True):
                console.print("[yellow][CANCELLED][/yellow] Execution paused at plan approval.")
                raise typer.Exit(code=0)

            console.print()
            state, output_dir = resume_research_session(
                run_id=state["run_id"],
                approved_plan=plan,
                settings=settings,
                mock=mock,
                top_k_papers=top_k,
                max_papers_per_query=max_per_query,
                max_iterations=max_iterations,
                download_pdfs=not no_pdf_download,
                local_pdf_dir=local_pdfs,
                on_event=on_event,
            )
    except typer.Exit:
        raise
    except Exception as exc:
        console.print(f"[bold red][ERROR][/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    selected = state.get("selected_papers") or []
    errors = state.get("errors") or []
    eval_history = state.get("evaluation_history") or []
    rewrite_history = state.get("rewrite_history") or []
    ver_report = state.get("verification_report")

    console.print()
    table = Table(title=f"Selected Literature ({len(selected)} papers)")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Year", style="magenta", no_wrap=True)
    table.add_column("Source", style="blue", no_wrap=True)
    table.add_column("Ingest", style="yellow", no_wrap=True)
    table.add_column("Title", style="white")

    for paper in selected:
        table.add_row(
            paper.paper_id,
            str(paper.year or "n.d."),
            paper.source,
            paper.ingest_status,
            paper.title,
        )
    console.print(table)

    console.print(
        f"\n[bold]Reflection Summary:[/bold] {state.get('iteration', 0)} supplemental round(s), "
        f"{len(eval_history)} evaluation(s), {len(rewrite_history)} rewrite batch(es)."
    )
    if ver_report is not None:
        invalid_str = ", ".join(ver_report.invalid_citations) or "None"
        console.print(
            f"[bold]Citation Audit:[/bold] validity={ver_report.reference_validity_rate:.0%}, "
            f"claim_support={ver_report.claim_support_rate:.0%} "
            f"({ver_report.supported_count}/{ver_report.checked_claims} supported), "
            f"invalid={invalid_str}"
        )

    if errors:
        console.print(f"\n[yellow]Warnings / Non-fatal issues ({len(errors)}):[/yellow]")
        for err in errors:
            console.print(f"  - {err}")

    console.print(f"\n[bold green][DONE][/bold green] Report saved to: {output_dir / 'report.md'}")
    console.print(f"[bold green][DONE][/bold green] State saved to:  {output_dir / 'state.json'}")


if __name__ == "__main__":
    app()
