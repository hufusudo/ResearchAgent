"""REST and SSE routes for ReacherAgent survey tasks."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse, StreamingResponse

from api.manager import TaskManager
from api.schemas import (
    ApprovePlanRequest,
    ChunksResponse,
    CreateTaskRequest,
    DeleteTaskResponse,
    HealthResponse,
    TaskDetailResponse,
    TaskSummary,
    UploadResponse,
)
from api.settings_store import RuntimeSettingsStore

router = APIRouter(prefix="/api", tags=["tasks"])


def _get_manager(request: Request) -> TaskManager:
    return request.app.state.task_manager


def _get_store(request: Request) -> RuntimeSettingsStore:
    return request.app.state.settings_store


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request) -> HealthResponse:
    """Return service health and live non-secret configuration state."""
    store = _get_store(request)
    settings = store.get_effective_settings()
    pub = store.get_public_settings()
    return HealthResponse(
        status="ok",
        version="0.4.0",
        llm_model=settings.llm_model,
        llm_api_key_configured=bool(settings.llm_api_key.strip()),
        enabled_llm_models=pub.enabled_llm_models,
        embedding_provider=settings.embedding_provider,
        academic_sources=settings.academic_sources_list,
        openalex_email_configured=bool(settings.openalex_email.strip()),
        semantic_scholar_key_configured=bool(settings.semantic_scholar_api_key.strip()),
    )


@router.post("/uploads", response_model=UploadResponse, status_code=201)
async def upload_local_pdfs(
    request: Request,
    files: list[UploadFile] = File(...),
) -> UploadResponse:
    """Upload one or more local PDF files into a temporary batch directory."""
    manager = _get_manager(request)
    payload: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        payload.append((f.filename or "document.pdf", content))
    try:
        return manager.save_uploaded_pdfs(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/tasks", response_model=TaskSummary, status_code=202)
def create_task(payload: CreateTaskRequest, request: Request) -> TaskSummary:
    """Create and asynchronously start a literature survey task."""
    manager = _get_manager(request)
    return manager.create_task(payload)


@router.get("/tasks", response_model=list[TaskSummary])
def list_tasks(request: Request) -> list[TaskSummary]:
    """List recent survey tasks and discovered CLI runs."""
    manager = _get_manager(request)
    return manager.list_tasks()


@router.get("/tasks/{run_id}", response_model=TaskDetailResponse)
def get_task(run_id: str, request: Request) -> TaskDetailResponse:
    """Fetch complete details for a single survey run."""
    manager = _get_manager(request)
    detail = manager.get_task(run_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"Task '{run_id}' not found.")
    return detail


@router.delete("/tasks/{run_id}", response_model=DeleteTaskResponse)
def delete_task(run_id: str, request: Request) -> DeleteTaskResponse:
    """Delete a completed, paused, or failed historical task."""
    manager = _get_manager(request)
    try:
        manager.delete_task(run_id)
        return DeleteTaskResponse(deleted=True, run_id=run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/tasks/{run_id}/approve", response_model=TaskSummary, status_code=202)
def approve_task(
    run_id: str,
    payload: ApprovePlanRequest,
    request: Request,
) -> TaskSummary:
    """Approve or edit a paused HITL plan and resume execution."""
    manager = _get_manager(request)
    try:
        return manager.approve_task(run_id, approved_plan=payload.plan)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/tasks/{run_id}/events")
async def stream_task_events(run_id: str, request: Request) -> StreamingResponse:
    """Stream Server-Sent Events (trace & status updates) for a survey run."""
    manager = _get_manager(request)
    if manager.get_task(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Task '{run_id}' not found.")
    return StreamingResponse(
        manager.subscribe(run_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/tasks/{run_id}/report", response_class=PlainTextResponse)
def get_task_report(
    run_id: str,
    request: Request,
    download: bool = Query(default=False, description="Set true to trigger file download."),
) -> PlainTextResponse:
    """Return the generated Markdown literature survey report."""
    manager = _get_manager(request)
    report = manager.get_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report for '{run_id}' not found.")
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="reacher_survey_{run_id}.md"'
    return PlainTextResponse(
        content=report,
        media_type="text/markdown; charset=utf-8",
        headers=headers,
    )


@router.get("/tasks/{run_id}/bibtex", response_class=PlainTextResponse)
def get_task_bibtex(
    run_id: str,
    request: Request,
    download: bool = Query(default=False, description="Set true to trigger .bib download."),
) -> PlainTextResponse:
    """Return BibTeX entries for all selected papers in a survey run."""
    manager = _get_manager(request)
    bib = manager.get_bibtex(run_id)
    if bib is None:
        raise HTTPException(status_code=404, detail=f"Task '{run_id}' not found.")
    headers = {}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="reacher_refs_{run_id}.bib"'
    return PlainTextResponse(
        content=bib,
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )


@router.get("/tasks/{run_id}/chunks", response_model=ChunksResponse)
def get_task_chunks(
    run_id: str,
    request: Request,
    paper_id: str | None = Query(default=None, description="Optional filter like P01."),
) -> ChunksResponse:
    """Return indexed source chunks for citation provenance inspection."""
    manager = _get_manager(request)
    chunks = manager.get_chunks(run_id, paper_id=paper_id)
    if chunks is None:
        raise HTTPException(status_code=404, detail=f"Task '{run_id}' not found.")
    return ChunksResponse(run_id=run_id, total=len(chunks), chunks=chunks)
