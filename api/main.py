"""FastAPI application factory and ASGI entrypoint with built SPA static hosting."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.manager import TaskManager
from api.routes.settings import router as settings_router
from api.routes.tasks import router as tasks_router
from api.settings_store import RuntimeSettingsStore
from core.config import Settings, get_settings

DEFAULT_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def create_app(
    settings: Settings | None = None,
    persist_rag: bool = True,
    static_dir: Path | None = None,
) -> FastAPI:
    """Create a configured FastAPI instance with CORS, TaskManager, SettingsStore, and SPA hosting."""
    cfg = settings or get_settings()
    settings_store = RuntimeSettingsStore(cfg)

    application = FastAPI(
        title="ReacherAgent API",
        version="0.4.0",
        description="REST and SSE API for autonomous academic literature surveys.",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.state.settings = cfg
    application.state.settings_store = settings_store
    application.state.task_manager = TaskManager(
        settings=cfg,
        persist_rag=persist_rag,
        settings_store=settings_store,
    )
    application.include_router(tasks_router)
    application.include_router(settings_router)

    dist_path = static_dir or DEFAULT_FRONTEND_DIST
    index_html = dist_path / "index.html"
    if dist_path.exists() and index_html.is_file():
        dist_resolved = dist_path.resolve()
        assets_dir = dist_path / "assets"
        if assets_dir.exists() and assets_dir.is_dir():
            application.mount(
                "/assets",
                StaticFiles(directory=str(assets_dir)),
                name="frontend-assets",
            )

        @application.get("/{full_path:path}", include_in_schema=False)
        def serve_spa(full_path: str) -> FileResponse:
            if full_path == "api" or full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="API endpoint not found.")
            if full_path:
                candidate = (dist_path / full_path).resolve()
                if candidate.is_relative_to(dist_resolved) and candidate.is_file():
                    return FileResponse(candidate)
            return FileResponse(index_html)

    return application


app = create_app()
