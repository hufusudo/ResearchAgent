"""API routes for runtime LLM/Embedding configuration, model discovery, and connectivity tests."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from api.schemas import (
    FetchModelsRequest,
    FetchModelsResponse,
    RuntimeSettingsResponse,
    TestConnectionRequest,
    TestConnectionResponse,
    UpdateSettingsRequest,
)
from api.settings_store import (
    RuntimeSettingsStore,
    fetch_provider_models,
    test_provider_connection,
)

router = APIRouter(prefix="/api/settings", tags=["settings"])


def _get_store(request: Request) -> RuntimeSettingsStore:
    return request.app.state.settings_store


@router.get("", response_model=RuntimeSettingsResponse)
def get_runtime_settings(request: Request) -> RuntimeSettingsResponse:
    """Return active masked LLM, Embedding, and academic source settings."""
    return _get_store(request).get_public_settings()


@router.put("", response_model=RuntimeSettingsResponse)
def update_runtime_settings(
    payload: UpdateSettingsRequest,
    request: Request,
) -> RuntimeSettingsResponse:
    """Persist updated LLM, Embedding, and academic source settings."""
    return _get_store(request).update(payload)


@router.post("/models", response_model=FetchModelsResponse)
def discover_models(
    payload: FetchModelsRequest,
    request: Request,
) -> FetchModelsResponse:
    """Query an OpenAI-compatible provider's /models endpoint using draft or saved credentials."""
    store = _get_store(request)
    eff_url, eff_key = store.resolve_credentials(
        target=payload.target,
        base_url=payload.base_url,
        api_key=payload.api_key,
    )
    try:
        models = fetch_provider_models(base_url=eff_url, api_key=eff_key)
        return FetchModelsResponse(models=models, count=len(models))
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/test", response_model=TestConnectionResponse)
def test_connection(
    payload: TestConnectionRequest,
    request: Request,
) -> TestConnectionResponse:
    """Run a live 1-token chat or embedding connectivity check."""
    store = _get_store(request)
    eff_url, eff_key = store.resolve_credentials(
        target=payload.target,
        base_url=payload.base_url,
        api_key=payload.api_key,
    )
    cfg = store.get_effective_settings()
    model = payload.model.strip() or (
        cfg.llm_model if payload.target == "llm" else cfg.embedding_model
    )
    return test_provider_connection(
        target=payload.target,
        base_url=eff_url,
        api_key=eff_key,
        model=model,
    )
