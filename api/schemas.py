"""Pydantic request and response schemas for the FastAPI REST and SSE endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from core.schemas import (
    CitationVerificationReport,
    EvaluationResult,
    EvidenceItem,
    Paper,
    QueryRewriteResult,
    ResearchPlan,
    TextChunk,
    TraceEvent,
)


class HealthResponse(BaseModel):
    """Service health and non-secret runtime configuration summary."""

    status: str = "ok"
    version: str = "0.4.0"
    llm_model: str
    llm_api_key_configured: bool
    enabled_llm_models: list[str] = Field(default_factory=list)
    embedding_provider: str
    academic_sources: list[str] = Field(default_factory=list)
    openalex_email_configured: bool = False
    semantic_scholar_key_configured: bool = False


class UploadedFileInfo(BaseModel):
    """Metadata of a single uploaded local PDF file."""

    filename: str
    size_bytes: int


class UploadResponse(BaseModel):
    """Response returned after uploading local PDFs for a survey run."""

    upload_batch_id: str
    files: list[UploadedFileInfo] = Field(default_factory=list)


class CreateTaskRequest(BaseModel):
    """Payload to launch a new literature survey task."""

    topic: str = Field(..., min_length=2, description="Research topic or question.")
    top_k_papers: int | None = Field(default=None, ge=1, le=30)
    max_papers_per_query: int | None = Field(default=None, ge=1, le=30)
    max_iterations: int | None = Field(default=None, ge=0, le=5)
    download_pdfs: bool = True
    require_plan_approval: bool = False
    mock: bool = False
    upload_batch_id: str | None = None
    sources: list[str] | None = None
    llm_model: str | None = None


class ApprovePlanRequest(BaseModel):
    """Payload to approve or revise a paused HITL research plan."""

    plan: ResearchPlan | None = None


class TaskSummary(BaseModel):
    """Lightweight task metadata row for task history listing."""

    run_id: str
    topic: str
    status: str
    created_at: str
    updated_at: str
    paper_count: int = 0
    chunk_count: int = 0
    iteration: int = 0
    error_message: str | None = None


class TaskDetailResponse(BaseModel):
    """Full survey task state exposed to decoupled frontends."""

    run_id: str
    topic: str
    status: str
    created_at: str
    updated_at: str
    output_dir: str
    error_message: str | None = None
    plan: ResearchPlan | None = None
    selected_papers: list[Paper] = Field(default_factory=list)
    evidence_by_subquestion: dict[str, list[EvidenceItem]] = Field(default_factory=dict)
    evaluation_history: list[EvaluationResult] = Field(default_factory=list)
    rewrite_history: list[QueryRewriteResult] = Field(default_factory=list)
    latest_evaluation: EvaluationResult | None = None
    verification_report: CitationVerificationReport | None = None
    trace: list[TraceEvent] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    draft_markdown: str = ""


class ChunksResponse(BaseModel):
    """Indexed source chunks returned for citation provenance inspection."""

    run_id: str
    total: int
    chunks: list[TextChunk] = Field(default_factory=list)


class DeleteTaskResponse(BaseModel):
    """Confirmation returned after deleting a historical survey task."""

    deleted: bool = True
    run_id: str


class RuntimeSettingsResponse(BaseModel):
    """Non-secret runtime API and model settings exposed to the frontend."""

    llm_base_url: str
    llm_model: str
    llm_api_key_configured: bool
    llm_api_key_masked: str
    available_llm_models: list[str] = Field(default_factory=list)
    enabled_llm_models: list[str] = Field(default_factory=list)
    embedding_provider: Literal["openai", "local_hash"]
    embedding_base_url: str
    embedding_model: str
    embedding_api_key_configured: bool
    embedding_api_key_masked: str
    available_embedding_models: list[str] = Field(default_factory=list)
    academic_sources: list[str] = Field(default_factory=list)
    openalex_email: str = ""
    semantic_scholar_key_configured: bool = False


class UpdateSettingsRequest(BaseModel):
    """Partial update payload for runtime API and model configuration."""

    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str | None = None
    available_llm_models: list[str] | None = None
    enabled_llm_models: list[str] | None = None
    embedding_provider: Literal["openai", "local_hash"] | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_model: str | None = None
    available_embedding_models: list[str] | None = None
    academic_sources: list[str] | None = None
    openalex_email: str | None = None
    semantic_scholar_api_key: str | None = None


class FetchModelsRequest(BaseModel):
    """Request payload to fetch available model IDs from an OpenAI-compatible API."""

    target: Literal["llm", "embedding"] = "llm"
    base_url: str = ""
    api_key: str = ""


class FetchModelsResponse(BaseModel):
    """Available model IDs retrieved from the provider."""

    models: list[str] = Field(default_factory=list)
    count: int = 0


class TestConnectionRequest(BaseModel):
    """Request payload to test live API connectivity for LLM or Embedding."""

    target: Literal["llm", "embedding"] = "llm"
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class TestConnectionResponse(BaseModel):
    """Result of a live API connectivity test."""

    ok: bool
    message: str
    latency_ms: int = 0
    model_used: str = ""
    detail: str = ""
