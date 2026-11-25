"""Core Pydantic schemas and LangGraph state definitions."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class SubQuestion(BaseModel):
    """A single research sub-question decomposed from the main topic."""

    id: str = Field(..., description="Unique identifier such as SQ1, SQ2.")
    title: str = Field(..., description="Short title of the sub-question.")
    description: str = Field(..., description="Detailed scope of what this sub-question investigates.")
    search_queries: list[str] = Field(
        default_factory=list,
        description="English academic search queries targeting this sub-question.",
    )


class ResearchPlan(BaseModel):
    """Structured decomposition of the user's research topic."""

    topic: str
    refined_scope: str
    sub_questions: list[SubQuestion] = Field(default_factory=list)


class Paper(BaseModel):
    """Normalized metadata for an academic paper."""

    paper_id: str = ""
    external_id: str
    source: Literal["arxiv", "local_pdf", "semantic_scholar", "openalex"]
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int
    abstract: str
    url: str
    pdf_url: str | None = None
    categories: list[str] = Field(default_factory=list)
    citation_count: int | None = None
    venue: str = ""
    relevance_score: float = 0.0
    relevance_reason: str = ""
    full_text_available: bool = False
    ingest_status: Literal["pending", "full_text", "abstract_only", "failed"] = "pending"


class TextChunk(BaseModel):
    """A chunk of text from a paper bound to provenance metadata."""

    chunk_id: str
    paper_id: str
    title: str
    year: int
    url: str
    section: str
    page: int
    text: str


class RetrievedChunk(BaseModel):
    """A retrieved chunk with dense, sparse BM25, and RRF scores."""

    chunk: TextChunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0


class EvidenceItem(BaseModel):
    """Structured evidence extracted from retrieved paper chunks."""

    paper_id: str
    sub_question_id: str
    method: str
    dataset_or_setting: str
    key_findings: str
    limitations: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    """Execution trace event emitted during LangGraph execution."""

    step: str
    node: str
    message: str
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScreeningScore(BaseModel):
    """Relevance judgment for a single candidate paper."""

    paper_id: str
    relevance_score: float
    relevance_reason: str


class ScreeningResult(BaseModel):
    """Batch relevance screening output from the LLM."""

    scores: list[ScreeningScore] = Field(default_factory=list)


class EvidenceExtractionResult(BaseModel):
    """Structured evidence extraction output for a sub-question."""

    items: list[EvidenceItem] = Field(default_factory=list)


class SubQuestionEvaluation(BaseModel):
    """Coverage and sufficiency judgment for a single sub-question."""

    sub_question_id: str
    evidence_count: int = 0
    unique_paper_count: int = 0
    coverage_score: float
    sufficient: bool
    gaps: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Critic node evaluation determining whether supplemental retrieval is needed."""

    iteration: int = 0
    overall_sufficient: bool
    needs_more_retrieval: bool
    weak_subquestion_ids: list[str] = Field(default_factory=list)
    subquestion_evaluations: list[SubQuestionEvaluation] = Field(default_factory=list)
    stop_reason: str = ""
    reasoning: str = ""


class RewrittenQuery(BaseModel):
    """A reformulated academic search query targeting a weak sub-question."""

    sub_question_id: str
    query: str
    rationale: str = ""


class QueryRewriteResult(BaseModel):
    """Query rewriter node output for supplemental retrieval."""

    iteration: int = 0
    rewritten_queries: list[RewrittenQuery] = Field(default_factory=list)
    new_sub_question: SubQuestion | None = None
    reasoning: str = ""


class ClaimVerificationItem(BaseModel):
    """Verification verdict for a single cited claim against source chunks."""

    claim: str
    cited_paper_ids: list[str] = Field(default_factory=list)
    matched_chunk_ids: list[str] = Field(default_factory=list)
    verdict: Literal["supported", "partially_supported", "unsupported"]
    reason: str = ""


class ClaimVerificationBatch(BaseModel):
    """Batch claim verification output from the LLM."""

    items: list[ClaimVerificationItem] = Field(default_factory=list)


class CitationVerificationReport(BaseModel):
    """Hard and soft citation provenance audit for the generated survey."""

    total_cited_claims: int = 0
    checked_claims: int = 0
    supported_count: int = 0
    partially_supported_count: int = 0
    unsupported_count: int = 0
    invalid_citations: list[str] = Field(default_factory=list)
    cited_paper_ids: list[str] = Field(default_factory=list)
    uncited_paper_ids: list[str] = Field(default_factory=list)
    reference_validity_rate: float = 1.0
    claim_support_rate: float = 1.0
    details: list[ClaimVerificationItem] = Field(default_factory=list)


class ResearchState(TypedDict, total=False):
    """Mutable state passed across LangGraph nodes."""

    run_id: str
    topic: str
    plan: ResearchPlan
    candidate_papers: list[Paper]
    selected_papers: list[Paper]
    chunks: list[TextChunk]
    retrieved_chunks_by_subquestion: dict[str, list[RetrievedChunk]]
    evidence_by_subquestion: dict[str, list[EvidenceItem]]
    executed_queries: list[str]
    pending_queries: list[str]
    target_subquestion_ids: list[str]
    last_new_paper_count: int
    last_overlap_ratio: float
    evaluation_history: list[EvaluationResult]
    rewrite_history: list[QueryRewriteResult]
    latest_evaluation: EvaluationResult | None
    verification_report: CitationVerificationReport | None
    pre_revision_verification_report: CitationVerificationReport | None
    revision_count: int
    max_verification_revisions: int
    iteration: int
    max_iterations: int
    draft_markdown: str
    trace: list[TraceEvent]
    errors: list[str]
