export interface HealthResponse {
  status: string;
  version: string;
  llm_model: string;
  llm_api_key_configured: boolean;
  enabled_llm_models: string[];
  embedding_provider: string;
  academic_sources: string[];
  openalex_email_configured: boolean;
  semantic_scholar_key_configured: boolean;
}

export interface UploadedFileInfo {
  filename: string;
  size_bytes: number;
}

export interface UploadResponse {
  upload_batch_id: string;
  files: UploadedFileInfo[];
}

export interface SubQuestion {
  id: string;
  title: string;
  description: string;
  search_queries: string[];
}

export interface ResearchPlan {
  topic: string;
  refined_scope: string;
  sub_questions: SubQuestion[];
}

export interface Paper {
  paper_id: string;
  external_id: string;
  source: "arxiv" | "local_pdf" | "semantic_scholar" | "openalex";
  title: string;
  authors: string[];
  year: number;
  abstract: string;
  url: string;
  pdf_url: string | null;
  categories: string[];
  citation_count: number | null;
  venue: string;
  relevance_score: number;
  relevance_reason: string;
  full_text_available: boolean;
  ingest_status: "pending" | "full_text" | "abstract_only" | "failed";
}

export interface TextChunk {
  chunk_id: string;
  paper_id: string;
  title: string;
  year: number;
  url: string;
  section: string;
  page: number;
  text: string;
}

export interface EvidenceItem {
  paper_id: string;
  sub_question_id: string;
  method: string;
  dataset_or_setting: string;
  key_findings: string;
  limitations: string;
  source_chunk_ids: string[];
}

export interface SubQuestionEvaluation {
  sub_question_id: string;
  evidence_count: number;
  unique_paper_count: number;
  coverage_score: number;
  sufficient: boolean;
  gaps: string[];
}

export interface EvaluationResult {
  iteration: number;
  overall_sufficient: boolean;
  needs_more_retrieval: boolean;
  weak_subquestion_ids: string[];
  subquestion_evaluations: SubQuestionEvaluation[];
  stop_reason: string;
  reasoning: string;
}

export interface RewrittenQuery {
  sub_question_id: string;
  query: string;
  rationale: string;
}

export interface QueryRewriteResult {
  iteration: number;
  rewritten_queries: RewrittenQuery[];
  new_sub_question: SubQuestion | null;
  reasoning: string;
}

export interface ClaimVerificationItem {
  claim: string;
  cited_paper_ids: string[];
  matched_chunk_ids: string[];
  verdict: "supported" | "partially_supported" | "unsupported";
  reason: string;
}

export interface CitationVerificationReport {
  total_cited_claims: number;
  checked_claims: number;
  supported_count: number;
  partially_supported_count: number;
  unsupported_count: number;
  invalid_citations: string[];
  cited_paper_ids: string[];
  uncited_paper_ids: string[];
  reference_validity_rate: number;
  claim_support_rate: number;
  details: ClaimVerificationItem[];
}

export interface TraceEvent {
  step: string;
  node: string;
  message: string;
  timestamp: string;
  metadata: Record<string, unknown>;
}

export interface TaskSummary {
  run_id: string;
  topic: string;
  status: string;
  created_at: string;
  updated_at: string;
  paper_count: number;
  chunk_count: number;
  iteration: number;
  error_message: string | null;
}

export interface TaskDetailResponse {
  run_id: string;
  topic: string;
  status: string;
  created_at: string;
  updated_at: string;
  output_dir: string;
  error_message: string | null;
  plan: ResearchPlan | null;
  selected_papers: Paper[];
  evidence_by_subquestion: Record<string, EvidenceItem[]>;
  evaluation_history: EvaluationResult[];
  rewrite_history: QueryRewriteResult[];
  latest_evaluation: EvaluationResult | null;
  verification_report: CitationVerificationReport | null;
  trace: TraceEvent[];
  errors: string[];
  draft_markdown: string;
}

export interface CreateTaskPayload {
  topic: string;
  top_k_papers: number;
  max_papers_per_query: number;
  max_iterations: number;
  download_pdfs: boolean;
  require_plan_approval: boolean;
  mock: boolean;
  upload_batch_id: string | null;
  sources: string[];
  llm_model?: string | null;
}

export interface RuntimeSettingsResponse {
  llm_base_url: string;
  llm_model: string;
  llm_api_key_configured: boolean;
  llm_api_key_masked: string;
  available_llm_models: string[];
  enabled_llm_models: string[];
  embedding_provider: "openai" | "local_hash";
  embedding_base_url: string;
  embedding_model: string;
  embedding_api_key_configured: boolean;
  embedding_api_key_masked: string;
  available_embedding_models: string[];
  academic_sources: string[];
  openalex_email: string;
  semantic_scholar_key_configured: boolean;
}

export interface UpdateSettingsRequest {
  llm_base_url?: string;
  llm_api_key?: string;
  llm_model?: string;
  available_llm_models?: string[];
  enabled_llm_models?: string[];
  embedding_provider?: "openai" | "local_hash";
  embedding_base_url?: string;
  embedding_api_key?: string;
  embedding_model?: string;
  available_embedding_models?: string[];
  academic_sources?: string[];
  openalex_email?: string;
  semantic_scholar_api_key?: string;
}

export interface FetchModelsResponse {
  models: string[];
  count: number;
}

export interface TestConnectionResponse {
  ok: boolean;
  message: string;
  latency_ms: number;
  model_used: string;
  detail: string;
}
