"""OpenAI-compatible LLM client and deterministic mock client."""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Protocol, TypeVar, runtime_checkable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from core.config import Settings
from core.schemas import (
    ClaimVerificationBatch,
    ClaimVerificationItem,
    EvaluationResult,
    EvidenceExtractionResult,
    EvidenceItem,
    QueryRewriteResult,
    ResearchPlan,
    RewrittenQuery,
    ScreeningResult,
    ScreeningScore,
    SubQuestion,
    SubQuestionEvaluation,
)

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol implemented by real and mock LLM clients."""

    def invoke_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        """Return plain text or Markdown output."""
        ...

    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        temperature: float = 0.1,
    ) -> T:
        """Return a Pydantic-validated structured output."""
        ...


def _extract_json_payload(raw_text: str) -> str:
    """Strip markdown code fences and isolate the outermost JSON object or array."""
    text = raw_text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    if text.startswith("{") and text.endswith("}"):
        return text
    if text.startswith("[") and text.endswith("]"):
        return text

    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
        return text[obj_start : obj_end + 1]

    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
        return text[arr_start : arr_end + 1]

    return text


class OpenAILLMClient:
    """LLM client backed by any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise ValueError(
                "LLM_API_KEY is required when running without --mock. "
                "Set it in .env or use --mock for offline testing."
            )
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self.max_retries = max(1, max_retries)
        self._stats_lock = threading.Lock()
        self._usage_stats: dict[str, Any] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "elapsed_seconds": 0.0,
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Return thread-safe cumulative LLM call and token usage statistics."""
        with self._stats_lock:
            return {
                "calls": int(self._usage_stats["calls"]),
                "prompt_tokens": int(self._usage_stats["prompt_tokens"]),
                "completion_tokens": int(self._usage_stats["completion_tokens"]),
                "total_tokens": int(self._usage_stats["total_tokens"]),
                "elapsed_seconds": round(float(self._usage_stats["elapsed_seconds"]), 3),
            }

    def reset_usage_stats(self) -> None:
        """Reset cumulative LLM usage counters."""
        with self._stats_lock:
            self._usage_stats = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "elapsed_seconds": 0.0,
            }

    def _record_usage(self, response: Any, elapsed: float) -> None:
        prompt_toks = 0
        completion_toks = 0
        total_toks = 0
        usage_meta = getattr(response, "usage_metadata", None)
        if isinstance(usage_meta, dict):
            prompt_toks = int(usage_meta.get("input_tokens") or usage_meta.get("prompt_tokens") or 0)
            completion_toks = int(
                usage_meta.get("output_tokens") or usage_meta.get("completion_tokens") or 0
            )
            total_toks = int(usage_meta.get("total_tokens") or (prompt_toks + completion_toks))
        else:
            resp_meta = getattr(response, "response_metadata", None)
            if isinstance(resp_meta, dict):
                tok_usage = resp_meta.get("token_usage") or resp_meta.get("usage") or {}
                if isinstance(tok_usage, dict):
                    prompt_toks = int(tok_usage.get("prompt_tokens") or 0)
                    completion_toks = int(tok_usage.get("completion_tokens") or 0)
                    total_toks = int(
                        tok_usage.get("total_tokens") or (prompt_toks + completion_toks)
                    )

        with self._stats_lock:
            self._usage_stats["calls"] += 1
            self._usage_stats["prompt_tokens"] += prompt_toks
            self._usage_stats["completion_tokens"] += completion_toks
            self._usage_stats["total_tokens"] += total_toks
            self._usage_stats["elapsed_seconds"] += elapsed

    def _build_chat_model(self, temperature: float) -> ChatOpenAI:
        return ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=temperature,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def invoke_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        model = self._build_chat_model(temperature=temperature)
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                t0 = time.perf_counter()
                response = model.invoke(
                    [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt),
                    ]
                )
                self._record_usage(response, time.perf_counter() - t0)
                content = response.content
                if isinstance(content, list):
                    return "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                return str(content)
            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                transient = any(
                    kw in msg
                    for kw in ("connection", "timeout", "timed out", "429", "502", "503", "504", "reset", "eof")
                )
                if attempt < 2 and transient:
                    time.sleep(2.0 * (2**attempt))
                    continue
                raise
        raise last_exc  # type: ignore[misc]

    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        temperature: float = 0.1,
    ) -> T:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
        structured_system = (
            f"{system_prompt}\n\n"
            "You MUST respond with a single valid JSON object only (no extra prose, "
            "no markdown explanation) strictly conforming to this JSON Schema:\n"
            f"{schema_json}"
        )

        raw_output = self.invoke_text(
            system_prompt=structured_system,
            user_prompt=user_prompt,
            temperature=temperature,
        )
        payload = _extract_json_payload(raw_output)
        try:
            return schema.model_validate_json(payload)
        except (ValidationError, ValueError) as first_error:
            repair_prompt = (
                f"Your previous response could not be parsed/validated against the JSON schema.\n"
                f"Validation error:\n{first_error}\n\n"
                f"Previous output:\n{raw_output}\n\n"
                "Please fix the JSON and output ONLY a valid JSON object matching the schema."
            )
            repaired_output = self.invoke_text(
                system_prompt=structured_system,
                user_prompt=repair_prompt,
                temperature=0.0,
            )
            repaired_payload = _extract_json_payload(repaired_output)
            return schema.model_validate_json(repaired_payload)


class MockLLMClient:
    """Deterministic offline LLM client for unit tests and --mock dry runs."""

    def __init__(self, trigger_reflection_on_first_round: bool = True) -> None:
        self.trigger_reflection_on_first_round = trigger_reflection_on_first_round
        self._stats_lock = threading.Lock()
        self._usage_stats: dict[str, Any] = {
            "calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "elapsed_seconds": 0.0,
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Return thread-safe cumulative mock LLM usage statistics."""
        with self._stats_lock:
            return {
                "calls": int(self._usage_stats["calls"]),
                "prompt_tokens": int(self._usage_stats["prompt_tokens"]),
                "completion_tokens": int(self._usage_stats["completion_tokens"]),
                "total_tokens": int(self._usage_stats["total_tokens"]),
                "elapsed_seconds": round(float(self._usage_stats["elapsed_seconds"]), 3),
            }

    def reset_usage_stats(self) -> None:
        """Reset cumulative mock LLM usage counters."""
        with self._stats_lock:
            self._usage_stats = {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "elapsed_seconds": 0.0,
            }

    def _record_mock_usage(self, prompt: str, output: str) -> None:
        pt = max(1, len(prompt) // 4)
        ct = max(1, len(output) // 4)
        with self._stats_lock:
            self._usage_stats["calls"] += 1
            self._usage_stats["prompt_tokens"] += pt
            self._usage_stats["completion_tokens"] += ct
            self._usage_stats["total_tokens"] += pt + ct
            self._usage_stats["elapsed_seconds"] += 0.001

    def invoke_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
    ) -> str:
        if "citation revision editor" in system_prompt.lower():
            valid_match = re.search(r"Valid Paper IDs:\s*(.+)", user_prompt)
            valid_pids = (
                re.findall(r"\b(P\d{2,})\b", valid_match.group(1))
                if valid_match
                else []
            )
            if not valid_pids:
                valid_pids = ["P01"]
            valid_set = set(valid_pids)
            body_match = re.search(
                r"Current Survey Body:\s*\n([\s\S]*?)(?:\n+Flagged Issues:|\Z)",
                user_prompt,
            )
            body_text = body_match.group(1).strip() if body_match else user_prompt
            # Strip invalid bracketed citations and replace unsupported claims cleanly.
            def _clean_cite_group(match: re.Match[str]) -> str:
                ids = [pid for pid in re.findall(r"\b(P\d{2,})\b", match.group(1)) if pid in valid_set]
                return f"[{', '.join(ids)}]" if ids else f"[{valid_pids[0]}]"

            revised = re.sub(r"\[((?:P\d{2,})(?:\s*,\s*P\d{2,})*)\]", _clean_cite_group, body_text)
            revised = re.sub(
                r"[^.!?\n]*\bphantom\b[^.!?\n]*[.!?]?",
                f"Evidence from [{valid_pids[0]}] grounds the synthesis in indexed literature.",
                revised,
                flags=re.IGNORECASE,
            )
            self._record_mock_usage(user_prompt, revised)
            return revised.strip()

        return self._finalize_mock_text(user_prompt)

    def _finalize_mock_text(self, user_prompt: str) -> str:
        topic_match = re.search(r"Research Topic:\s*(.+)", user_prompt)
        topic = topic_match.group(1).strip() if topic_match else "Academic Literature Survey"
        paper_ids = list(dict.fromkeys(re.findall(r"\b(P\d{2,})\b", user_prompt))) or ["P01", "P02"]
        sq_titles = re.findall(r"\[(SQ\d+)\]\s*(.+)", user_prompt) or [
            ("SQ1", "Problem Formulation and Core Architectures"),
            ("SQ2", "Retrieval, Grounding, and Verification Mechanisms"),
            ("SQ3", "Evaluation Benchmarks and Open Limitations"),
        ]
        cite_a = f"[{paper_ids[0]}]"
        cite_b = f"[{paper_ids[1]}]" if len(paper_ids) > 1 else cite_a
        cite_c = f"[{paper_ids[2]}]" if len(paper_ids) > 2 else cite_b
        table_rows = [
            f"| [{pid}] | Modular retrieval-augmented pipeline #{idx} | "
            f"Academic QA & multi-hop benchmarks | Improved factual grounding over baselines | "
            f"Higher inference latency and sensitivity to noisy chunks |"
            for idx, pid in enumerate(paper_ids[:8], start=1)
        ]
        comparison_table = "\n".join(
            [
                "| Paper | Core Method | Dataset / Setting | Key Findings | Limitations |",
                "| :--- | :--- | :--- | :--- | :--- |",
                *table_rows,
            ]
        )
        sq_sections = [
            f"### {sq_id}: {sq_title}\n"
            f"Recent studies investigate {sq_title.lower()} within the context of {topic} "
            f"{cite_a} {cite_b}. Specifically, {cite_a} introduces structured multi-step decomposition "
            f"and evidence alignment, while {cite_b} emphasizes hybrid retrieval across semantic and "
            f"lexical signals to mitigate unsupported claims {cite_c}."
            for sq_id, sq_title in sq_titles
        ]
        out = (
            f"# Literature Survey: {topic}\n\n"
            "## 1. Background and Scope\n"
            f"Research on **{topic}** has expanded rapidly as autonomous workflows combine "
            f"stateful orchestration with grounded retrieval {cite_a} {cite_b}. Rather than relying on "
            f"single-pass generation, modern systems coordinate planning, multi-channel evidence "
            f"retrieval, and structured synthesis {cite_c}.\n\n"
            "## 2. Thematic Synthesis by Sub-Question\n"
            + "\n\n".join(sq_sections)
            + "\n\n## 3. Cross-Literature Comparison Matrix\n"
            + comparison_table
            + "\n\n## 4. Research Gaps and Future Directions\n"
            f"Across the reviewed works {cite_a} {cite_b} {cite_c}, three open challenges remain: "
            "(1) automated detection of evidence gaps during iterative retrieval, "
            "(2) strict claim-level citation verification against source passages, and "
            "(3) standardized multi-aspect evaluation beyond surface fluency."
        )
        self._record_mock_usage(user_prompt, out)
        return out

    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[T],
        temperature: float = 0.1,
    ) -> T:
        if schema is ResearchPlan:
            self._record_mock_usage(user_prompt, "ResearchPlan")
            topic_match = re.search(r"Topic:\s*(.+)", user_prompt)
            topic = topic_match.group(1).strip() if topic_match else user_prompt.strip()
            short_topic = " ".join(topic.split()[:6]) or "agentic retrieval augmented generation"
            plan = ResearchPlan(
                topic=topic,
                refined_scope=(
                    f"Systematic academic survey of {topic}, covering architectures, "
                    "hybrid retrieval & grounding, and evaluation methodologies."
                ),
                sub_questions=[
                    SubQuestion(
                        id="SQ1",
                        title="Task Decomposition and Agentic Orchestration",
                        description=f"How are research workflows decomposed and orchestrated for {topic}?",
                        search_queries=[
                            f"{short_topic} agent workflow",
                            "multi-step research agent state graph planning",
                        ],
                    ),
                    SubQuestion(
                        id="SQ2",
                        title="Hybrid Retrieval and Hallucination Mitigation",
                        description="How do dense and sparse retrieval methods preserve citation provenance?",
                        search_queries=[
                            "hybrid retrieval dense BM25 citation grounding",
                            "retrieval augmented generation attribution verification",
                        ],
                    ),
                    SubQuestion(
                        id="SQ3",
                        title="Evaluation Benchmarks and Iterative Reflection",
                        description="What evaluation metrics and self-reflection loops measure survey quality?",
                        search_queries=[
                            "self-reflective RAG evaluation benchmark",
                            "automated literature review evaluation faithfulness",
                        ],
                    ),
                ],
            )
            return schema.model_validate(plan.model_dump())

        if schema is ScreeningResult:
            self._record_mock_usage(user_prompt, "ScreeningResult")
            paper_ids = list(dict.fromkeys(re.findall(r"\b(P\d{2,})\b", user_prompt)))
            scores = [
                ScreeningScore(
                    paper_id=pid,
                    relevance_score=round(max(5.0, 9.6 - idx * 0.35), 2),
                    relevance_reason=f"Directly addresses core methodology and empirical evaluation ({pid}).",
                )
                for idx, pid in enumerate(paper_ids)
            ]
            return schema.model_validate(ScreeningResult(scores=scores).model_dump())

        if schema is EvidenceExtractionResult:
            self._record_mock_usage(user_prompt, "EvidenceExtractionResult")
            sq_match = re.search(r"Sub-Question ID:\s*(SQ\d+)", user_prompt)
            sq_id = sq_match.group(1) if sq_match else "SQ1"
            chunk_ids = list(dict.fromkeys(re.findall(r"\b(P\d{2,}-C\d{2,})\b", user_prompt)))

            chunks_by_paper: dict[str, list[str]] = {}
            for cid in chunk_ids:
                pid = cid.split("-")[0]
                chunks_by_paper.setdefault(pid, []).append(cid)

            items = [
                EvidenceItem(
                    paper_id=pid,
                    sub_question_id=sq_id,
                    method=f"Grounded retrieval and structured reasoning architecture in {pid}",
                    dataset_or_setting="English academic benchmark corpora and multi-hop QA tasks",
                    key_findings=(
                        f"Demonstrates improved factual consistency and traceable evidence attribution "
                        f"for {sq_id}."
                    ),
                    limitations="Requires careful chunk sizing and bounded iterative retrieval steps.",
                    source_chunk_ids=cids[:2],
                )
                for pid, cids in list(chunks_by_paper.items())[:5]
            ]
            return schema.model_validate(EvidenceExtractionResult(items=items).model_dump())

        if schema is EvaluationResult:
            self._record_mock_usage(user_prompt, "EvaluationResult")
            iter_match = re.search(r"Current Iteration:\s*(\d+)", user_prompt)
            iteration = int(iter_match.group(1)) if iter_match else 0

            sq_stats = re.findall(
                r"\[(SQ\d+)\][^\n]*?evidence=(\d+),\s*unique_papers=(\d+)",
                user_prompt,
            )
            if not sq_stats:
                sq_ids = list(dict.fromkeys(re.findall(r"\b(SQ\d+)\b", user_prompt))) or ["SQ1", "SQ2"]
                sq_stats = [(sq_id, "3", "2") for sq_id in sq_ids]

            should_reflect = iteration == 0 and self.trigger_reflection_on_first_round
            weak_target = "SQ2" if any(s[0] == "SQ2" for s in sq_stats) else sq_stats[0][0]

            evals: list[SubQuestionEvaluation] = []
            weak_ids: list[str] = []
            for sq_id, ev_cnt, uniq_cnt in sq_stats:
                is_weak = should_reflect and sq_id == weak_target
                if is_weak:
                    weak_ids.append(sq_id)
                evals.append(
                    SubQuestionEvaluation(
                        sub_question_id=sq_id,
                        evidence_count=int(ev_cnt),
                        unique_paper_count=int(uniq_cnt),
                        coverage_score=5.2 if is_weak else 8.7,
                        sufficient=not is_weak,
                        gaps=(
                            ["Need additional empirical benchmarks on hybrid retrieval and citation attribution."]
                            if is_weak
                            else []
                        ),
                    )
                )

            result = EvaluationResult(
                iteration=iteration,
                overall_sufficient=not bool(weak_ids),
                needs_more_retrieval=bool(weak_ids),
                weak_subquestion_ids=weak_ids,
                subquestion_evaluations=evals,
                stop_reason="insufficient_coverage" if weak_ids else "evidence_sufficient",
                reasoning=(
                    f"Iteration {iteration}: {weak_ids[0]} requires supplemental retrieval for stronger comparison."
                    if weak_ids
                    else f"Iteration {iteration}: All sub-questions have sufficient multi-paper coverage."
                ),
            )
            return schema.model_validate(result.model_dump())

        if schema is QueryRewriteResult:
            self._record_mock_usage(user_prompt, "QueryRewriteResult")
            iter_match = re.search(r"Current Iteration:\s*(\d+)", user_prompt)
            iteration = int(iter_match.group(1)) if iter_match else 0
            weak_line = re.search(r"Weak Sub-Questions:\s*(.+)", user_prompt)
            if weak_line:
                weak_ids = re.findall(r"\b(SQ\d+)\b", weak_line.group(1))
            else:
                weak_ids = list(dict.fromkeys(re.findall(r"\b(SQ\d+)\b", user_prompt)))[:2]
            if not weak_ids:
                weak_ids = ["SQ2"]

            rewritten = [
                RewrittenQuery(
                    sub_question_id=sq_id,
                    query=f"hybrid BM25 dense retrieval citation attribution {sq_id.lower()} benchmark",
                    rationale=f"Broaden empirical benchmark coverage and provenance studies for {sq_id}.",
                )
                for sq_id in weak_ids
            ]
            result = QueryRewriteResult(
                iteration=iteration + 1,
                rewritten_queries=rewritten,
                new_sub_question=None,
                reasoning="Generated targeted non-duplicate English queries for weak sub-questions.",
            )
            return schema.model_validate(result.model_dump())

        if schema is ClaimVerificationBatch:
            self._record_mock_usage(user_prompt, "ClaimVerificationBatch")
            blocks = re.split(r"(?=Claim\s+\d+:)", user_prompt)
            items: list[ClaimVerificationItem] = []
            for block in blocks:
                claim_match = re.search(r"Claim\s+\d+:\s*(.+)", block)
                if not claim_match:
                    continue
                claim_text = claim_match.group(1).strip()
                cited_pids = list(dict.fromkeys(re.findall(r"\b(P\d{2,})\b", block)))
                chunk_ids = list(dict.fromkeys(re.findall(r"\b(P\d{2,}-C\d{2,})\b", block)))
                items.append(
                    ClaimVerificationItem(
                        claim=claim_text,
                        cited_paper_ids=cited_pids,
                        matched_chunk_ids=chunk_ids[:2],
                        verdict="supported",
                        reason="Claim aligns with retrieved source chunks.",
                    )
                )
            return schema.model_validate(ClaimVerificationBatch(items=items).model_dump())

        raise ValueError(f"MockLLMClient does not support schema: {schema.__name__}")


def create_llm_client(settings: Settings, mock: bool = False) -> LLMClientProtocol:
    """Factory creating either MockLLMClient or OpenAILLMClient from settings."""
    if mock:
        return MockLLMClient()
    return OpenAILLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
    )
