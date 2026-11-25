"""Evaluation metrics, external reference verification, and blind LLM-as-a-Judge rubrics."""

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from typing import Any

from pydantic import BaseModel, Field

from core.graph.workflow import MockArxivClient
from core.llm.client import LLMClientProtocol
from core.retrieval.pdf_parser import chunk_abstract_only
from core.schemas import (
    CitationVerificationReport,
    Paper,
    ResearchPlan,
    RetrievedChunk,
    TextChunk,
)
from core.verifier.citation_verifier import (
    _extract_citations_from_text,
    _extract_survey_body,
    verify_citations,
)


class ExternalReferenceCheckItem(BaseModel):
    """Verification outcome for a single reference listed by a baseline (e.g., Direct LLM)."""

    ref_id: str
    raw_citation: str
    extracted_title: str
    matched: bool = False
    matched_source: str = ""
    matched_title: str = ""
    similarity: float = 0.0


class ExternalVerificationSummary(BaseModel):
    """Combined external reference existence + claim grounding audit for Direct LLM."""

    total_listed_references: int = 0
    verified_references: int = 0
    hallucinated_references: int = 0
    external_reference_validity_rate: float = 0.0
    inline_reference_validity_rate: float = 0.0
    effective_reference_validity_rate: float = 0.0
    claim_support_rate: float = 0.0
    effective_claim_support_rate: float = 0.0
    checked_claims: int = 0
    hallucinated_only_claims: int = 0
    supported_count: int = 0
    partially_supported_count: int = 0
    unsupported_count: int = 0
    details: list[ExternalReferenceCheckItem] = Field(default_factory=list)


class SubQuestionCoverageJudgeItem(BaseModel):
    """LLM-as-a-Judge coverage rating for one canonical sub-question."""

    sub_question_id: str
    coverage_score: float = Field(..., ge=0.0, le=10.0)
    adequately_covered: bool
    distinct_cited_papers: int = Field(default=0, ge=0)
    rationale: str = ""


class DraftCoverageBatch(BaseModel):
    """Structured batch output for canonical sub-question coverage evaluation."""

    items: list[SubQuestionCoverageJudgeItem] = Field(default_factory=list)


class SubQuestionCoverageReport(BaseModel):
    """Aggregated sub-question coverage score across a generated survey."""

    total_subquestions: int = 0
    covered_count: int = 0
    subquestion_coverage_rate: float = 0.0
    avg_coverage_score: float = 0.0
    details: list[SubQuestionCoverageJudgeItem] = Field(default_factory=list)


class SynthesisRubricScore(BaseModel):
    """Blind 1-5 multi-aspect rubric rating for survey synthesis quality."""

    comparative_depth: float = Field(..., ge=1.0, le=5.0)
    non_sequential_synthesis: float = Field(..., ge=1.0, le=5.0)
    factual_rigor: float = Field(..., ge=1.0, le=5.0)
    structural_completeness: float = Field(..., ge=1.0, le=5.0)
    overall_score: float = Field(..., ge=1.0, le=5.0)
    rationale: str = ""


class ChunkRelevanceItem(BaseModel):
    """Relevance grade (0=irrelevant, 1=partially relevant, 2=highly relevant) for a chunk."""

    chunk_id: str
    grade: int = Field(..., ge=0, le=2)


class ChunkRelevanceBatch(BaseModel):
    """Pooled chunk relevance judgments for retrieval ablation."""

    items: list[ChunkRelevanceItem] = Field(default_factory=list)


def _normalize_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _title_similarity(a: str, b: str) -> float:
    na = _normalize_title(a)
    nb = _normalize_title(b)
    if not na or not nb:
        return 0.0
    seq_ratio = SequenceMatcher(None, na, nb).ratio()
    toks_a = {t for t in na.split() if len(t) >= 2}
    toks_b = {t for t in nb.split() if len(t) >= 2}
    jaccard = len(toks_a & toks_b) / len(toks_a | toks_b) if (toks_a and toks_b) else 0.0
    return round(max(seq_ratio, jaccard), 4)


def normalize_numeric_citations_to_pxx(draft_markdown: str) -> str:
    """Normalize [1], [1, 2], or numbered references to [P01] format if [Pxx] is absent."""
    if re.search(r"\[P\d{2,}\]", draft_markdown):
        return draft_markdown

    def _replace_num_group(match: re.Match[str]) -> str:
        nums = [int(n) for n in re.findall(r"\d+", match.group(1))]
        if not nums or any(n <= 0 or n > 99 for n in nums):
            return match.group(0)
        return "[" + ", ".join(f"P{n:02d}" for n in nums) + "]"

    parts = re.split(r"(?=\n##\s+References\b)", draft_markdown, maxsplit=1, flags=re.IGNORECASE)
    body = re.sub(r"\[(\d+(?:\s*,\s*\d+)*)\]", _replace_num_group, parts[0])
    if len(parts) == 1:
        return body

    ref_lines: list[str] = []
    for line in parts[1].splitlines():
        m_br = re.match(r"^\s*(?:[-*]\s*)?\[(\d{1,2})\]\s*(.+)$", line)
        m_dot = re.match(r"^\s*(\d{1,2})\.\s+(.+)$", line)
        if m_br:
            num = int(m_br.group(1))
            ref_lines.append(f"- **[P{num:02d}]** {m_br.group(2).strip()}")
        elif m_dot:
            num = int(m_dot.group(1))
            ref_lines.append(f"- **[P{num:02d}]** {m_dot.group(2).strip()}")
        else:
            ref_lines.append(line)
    return body + "\n" + "\n".join(ref_lines)


def _extract_candidate_title(raw_citation: str) -> str:
    cleaned = re.sub(r"\*{1,2}", "", raw_citation).strip()
    cleaned = re.sub(r"^\[?P?\d{1,3}\]?\s*[:.\-]?\s*", "", cleaned).strip()
    cleaned = re.sub(r"https?://\S+", "", cleaned).strip()
    quoted = re.search(r'["“]([^"”]{10,220})["”]', cleaned)
    if quoted:
        return quoted.group(1).strip(" .,:;")

    # Handle both "Title (2024). Authors..." and "Authors (2024). Title. Venue"
    year_split = re.split(r"\s*\(\d{4}[a-z]?\)\.?\s*", cleaned, maxsplit=1)
    if len(year_split) == 2:
        before_year = year_split[0].strip(" .,:;")
        after_year = year_split[1].strip()
        after_first = re.split(r"\.\s+(?=[A-Z0-9])", after_year, maxsplit=1)[0].strip(" .,:;")
        looks_like_authors = bool(
            re.search(r"\bet al\b|\b[A-Z]\.|,\s*[A-Z]\.", before_year)
            or before_year.count(",") >= 1
        )
        if len(before_year) >= 12 and not looks_like_authors and len(before_year) >= len(after_first) * 0.75:
            return before_year
        if len(after_first) >= 12:
            return after_first

    before_year = re.split(r"\s*\(\d{4}[a-z]?\)", cleaned, maxsplit=1)[0].strip()
    if len(before_year) >= 12 and "," not in before_year[:25]:
        return before_year.strip(" .,:;")

    segments = [seg.strip(" .,:;") for seg in re.split(r"\.\s+", cleaned) if len(seg.strip()) >= 12]
    if len(segments) >= 2:
        # Prefer the longest title-like segment among early segments (skipping short author lists)
        for seg in segments[:3]:
            if not re.search(r"\b(?:et al|arxiv|proceedings|conference|journal)\b", seg.lower()):
                if seg.count(",") <= 3:
                    return seg
        return segments[1]
    return segments[0] if segments else cleaned[:180]


def _parse_listed_references(draft_markdown: str) -> list[tuple[str, str, str]]:
    """Extract (ref_id, raw_citation, extracted_title) from the References section."""
    ref_split = re.split(
        r"\n##\s+References\b",
        draft_markdown,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    if len(ref_split) < 2:
        return []
    ref_block = re.split(
        r"\n##\s+",
        ref_split[1],
        maxsplit=1,
    )[0]

    parsed: list[tuple[str, str, str]] = []
    seen_ids: set[str] = set()
    auto_idx = 1
    for raw_line in ref_block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        m_pxx = re.search(r"\b(P\d{2,})\b", line)
        if m_pxx:
            ref_id = m_pxx.group(1)
        else:
            m_num = re.match(r"^(?:[-*]\s*)?(?:\[(\d{1,2})\]|(\d{1,2})\.)\s+", line)
            if m_num:
                num = int(m_num.group(1) or m_num.group(2))
                ref_id = f"P{num:02d}"
            elif line.startswith(("-", "*")):
                ref_id = f"P{auto_idx:02d}"
                auto_idx += 1
            else:
                continue

        if ref_id in seen_ids:
            continue
        seen_ids.add(ref_id)
        title = _extract_candidate_title(line)
        parsed.append((ref_id, line, title))
    return parsed


def _lookup_external_paper(
    title_query: str,
    openalex_client: Any,
    arxiv_client: Any,
    mock: bool = False,
) -> tuple[Paper | None, float]:
    if not title_query.strip():
        return None, 0.0

    if mock:
        catalog_papers = MockArxivClient().search(title_query, max_results=10)
        best_p: Paper | None = None
        best_sim = 0.0
        for p in catalog_papers:
            sim = _title_similarity(title_query, p.title)
            if sim > best_sim:
                best_sim = sim
                best_p = p
        return best_p, best_sim

    candidates: list[Paper] = []
    if openalex_client is not None:
        try:
            candidates.extend(openalex_client.search(query=title_query, max_results=5))
        except Exception:
            pass

    best_paper: Paper | None = None
    best_sim = 0.0
    for p in candidates:
        sim = _title_similarity(title_query, p.title)
        if sim > best_sim:
            best_sim = sim
            best_paper = p

    if best_sim < 0.78 and arxiv_client is not None:
        try:
            arx_hits = arxiv_client.search(query=f'ti:"{title_query[:120]}"', max_results=5)
            if not arx_hits:
                arx_hits = arxiv_client.search(query=title_query[:120], max_results=5)
            for p in arx_hits:
                sim = _title_similarity(title_query, p.title)
                if sim > best_sim:
                    best_sim = sim
                    best_paper = p
        except Exception:
            pass

    return best_paper, best_sim


def verify_external_references(
    draft_markdown: str,
    openalex_client: Any,
    arxiv_client: Any,
    llm: LLMClientProtocol | None,
    *,
    mock: bool = False,
    similarity_threshold: float = 0.78,
) -> ExternalVerificationSummary:
    """Verify whether Direct LLM's cited references actually exist in real academic catalogs."""
    normalized_draft = normalize_numeric_citations_to_pxx(draft_markdown)
    listed = _parse_listed_references(normalized_draft)

    def _check_one(item: tuple[str, str, str]) -> tuple[ExternalReferenceCheckItem, Paper | None]:
        ref_id, raw_line, extracted_title = item
        matched_paper, sim = _lookup_external_paper(
            extracted_title,
            openalex_client=openalex_client,
            arxiv_client=arxiv_client,
            mock=mock,
        )
        is_match = matched_paper is not None and sim >= similarity_threshold
        check_item = ExternalReferenceCheckItem(
            ref_id=ref_id,
            raw_citation=raw_line,
            extracted_title=extracted_title,
            matched=is_match,
            matched_source=matched_paper.source if (is_match and matched_paper) else "",
            matched_title=matched_paper.title if (is_match and matched_paper) else "",
            similarity=round(sim, 4),
        )
        if is_match and matched_paper is not None:
            return check_item, matched_paper.model_copy(
                update={"paper_id": ref_id, "ingest_status": "abstract_only"}
            )
        return check_item, None

    results: list[tuple[ExternalReferenceCheckItem, Paper | None]] = []
    if listed:
        if len(listed) == 1 or mock:
            results = [_check_one(it) for it in listed]
        else:
            with ThreadPoolExecutor(max_workers=min(4, len(listed))) as executor:
                results = list(executor.map(_check_one, listed))

    details = [r[0] for r in results]
    verified_papers = [r[1] for r in results if r[1] is not None]
    verified_chunks: list[TextChunk] = []
    for vp in verified_papers:
        verified_chunks.extend(chunk_abstract_only(vp))

    total_listed = len(listed)
    verified_count = len(verified_papers)
    hallucinated_count = total_listed - verified_count
    external_validity = round(verified_count / total_listed, 4) if total_listed > 0 else 0.0

    ver_report: CitationVerificationReport = verify_citations(
        draft_markdown=normalized_draft,
        selected_papers=verified_papers,
        chunks=verified_chunks,
        llm=llm,
        max_claims=16,
    )

    body = _extract_survey_body(normalized_draft)
    all_inline_cites = _extract_citations_from_text(body)
    valid_id_set = {p.paper_id for p in verified_papers}
    if all_inline_cites:
        inline_validity = ver_report.reference_validity_rate
    else:
        inline_validity = 0.0 if total_listed == 0 else external_validity

    # Count sentences/rows in body that cite ONLY hallucinated/unverified IDs.
    hallucinated_only_claims = 0
    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or re.match(r"^\|[\s:|-]+\|$", line):
            continue
        candidates = (
            [line.strip("| ").replace("|", " ; ")]
            if (line.startswith("|") and line.endswith("|"))
            else re.split(r"(?<=[.!?])\s+", line)
        )
        for sent in candidates:
            cleaned = " ".join(sent.split())
            if len(cleaned) < 20:
                continue
            pids = _extract_citations_from_text(cleaned)
            if pids and not any(pid in valid_id_set for pid in pids):
                hallucinated_only_claims += 1

    denom = ver_report.checked_claims + hallucinated_only_claims
    if denom > 0:
        effective_support = round(
            (ver_report.supported_count + 0.5 * ver_report.partially_supported_count) / denom,
            4,
        )
    else:
        effective_support = 0.0

    # Combine inline citation validity and listed-reference verification faithfully.
    effective_validity = (
        round(min(external_validity, inline_validity), 4)
        if all_inline_cites
        else external_validity
    )

    return ExternalVerificationSummary(
        total_listed_references=total_listed,
        verified_references=verified_count,
        hallucinated_references=hallucinated_count,
        external_reference_validity_rate=external_validity,
        inline_reference_validity_rate=round(inline_validity, 4),
        effective_reference_validity_rate=effective_validity,
        claim_support_rate=ver_report.claim_support_rate,
        effective_claim_support_rate=effective_support,
        checked_claims=ver_report.checked_claims,
        hallucinated_only_claims=hallucinated_only_claims,
        supported_count=ver_report.supported_count,
        partially_supported_count=ver_report.partially_supported_count,
        unsupported_count=ver_report.unsupported_count,
        details=details,
    )


def _strip_for_blind_judge(draft_markdown: str) -> str:
    body = _extract_survey_body(draft_markdown)
    # Remove any top-level system-identifying notes if present, keep survey content intact.
    return body.strip()


def _truncate_preserving_structure(body: str, max_chars: int = 48000) -> str:
    """Keep head and tail sections (including comparison matrix & gaps) intact if oversized."""
    if len(body) <= max_chars:
        return body
    head_len = int(max_chars * 0.65)
    tail_len = max_chars - head_len
    return f"{body[:head_len]}\n\n[...middle condensed...]\n\n{body[-tail_len:]}"


def evaluate_subquestion_coverage(
    draft_markdown: str,
    canonical_plan: ResearchPlan,
    llm: LLMClientProtocol,
    *,
    mock: bool = False,
) -> SubQuestionCoverageReport:
    """Evaluate how thoroughly a survey draft covers each canonical sub-question."""
    body = _strip_for_blind_judge(draft_markdown)
    sqs = canonical_plan.sub_questions
    if not sqs:
        return SubQuestionCoverageReport()

    if mock:
        items: list[SubQuestionCoverageJudgeItem] = []
        body_lower = body.lower()
        for sq in sqs:
            has_heading = sq.id.lower() in body_lower or any(
                tok in body_lower for tok in _normalize_title(sq.title).split() if len(tok) > 4
            )
            cites = len(set(_extract_citations_from_text(body)))
            score = 8.4 if (has_heading and cites >= 3) else (6.8 if has_heading else 4.5)
            items.append(
                SubQuestionCoverageJudgeItem(
                    sub_question_id=sq.id,
                    coverage_score=score,
                    adequately_covered=score >= 6.5,
                    distinct_cited_papers=cites,
                    rationale="Mock deterministic coverage check.",
                )
            )
        covered = sum(1 for it in items if it.adequately_covered)
        return SubQuestionCoverageReport(
            total_subquestions=len(items),
            covered_count=covered,
            subquestion_coverage_rate=round(covered / len(items), 4),
            avg_coverage_score=round(sum(it.coverage_score for it in items) / len(items), 4),
            details=items,
        )

    sq_list_text = "\n".join(
        f"- [{sq.id}] {sq.title}: {sq.description}" for sq in sqs
    )
    system_prompt = (
        "You are an objective academic survey evaluator.\n"
        "Given a literature survey draft (without references) and a list of canonical research sub-questions:\n"
        "For EACH sub-question, score how thoroughly and specifically the draft addresses it (0.0 to 10.0), "
        "whether it is adequately covered (`adequately_covered`, true if score >= 6.5 with concrete multi-paper findings), "
        "and how many distinct inline citations (like [P01], [P02]) appear in its discussion."
    )
    user_prompt = (
        f"Research Topic: {canonical_plan.topic}\n\n"
        f"Canonical Sub-Questions:\n{sq_list_text}\n\n"
        f"Survey Draft Body:\n{_truncate_preserving_structure(body, 48000)}"
    )
    batch = llm.invoke_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=DraftCoverageBatch,
        temperature=0.0,
    )
    by_id = {it.sub_question_id: it for it in batch.items}
    items = []
    for sq in sqs:
        it = by_id.get(sq.id)
        if it is None:
            it = SubQuestionCoverageJudgeItem(
                sub_question_id=sq.id,
                coverage_score=5.0,
                adequately_covered=False,
                distinct_cited_papers=0,
                rationale="Sub-question omitted from judge response.",
            )
        items.append(it)
    covered = sum(1 for it in items if it.adequately_covered and it.coverage_score >= 6.5)
    return SubQuestionCoverageReport(
        total_subquestions=len(items),
        covered_count=covered,
        subquestion_coverage_rate=round(covered / len(items), 4),
        avg_coverage_score=round(sum(it.coverage_score for it in items) / len(items), 4),
        details=items,
    )


def evaluate_synthesis_rubric(
    draft_markdown: str,
    topic: str,
    llm: LLMClientProtocol,
    *,
    mock: bool = False,
) -> SynthesisRubricScore:
    """Blind LLM-as-a-Judge 1-5 rubric scoring across 4 synthesis quality dimensions."""
    body = _strip_for_blind_judge(draft_markdown)
    if mock:
        has_table = "| paper |" in body.lower() or "| method" in body.lower() or "\n|---" in body.replace(" ", "").replace(":", "")
        co_cites = len(re.findall(r"\[P\d{2,}\][,; ]+\[P\d{2,}\]|\[P\d{2,}\s*,\s*P\d{2,}\]", body))
        has_sections = all(
            sec in body.lower()
            for sec in ("background", "synthesis", "comparison", "gap")
        )
        comp = 4.4 if has_table else 2.8
        non_seq = 4.3 if (has_table and co_cites >= 2) else 3.0
        rigor = 4.2 if len(re.findall(r"\[P\d{2,}\]", body)) >= 5 else 2.9
        struct = 4.5 if has_sections else 3.4
        overall = round((comp + non_seq + rigor + struct) / 4.0, 4)
        return SynthesisRubricScore(
            comparative_depth=comp,
            non_sequential_synthesis=non_seq,
            factual_rigor=rigor,
            structural_completeness=struct,
            overall_score=overall,
            rationale="Mock deterministic synthesis rubric evaluation.",
        )

    system_prompt = (
        "You are a senior CS/AI peer reviewer performing a blind evaluation of an academic literature survey draft.\n"
        "Score the survey from 1.0 to 5.0 (decimals like 3.5 or 4.2 allowed) on four independent criteria:\n"
        "1. `comparative_depth`: Does it explicitly compare methods, datasets, metrics, and trade-offs across papers (e.g., structured comparison matrix, contrasting results)?\n"
        "2. `non_sequential_synthesis`: Does it organize insights thematically across papers (good=4-5) rather than summarizing papers one-by-one sequentially (poor=1-2)?\n"
        "3. `factual_rigor`: Are claims concrete, appropriately scoped, properly attributed with inline citations, and free of vague hype or unsupported superlatives?\n"
        "4. `structural_completeness`: Does it cleanly cover scope, thematic synthesis, comparative matrix, and concrete open challenges/future directions?\n"
        "Set `overall_score` to the exact arithmetic mean of the four scores."
    )
    user_prompt = (
        f"Research Topic: {topic}\n\n"
        f"Blind Survey Draft Body:\n{_truncate_preserving_structure(body, 48000)}"
    )
    score = llm.invoke_structured(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        schema=SynthesisRubricScore,
        temperature=0.0,
    )
    mean_score = round(
        (
            score.comparative_depth
            + score.non_sequential_synthesis
            + score.factual_rigor
            + score.structural_completeness
        )
        / 4.0,
        4,
    )
    return score.model_copy(update={"overall_score": mean_score})


def compute_retrieval_ablation_metrics(
    query: str,
    ranked_by_mode: dict[str, list[RetrievedChunk]],
    relevance_grades: dict[str, int],
    *,
    top_k: int = 8,
) -> dict[str, dict[str, float]]:
    """Compute Precision@K, Recall@K, nDCG@K, MRR, diversity, and term-hit metrics per mode."""
    query_terms = {
        tok
        for tok in re.findall(r"[a-z0-9]+", query.lower())
        if len(tok) >= 4
    }
    total_relevant = sum(1 for g in relevance_grades.values() if g >= 1)
    ideal_grades = sorted((g for g in relevance_grades.values() if g > 0), reverse=True)[:top_k]
    idcg = sum((2**g - 1) / math.log2(idx + 2) for idx, g in enumerate(ideal_grades))

    metrics_by_mode: dict[str, dict[str, float]] = {}
    for mode_name, hits in ranked_by_mode.items():
        top_hits = hits[:top_k]
        rel_hits = 0
        dcg = 0.0
        mrr = 0.0
        paper_counts: dict[str, int] = {}
        matched_terms: set[str] = set()

        for idx, item in enumerate(top_hits):
            cid = item.chunk.chunk_id
            grade = int(relevance_grades.get(cid, 0))
            if grade >= 1:
                rel_hits += 1
                if mrr == 0.0:
                    mrr = 1.0 / (idx + 1)
            if grade > 0:
                dcg += (2**grade - 1) / math.log2(idx + 2)

            pid = item.chunk.paper_id
            paper_counts[pid] = paper_counts.get(pid, 0) + 1

            chunk_text_lower = f"{item.chunk.title} {item.chunk.section} {item.chunk.text}".lower()
            for term in query_terms:
                if term in chunk_text_lower:
                    matched_terms.add(term)

        k_denom = max(1, len(top_hits))
        precision = rel_hits / k_denom
        recall = (rel_hits / total_relevant) if total_relevant > 0 else 0.0
        ndcg = (dcg / idcg) if idcg > 0 else 0.0
        unique_papers = float(len(paper_counts))
        max_share = (max(paper_counts.values()) / k_denom) if paper_counts else 0.0
        term_hit_rate = (len(matched_terms) / len(query_terms)) if query_terms else 1.0

        metrics_by_mode[mode_name] = {
            "precision_at_k": round(precision, 4),
            "recall_at_k": round(recall, 4),
            "ndcg_at_k": round(ndcg, 4),
            "mrr": round(mrr, 4),
            "unique_papers_at_k": round(unique_papers, 4),
            "max_single_paper_share": round(max_share, 4),
            "term_hit_rate": round(term_hit_rate, 4),
        }
    return metrics_by_mode
