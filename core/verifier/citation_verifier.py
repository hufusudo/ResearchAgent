"""Hard and soft citation provenance auditing for generated literature surveys."""

from __future__ import annotations

import re

from core.graph.prompts import CITATION_VERIFIER_SYSTEM_PROMPT
from core.llm.client import LLMClientProtocol
from core.schemas import (
    CitationVerificationReport,
    ClaimVerificationBatch,
    ClaimVerificationItem,
    Paper,
    TextChunk,
)

_BRACKET_CITE_GROUP_RE = re.compile(r"\[((?:P\d{2,})(?:\s*,\s*P\d{2,})*)\]")
_SINGLE_PAPER_ID_RE = re.compile(r"\b(P\d{2,})\b")
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "by",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "using",
    "based",
    "across",
    "papers",
    "paper",
    "study",
    "studies",
}


def _extract_survey_body(draft_markdown: str) -> str:
    body = re.split(
        r"\n##\s+(?:References|Citation Verification Summary)\b",
        draft_markdown,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return body.strip()


def _extract_citations_from_text(text: str) -> list[str]:
    citations: list[str] = []
    for match in _BRACKET_CITE_GROUP_RE.finditer(text):
        citations.extend(_SINGLE_PAPER_ID_RE.findall(match.group(1)))
    return citations


def _tokenize_content(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+", text.lower())
        if len(tok) >= 3 and tok not in _STOPWORDS
    }


def _extract_cited_claims(body: str, valid_paper_ids: set[str]) -> list[tuple[str, list[str]]]:
    claims: list[tuple[str, list[str]]] = []
    seen_claims: set[str] = set()

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or re.match(r"^\|[\s:|-]+\|$", line):
            continue
        if line.startswith("|") and line.endswith("|"):
            # Treat comparison table rows cleanly as single claim strings.
            candidates = [line.strip("| ").replace("|", " ; ")]
        else:
            candidates = re.split(r"(?<=[.!?])\s+", line)

        for sent in candidates:
            cleaned = " ".join(sent.split())
            if len(cleaned) < 20:
                continue
            cited = [pid for pid in dict.fromkeys(_extract_citations_from_text(cleaned)) if pid in valid_paper_ids]
            if not cited:
                continue
            if cleaned not in seen_claims:
                seen_claims.add(cleaned)
                claims.append((cleaned, cited))

    return claims


def _select_stratified_claims(
    claims: list[tuple[str, list[str]]],
    max_claims: int,
) -> list[tuple[str, list[str]]]:
    """Select evenly spaced claims across the full survey so all sections are audited."""
    if max_claims <= 0 or len(claims) <= max_claims:
        return claims[: max(0, max_claims)] if max_claims >= 0 else claims
    if max_claims == 1:
        return [claims[0]]

    n = len(claims)
    selected_indices: list[int] = []
    seen_indices: set[int] = set()
    for i in range(max_claims):
        idx = round(i * (n - 1) / (max_claims - 1))
        if idx not in seen_indices:
            seen_indices.add(idx)
            selected_indices.append(idx)
    return [claims[idx] for idx in selected_indices]


def _rank_supporting_chunks(
    claim_text: str,
    cited_pids: list[str],
    chunks_by_paper: dict[str, list[TextChunk]],
    limit: int = 2,
    max_total_chunks: int = 8,
) -> list[TextChunk]:
    claim_tokens = _tokenize_content(claim_text)

    ranked_by_paper: dict[str, list[tuple[int, int, TextChunk]]] = {}
    for pid in cited_pids:
        paper_chunks = chunks_by_paper.get(pid, [])
        if not paper_chunks:
            continue
        scored = [
            (len(claim_tokens & _tokenize_content(chunk.text)), len(chunk.text), chunk)
            for chunk in paper_chunks
        ]
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        ranked_by_paper[pid] = scored

    if not ranked_by_paper:
        return []

    if len(cited_pids) == 1:
        only_pid = cited_pids[0]
        return [item[2] for item in ranked_by_paper.get(only_pid, [])[: max(limit, 2)]]

    selected: list[TextChunk] = []
    seen_ids: set[str] = set()

    # First pass: include the best-matching chunk for every cited paper.
    for pid in cited_pids:
        if len(selected) >= max_total_chunks:
            break
        scored = ranked_by_paper.get(pid)
        if not scored:
            continue
        chunk = scored[0][2]
        if chunk.chunk_id not in seen_ids:
            seen_ids.add(chunk.chunk_id)
            selected.append(chunk)

    # Second pass: for 2-paper claims, include a second positive-overlap chunk if budget permits.
    if len(cited_pids) <= 2:
        budget = min(max_total_chunks, len(cited_pids) * limit)
        for pid in cited_pids:
            if len(selected) >= budget:
                break
            scored = ranked_by_paper.get(pid, [])
            for overlap, _, chunk in scored[1:]:
                if len(selected) >= budget:
                    break
                if overlap > 0 and chunk.chunk_id not in seen_ids:
                    seen_ids.add(chunk.chunk_id)
                    selected.append(chunk)

    return selected


def _heuristic_verify_claims(
    claim_contexts: list[tuple[str, list[str], list[TextChunk]]],
) -> list[ClaimVerificationItem]:
    items: list[ClaimVerificationItem] = []
    for claim_text, cited_pids, matched_chunks in claim_contexts:
        if not matched_chunks:
            items.append(
                ClaimVerificationItem(
                    claim=claim_text,
                    cited_paper_ids=cited_pids,
                    matched_chunk_ids=[],
                    verdict="unsupported",
                    reason="No chunks found for cited paper IDs.",
                )
            )
            continue

        claim_tokens = _tokenize_content(claim_text)
        best_overlap = max(
            len(claim_tokens & _tokenize_content(f"{c.title} {c.text}"))
            for c in matched_chunks
        )
        if best_overlap >= 2 or not claim_tokens:
            verdict = "supported"
            reason = f"Matched {best_overlap} content terms in source chunk(s)."
        elif best_overlap == 1:
            verdict = "partially_supported"
            reason = "Limited lexical overlap with cited source chunk(s)."
        else:
            verdict = "unsupported"
            reason = "Zero content token overlap with cited source chunk(s)."

        items.append(
            ClaimVerificationItem(
                claim=claim_text,
                cited_paper_ids=cited_pids,
                matched_chunk_ids=[c.chunk_id for c in matched_chunks],
                verdict=verdict,  # type: ignore[arg-type]
                reason=reason,
            )
        )
    return items


def verify_citations(
    draft_markdown: str,
    selected_papers: list[Paper],
    chunks: list[TextChunk],
    llm: LLMClientProtocol | None = None,
    max_claims: int = 16,
) -> CitationVerificationReport:
    """Audit bracketed citations and semantic claim grounding against indexed chunks."""
    body = _extract_survey_body(draft_markdown)
    valid_ids = [p.paper_id for p in selected_papers if p.paper_id]
    valid_id_set = set(valid_ids)

    all_cited = _extract_citations_from_text(body)
    cited_paper_ids = [pid for pid in dict.fromkeys(all_cited) if pid in valid_id_set]
    invalid_citations = list(dict.fromkeys(pid for pid in all_cited if pid not in valid_id_set))
    uncited_paper_ids = [pid for pid in valid_ids if pid not in set(cited_paper_ids)]
    valid_occurrences = sum(1 for pid in all_cited if pid in valid_id_set)
    reference_validity_rate = round(valid_occurrences / len(all_cited), 4) if all_cited else 1.0

    chunks_by_paper: dict[str, list[TextChunk]] = {}
    for chunk in chunks:
        chunks_by_paper.setdefault(chunk.paper_id, []).append(chunk)

    extracted_claims = _extract_cited_claims(body, valid_id_set)
    total_cited_claims = len(extracted_claims)
    sampled_claims = _select_stratified_claims(extracted_claims, max_claims)

    claim_contexts: list[tuple[str, list[str], list[TextChunk]]] = [
        (
            claim_text,
            cited_pids,
            _rank_supporting_chunks(
                claim_text,
                cited_pids,
                chunks_by_paper,
                limit=2,
                max_total_chunks=8,
            ),
        )
        for claim_text, cited_pids in sampled_claims
    ]

    verified_items: list[ClaimVerificationItem] = []
    if claim_contexts:
        if llm is not None:
            prompt_blocks: list[str] = []
            allowed_chunks_per_idx: list[set[str]] = []
            for idx, (claim_text, cited_pids, matched_chunks) in enumerate(claim_contexts, start=1):
                allowed_ids = {c.chunk_id for c in matched_chunks}
                allowed_chunks_per_idx.append(allowed_ids)
                excerpts = "\n".join(
                    f"  [{c.chunk_id}] ({c.title}): {c.text[:1800]}"
                    for c in matched_chunks
                ) or "  (No source excerpts found)"
                prompt_blocks.append(
                    f"Claim {idx}: {claim_text}\n"
                    f"Cited Papers: {', '.join(f'[{pid}]' for pid in cited_pids)}\n"
                    f"Candidate Chunks:\n{excerpts}"
                )
            try:
                batch = llm.invoke_structured(
                    system_prompt=CITATION_VERIFIER_SYSTEM_PROMPT,
                    user_prompt="\n\n".join(prompt_blocks),
                    schema=ClaimVerificationBatch,
                )
                if len(batch.items) == len(claim_contexts):
                    for idx, item in enumerate(batch.items):
                        claim_text, cited_pids, matched_chunks = claim_contexts[idx]
                        allowed_ids = allowed_chunks_per_idx[idx]
                        clean_cids = [cid for cid in item.matched_chunk_ids if cid in allowed_ids]
                        if not clean_cids and matched_chunks:
                            clean_cids = [matched_chunks[0].chunk_id]
                        verified_items.append(
                            item.model_copy(
                                update={
                                    "claim": claim_text,
                                    "cited_paper_ids": cited_pids,
                                    "matched_chunk_ids": clean_cids,
                                }
                            )
                        )
                else:
                    verified_items = _heuristic_verify_claims(claim_contexts)
            except Exception:
                verified_items = _heuristic_verify_claims(claim_contexts)
        else:
            verified_items = _heuristic_verify_claims(claim_contexts)

    checked_claims = len(verified_items)
    supported_count = sum(1 for item in verified_items if item.verdict == "supported")
    partially_supported_count = sum(
        1 for item in verified_items if item.verdict == "partially_supported"
    )
    unsupported_count = sum(1 for item in verified_items if item.verdict == "unsupported")
    claim_support_rate = (
        round((supported_count + 0.5 * partially_supported_count) / checked_claims, 4)
        if checked_claims > 0
        else 1.0
    )

    return CitationVerificationReport(
        total_cited_claims=total_cited_claims,
        checked_claims=checked_claims,
        supported_count=supported_count,
        partially_supported_count=partially_supported_count,
        unsupported_count=unsupported_count,
        invalid_citations=invalid_citations,
        cited_paper_ids=cited_paper_ids,
        uncited_paper_ids=uncited_paper_ids,
        reference_validity_rate=reference_validity_rate,
        claim_support_rate=claim_support_rate,
        details=verified_items,
    )


def format_verification_summary_markdown(report: CitationVerificationReport) -> str:
    """Format a concise Markdown summary block appended after References."""
    invalid_str = ", ".join(f"[{c}]" for c in report.invalid_citations) or "None"
    uncited_str = ", ".join(f"[{p}]" for p in report.uncited_paper_ids) or "None"
    return "\n".join(
        [
            "## Citation Verification Summary",
            "",
            f"- **Reference Validity Rate**: {report.reference_validity_rate * 100:.1f}% (Invalid citations: {invalid_str})",
            f"- **Cited Papers**: {len(report.cited_paper_ids)}/{len(report.cited_paper_ids) + len(report.uncited_paper_ids)} (Uncited: {uncited_str})",
            (
                f"- **Claim Support Rate**: {report.claim_support_rate * 100:.1f}% "
                f"({report.supported_count} supported, {report.partially_supported_count} partial, "
                f"{report.unsupported_count} unsupported across {report.checked_claims} checked / "
                f"{report.total_cited_claims} cited claims)"
            ),
        ]
    )
