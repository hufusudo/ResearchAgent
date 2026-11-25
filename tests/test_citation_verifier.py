"""Unit tests for hard and soft Citation Verifier auditing."""

from __future__ import annotations

from core.llm.client import MockLLMClient
from core.schemas import Paper, TextChunk
from core.verifier.citation_verifier import (
    format_verification_summary_markdown,
    verify_citations,
)


def _sample_papers_and_chunks() -> tuple[list[Paper], list[TextChunk]]:
    papers = [
        Paper(
            paper_id="P01",
            external_id="2401.00001",
            source="arxiv",
            title="Hybrid Retrieval with BM25 and RRF",
            year=2024,
            abstract="BM25 plus dense retrieval via RRF improves academic QA recall.",
            url="https://arxiv.org/abs/2401.00001",
        ),
        Paper(
            paper_id="P02",
            external_id="2402.00002",
            source="arxiv",
            title="Stateful Agentic Planning",
            year=2024,
            abstract="Directed state graphs support bounded reflection loops.",
            url="https://arxiv.org/abs/2402.00002",
        ),
    ]
    chunks = [
        TextChunk(
            chunk_id="P01-C01",
            paper_id="P01",
            title="Hybrid Retrieval with BM25 and RRF",
            year=2024,
            url="https://arxiv.org/abs/2401.00001",
            section="Results",
            page=4,
            text="Combining BM25 lexical matching with dense vectors via RRF improves academic QA recall.",
        ),
        TextChunk(
            chunk_id="P02-C01",
            paper_id="P02",
            title="Stateful Agentic Planning",
            year=2024,
            url="https://arxiv.org/abs/2402.00002",
            section="Method",
            page=2,
            text="Directed state graphs orchestrate bounded reflection loops for literature synthesis.",
        ),
    ]
    return papers, chunks


def test_verify_citations_detects_invalid_and_uncited_papers() -> None:
    papers, chunks = _sample_papers_and_chunks()
    draft = (
        "# Literature Survey\n\n"
        "Combining BM25 lexical matching with dense vectors via RRF improves academic QA recall [P01]. "
        "However, an unverified phantom study reports magical gains [P99].\n\n"
        "## References\n\n"
        "- **[P01]** Hybrid Retrieval with BM25 and RRF\n"
        "- **[P02]** Stateful Agentic Planning\n"
    )

    report = verify_citations(draft, papers, chunks, llm=None)
    assert report.invalid_citations == ["P99"]
    assert report.cited_paper_ids == ["P01"]
    assert report.uncited_paper_ids == ["P02"]
    assert report.reference_validity_rate == 0.5
    assert report.checked_claims == 1
    assert report.supported_count == 1
    assert report.claim_support_rate == 1.0


def test_verify_citations_with_mock_llm_and_markdown_summary() -> None:
    papers, chunks = _sample_papers_and_chunks()
    draft = (
        "# Literature Survey\n\n"
        "BM25 and dense vector RRF fusion improves academic QA recall [P01]. "
        "Directed state graphs orchestrate bounded reflection loops for synthesis [P02].\n\n"
        "## References\n\n"
        "- **[P01]** Hybrid Retrieval\n"
        "- **[P02]** Stateful Agentic Planning\n"
    )

    report = verify_citations(draft, papers, chunks, llm=MockLLMClient())
    assert report.reference_validity_rate == 1.0
    assert report.invalid_citations == []
    assert report.uncited_paper_ids == []
    assert report.checked_claims == 2
    assert report.claim_support_rate == 1.0

    summary_md = format_verification_summary_markdown(report)
    assert "## Citation Verification Summary" in summary_md
    assert "100.0%" in summary_md


def test_verify_citations_preserves_long_chunk_tail_and_multi_paper_chunks() -> None:
    tail_fact = "verification-tail-marker-999 confirms provenance across four agents."
    long_abstract = ("Background context about literature review. " * 25) + tail_fact
    assert len(long_abstract) > 950

    papers = [
        Paper(
            paper_id=f"P0{i}",
            external_id=f"2501.0000{i}",
            source="arxiv",
            title=f"Paper {i}",
            year=2025,
            abstract=long_abstract if i == 1 else f"Paper {i} abstract.",
            url=f"https://arxiv.org/abs/2501.0000{i}",
        )
        for i in range(1, 5)
    ]
    chunks = [
        TextChunk(
            chunk_id=f"P0{i}-C01",
            paper_id=f"P0{i}",
            title=f"Paper {i}",
            year=2025,
            url=f"https://arxiv.org/abs/2501.0000{i}",
            section="Abstract",
            page=1,
            text=long_abstract if i == 1 else f"Unique evidence from paper {i} on synthesis.",
        )
        for i in range(1, 5)
    ]

    captured_prompts: list[str] = []

    class _SpyLLM:
        def invoke_structured(self, system_prompt: str, user_prompt: str, schema, temperature: float = 0.0):
            captured_prompts.append(user_prompt)
            from core.schemas import ClaimVerificationBatch, ClaimVerificationItem

            return ClaimVerificationBatch(
                items=[
                    ClaimVerificationItem(
                        claim="Multi-paper synthesis claim",
                        cited_paper_ids=["P01", "P02", "P03", "P04"],
                        matched_chunk_ids=["P01-C01", "P02-C01", "P03-C01", "P04-C01"],
                        verdict="supported",
                        reason="All four chunks provided.",
                    )
                ]
            )

        def invoke_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
            return ""

    draft = (
        "# Literature Survey\n\n"
        "Multi-agent workflows synthesize literature with provenance tracking [P01], [P02], [P03], [P04].\n"
    )
    report = verify_citations(draft, papers, chunks, llm=_SpyLLM())  # type: ignore[arg-type]
    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert tail_fact in prompt
    for cid in ("P01-C01", "P02-C01", "P03-C01", "P04-C01"):
        assert cid in prompt
    assert report.claim_support_rate == 1.0


def test_verify_citations_stratified_sampling_covers_full_document() -> None:
    papers, chunks = _sample_papers_and_chunks()
    sentences = [
        f"Claim index {idx:02d} discusses grounded retrieval and synthesis [P01]."
        for idx in range(30)
    ]
    draft = "# Literature Survey\n\n" + " ".join(sentences)
    report = verify_citations(draft, papers, chunks, llm=None, max_claims=6)
    assert report.total_cited_claims == 30
    assert report.checked_claims == 6
    checked_texts = [item.claim for item in report.details]
    assert any("Claim index 00" in c for c in checked_texts)
    assert any("Claim index 29" in c for c in checked_texts)
    assert any("Claim index 12" in c or "Claim index 17" in c or "Claim index 18" in c for c in checked_texts)
