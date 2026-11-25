"""Unit tests for arXiv Atom XML parsing and paper deduplication."""

from __future__ import annotations

from core.retrieval.arxiv_client import deduplicate_and_number_papers, parse_arxiv_atom
from core.schemas import Paper

SAMPLE_ATOM_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <published>2024-01-22T18:00:00Z</published>
    <title>
      Agentic   Retrieval-Augmented
      Generation Survey
    </title>
    <summary>
      We investigate   cyclic orchestration and
      hybrid retrieval for literature synthesis.
    </summary>
    <author><name>Alice Smith</name></author>
    <author><name>Bob Jones</name></author>
    <link rel="alternate" type="text/html" href="https://arxiv.org/abs/2401.12345v2"/>
    <link title="pdf" rel="related" type="application/pdf" href="https://arxiv.org/pdf/2401.12345v2"/>
    <category term="cs.CL"/>
    <category term="cs.AI"/>
  </entry>
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id</id>
    <title>Error</title>
    <summary>Malformed query</summary>
  </entry>
</feed>
"""


def test_parse_arxiv_atom_normalizes_whitespace_and_skips_errors() -> None:
    papers = parse_arxiv_atom(SAMPLE_ATOM_XML)
    assert len(papers) == 1

    paper = papers[0]
    assert paper.external_id == "2401.12345"
    assert paper.source == "arxiv"
    assert paper.title == "Agentic Retrieval-Augmented Generation Survey"
    assert paper.abstract == (
        "We investigate cyclic orchestration and hybrid retrieval for literature synthesis."
    )
    assert paper.year == 2024
    assert paper.authors == ["Alice Smith", "Bob Jones"]
    assert paper.url == "https://arxiv.org/abs/2401.12345v2"
    assert paper.pdf_url == "https://arxiv.org/pdf/2401.12345v2"
    assert paper.categories == ["cs.CL", "cs.AI"]
    assert paper.full_text_available is True


def test_deduplicate_and_number_papers() -> None:
    p1 = Paper(
        external_id="2401.00001",
        source="arxiv",
        title="Hybrid RAG: A Survey",
        year=2024,
        abstract="First copy.",
        url="https://arxiv.org/abs/2401.00001",
    )
    p1_dup_id = Paper(
        external_id="2401.00001",
        source="arxiv",
        title="Hybrid RAG: A Survey (Revised)",
        year=2024,
        abstract="Duplicate by ID.",
        url="https://arxiv.org/abs/2401.00001",
    )
    p1_dup_title = Paper(
        external_id="2402.99999",
        source="arxiv",
        title="hybrid rag  a survey!",
        year=2024,
        abstract="Duplicate by normalized title.",
        url="https://arxiv.org/abs/2402.99999",
    )
    p2 = Paper(
        external_id="2403.00002",
        source="arxiv",
        title="Self-Reflective Agents",
        year=2025,
        abstract="Second unique paper.",
        url="https://arxiv.org/abs/2403.00002",
    )

    result = deduplicate_and_number_papers([p1, p1_dup_id, p1_dup_title, p2])
    assert len(result) == 2
    assert [p.paper_id for p in result] == ["P01", "P02"]
    assert result[0].title == "Hybrid RAG: A Survey"
    assert result[1].title == "Self-Reflective Agents"
