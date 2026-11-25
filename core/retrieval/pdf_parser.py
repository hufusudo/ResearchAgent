"""PDF parsing, section detection, and provenance-preserving text chunking."""

from __future__ import annotations

import re
from pathlib import Path

import pymupdf

from core.schemas import Paper, TextChunk

_KNOWN_SECTIONS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "preliminaries",
    "method",
    "methods",
    "methodology",
    "approach",
    "framework",
    "architecture",
    "system design",
    "implementation",
    "experiments",
    "experimental setup",
    "evaluation",
    "results",
    "analysis",
    "discussion",
    "limitations",
    "conclusion",
    "conclusions",
    "future work",
    "acknowledgments",
    "acknowledgements",
    "references",
    "appendix",
}

_NUMBERED_HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)|(?:[IVX]+))\.?\s+([A-Za-z][A-Za-z0-9\s,&:/-]{2,70})$"
)


def detect_section_heading(line: str) -> str | None:
    """Detect whether a single line looks like an academic paper section heading."""
    cleaned = " ".join(line.strip().split())
    if not cleaned or len(cleaned) > 90 or cleaned.endswith("."):
        return None

    lowered = cleaned.lower().rstrip(":")
    if lowered in _KNOWN_SECTIONS:
        return cleaned.rstrip(":")

    match = _NUMBERED_HEADING_RE.match(cleaned)
    if match:
        title_part = match.group(1).strip().rstrip(":")
        words = title_part.split()
        if title_part.lower() in _KNOWN_SECTIONS or (1 <= len(words) <= 6 and title_part[0].isupper()):
            return cleaned.rstrip(":")

    return None


def _split_long_block(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    parts: list[str] = []
    start = 0
    step_floor = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            split_at = max(
                text.rfind("\n\n", start + step_floor // 2, end),
                text.rfind(". ", start + step_floor // 2, end),
                text.rfind(" ", start + step_floor // 2, end),
            )
            if split_at > start:
                end = split_at + 1
        segment = text[start:end].strip()
        if segment:
            parts.append(segment)
        if end >= len(text):
            break
        next_start = max(start + 1, end - chunk_overlap)
        start = next_start
    return parts


_REFERENCE_HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*)|(?:[IVX]+))?\.?\s*(?:references(?:\s+cited)?|bibliography|works\s+cited)$",
    re.IGNORECASE,
)


def _is_reference_heading(heading: str) -> bool:
    return bool(_REFERENCE_HEADING_RE.match(heading.strip()))


def chunk_text_blocks(
    paper: Paper,
    pages: list[tuple[int, str]],
    chunk_size: int = 1600,
    chunk_overlap: int = 250,
    skip_references: bool = True,
    max_chunks: int | None = 120,
) -> list[TextChunk]:
    """Split page-tagged text into overlapping chunks while preserving section and page."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size")

    segments: list[tuple[int, str, str]] = []
    current_section = "Body"
    stop_at_references = False

    for page_num, raw_text in pages:
        if stop_at_references:
            break
        if not raw_text or not raw_text.strip():
            continue
        lines = raw_text.splitlines()
        buffer: list[str] = []

        for line in lines:
            heading = detect_section_heading(line)
            if heading:
                if buffer:
                    block_text = " ".join(buffer).strip()
                    if block_text:
                        segments.append((page_num, current_section, block_text))
                    buffer = []
                if skip_references and page_num >= 2 and segments and _is_reference_heading(heading):
                    stop_at_references = True
                    break
                current_section = heading
            else:
                stripped = line.strip()
                if stripped:
                    buffer.append(stripped)
                elif buffer:
                    block_text = " ".join(buffer).strip()
                    if block_text:
                        segments.append((page_num, current_section, block_text))
                    buffer = []

        if buffer and not stop_at_references:
            block_text = " ".join(buffer).strip()
            if block_text:
                segments.append((page_num, current_section, block_text))

    if not segments:
        return []

    chunks: list[TextChunk] = []
    current_text = ""
    current_page = segments[0][0]
    current_sec = segments[0][1]

    def flush_chunk(text_to_flush: str, page_val: int, sec_val: str) -> None:
        if max_chunks is not None and len(chunks) >= max_chunks:
            return
        cleaned = text_to_flush.strip()
        if not cleaned:
            return
        idx = len(chunks) + 1
        pid = paper.paper_id or "P00"
        chunks.append(
            TextChunk(
                chunk_id=f"{pid}-C{idx:02d}",
                paper_id=pid,
                title=paper.title,
                year=paper.year,
                url=paper.url,
                section=sec_val,
                page=page_val,
                text=cleaned,
            )
        )

    for page_num, section, block in segments:
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        if current_text and section != current_sec:
            flush_chunk(current_text, current_page, current_sec)
            current_text = ""
            if max_chunks is not None and len(chunks) >= max_chunks:
                break

        if len(block) > chunk_size:
            if current_text:
                flush_chunk(current_text, current_page, current_sec)
                current_text = ""
            for piece in _split_long_block(block, chunk_size, chunk_overlap):
                flush_chunk(piece, page_num, section)
                if max_chunks is not None and len(chunks) >= max_chunks:
                    break
            current_page = page_num
            current_sec = section
            continue

        candidate = f"{current_text}\n\n{block}".strip() if current_text else block
        if len(candidate) <= chunk_size:
            if not current_text:
                current_page = page_num
                current_sec = section
            current_text = candidate
        else:
            flush_chunk(current_text, current_page, current_sec)
            overlap_tail = current_text[-chunk_overlap:].strip() if chunk_overlap > 0 else ""
            if overlap_tail and len(overlap_tail) + len(block) + 2 <= chunk_size:
                current_text = f"{overlap_tail}\n\n{block}".strip()
            else:
                current_text = block
            current_page = page_num
            current_sec = section

    if current_text and (max_chunks is None or len(chunks) < max_chunks):
        flush_chunk(current_text, current_page, current_sec)

    return chunks


def parse_pdf_to_chunks(
    paper: Paper,
    pdf_path: Path,
    chunk_size: int = 1600,
    chunk_overlap: int = 250,
    skip_references: bool = True,
    max_chunks_per_paper: int = 120,
) -> list[TextChunk]:
    """Extract text from a PDF using PyMuPDF and split into TextChunk objects.

    Returns an empty list on corrupt or unreadable PDFs so callers can gracefully
    degrade to abstract-only ingestion.
    """
    if not pdf_path.exists() or not pdf_path.is_file():
        return []

    try:
        pages: list[tuple[int, str]] = []
        with pymupdf.open(pdf_path) as doc:
            for idx, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    pages.append((idx, text))
        return chunk_text_blocks(
            paper=paper,
            pages=pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            skip_references=skip_references,
            max_chunks=max_chunks_per_paper,
        )
    except Exception:
        return []


def chunk_abstract_only(paper: Paper) -> list[TextChunk]:
    """Create a fallback single chunk from title and abstract when full text is unavailable."""
    pid = paper.paper_id or "P00"
    body = "\n\n".join(part for part in [paper.title.strip(), paper.abstract.strip()] if part)
    if not body:
        body = paper.title.strip() or "No text available."
    return [
        TextChunk(
            chunk_id=f"{pid}-C01",
            paper_id=pid,
            title=paper.title,
            year=paper.year,
            url=paper.url,
            section="Abstract",
            page=1,
            text=body,
        )
    ]


def rebind_chunk_paper_id(chunks: list[TextChunk], paper_id: str) -> list[TextChunk]:
    """Update chunk paper_id and sequential chunk_id after paper numbering."""
    rebound: list[TextChunk] = []
    for idx, chunk in enumerate(chunks, start=1):
        rebound.append(
            chunk.model_copy(
                update={
                    "paper_id": paper_id,
                    "chunk_id": f"{paper_id}-C{idx:02d}",
                }
            )
        )
    return rebound


def load_local_pdfs(
    pdf_dir: Path,
    chunk_size: int = 1600,
    chunk_overlap: int = 250,
) -> list[tuple[Paper, list[TextChunk]]]:
    """Load and parse all PDFs in a local directory as supplementary papers."""
    if not pdf_dir.exists() or not pdf_dir.is_dir():
        return []

    results: list[tuple[Paper, list[TextChunk]]] = []
    for idx, pdf_path in enumerate(sorted(pdf_dir.glob("*.pdf")), start=1):
        title = pdf_path.stem.replace("_", " ").replace("-", " ").strip()
        authors: list[str] = []
        abstract_snippet = ""
        year = 0

        try:
            with pymupdf.open(pdf_path) as doc:
                meta = doc.metadata or {}
                if meta.get("title") and len(str(meta["title"]).strip()) > 3:
                    title = str(meta["title"]).strip()
                if meta.get("author"):
                    authors = [a.strip() for a in str(meta["author"]).split(",") if a.strip()]
                creation_date = str(meta.get("creationDate", ""))
                year_match = re.search(r"(\d{4})", creation_date)
                if year_match:
                    year = int(year_match.group(1))
                if len(doc) > 0:
                    first_page = " ".join((doc[0].get_text("text") or "").split())
                    abstract_snippet = first_page[:800]
        except Exception:
            continue

        paper = Paper(
            paper_id=f"L{idx:02d}",
            external_id=f"local:{pdf_path.name}",
            source="local_pdf",
            title=title,
            authors=authors,
            year=year,
            abstract=abstract_snippet or f"Local PDF document: {pdf_path.name}",
            url=pdf_path.resolve().as_uri(),
            pdf_url=str(pdf_path.resolve()),
            full_text_available=True,
            ingest_status="full_text",
        )
        chunks = parse_pdf_to_chunks(
            paper=paper,
            pdf_path=pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if not chunks:
            paper.ingest_status = "abstract_only"
            chunks = chunk_abstract_only(paper)

        results.append((paper, chunks))

    return results
