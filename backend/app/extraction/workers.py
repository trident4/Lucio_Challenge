"""Phase 1b: Multi-core document extraction with focused chunking.

All functions are top-level (picklable) for ProcessPoolExecutor.
PDF: PyMuPDF layout mode, ~2000-char chunks with 200-char overlap.
DOCX: python-docx, ~2000-char chunks with 200-char overlap.

Chunks are small and focused (~half a page) so that:
- Embeddings capture specific content, not diffuse 5-page averages
- BM25 matches are precise (fewer irrelevant terms per chunk)
- Full chunk text can be sent to the LLM without truncation
"""

import io
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor

import docx
import fitz  # PyMuPDF

logger = logging.getLogger("lucio.workers")

# Chunking parameters — small, focused chunks for precise retrieval
CHUNK_SIZE = 4000  # Increased from 2000 to better capture full financial tables

# Document type classification patterns
SCOTUS_PATTERN = re.compile(r"\d+\s+U\.S\.\s+\d+")


def _classify_document(filename: str) -> str:
    """Classify document type from filename using regex patterns.

    Reliable code-based classification instead of asking the LLM to
    pattern-match filenames in the prompt.
    """
    if SCOTUS_PATTERN.search(filename):
        return "SCOTUS case"
    if " v. " in filename or " v " in filename:
        return "Legal case"
    if filename.lower().endswith(".docx"):
        return "Agreement/Contract"
    return "Document"


CHUNK_OVERLAP = 200  # chars overlap between chunks
HEADER_CHARS = 400  # chars from page 1 for identifying the document


def run_extraction(
    file_tuples: list[tuple[str, bytes]],
) -> tuple[list[dict], list[dict]]:
    """Run extraction across all CPU cores via ProcessPoolExecutor.

    Args:
        file_tuples: List of (filename, raw_bytes) from unzip_to_tuples.

    Returns:
        (all_chunks, all_metadata) where each worker produces chunks + metadata.
    """
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(_extract_document_wrapper, file_tuples))

    all_chunks: list[dict] = []
    all_metadata: list[dict] = []
    for chunks, meta in results:
        all_chunks.extend(chunks)
        all_metadata.append(meta)

    logger.info(
        f"Extracted {len(all_chunks)} chunks from {len(all_metadata)} documents"
    )
    return all_chunks, all_metadata


def _extract_document_wrapper(args: tuple[str, bytes]) -> tuple[list[dict], dict]:
    """Wrapper to unpack tuple for pool.map (which passes single arg)."""
    filename, file_bytes = args
    return extract_document(filename, file_bytes)


def extract_document(filename: str, file_bytes: bytes) -> tuple[list[dict], dict]:
    """Dispatch extraction based on file extension.

    Returns:
        (chunks, metadata) tuple.
    """
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return _extract_pdf(filename, file_bytes)
    elif ext == ".docx":
        return _extract_docx(filename, file_bytes)
    else:
        return [], {"filename": filename, "title": "", "page_count": 0}


# ── PDF Extraction ──────────────────────────────────────────────────────────


def _extract_pdf(filename: str, file_bytes: bytes) -> tuple[list[dict], dict]:
    """Extract text from a PDF using PyMuPDF layout mode.

    Extracts each page's text, then splits into ~2000-char chunks
    with 200-char overlap. Tracks which page(s) each chunk spans.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_count = len(doc)

    # Extract all page texts using layout mode (preserves table columns)
    page_texts: list[str] = []
    for page in doc:
        page_texts.append(page.get_text("layout"))

    # Header: first 400 chars of page 1, sanitized
    header = _sanitize_header(page_texts[0][:HEADER_CHARS] if page_texts else "")

    # Title: first non-empty line
    title = _extract_title(page_texts[0] if page_texts else "")

    # Build page boundary map: (start_char_offset, page_number)
    # This lets us track which page(s) each chunk falls on.
    page_boundaries: list[tuple[int, int]] = []
    offset = 0
    for i, pt in enumerate(page_texts):
        page_boundaries.append((offset, i + 1))  # 1-indexed page numbers
        offset += len(pt) + 1  # +1 for the newline joiner

    # Join all pages into one continuous string
    full_text = "\n".join(page_texts)

    # Split into focused chunks with overlap
    chunks = _split_into_chunks(full_text, filename, header, page_boundaries)

    doc.close()

    metadata = {
        "filename": filename,
        "title": title,
        "type": _classify_document(filename),
        "page_count": page_count,
    }
    return chunks, metadata


# ── DOCX Extraction ─────────────────────────────────────────────────────────


def _extract_docx(filename: str, file_bytes: bytes) -> tuple[list[dict], dict]:
    """Extract text from a DOCX using python-docx.

    Joins paragraphs into a single string, then splits into
    ~2000-char chunks with 200-char overlap.
    """
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]

    # Header: first 400 chars
    raw_header = " ".join(paragraphs)[:HEADER_CHARS]
    header = _sanitize_header(raw_header)

    # Title: first non-empty paragraph
    title = paragraphs[0].strip() if paragraphs else ""

    # Join all paragraphs
    full_text = "\n".join(paragraphs)

    # DOCX has no reliable page numbers, so no page boundaries
    chunks = _split_into_chunks(full_text, filename, header, page_boundaries=None)

    metadata = {
        "filename": filename,
        "title": title,
        "type": _classify_document(filename),
        "page_count": 0,
    }
    return chunks, metadata


# ── Core Chunking Logic ────────────────────────────────────────────────────


def _split_into_chunks(
    full_text: str,
    filename: str,
    header: str,
    page_boundaries: list[tuple[int, int]] | None,
) -> list[dict]:
    """Split text into small, focused chunks with overlap.

    Tries to break at paragraph boundaries (double newline) rather
    than mid-sentence for cleaner chunks.

    Args:
        full_text: The complete document text.
        filename: Source filename.
        header: Document header for identification.
        page_boundaries: List of (char_offset, page_num) for PDFs.
                         None for DOCX.

    Returns:
        List of chunk dicts.
    """
    if not full_text.strip():
        return []

    chunks = []
    text_len = len(full_text)
    start = 0
    chunk_idx = 0

    while start < text_len:
        # Determine chunk end
        end = min(start + CHUNK_SIZE, text_len)

        # Try to break at a paragraph boundary (double newline)
        if end < text_len:
            break_point = full_text.rfind("\n\n", start + CHUNK_SIZE // 2, end)
            if break_point > start:
                end = break_point

        content = full_text[start:end].strip()
        if content:
            # Determine which pages this chunk spans
            page_nums = _get_page_nums(start, end, page_boundaries)

            chunks.append(_make_chunk(filename, header, content, page_nums, chunk_idx))
            chunk_idx += 1

        # Advance with overlap
        start = end - CHUNK_OVERLAP if end < text_len else text_len

    return chunks


def _get_page_nums(
    start: int, end: int, page_boundaries: list[tuple[int, int]] | None
) -> list[int]:
    """Determine which pages a chunk spans based on character offsets."""
    if page_boundaries is None:
        return [0]

    pages = set()
    for i, (offset, page_num) in enumerate(page_boundaries):
        # Check if this page overlaps with our chunk [start, end)
        next_offset = (
            page_boundaries[i + 1][0] if i + 1 < len(page_boundaries) else float("inf")
        )
        if offset < end and next_offset > start:
            pages.add(page_num)

    return sorted(pages) if pages else [0]


# ── Helpers ─────────────────────────────────────────────────────────────────


def _sanitize_header(raw: str) -> str:
    """Remove excessive whitespace from header text."""
    return re.sub(r"\n{2,}", "\n", raw).strip()


def _extract_title(page1_text: str) -> str:
    """Extract the first non-empty line as the document title."""
    for line in page1_text.split("\n"):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _make_chunk(
    filename: str,
    header: str,
    content: str,
    page_nums: list[int],
    index: int,
) -> dict:
    """Build a chunk dict with enriched text and raw content.

    - text:    enriched with [HEADER]/[SOURCE]/[PAGES] — for BM25 indexing and LLM context
    - content: raw document text only — for embedding (no metadata noise)
    """
    chunk_id = f"{filename}::chunk_{index}"
    text = (
        f"[HEADER: {header}] "
        f"[SOURCE: {filename}] "
        f"[PAGES: {page_nums}]\n\n"
        f"{content}"
    )
    return {
        "chunk_id": chunk_id,
        "filename": filename,
        "page_nums": page_nums,
        "text": text,
        "content": content,
    }
