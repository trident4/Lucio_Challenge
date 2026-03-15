"""Phase 1b: Multi-core document extraction with focused chunking.

All functions are top-level (picklable) for ProcessPoolExecutor.
PDF: PyMuPDF layout mode, ~2000-char chunks with 200-char overlap.
DOCX: python-docx, ~2000-char chunks with 200-char overlap.
XLSX: openpyxl, table-aware markdown chunks with header repetition.

Chunks are small and focused (~half a page) so that:
- Embeddings capture specific content, not diffuse 5-page averages
- BM25 matches are precise (fewer irrelevant terms per chunk)
- Full chunk text can be sent to the LLM without truncation
"""

import datetime
import io
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor

import docx
import fitz  # PyMuPDF
import openpyxl

logger = logging.getLogger("lucio.workers")

# Chunking parameters — small, focused chunks for precise retrieval
CHUNK_SIZE = 2000  # chars per chunk (~500 tokens, ~half a page)

# Document type classification patterns
SCOTUS_PATTERN = re.compile(r"\d+\s+U\.\s*S\.\s+\d+")
EARNINGS_PATTERN = re.compile(r"earnings|transcript", re.IGNORECASE)


def _classify_document(filename: str) -> str:
    """Classify document type from filename using regex patterns.

    Reliable code-based classification instead of asking the LLM to
    pattern-match filenames in the prompt.
    """
    if SCOTUS_PATTERN.search(filename):
        return "SCOTUS case"
    if EARNINGS_PATTERN.search(filename):
        return "Earnings transcript"
    if " v. " in filename or " v " in filename:
        return "Legal case"
    if filename.lower().endswith(".xlsx"):
        return "Financial data"
    if filename.lower().endswith(".docx"):
        return "Agreement/Contract"
    return "Document"


CHUNK_OVERLAP = 200  # chars overlap between chunks
HEADER_CHARS = 400  # chars from page 1 for identifying the document


def run_extraction(
    file_tuples: list[tuple[str, bytes]],
    pool: ProcessPoolExecutor | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run extraction across all CPU cores via ProcessPoolExecutor.

    Args:
        file_tuples: List of (filename, raw_bytes) from unzip_to_tuples.
        pool: Optional pre-created pool to reuse (avoids 2-3s macOS spawn).

    Returns:
        (all_chunks, all_metadata) where each worker produces chunks + metadata.
    """
    _pool = pool or ProcessPoolExecutor(max_workers=os.cpu_count())
    try:
        results = list(_pool.map(_extract_document_wrapper, file_tuples))
    finally:
        if not pool:
            _pool.shutdown(wait=False)

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
    elif ext == ".xlsx":
        return _extract_xlsx(filename, file_bytes)
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


# ── XLSX Extraction ─────────────────────────────────────────────────────────


def _extract_xlsx(filename: str, file_bytes: bytes) -> tuple[list[dict], dict]:
    """Extract tabular data from an Excel workbook as markdown table chunks.

    Each sheet is converted to a markdown table and split into chunks
    that respect row boundaries. The header row is repeated in every chunk.
    """
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)

    all_chunks: list[dict] = []
    sheet_count = 0
    chunk_idx = 0

    for sheet_num, ws in enumerate(wb.worksheets, start=1):
        rows = list(ws.iter_rows(values_only=True))
        trimmed, col_start, col_end = _trim_empty(rows)
        if len(trimmed) < 2:  # need header + at least one data row
            continue

        sheet_count += 1
        header_row = trimmed[0]
        data_rows = trimmed[1:]

        md_header = _make_md_header(header_row, col_start, col_end)
        header_tag = f"Excel: {filename} | Sheet: {ws.title}"

        chunks = _split_table_into_chunks(
            data_rows, md_header, header_tag, filename, col_start, col_end,
            sheet_num, chunk_idx,
        )
        chunk_idx += len(chunks)
        all_chunks.extend(chunks)

    wb.close()

    metadata = {
        "filename": filename,
        "title": filename,
        "type": _classify_document(filename),
        "page_count": sheet_count,
    }
    return all_chunks, metadata


def _trim_empty(rows: list[tuple]) -> tuple[list[tuple], int, int]:
    """Remove fully-empty rows and find column bounds.

    Returns:
        (non_empty_rows, col_start, col_end) where col_start/col_end
        are the min/max column indices with data (inclusive).
    """
    non_empty = [r for r in rows if any(c is not None for c in r)]
    if not non_empty:
        return [], 0, 0

    col_start = min(
        i for r in non_empty for i, c in enumerate(r) if c is not None
    )
    col_end = max(
        i for r in non_empty for i, c in enumerate(r) if c is not None
    )
    return non_empty, col_start, col_end


def _format_cell(value) -> str:
    """Format a cell value for markdown table output."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return str(value)
    if isinstance(value, datetime.datetime):
        if value.hour == 0 and value.minute == 0 and value.second == 0:
            return value.strftime("%Y-%m-%d")
        return str(value)
    if isinstance(value, datetime.date):
        return value.strftime("%Y-%m-%d")
    s = str(value)
    return s.replace("|", "\\|")


def _make_md_header(header_row: tuple, col_start: int, col_end: int) -> str:
    """Create markdown table header and separator lines."""
    cells = [_format_cell(header_row[i]) for i in range(col_start, col_end + 1)]
    header_line = "| " + " | ".join(cells) + " |"
    sep_line = "| " + " | ".join("---" for _ in cells) + " |"
    return header_line + "\n" + sep_line


def _format_row(row: tuple, col_start: int, col_end: int) -> str:
    """Format one data row as a markdown table row."""
    cells = [_format_cell(row[i]) if i < len(row) else "" for i in range(col_start, col_end + 1)]
    return "| " + " | ".join(cells) + " |"


def _split_table_into_chunks(
    data_rows: list[tuple],
    md_header: str,
    header_tag: str,
    filename: str,
    col_start: int,
    col_end: int,
    sheet_num: int,
    start_chunk_idx: int,
) -> list[dict]:
    """Split table data rows into chunks, repeating the markdown header in each.

    No overlap — repeating financial data rows would confuse the LLM
    with duplicate numbers.
    """
    header_overhead = len(md_header) + 1  # +1 for newline after header
    budget = CHUNK_SIZE - header_overhead

    chunks: list[dict] = []
    current_rows: list[str] = []
    current_size = 0
    chunk_idx = start_chunk_idx

    for row in data_rows:
        formatted = _format_row(row, col_start, col_end)
        row_size = len(formatted) + 1  # +1 for newline

        # If adding this row exceeds budget and we have rows, flush the chunk
        if current_rows and current_size + row_size > budget:
            content = md_header + "\n" + "\n".join(current_rows)
            chunks.append(_make_chunk(filename, header_tag, content, [sheet_num], chunk_idx))
            chunk_idx += 1
            current_rows = []
            current_size = 0

        # Always add the row (handles oversize single rows)
        current_rows.append(formatted)
        current_size += row_size

    # Flush remaining rows
    if current_rows:
        content = md_header + "\n" + "\n".join(current_rows)
        chunks.append(_make_chunk(filename, header_tag, content, [sheet_num], chunk_idx))

    return chunks


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
