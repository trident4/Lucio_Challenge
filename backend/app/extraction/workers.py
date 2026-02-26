"""Phase 1b: Multi-core document extraction with high-fidelity chunking.

All functions are top-level (picklable) for ProcessPoolExecutor.
PDF: PyMuPDF layout mode, 5-page sliding window, 2-page overlap.
DOCX: python-docx, 25-paragraph batch, 10-paragraph overlap.
"""

import io
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor

import docx
import fitz  # PyMuPDF

logger = logging.getLogger("lucio.workers")

# Chunking parameters
PDF_WINDOW = 5  # pages per chunk
PDF_STRIDE = 3  # stride (window - overlap)
DOCX_WINDOW = 25  # paragraphs per chunk
DOCX_STRIDE = 15  # stride (window - overlap)
HEADER_CHARS = 400  # chars from page 1 for header


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
        # Unsupported format — return empty
        return [], {"filename": filename, "title": "", "page_count": 0}


# ── PDF Extraction ──────────────────────────────────────────────────────────


def _extract_pdf(filename: str, file_bytes: bytes) -> tuple[list[dict], dict]:
    """Extract text from a PDF using PyMuPDF layout mode.

    Uses a 5-page sliding window with 2-page overlap to preserve
    financial table context and footnote associations.
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

    # Build chunks with sliding window
    chunks: list[dict] = []
    if page_count <= PDF_WINDOW:
        # Small file: single chunk
        text = "\n".join(page_texts)
        pages = list(range(1, page_count + 1))
        chunks.append(_make_chunk(filename, header, text, pages, 0))
    else:
        chunk_idx = 0
        for start in range(0, page_count, PDF_STRIDE):
            end = min(start + PDF_WINDOW, page_count)
            text = "\n".join(page_texts[start:end])
            pages = list(range(start + 1, end + 1))  # 1-indexed
            chunks.append(_make_chunk(filename, header, text, pages, chunk_idx))
            chunk_idx += 1
            # Stop if we've reached the end
            if end == page_count:
                break

    doc.close()

    metadata = {
        "filename": filename,
        "title": title,
        "page_count": page_count,
    }
    return chunks, metadata


# ── DOCX Extraction ─────────────────────────────────────────────────────────


def _extract_docx(filename: str, file_bytes: bytes) -> tuple[list[dict], dict]:
    """Extract text from a DOCX using python-docx.

    Uses 25-paragraph batching with 10-paragraph overlap to keep
    legal definitions in context with the clauses that reference them.
    """
    document = docx.Document(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    para_count = len(paragraphs)

    # Header: first 400 chars of all paragraphs joined
    raw_header = " ".join(paragraphs)[:HEADER_CHARS]
    header = _sanitize_header(raw_header)

    # Title: first non-empty paragraph
    title = paragraphs[0].strip() if paragraphs else ""

    # Build chunks
    chunks: list[dict] = []
    if para_count <= DOCX_WINDOW:
        # Small file: single chunk
        text = "\n".join(paragraphs)
        chunks.append(_make_chunk(filename, header, text, [0], 0))
    else:
        chunk_idx = 0
        for start in range(0, para_count, DOCX_STRIDE):
            end = min(start + DOCX_WINDOW, para_count)
            text = "\n".join(paragraphs[start:end])
            # DOCX has no reliable page numbers
            chunks.append(_make_chunk(filename, header, text, [0], chunk_idx))
            chunk_idx += 1
            if end == para_count:
                break

    metadata = {
        "filename": filename,
        "title": title,
        "page_count": 0,  # DOCX has no reliable page count
    }
    return chunks, metadata


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
    """Build a chunk dict with the required text template."""
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
    }
