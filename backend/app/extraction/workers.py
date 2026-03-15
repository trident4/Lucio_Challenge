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

import base64
import datetime
import io
import logging
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import combinations

import docx
import fitz  # PyMuPDF
import openpyxl

logger = logging.getLogger("lucio.workers")

# Chunking parameters — small, focused chunks for precise retrieval
CHUNK_SIZE = 2000  # chars per chunk (~500 tokens, ~half a page)

# OCR parameters for scanned PDFs
SCANNED_THRESHOLD = 50  # chars — pages with less text are considered scanned
OCR_MODEL = "google/gemini-2.0-flash-001"
OCR_DPI = 72   # lowest readable DPI — fewer image tokens = faster processing
OCR_MAX_PAGES = 15  # cap pages to keep OCR under ~10s
OCR_MAX_WORKERS = 15  # match max pages
OCR_TIMEOUT = 15.0

# Filename normalization: strip leading "4. " or "11. " numbering prefixes
_LEADING_JUNK = re.compile(r'^[\d.\s\-_]+')

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


def deduplicate_scanned(
    file_tuples: list[tuple[str, bytes]],
) -> list[tuple[str, bytes]]:
    """Remove scanned PDFs that have a digital twin in the corpus.

    For each PDF, quick-checks page 1 text. If < SCANNED_THRESHOLD chars,
    it's a scanned candidate. Groups by normalized filename (stripping
    leading numbering like "4. "). If a group has both scanned and digital
    versions, drops the scanned one.
    """
    pdf_info: list[tuple[str, bytes, str, bool]] = []  # (filename, bytes, norm_name, is_scanned)

    for filename, file_bytes in file_tuples:
        if not filename.lower().endswith(".pdf"):
            continue
        norm = _LEADING_JUNK.sub('', filename)
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            text = doc.load_page(0).get_text() if len(doc) > 0 else ""
            doc.close()
            is_scanned = len(text.strip()) < SCANNED_THRESHOLD
        except Exception:
            is_scanned = False
        pdf_info.append((filename, file_bytes, norm, is_scanned))

    # Group by normalized name
    groups: dict[str, list[tuple[str, bool]]] = {}
    for filename, _, norm, is_scanned in pdf_info:
        groups.setdefault(norm, []).append((filename, is_scanned))

    # Find scanned PDFs that have a digital twin
    skip = set()
    for norm, members in groups.items():
        if len(members) < 2:
            continue
        has_digital = any(not s for _, s in members)
        if has_digital:
            for fn, is_scanned in members:
                if is_scanned:
                    twin = next(f for f, s in members if not s)
                    logger.info(f"Skipping scanned duplicate: {fn} (twin: {twin})")
                    skip.add(fn)

    if skip:
        return [(fn, fb) for fn, fb in file_tuples if fn not in skip]
    return file_tuples


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


def _extract_document_wrapper(args: tuple) -> tuple[list[dict], dict]:
    """Wrapper to unpack tuple for pool.map (which passes single arg)."""
    filename, file_bytes = args
    return extract_document(filename, file_bytes)


def extract_document(
    filename: str, file_bytes: bytes, ocr_config: dict | None = None,
) -> tuple[list[dict], dict]:
    """Dispatch extraction based on file extension.

    Returns:
        (chunks, metadata) tuple.
    """
    ext = os.path.splitext(filename.lower())[1]
    if ext == ".pdf":
        return _extract_pdf(filename, file_bytes, ocr_config)
    elif ext == ".docx":
        return _extract_docx(filename, file_bytes)
    elif ext == ".xlsx":
        return _extract_xlsx(filename, file_bytes)
    else:
        return [], {"filename": filename, "title": "", "page_count": 0}


# ── PDF Extraction ──────────────────────────────────────────────────────────


def _extract_pdf(
    filename: str, file_bytes: bytes, ocr_config: dict | None = None,
) -> tuple[list[dict], dict]:
    """Extract text from a PDF using PyMuPDF layout mode.

    Extracts each page's text, then splits into ~2000-char chunks
    with 200-char overlap. Tracks which page(s) each chunk spans.

    For scanned pages (< SCANNED_THRESHOLD chars), falls back to
    vision-based OCR via API if ocr_config is provided.
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    page_count = len(doc)

    # Extract all page texts using layout mode (preserves table columns)
    page_texts: list[str] = []
    for page in doc:
        page_texts.append(page.get_text("layout"))

    # Detect scanned pages
    scanned = [i for i, t in enumerate(page_texts) if len(t.strip()) < SCANNED_THRESHOLD]
    if ocr_config and scanned:
        # Background OCR path: actually perform OCR
        if len(scanned) > OCR_MAX_PAGES:
            logger.info(
                f"{filename}: {len(scanned)} scanned pages, capping OCR to first {OCR_MAX_PAGES}"
            )
            scanned = scanned[:OCR_MAX_PAGES]
        logger.info(f"{filename}: OCR {len(scanned)}/{page_count} scanned pages")
        ocr_texts = _ocr_scanned_pages(doc, scanned, ocr_config)
        for i, text in zip(scanned, ocr_texts):
            page_texts[i] = text
        scanned = []  # OCR done, no longer scanned
    elif scanned:
        # Extraction path: flag scanned pages, defer OCR to background
        logger.info(f"{filename}: {len(scanned)}/{page_count} scanned pages (OCR deferred to background)")

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
        "scanned_pages": scanned,
    }
    return chunks, metadata


def _ocr_scanned_pages(
    doc, page_indices: list[int], ocr_config: dict,
) -> list[str]:
    """OCR scanned pages by sending each page image to a vision LLM.

    Fires all API calls concurrently via ThreadPoolExecutor — the calls
    are I/O-bound so 30+ in-flight requests complete in ~13-15s total
    regardless of page count.
    """
    import httpx

    api_key = ocr_config["api_key"]
    base_url = ocr_config["base_url"]
    model = ocr_config.get("model", OCR_MODEL)

    # Render pages to base64 JPEG (3-5x smaller than PNG for scanned docs)
    page_images: list[tuple[int, str]] = []
    for idx in page_indices:
        pix = doc[idx].get_pixmap(dpi=OCR_DPI)
        b64 = base64.b64encode(pix.tobytes("jpeg")).decode()
        page_images.append((idx, b64))

    logger.info(f"OCR: {len(page_images)} pages at {OCR_DPI} DPI")

    def _ocr_one(args: tuple[int, str]) -> tuple[int, str]:
        page_idx, b64_image = args
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Extract ALL text from this document page "
                                    "exactly as written. Preserve headings, "
                                    "paragraphs, and structure. Output only "
                                    "the extracted text."
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64_image}",
                                },
                            },
                        ],
                    }],
                    "max_tokens": 4000,
                    "temperature": 0.0,
                },
                timeout=OCR_TIMEOUT,
            )
            resp.raise_for_status()
            return page_idx, resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"OCR failed for page {page_idx + 1}: {e}")
            return page_idx, ""

    workers = min(len(page_images), OCR_MAX_WORKERS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_ocr_one, page_images))

    results.sort(key=lambda x: x[0])
    texts = [text for _, text in results]

    ocr_chars = sum(len(t) for t in texts)
    logger.info(f"OCR complete: {ocr_chars:,d} chars from {len(texts)} pages")
    return texts


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

_AGGREGATE_ROW_RE = re.compile(r'\b(Total|Net|Free|Gross|Operating|Grand)\b', re.IGNORECASE)
MAX_CROSSTAB_ROWS = 50


def _detect_dimensions_and_measures(
    header_row: tuple, data_rows: list[tuple], col_start: int, col_end: int,
) -> tuple[list[tuple], list[tuple]]:
    """Auto-detect categorical dimensions vs numeric measures.

    Dimension: text column with ≤20 unique values that REPEAT (unique/total ≤ 50%).
    Measure: purely numeric column.
    """
    dimensions = []  # (col_idx, col_name, sorted_unique_values)
    measures = []    # (col_idx, col_name)

    for col_idx in range(col_start, col_end + 1):
        col_name = _format_cell(header_row[col_idx])
        numeric_count = 0
        text_values = set()

        for row in data_rows:
            val = row[col_idx] if col_idx < len(row) else None
            if val is None:
                continue
            if isinstance(val, (int, float)):
                numeric_count += 1
            else:
                text_values.add(str(val).strip())

        if text_values and numeric_count == 0:
            unique_ratio = len(text_values) / len(data_rows) if data_rows else 1
            if len(text_values) <= 20 and unique_ratio <= 0.5:
                dimensions.append((col_idx, col_name, sorted(text_values)))
        elif numeric_count > 0 and not text_values:
            measures.append((col_idx, col_name))

    return dimensions, measures


def _build_sheet_summary(
    ws_title: str, header_row: tuple, data_rows: list[tuple],
    col_start: int, col_end: int,
    dimensions: list[tuple], measures: list[tuple],
) -> str:
    """Build a summary chunk for one sheet.

    For dimensional sheets: lists unique values per dimension.
    For non-dimensional sheets: lists row labels and key aggregate metrics.
    """
    col_names = [_format_cell(header_row[i]) for i in range(col_start, col_end + 1)]
    lines = [
        f"SHEET SUMMARY: {ws_title} | {len(data_rows)} data rows",
        f"Columns: {' | '.join(col_names)}",
        "",
    ]

    if dimensions:
        # Dimensional sheet — list unique values per dimension
        lines.append("Unique values:")
        for _, dim_name, unique_vals in dimensions:
            lines.append(f"- {dim_name} ({len(unique_vals)}): {', '.join(unique_vals)}")
    else:
        # Non-dimensional sheet — row labels + key metrics
        first_col_idx = col_start
        row_labels = []
        for row in data_rows:
            val = row[first_col_idx] if first_col_idx < len(row) else None
            if val is not None:
                row_labels.append(str(val).strip())

        if row_labels and len(row_labels) <= 30:
            lines.append(f"Row labels ({len(row_labels)}): {', '.join(row_labels)}")
            lines.append("")

        # Key metrics: find aggregate rows and report last numeric column value
        last_measure_idx = None
        last_measure_name = None
        for col_idx in range(col_end, col_start - 1, -1):
            has_numeric = any(
                isinstance(row[col_idx] if col_idx < len(row) else None, (int, float))
                for row in data_rows
            )
            if has_numeric:
                last_measure_idx = col_idx
                last_measure_name = _format_cell(header_row[col_idx])
                break

        if last_measure_idx is not None:
            agg_metrics = []
            for row in data_rows:
                label_val = row[first_col_idx] if first_col_idx < len(row) else None
                if label_val is None:
                    continue
                label = str(label_val).strip()
                if _AGGREGATE_ROW_RE.search(label):
                    metric_val = row[last_measure_idx] if last_measure_idx < len(row) else None
                    if metric_val is not None:
                        agg_metrics.append((label, _format_cell(metric_val)))
            if agg_metrics:
                lines.append(f"Key metrics ({last_measure_name}):")
                for label, val in agg_metrics:
                    lines.append(f"- {label}: {val}")

    return "\n".join(lines)


def _build_aggregation_chunks(
    dimensions: list[tuple], measures: list[tuple],
    header_row: tuple, data_rows: list[tuple],
    col_start: int, col_end: int,
    filename: str, header_tag: str, ws_title: str,
    sheet_num: int, start_chunk_idx: int,
) -> tuple[list[dict], int]:
    """Build pre-computed aggregation chunks for dimensional sheets.

    Generates single-dimension totals and two-way cross-tabs.
    Excludes subtotal rows (matching _AGGREGATE_ROW_RE) to prevent double-counting.
    """
    # Filter out subtotal rows
    agg_rows = [
        row for row in data_rows
        if not any(
            isinstance(row[d_idx] if d_idx < len(row) else None, str)
            and _AGGREGATE_ROW_RE.search(str(row[d_idx]))
            for d_idx, _, _ in dimensions
        )
    ]

    all_chunks: list[dict] = []
    chunk_idx = start_chunk_idx

    # Single-dimension totals
    for d_idx, d_name, d_vals in dimensions:
        groups: dict[str, list[tuple]] = defaultdict(list)
        for row in agg_rows:
            key = str(row[d_idx]).strip() if d_idx < len(row) and row[d_idx] is not None else ""
            if key:
                groups[key].append(row)

        # Build aggregated rows
        agg_header_cells = [d_name] + [m_name for _, m_name in measures]
        agg_md_header = "| " + " | ".join(agg_header_cells) + " |\n"
        agg_md_header += "| " + " | ".join("---" for _ in agg_header_cells) + " |"

        agg_data_rows = []
        for val in sorted(groups.keys()):
            row_cells = [val]
            for m_idx, _ in measures:
                total = sum(
                    r[m_idx] for r in groups[val]
                    if m_idx < len(r) and isinstance(r[m_idx], (int, float))
                )
                row_cells.append(_format_cell(total))
            agg_data_rows.append("| " + " | ".join(row_cells) + " |")

        content = f"AGGREGATION: {ws_title} — Totals by {d_name}\n{agg_md_header}\n" + "\n".join(agg_data_rows)

        if len(content) <= CHUNK_SIZE:
            all_chunks.append(_make_chunk(filename, header_tag, content, [sheet_num], chunk_idx))
            chunk_idx += 1
        else:
            # Split large aggregation tables
            sub_rows = [tuple(c.strip() for c in r.strip("| ").split("|")) for r in agg_data_rows]
            sub_chunks = _split_table_into_chunks(
                sub_rows, agg_md_header, header_tag, filename,
                0, len(agg_header_cells) - 1, sheet_num, chunk_idx,
            )
            chunk_idx += len(sub_chunks)
            all_chunks.extend(sub_chunks)

    # Two-way cross-tabs (only when 2+ dimensions)
    if len(dimensions) >= 2:
        for (d1_idx, d1_name, d1_vals), (d2_idx, d2_name, d2_vals) in combinations(dimensions, 2):
            max_rows = len(d1_vals) * len(d2_vals)
            if max_rows > MAX_CROSSTAB_ROWS:
                logger.info(f"Skipping cross-tab {d1_name}×{d2_name}: {max_rows} rows exceeds cap")
                continue

            groups: dict[tuple[str, str], list[tuple]] = defaultdict(list)
            for row in agg_rows:
                k1 = str(row[d1_idx]).strip() if d1_idx < len(row) and row[d1_idx] is not None else ""
                k2 = str(row[d2_idx]).strip() if d2_idx < len(row) and row[d2_idx] is not None else ""
                if k1 and k2:
                    groups[(k1, k2)].append(row)

            ct_header_cells = [d1_name, d2_name] + [m_name for _, m_name in measures]
            ct_md_header = "| " + " | ".join(ct_header_cells) + " |\n"
            ct_md_header += "| " + " | ".join("---" for _ in ct_header_cells) + " |"

            ct_data_rows = []
            for (v1, v2) in sorted(groups.keys()):
                row_cells = [v1, v2]
                for m_idx, _ in measures:
                    total = sum(
                        r[m_idx] for r in groups[(v1, v2)]
                        if m_idx < len(r) and isinstance(r[m_idx], (int, float))
                    )
                    row_cells.append(_format_cell(total))
                ct_data_rows.append("| " + " | ".join(row_cells) + " |")

            content = f"AGGREGATION: {ws_title} — Totals by {d1_name} × {d2_name}\n{ct_md_header}\n" + "\n".join(ct_data_rows)

            if len(content) <= CHUNK_SIZE:
                all_chunks.append(_make_chunk(filename, header_tag, content, [sheet_num], chunk_idx))
                chunk_idx += 1
            else:
                sub_rows = [tuple(c.strip() for c in r.strip("| ").split("|")) for r in ct_data_rows]
                sub_chunks = _split_table_into_chunks(
                    sub_rows, ct_md_header, header_tag, filename,
                    0, len(ct_header_cells) - 1, sheet_num, chunk_idx,
                )
                chunk_idx += len(sub_chunks)
                all_chunks.extend(sub_chunks)

    return all_chunks, chunk_idx


def _extract_xlsx(filename: str, file_bytes: bytes) -> tuple[list[dict], dict]:
    """Extract tabular data from an Excel workbook as markdown table chunks.

    Each sheet produces:
    1. A summary chunk (column headers, row count, key metrics or unique values)
    2. Aggregation chunks (pre-computed totals for dimensional sheets)
    3. Data chunks (existing markdown table chunks with header repetition)
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

        # 1. Detect structure
        dimensions, measures = _detect_dimensions_and_measures(
            header_row, data_rows, col_start, col_end,
        )

        # 2. Summary chunk (always first)
        summary = _build_sheet_summary(
            ws.title, header_row, data_rows, col_start, col_end,
            dimensions, measures,
        )
        all_chunks.append(_make_chunk(filename, header_tag, summary, [sheet_num], chunk_idx))
        chunk_idx += 1

        # 3. Aggregation chunks (for sheets with dimensions + measures)
        if dimensions and measures:
            agg_chunks, chunk_idx = _build_aggregation_chunks(
                dimensions, measures, header_row, data_rows,
                col_start, col_end, filename, header_tag, ws.title,
                sheet_num, chunk_idx,
            )
            all_chunks.extend(agg_chunks)

        # 4. Data chunks (existing logic, unchanged)
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
