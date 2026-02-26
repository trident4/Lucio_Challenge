# Phase 1: The In-Memory Heist (Extraction & Chunking)

## 📌 Objective

Download a 250MB zip file into RAM and aggressively extract its text across multiple CPU cores. This phase is the foundation of accuracy; it must preserve financial table structures and legal context at all costs.

## 🧰 Dependencies

- `httpx`, `zipfile`, `io`, `os`, `concurrent.futures`, `pymupdf` (fitz), `python-docx`

## 🏗 Architectural Requirements

### 1. Smart Asset Loading (Hybrid Local/Remote)

The system must be able to load the corpus from either a local path or a remote URL without code changes.

- **Logic:**
  1. Check if `corpus_url` is a local file path using `os.path.exists()`.
  2. If local: `with open(path, "rb") as f: content = f.read()`.
  3. If remote: `async with httpx.AsyncClient() as client: resp = await client.get(url); content = resp.content`.
- **RAM-Only:** Always wrap the resulting bytes in `io.BytesIO(content)` before passing to `zipfile.ZipFile`.

### 2. The Multiprocessing Shredder

To bypass the Python Global Interpreter Lock (GIL), use `concurrent.futures.ProcessPoolExecutor()`.

- Map the `extract_document` worker function to the list of file-byte tuples from the zip.

### 3. High-Fidelity Worker Logic (Document Specific)

The worker must return a list of: `{"chunk_id": str, "filename": str, "page_nums": List[int], "text": str}`.

#### A. Pre-Processing (Metadata Grab)

- For every file, extract the first 400 characters of Page 1.
- Sanitize (remove double newlines).
- Save as `doc_header`. This string MUST be prepended to every chunk generated from this document to provide global context (e.g., Company Name, Date).

#### B. PDF Extraction (PyMuPDF / fitz)

- **Table Preservation:** Use `page.get_text("layout")`. This is non-negotiable as it maintains horizontal row integrity for financial data.
- **5-Page Sliding Window:** - Create chunks of 5 consecutive pages.
  - Apply a **2-page overlap** (e.g., Chunk 1: Pages 1-5, Chunk 2: Pages 4-8, Chunk 3: Pages 7-11).
  - _Banker's Logic:_ This ensures footnotes at the bottom of a page and headers at the top of the next page are trapped in the same context window.

#### C. DOCX Extraction (python-docx)

- **Paragraph Batching:** Iterate through `doc.paragraphs`.
  - Create chunks of **25 paragraphs**.
  - Apply a **10-paragraph overlap** between chunks.
  - _Legal Logic:_ This ensures that "Defined Terms" in Article I are likely to be caught in the first several chunks alongside the clauses that use them.

### 4. Final Data Assembly

The `text` field of each chunk must follow this exact template:
`"[HEADER: {doc_header}] [SOURCE: {filename}] [PAGES: {page_nums}] \n\n {extracted_content}"`
