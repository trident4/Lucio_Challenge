# Phase 2: Tantivy BM25 Indexing

## 📌 Objective

Build an ultra-fast, in-memory keyword inverted index using Rust-backed Tantivy to bypass the Python GIL entirely.

## 🧰 Dependencies

- `tantivy`

## 🏗 Architectural Requirements

### 1. Schema Definition

- Initialize a `tantivy.SchemaBuilder()`.
- Add `chunk_id`: `add_text_field("chunk_id", stored=True)`
- Add `text`: `add_text_field("text", stored=True, index_option="position")`
- Add `filename`: `add_text_field("filename", stored=True)`
- Add `page_nums`: `add_json_field("page_nums", stored=True)` (Store the list of pages as JSON).

### 2. Index Instantiation

- Build the schema.
- Create the index purely in RAM: `tantivy.Index(schema)`. Do not provide a directory path.
- Initialize the writer: `writer = index.writer()`.

### 3. Population & Commit

- Iterate through the flat list of chunk dictionaries returned by Phase 1.
- Add them as Tantivy Documents: `writer.add_document(...)`.
- Execute `writer.commit()` immediately after the loop.
