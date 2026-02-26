# Workflow 2: The "Banker's Standard" (Pre-Vectorized Semantic Architecture)

## 📌 Overview

This architecture prioritizes flawless, zero-hallucination semantic retrieval. It is designed for scenarios where the 250MB corpus can be fully ingested, parsed, and embedded _before_ the 30-second query timer begins. It utilizes heavy natural language processing and dense vector math to map complex corporate documents perfectly.

## 🛠 Hardware & Model Stack

- **Data Tier (Mac M1 - 16GB RAM):** Handles persistent vector storage, layout parsing, and FastAPI query routing.
  - _Vector Database:_ PostgreSQL with `pgvector` or local Qdrant instance.
- **Intelligence Tier (Mac Studio - Remote):** Handles heavy embedding, vision parsing, and multi-hop reasoning.
  - _Vision/Document Parsing:_ `granite-docling-258m-mlx`
  - _Semantic Embedding:_ `text-embedding-nomic-embed-text-v1.5` (Full 768 dimensions).
  - _Deep Reasoning:_ `qwen/qwen3-30b-a3b-2507`

## ⏱ Phase 1: Pre-Processing (Executed Before the Clock Starts)

### Step 1: High-Fidelity DOM Parsing (Time: ~2 Mins)

- The corpus is downloaded to the M1's solid-state drive.
- Complex PDFs are routed to `granite-docling-258m-mlx` (or heavy PyMuPDF scripts) to reconstruct a perfect Document Object Model (DOM). The system explicitly identifies Headers, Footnotes, Tables, and Captions.

### Step 2: Semantic NLP Chunking (Time: ~4 Mins)

- Raw text is processed through an NLP pipeline (e.g., SpaCy).
- Chunks are not bound by physical pages or token limits; they are bound by **logical topic shifts**.
- _Footnote Resolution:_ Floating footnotes are semantically mapped and injected directly into the parent table's text to prevent orphaned data during retrieval.

### Step 3: Massive Upfront Embedding (Time: ~10 Mins)

- Every semantic chunk (potentially thousands) is batched and sent to the Nomic embedding model.
- Full 768-dimensional vectors are generated to capture the maximum mathematical nuance of the legal jargon.

### Step 4: Persistent Database Storage (Time: ~10 Secs)

- Vectors, raw text, and enriched DOM metadata (Document Type, Year, Governing Law) are committed to the Vector Database.
- An HNSW (Hierarchical Navigable Small World) index is generated to map the vector space for instantaneous search.

---

## ⏱ Phase 2: Query Execution (The 30-Second Clock)

### Step 1: Query Embedding (Seconds 0.0 - 0.5)

- The 15 incoming Lucio questions hit the FastAPI router.
- Questions are immediately forwarded to the Nomic endpoint to be converted into 768-dimensional vectors.

### Step 2: Dense Vector ANN Search (Seconds 0.5 - 1.5)

- The M1 executes an Approximate Nearest Neighbor (ANN) search against the persistent Vector DB.
- Pre-filtering logic uses metadata (e.g., `WHERE doc_type = 'financial'`) to narrow the search space before calculating vector distance.
- The database returns the top semantically relevant chunks, even if exact keyword overlap is zero.

### Step 3: Multi-Agent Deep Reasoning (Seconds 1.5 - 25.0)

- Because the retrieved chunks are semantically perfect, smaller, and highly concentrated, the context window is highly optimized.
- The 15 context payloads are sent asynchronously to the remote Mac Studio.
- `qwen3-30b` executes deep multi-hop reasoning across the provided clauses (e.g., calculating exact valuation impacts based on retrieved M&A term sheets).

### Step 4: The Payload Drop (Seconds 25.0 - 30.0)

- Answers are aggregated, formatted into the exact JSON schema required by Lucio, and POSTed to the target endpoint to stop the clock.
