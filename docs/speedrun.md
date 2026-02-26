# Workflow 1: The "30-Second Speedrun" (Just-In-Time Architecture)

## 📌 Overview

This architecture is engineered for extreme low-latency execution. It assumes the 250MB corpus `.zip` is provided at the exact millisecond the 30-second query timer begins. To survive the clock, this workflow strictly avoids disk I/O, heavy upfront vectorization, and Python GIL bottlenecks by utilizing in-memory processing, Just-In-Time (JIT) embeddings, and decoupled remote LLM inference.

## 🛠 Hardware & Model Stack

- **Orchestrator (Mac M1 - 16GB RAM):** Handles FastAPI orchestration, `.zip` extraction, multiprocessing, and exact-math vector operations entirely in unified memory.
- **Inference Engine (Mac Studio - Remote):** Handles all neural network computations via an OpenAI-compatible API with continuous batching enabled.
  - _Embedding:_ `text-embedding-nomic-embed-text-v1.5` (Truncated to 256 dimensions).
  - _Deep Reasoning:_ `qwen/qwen3-30b-a3b-2507`
  - _Fast Fact Extraction:_ `ibm/granite-4-h-tiny`
  - _Counting/JSON Logic:_ `qwen/qwen3-coder-next`

## ⏱ Phase-by-Phase Execution Plan

### Phase 1: The In-Memory Heist (Seconds 0.0 - 5.0)

- **Network Fetch:** FastAPI router triggers `httpx` to download the 250MB `.zip` directly into `io.BytesIO`.
- **Multiprocessing Shredder:** `ProcessPoolExecutor` spins up across all M1 cores to bypass the GIL.
- **Banker-Grade Chunking:**
  - PDFs are parsed using PyMuPDF `layout` mode to preserve table row integrity horizontally.
  - Text is chunked using a **5-Page Sliding Window with a 2-Page Overlap** (roughly 3,500 tokens) to ensure massive financial tables and trailing footnotes are never severed.
- **Zero-Latency Metadata Injection:** \* Workers parse Page 1 of each document, extracting Titles/Parties via regex and prepending a `[HEADER]` string to every generated chunk.
  - "Article I / Definitions" sections are regex-stripped and pushed to a global JSON payload in memory.

### Phase 2: The Tantivy BM25 Index (Seconds 5.0 - 7.0)

- **Index Build:** The enriched, overlapping chunks are flushed directly into a `tantivy` in-memory schema.
- **Rust Acceleration:** Tantivy's Rust backend builds the inverted BM25 index over the 250MB text instantly, sidestepping Python's single-thread limitations.

### Phase 3: Wide-Net Retrieval & JIT Embedding (Seconds 7.0 - 12.0)

- **Concurrent Retrieval:** 15 distinct questions are fired at Tantivy concurrently via `asyncio.gather()`. The top 75 chunks per question are retrieved.
- **Cache Check & Batch:** Retrieved `chunk_ids` are checked against a local Python `vector_cache` dictionary.
- **JIT Embedding:** Only unique, un-embedded chunks are batched and sent via HTTP to the remote Mac Studio running `text-embedding-nomic-embed-text-v1.5`. Outputs are strictly truncated to 256 dimensions.

### Phase 4: Exact Math Reranking (Seconds 12.0 - 14.0)

- **k-NN Search:** Using pure `numpy`, the M1 calculates the exact dot product ($A \cdot B$) between each question vector and its corresponding 75 chunk vectors.
- **Isolation:** The mathematically perfect Top 5 chunks are isolated for each of the 15 queries.

### Phase 5: Agentic Routing & Inference (Seconds 14.0 - 25.0)

- **Heuristic Routing:** A regex/keyword router analyzes the 15 questions and dispatches them to the Mac Studio:
  - _Complex M&A/Regulatory:_ Routes to `qwen3-30b` alongside the Top 5 chunks + Global Definitions JSON.
  - _Simple Facts:_ Routes to `granite-4-h-tiny` with the Top 5 chunks for sub-second extraction.
  - _Document Counting:_ Routes to `qwen3-coder` alongside the Global Metadata JSON (bypassing chunks entirely).
- **Batch Execution:** `AsyncOpenAI` client fires all 15 payloads simultaneously to the Mac Studio.

### Phase 6: The Payload Drop (Seconds 25.0 - 30.0)

- **Aggregation:** Answers return to the M1.
- **Formatting:** System maps the answers to the required Lucio JSON schema, including precise document names and page numbers.
- **Submission:** Final POST request is sent to stop the clock.
