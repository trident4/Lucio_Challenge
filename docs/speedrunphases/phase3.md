# Phase 3: Wide-Net Retrieval & JIT Vectorization

## 📌 Objective

Execute concurrent BM25 searches for all 15 questions, pull the top 75 chunks per question, check the global cache, and dynamically embed only the missing chunks using the remote Mac Studio.

## 🧰 Dependencies

- `asyncio`, `openai` (AsyncOpenAI)

## 🏗 Architectural Requirements

### 1. Concurrent Tantivy Search

- Create an async function that accepts the Tantivy searcher and a `Question` object.
- Execute a keyword search against the `text` field.
- Retrieve the **Top 75** documents. Return a list of their `chunk_id`s and raw text.
- Use `asyncio.gather` to run this for all 15 questions simultaneously.

### 2. The Vector Cache Check

- Flatten all retrieved `chunk_id`s from the 15 queries.
- Compare against `global_vector_cache`. Isolate the `chunk_id`s that are NOT in the cache.

### 3. Batched Embedding (Mac Studio)

- Initialize `AsyncOpenAI` pointing to the Mac Studio base URL.
- If there are missing chunks, batch their text and call the embedding API.
  - Model: `text-embedding-nomic-embed-text-v1.5`
  - **Strict Constraint:** You must specify the API parameter to truncate dimensions to `256`.
- Map the returned vectors back to their `chunk_id`s and update `global_vector_cache`.
