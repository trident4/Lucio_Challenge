# Phase 4: Exact Math Reranking

## 📌 Objective

Use raw CPU matrix math to find the mathematically perfect Top 5 chunks out of the 75 retrieved by BM25, acting as an exact-match reranker.

## 🧰 Dependencies

- `numpy`

## 🏗 Architectural Requirements

### 1. Vector Prep

For each of the 15 questions:

- Embed the user's question string using the Mac Studio endpoint (1x256 vector).
- Retrieve the 75 chunk vectors for this specific question from `global_vector_cache`.
- Stack them into a `numpy` matrix of shape `(75, 256)`.

### 2. Math Execution

- Convert the question vector and chunk matrix to `np.float32`.
- **L2 Normalization:** Ensure both the question vector and the chunk matrix are L2-normalized. (Divide by `np.linalg.norm(..., axis=1, keepdims=True)`).
- **Dot Product:** Calculate similarity scores using `scores = np.dot(chunk_matrix, question_vector.T)`.
- **Top K Selection:** Use `np.argsort(scores.flatten())[-5:][::-1]` to get the indices of the Top 5 highest scores.

### 3. Output Assembly

- Map those Top 5 indices back to the original chunk text, `filename`, and `page_nums`.
- Return a finalized context payload object for each question containing the concatenated string of the Top 5 chunks and their source metadata.
