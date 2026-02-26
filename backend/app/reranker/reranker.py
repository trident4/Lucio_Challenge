"""Phase 4: Exact math reranking via numpy dot product.

For each question, takes the BM25 top-K chunks and reranks them
using cosine similarity (L2-normalized dot product) against the
question vector. Returns the top rerank_top_k chunks with their
context and source metadata.
"""

import logging

import numpy as np

logger = logging.getLogger("lucio.reranker")

EPS = 1e-10  # Guard against zero-division in L2 normalization


def rerank_all(
    q_vectors: dict[str, np.ndarray],
    search_results: dict[str, list[dict]],
    vector_cache: dict[str, np.ndarray],
    top_k: int = 5,
) -> dict[str, dict]:
    """Rerank BM25 results using exact cosine similarity.

    For each question:
    1. Stack its chunk vectors into a (N, 256) matrix
    2. L2-normalize both question vector and chunk matrix
    3. Compute dot product scores
    4. Select top_k highest scoring chunks

    Args:
        q_vectors: question_id -> 256d question vector.
        search_results: question_id -> list of BM25 hit dicts.
        vector_cache: chunk_id -> 256d vector.
        top_k: Number of top chunks to keep per question.

    Returns:
        Dict mapping question_id -> {
            "context": concatenated top chunk texts,
            "sources": list of {filename, page_nums} dicts
        }
    """
    result = {}

    for q_id, q_vec in q_vectors.items():
        hits = search_results.get(q_id, [])
        if not hits:
            result[q_id] = {"context": "", "sources": []}
            continue

        # Filter to chunks that exist in the cache
        valid_hits = [h for h in hits if h["chunk_id"] in vector_cache]
        if not valid_hits:
            result[q_id] = {"context": "", "sources": []}
            continue

        # Stack chunk vectors into a matrix
        chunk_ids = [h["chunk_id"] for h in valid_hits]
        matrix = np.stack([vector_cache[cid] for cid in chunk_ids])  # (N, 256)

        # L2 normalize with epsilon guard
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + EPS
        matrix = matrix / norms
        q_norm = q_vec / (np.linalg.norm(q_vec) + EPS)

        # Dot product scores
        scores = matrix @ q_norm  # (N,)

        # Top-k selection
        actual_k = min(top_k, len(valid_hits))
        top_idx = np.argsort(scores)[-actual_k:][::-1]

        top_chunks = [valid_hits[i] for i in top_idx]
        result[q_id] = {
            "context": "\n\n---\n\n".join(c["text"] for c in top_chunks),
            "sources": [
                {"filename": c["filename"], "page_nums": c["page_nums"]}
                for c in top_chunks
            ],
        }

    logger.info(f"Reranked {len(q_vectors)} questions → top {top_k} chunks each")
    return result
