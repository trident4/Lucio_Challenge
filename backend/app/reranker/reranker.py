"""Phase 4: RRF hybrid reranking + neighbor context enrichment.

Two-stage reranking:
1. Compute BM25 rank (from Tantivy scores) and embedding rank (cosine similarity)
2. Combine via Reciprocal Rank Fusion for more robust scoring

Then enrich with context:
- Prose (PDF/DOCX): ±1 neighboring chunks for continuity
- Tabular (XLSX): no neighbors (avoids 3× duplication), sheet name labels instead
"""

import logging
import re
from collections import defaultdict

import numpy as np

logger = logging.getLogger("lucio.reranker")

EPS = 1e-10
RRF_K = 60  # Standard RRF constant (from Cormack et al.)


def rerank_all(
    questions: list,
    q_vectors: dict[str, np.ndarray],
    search_results: dict[str, list[dict]],
    vector_cache: dict[str, np.ndarray],
    all_chunks: list[dict],
    top_k: int = 8,
) -> dict[str, dict]:
    """RRF-rerank BM25 results using combined BM25 + embedding ranks.

    For each question:
    1. BM25 rank from Tantivy scores (already sorted)
    2. Embedding rank from cosine similarity (dot product on L2-normed vecs)
    3. RRF score = 1/(K + bm25_rank) + 1/(K + embed_rank)
    4. Select top_k by RRF score
    5. Attach ±1 neighbor chunks for context continuity

    Returns:
        Dict mapping question_id -> {context, sources}
    """
    # Build chunk index ONCE for all questions (immutable lookup)
    global_chunks_by_file = _build_chunk_index(all_chunks)

    result = {}
    q_text_map = {q.id: q.text for q in questions}

    for q_id, q_vec in q_vectors.items():
        hits = search_results.get(q_id, [])
        if not hits:
            result[q_id] = {"context": "", "sources": []}
            continue

        # Filter to chunks with embeddings
        valid_hits = [h for h in hits if h["chunk_id"] in vector_cache]
        if not valid_hits:
            result[q_id] = {"context": "", "sources": []}
            continue

        # ── BM25 ranking (already sorted by Tantivy score) ──
        bm25_rank = {h["chunk_id"]: i for i, h in enumerate(valid_hits)}

        # ── Embedding ranking (cosine similarity) ──
        chunk_ids = [h["chunk_id"] for h in valid_hits]
        matrix = np.stack([vector_cache[cid] for cid in chunk_ids])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + EPS
        matrix = matrix / norms
        q_norm = q_vec / (np.linalg.norm(q_vec) + EPS)
        cosine_scores = matrix @ q_norm

        # Sort by cosine to get embedding rank
        embed_order = np.argsort(cosine_scores)[::-1]
        embed_rank = {chunk_ids[idx]: rank for rank, idx in enumerate(embed_order)}

        # ── RRF fusion ──
        rrf_scores = {}
        for cid in chunk_ids:
            rrf_scores[cid] = 1.0 / (RRF_K + bm25_rank[cid]) + 1.0 / (
                RRF_K + embed_rank[cid]
            )

        # Select top-K by RRF
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)
        actual_k = min(top_k, len(sorted_ids))
        top_ids = sorted_ids[:actual_k]

        # Map back to hit dicts
        hit_lookup = {h["chunk_id"]: h for h in valid_hits}
        top_chunks = [hit_lookup[cid] for cid in top_ids]

        # ── Inject document headers (chunk_0) ──
        # For each unique document in top results, ensure chunk_0 is in context.
        # This provides case headers (justices), exec summaries, etc.
        seen_files = set()
        header_chunks = []
        for c in top_chunks:
            seen_files.add(c["filename"])

        for filename in seen_files:
            header_key = (filename, 0)
            header_content = global_chunks_by_file.get(header_key)
            # Only add if chunk_0 exists and isn't already in top chunks
            header_cid = f"{filename}::chunk_0"
            if header_content and header_cid not in top_ids:
                header_chunks.append(
                    {
                        "chunk_id": header_cid,
                        "filename": filename,
                        "content": header_content,
                        "page_nums": [1],
                    }
                )

        # ── Build context with ±1 neighbor enrichment ──
        context_parts = []

        # Headers first (document intros)
        for hc in header_chunks:
            context_parts.append(
                f"[SOURCE: {hc['filename']} — DOCUMENT HEADER]\n{hc['content']}"
            )

        # Then top reranked chunks with context enrichment
        for c in top_chunks:
            chunk_idx = _extract_chunk_index(c["chunk_id"])
            filename = c["filename"]

            if filename.lower().endswith(".xlsx"):
                # XLSX: no neighbor expansion — consecutive xlsx chunks in top-K
                # cause each chunk to appear ~3× via overlapping ±1 windows,
                # tripling context size with duplicate table rows.
                # Instead, label with sheet name so LLM can orient.
                sheet_name = _extract_sheet_name(c.get("text", ""))
                prefix = f"Sheet: {sheet_name}\n\n" if sheet_name else ""
                context_parts.append(
                    f"[SOURCE: {filename}]\n{prefix}{c['content']}"
                )
            else:
                # Prose: ±1 neighbor expansion for context continuity
                parts = []
                prev_content = global_chunks_by_file.get((filename, chunk_idx - 1))
                if prev_content:
                    parts.append(prev_content)

                parts.append(c["content"])

                next_content = global_chunks_by_file.get((filename, chunk_idx + 1))
                if next_content:
                    parts.append(next_content)

                context_parts.append(f"[SOURCE: {filename}]\n" + "\n---\n".join(parts))

        # Include header chunks in sources
        all_source_chunks = header_chunks + top_chunks
        result[q_id] = {
            "question": q_text_map.get(q_id, ""),
            "context": "\n\n===\n\n".join(context_parts),
            "sources": [
                {"filename": c["filename"], "page_nums": c["page_nums"]}
                for c in all_source_chunks
            ],
        }

    logger.info(f"Reranked {len(q_vectors)} questions → top {top_k} chunks each (RRF)")
    return result


def _build_chunk_index(
    chunks: list[dict],
) -> dict[tuple[str, int], str]:
    """Build (filename, chunk_index) -> content lookup from all available chunks.

    This guarantees we can find neighbor chunks (chunk-1, chunk+1) even
    if they were not retrieved in the top BM25 search results.
    """
    index = {}
    for h in chunks:
        chunk_idx = _extract_chunk_index(h["chunk_id"])
        key = (h["filename"], chunk_idx)
        if key not in index:
            index[key] = h["content"]
    return index


def _extract_chunk_index(chunk_id: str) -> int:
    """Extract the numeric chunk index from a chunk_id like 'file.pdf::chunk_4'."""
    try:
        return int(chunk_id.split("::chunk_")[-1])
    except (ValueError, IndexError):
        return -1


_SHEET_NAME_RE = re.compile(r"Sheet: ([^\]]+)\]")


def _extract_sheet_name(text: str) -> str:
    """Extract sheet name from an xlsx chunk's [HEADER: ... | Sheet: X] tag."""
    m = _SHEET_NAME_RE.search(text)
    return m.group(1).strip() if m else ""
