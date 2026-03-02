import logging
import re
import numpy as np
from openai import AsyncOpenAI

from app.config import Settings
from app.embeddings.embedder import embed_batch, _prepare_for_embedding

logger = logging.getLogger("lucio.compressor")

# Split on periods/question marks/exclamation marks followed by space and a Capital letter,
# or split on explicit newlines.
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n+")


async def compress_context(
    client: AsyncOpenAI,
    q_vectors: dict[str, np.ndarray],
    reranked: dict[str, dict],
    settings: Settings,
    top_n_sentences: int = 45,  # Increased to 45 to capture long justice lists (Q4)
    padding: int = 3,  # Increased padding to 3 to prevent severing context
) -> dict[str, dict]:
    """
    Compresses the massive chunk context into dense, padded sentence blocks.
    Ensures that [SOURCE: ...] tags remain untouched.
    """
    compressed_result = {}
    total_original_len = 0
    total_compressed_len = 0

    for q_id, data in reranked.items():
        q_vec = q_vectors.get(q_id)
        if q_vec is None or not data.get("context"):
            compressed_result[q_id] = data
            continue

        context_str = data["context"]
        total_original_len += len(context_str)
        blocks = context_str.split("\n\n===\n\n")

        all_sentences = []
        sentence_metadata = []

        for b_idx, block in enumerate(blocks):
            lines = block.split("\n")
            if not lines:
                continue

            source_line = lines[0]
            chunk_content = "\n".join(lines[1:])

            raw_sentences = SENTENCE_SPLIT_REGEX.split(chunk_content)

            # Aggressively filter out empty sentences and pure whitespace
            # The API will crash if we ask it to embed ""
            sentences = []
            for s in raw_sentences:
                clean_s = s.strip()
                if (
                    clean_s and len(clean_s) > 2
                ):  # Ignore punctuation-only or tiny glitches
                    sentences.append(clean_s)

            if not sentences:
                continue

            sentence_metadata.append(
                {
                    "block_idx": b_idx,
                    "text": source_line,
                    "is_source": True,
                    "local_idx": -1,
                }
            )

            for i, s in enumerate(sentences):
                sentence_metadata.append(
                    {
                        "block_idx": b_idx,
                        "text": s,
                        "is_source": False,
                        "local_idx": i,
                        "total_in_block": len(sentences),
                    }
                )
                all_sentences.append(s)

        if not all_sentences:
            compressed_result[q_id] = data
            continue

        # Batch embed all sentences
        # Nomic prefix: 'search_document: '
        sentence_vectors = []
        batch_size = settings.embedding_batch_size
        for i in range(0, len(all_sentences), batch_size):
            batch_texts = [
                _prepare_for_embedding(f"search_document: {s}")
                for s in all_sentences[i : i + batch_size]
            ]
            vecs = await embed_batch(client, batch_texts, settings)
            sentence_vectors.extend(vecs)

        # Cosine similarity
        matrix = np.stack(sentence_vectors)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix = matrix / norms
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-10)

        cosine_scores = matrix @ q_norm

        # Get top N indices
        actual_top_n = min(top_n_sentences, len(cosine_scores))
        top_indices = np.argsort(cosine_scores)[::-1][:actual_top_n]

        # Map embedded sentence index to sentence_metadata index
        embed_idx_to_meta_idx = {}
        curr_embed_idx = 0
        for m_idx, meta in enumerate(sentence_metadata):
            if not meta["is_source"]:
                embed_idx_to_meta_idx[curr_embed_idx] = m_idx
                curr_embed_idx += 1

        # Find exactly which m_idx we need to keep
        keep_m_indices = set()
        for embed_idx in top_indices:
            m_idx = embed_idx_to_meta_idx[embed_idx]
            meta = sentence_metadata[m_idx]
            block_idx = meta["block_idx"]
            local_idx = meta["local_idx"]

            # Keep padding [-padding, +padding]
            for offset in range(-padding, padding + 1):
                target_local = local_idx + offset
                if 0 <= target_local < meta["total_in_block"]:
                    # Find the corresponding m_idx
                    target_m_idx = m_idx + offset
                    if 0 <= target_m_idx < len(sentence_metadata):
                        if (
                            sentence_metadata[target_m_idx]["block_idx"] == block_idx
                            and not sentence_metadata[target_m_idx]["is_source"]
                        ):
                            keep_m_indices.add(target_m_idx)

        # Identify all document header blocks (which must be preserved entirely)
        header_blocks = {
            meta["block_idx"]
            for meta in sentence_metadata
            if meta["is_source"] and "DOCUMENT HEADER" in meta["text"]
        }

        # Always keep source lines if we kept at least one sentence from that block
        kept_blocks = {
            sentence_metadata[m_idx]["block_idx"] for m_idx in keep_m_indices
        }

        for m_idx, meta in enumerate(sentence_metadata):
            # If this block is a document header, keep ALL of its content unmodified
            if meta["block_idx"] in header_blocks:
                keep_m_indices.add(m_idx)
            # If this a normal chunk, keep its [SOURCE: ...] line if we kept any sentences
            elif meta["is_source"] and meta["block_idx"] in kept_blocks:
                keep_m_indices.add(m_idx)

        # Reconstruct context
        reconstructed_blocks = {}
        for m_idx in sorted(list(keep_m_indices)):
            meta = sentence_metadata[m_idx]
            b_idx = meta["block_idx"]
            if b_idx not in reconstructed_blocks:
                reconstructed_blocks[b_idx] = []
            reconstructed_blocks[b_idx].append(meta["text"])

        # Join pieces
        final_context_parts = []
        for b_idx in sorted(reconstructed_blocks.keys()):
            lines = reconstructed_blocks[b_idx]
            source = lines[0]  # The source line
            sentences_text = " ".join(lines[1:])
            final_context_parts.append(f"{source}\n{sentences_text}")

        compressed_context = "\n\n===\n\n".join(final_context_parts)
        total_compressed_len += len(compressed_context)

        compressed_result[q_id] = {
            "context": compressed_context,
            "sources": data["sources"],
        }

    if total_original_len > 0:
        reduction = 100 * (1 - total_compressed_len / total_original_len)
        logger.info(
            f"Context compression complete: reduced payload by {reduction:.1f}%"
        )

    return compressed_result
