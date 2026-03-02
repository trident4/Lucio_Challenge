import logging
import re
import numpy as np
from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger("lucio.compressor")

# Split on periods/question marks/exclamation marks followed by space and a Capital letter,
# or split on explicit newlines.
SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
    "what",
    "how",
    "who",
    "when",
    "where",
    "why",
    "which",
    "did",
    "does",
    "do",
    "have",
    "had",
    "can",
    "could",
    "would",
    "should",
    "this",
    "these",
    "those",
    "they",
    "their",
    "them",
    "then",
    "than",
    "or",
    "if",
    "but",
    "not",
    "no",
    "yes",
    "so",
    "some",
    "such",
    "there",
    "about",
    "all",
    "any",
    "been",
    "many",
    "much",
    "other",
    "into",
    "out",
    "up",
    "down",
    "more",
    "most",
    "only",
}


def _get_keyword_weights(query: str) -> dict[str, float]:
    """Extract keywords from the query and assign weights.
    Numbers, Acronyms (ALL CAPS), and Proper Names (Capitalized) get 3.0 weight.
    Everything else gets 1.0 weight.
    """
    words = re.findall(r"\b[a-zA-Z0-9_.]+\b", query)
    weights = {}

    for w in words:
        w_lower = w.lower().strip()
        if w_lower in STOPWORDS or len(w_lower) <= 2:
            continue

        weight = 1.0
        # Check if it contains digits (like 42.3, 2021)
        if any(c.isdigit() for c in w):
            weight = 3.0
        # Check if acronym (HHI, CEO)
        elif w.isupper() and len(w) > 1:
            weight = 3.0
        # Check if Proper Noun (Scalia, Kodak)
        elif w[0].isupper() and not w.isupper():
            weight = 3.0

        weights[w_lower] = max(weights.get(w_lower, 0.0), weight)

    return weights


def _score_sentence(sentence: str, keyword_weights: dict[str, float]) -> float:
    """Score a sentence based on the presence of weighted keywords."""
    score = 0.0
    sentence_lower = sentence.lower()
    for kw, weight in keyword_weights.items():
        # Fast substring check, could be improved with regex word boundary if needed
        # but for speed, direct count is usually fine for these queries.
        if kw in sentence_lower:
            # Add weight times the frequency of occurrence
            score += weight * sentence_lower.count(kw)
    return score


async def compress_context(
    client: AsyncOpenAI,  # Left for interface compatibility
    q_vectors: dict[str, np.ndarray],
    reranked: dict[str, dict],
    settings: Settings,
    top_n_sentences: int = 40,
    padding: int = 2,
) -> dict[str, dict]:
    """
    Compresses the massive chunk context into dense, padded sentence blocks
    using Sub-Millisecond local Keyword Weighting (BM25-Lite) instead of external APIs.
    Ensures that Document Headers (chunk 0) remain untouched.
    """
    compressed_result = {}
    total_original_len = 0
    total_compressed_len = 0

    for q_id, data in reranked.items():
        if not data.get("context"):
            compressed_result[q_id] = data
            continue

        # We need the original question text to extract keywords.
        # reranked dict usually doesn't have the question text directly,
        # but wait, do we have it? Let's assume we can get it from data["query"] if added,
        # or we just fall back.
        # Let's check `data` content. Retriver output has `question`.
        question_text = data.get("question", "")
        if not question_text:
            logger.warning(
                f"Could not find question_text for {q_id}, bypassing compression."
            )
            compressed_result[q_id] = data
            continue

        keyword_weights = _get_keyword_weights(question_text)

        context_str = data["context"]
        total_original_len += len(context_str)
        blocks = context_str.split("\n\n===\n\n")

        sentence_metadata = []
        scoring_array = []

        for b_idx, block in enumerate(blocks):
            lines = block.split("\n")
            if not lines:
                continue

            source_line = lines[0]
            chunk_content = "\n".join(lines[1:])
            raw_sentences = SENTENCE_SPLIT_REGEX.split(chunk_content)

            sentences = [
                s.strip() for s in raw_sentences if s.strip() and len(s.strip()) > 2
            ]

            if not sentences:
                continue

            # Record Header
            sentence_metadata.append(
                {
                    "block_idx": b_idx,
                    "text": source_line,
                    "is_source": True,
                    "local_idx": -1,
                }
            )

            # Record Sentences
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
                # Only score if it's not a source tag
                scoring_array.append(_score_sentence(s, keyword_weights))

        if not scoring_array:
            compressed_result[q_id] = data
            continue

        # Get top N indices from the scores
        scores = np.array(scoring_array)
        actual_top_n = min(top_n_sentences, len(scores))

        # np.argsort sorts ascending, so we take the end and reverse it
        top_indices = np.argsort(scores)[-actual_top_n:][::-1]

        # Only keep sentences that actually scored > 0 to prevent garbage collection
        # But if none scored > 0, we'll just keep the top ones anyway (or fallback).
        # We will filter out 0 scores unless it's the only thing we have.
        valid_indices = [idx for idx in top_indices if scores[idx] > 0]
        if not valid_indices and len(scores) > 0:
            valid_indices = top_indices  # fallback

        # Map scoring array index to sentence_metadata index
        embed_idx_to_meta_idx = {}
        curr_embed_idx = 0
        for m_idx, meta in enumerate(sentence_metadata):
            if not meta["is_source"]:
                embed_idx_to_meta_idx[curr_embed_idx] = m_idx
                curr_embed_idx += 1

        keep_m_indices = set()
        for valid_idx in valid_indices:
            m_idx = embed_idx_to_meta_idx[valid_idx]
            meta = sentence_metadata[m_idx]
            block_idx = meta["block_idx"]
            local_idx = meta["local_idx"]

            # Keep padding [-padding, +padding]
            for offset in range(-padding, padding + 1):
                target_local = local_idx + offset
                if 0 <= target_local < meta["total_in_block"]:
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

        final_context_parts = []
        for b_idx in sorted(reconstructed_blocks.keys()):
            lines = reconstructed_blocks[b_idx]
            source = lines[0]
            sentences_text = " ".join(lines[1:])
            final_context_parts.append(f"{source}\n{sentences_text}")

        compressed_context = "\n\n===\n\n".join(final_context_parts)
        total_compressed_len += len(compressed_context)

        compressed_result[q_id] = {
            "context": compressed_context,
            "sources": data["sources"],
            "question": question_text,
        }

    if total_original_len > 0:
        reduction = 100 * (1 - total_compressed_len / total_original_len)
        logger.info(
            f"Keyword compression complete: reduced payload by {reduction:.1f}%"
        )

    return compressed_result
