"""Phase 3b: JIT embedding with caching via Mac Studio.

Embeds only cache-miss chunks (deduped across questions),
batched in groups of EMBEDDING_BATCH_SIZE.
"""

import logging

import numpy as np
from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger("lucio.embedder")


async def embed_batch(
    client: AsyncOpenAI,
    texts: list[str],
    settings: Settings,
) -> list[np.ndarray]:
    """Embed a batch of texts via the Mac Studio embedding API.

    Handles the dimensions param gracefully:
    - If supported, passes dimensions=256 to the API.
    - If not (probed at startup), truncates manually and renormalizes.

    Args:
        client: AsyncOpenAI client pointing to Mac Studio.
        texts: List of text strings to embed.
        settings: App settings.

    Returns:
        List of 256-dimensional numpy arrays (float32).
    """
    if settings.supports_dimensions_param:
        resp = await client.embeddings.create(
            input=texts,
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )
        return [np.array(e.embedding, dtype=np.float32) for e in resp.data]
    else:
        resp = await client.embeddings.create(
            input=texts,
            model=settings.embedding_model,
        )
        vectors = []
        for e in resp.data:
            vec = np.array(
                e.embedding[: settings.embedding_dimensions], dtype=np.float32
            )
            # Re-normalize after truncation
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return vectors


async def embed_and_cache(
    client: AsyncOpenAI,
    search_results: dict[str, list[dict]],
    vector_cache: dict[str, np.ndarray],
    settings: Settings,
) -> None:
    """Embed all uncached chunks from search results.

    Deduplicates chunk_ids across all questions, then batches
    API calls in groups of embedding_batch_size.

    Args:
        client: AsyncOpenAI client.
        search_results: question_id -> list of hit dicts (with chunk_id, text).
        vector_cache: Global cache to update in-place.
        settings: App settings.
    """
    # Build text lookup from all search results (deduplicates naturally)
    text_lookup: dict[str, str] = {}
    for hits in search_results.values():
        for h in hits:
            text_lookup[h["chunk_id"]] = h["text"]

    # Find cache misses
    missing = [cid for cid in text_lookup if cid not in vector_cache]
    if not missing:
        logger.info("All chunks already cached — skipping embedding")
        return

    logger.info(
        f"Embedding {len(missing)} uncached chunks in batches of {settings.embedding_batch_size}"
    )

    # Batch embed
    for i in range(0, len(missing), settings.embedding_batch_size):
        batch_ids = missing[i : i + settings.embedding_batch_size]
        batch_texts = [text_lookup[cid] for cid in batch_ids]
        vectors = await embed_batch(client, batch_texts, settings)
        for cid, vec in zip(batch_ids, vectors):
            vector_cache[cid] = vec


async def embed_questions(
    client: AsyncOpenAI,
    questions,
    settings: Settings,
) -> dict[str, np.ndarray]:
    """Embed all question texts in a single batch.

    Args:
        client: AsyncOpenAI client.
        questions: List of Question objects.
        settings: App settings.

    Returns:
        Dict mapping question_id -> 256d numpy vector.
    """
    texts = [q.text for q in questions]
    vectors = await embed_batch(client, texts, settings)
    result = {q.id: v for q, v in zip(questions, vectors)}
    logger.info(f"Embedded {len(questions)} question vectors")
    return result
