"""Phase 3a: Concurrent BM25 retrieval via Tantivy.

Runs searches in threads (asyncio.to_thread) because Tantivy's
Python bindings are synchronous Rust FFI that would block the event loop.

Uses multi-query retrieval: searches with both the full question AND
extracted entity names, then merges results. This handles vocabulary
mismatch (e.g., "bench" won't match "BLACKMUN, J., delivered the opinion"
but "Eastman Kodak" will find the case header).
"""

import asyncio
import json
import logging
import re

logger = logging.getLogger("lucio.retriever")

# Tantivy reserved characters that must be escaped in user queries.
TANTIVY_SPECIAL = re.compile(r'([+\\^`~*?\\\\/()\\[\\]{}"!\\-:&|])')

# Pattern to extract proper nouns / named entities (capitalized multi-word phrases)
# Matches sequences of capitalized words like "Eastman Kodak", "Meta", "CCI", "NVCA IRA"
ENTITY_PATTERN = re.compile(
    r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b"  # Title Case: "Eastman Kodak"
    r"|\b([A-Z]{2,}(?:\s+[A-Z]{2,})*)\b"  # ALL CAPS: "CCI", "NVCA IRA", "SCOTUS"
)


def _escape_query(text: str) -> str:
    """Escape Tantivy reserved characters in query text."""
    return TANTIVY_SPECIAL.sub(r"\\\\\\1", text)


def _extract_entities(question_text: str) -> list[str]:
    """Extract proper nouns / named entities from a question.

    Returns unique entity strings for multi-query retrieval.
    Filters out common words that happen to be capitalized (What, How, etc.)
    """
    stop_words = {
        "What",
        "How",
        "Why",
        "Who",
        "Where",
        "When",
        "Which",
        "Name",
        "Does",
        "Did",
        "Was",
        "Were",
        "Are",
        "Is",
        "Has",
        "Have",
        "Had",
        "Can",
        "Could",
        "Would",
        "Should",
        "The",
        "This",
        "That",
        "If",
        "And",
        "For",
        "Not",
        "All",
        "Any",
        "Some",
    }

    entities = set()
    for match in ENTITY_PATTERN.finditer(question_text):
        entity = match.group(0).strip()
        if entity and entity not in stop_words and len(entity) > 1:
            entities.add(entity)

    return list(entities)


def _search_one(index, searcher, question_text: str, top_k: int) -> list[dict]:
    """Execute a single BM25 search (synchronous — called from thread)."""
    safe_query = _escape_query(question_text)
    query = index.parse_query(safe_query, ["text"])
    search_result = searcher.search(query, top_k)

    results = []
    for score, doc_addr in search_result.hits:
        doc = searcher.doc(doc_addr)
        results.append(
            {
                "chunk_id": doc["chunk_id"][0],
                "text": doc["text"][0],
                "content": doc["content"][0],
                "filename": doc["filename"][0],
                "page_nums": json.loads(doc["page_nums"][0]),
                "bm25_score": float(score),
            }
        )
    return results


def _merge_results(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Merge two result lists, deduplicating by chunk_id.

    Primary results keep their BM25 scores. Secondary results that
    are new get their own scores. If a chunk appears in both,
    keep the higher BM25 score.
    """
    seen = {}
    for hit in primary:
        seen[hit["chunk_id"]] = hit

    for hit in secondary:
        cid = hit["chunk_id"]
        if cid not in seen or hit["bm25_score"] > seen[cid]["bm25_score"]:
            seen[cid] = hit

    # Sort by BM25 score descending to maintain ranking
    return sorted(seen.values(), key=lambda h: h["bm25_score"], reverse=True)


async def search_all(index, questions, top_k: int = 150) -> dict[str, list[dict]]:
    """Run multi-query BM25 searches for all questions.

    For each question, runs two searches:
    1. Full question text (original)
    2. Extracted entity names only (for vocabulary mismatch)

    Then merges and deduplicates results.

    Args:
        index: Tantivy Index.
        questions: List of Question objects.
        top_k: Number of chunks per query.

    Returns:
        Dict mapping question_id -> list of result dicts.
    """
    searcher = index.searcher()

    # Build all search tasks (primary + entity queries)
    primary_tasks = []
    entity_queries = []

    for q in questions:
        # Primary: full question
        primary_tasks.append(
            asyncio.to_thread(_search_one, index, searcher, q.text, top_k)
        )
        # Secondary: entity-only query
        entities = _extract_entities(q.text)
        entity_queries.append(" ".join(entities) if entities else None)

    # Run primary searches concurrently
    all_primary = await asyncio.gather(*primary_tasks)

    # Run entity searches concurrently (only for questions that have entities)
    entity_tasks = []
    entity_indices = []
    for i, eq in enumerate(entity_queries):
        if eq:
            entity_tasks.append(
                asyncio.to_thread(_search_one, index, searcher, eq, top_k)
            )
            entity_indices.append(i)

    entity_results_list = await asyncio.gather(*entity_tasks) if entity_tasks else []

    # Map entity results back to question indices
    all_entity = [[] for _ in questions]
    for idx, entity_result in zip(entity_indices, entity_results_list):
        all_entity[idx] = entity_result

    # Merge results
    search_results = {}
    for q, primary, entity, eq in zip(
        questions, all_primary, all_entity, entity_queries
    ):
        if eq and entity:
            merged = _merge_results(primary, entity)
            logger.info(
                f"  {q.id}: {len(primary)} primary + {len(entity)} entity "
                f"({eq}) → {len(merged)} merged"
            )
            search_results[q.id] = merged
        else:
            search_results[q.id] = primary

    total_hits = sum(len(r) for r in search_results.values())
    logger.info(
        f"BM25 search: {len(questions)} queries, {total_hits} total hits (multi-query)"
    )
    return search_results
