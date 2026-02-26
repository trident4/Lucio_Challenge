"""Phase 3a: Concurrent BM25 retrieval via Tantivy.

Runs searches in threads (asyncio.to_thread) because Tantivy's
Python bindings are synchronous Rust FFI that would block the event loop.
"""

import asyncio
import logging
import re

logger = logging.getLogger("lucio.retriever")

# Tantivy reserved characters that must be escaped in user queries.
# Without escaping, M&A text like "Section 3.1(a)" would be parsed
# as query operators, returning wrong or zero results.
TANTIVY_SPECIAL = re.compile(r'([+\^`~*?\\/()\[\]{}"!\-:&|])')


def _escape_query(text: str) -> str:
    """Escape Tantivy reserved characters in query text."""
    return TANTIVY_SPECIAL.sub(r"\\\1", text)


def _search_one(index, searcher, question_text: str, top_k: int) -> list[dict]:
    """Execute a single BM25 search (synchronous — called from thread).

    Args:
        index: Tantivy Index (needed for parse_query).
        searcher: Tantivy Searcher instance.
        question_text: Raw question text.
        top_k: Number of top results to retrieve.

    Returns:
        List of dicts with chunk_id, text, filename, page_nums.
    """
    safe_query = _escape_query(question_text)
    query = index.parse_query(safe_query, ["text"])
    search_result = searcher.search(query, top_k)

    results = []
    for _score, doc_addr in search_result.hits:
        doc = searcher.doc(doc_addr)
        results.append(
            {
                "chunk_id": doc["chunk_id"][0],
                "text": doc["text"][0],
                "filename": doc["filename"][0],
                "page_nums": doc["page_nums"],
            }
        )
    return results


async def search_all(index, questions, top_k: int = 75) -> dict[str, list[dict]]:
    """Run concurrent BM25 searches for all questions.

    Uses asyncio.to_thread to avoid blocking the event loop
    (Tantivy is synchronous Rust FFI).

    Args:
        index: Tantivy Index.
        questions: List of Question objects.
        top_k: Number of chunks per question.

    Returns:
        Dict mapping question_id -> list of result dicts.
    """
    searcher = index.searcher()

    tasks = [
        asyncio.to_thread(_search_one, index, searcher, q.text, top_k)
        for q in questions
    ]
    results = await asyncio.gather(*tasks)

    search_results = {q.id: r for q, r in zip(questions, results)}
    total_hits = sum(len(r) for r in search_results.values())
    logger.info(f"BM25 search: {len(questions)} queries, {total_hits} total hits")
    return search_results
