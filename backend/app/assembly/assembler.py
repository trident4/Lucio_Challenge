"""Phase 6: Payload assembly — format results into ChallengeResponse.

Maps LLM answers + source tracking into the required Pydantic schema,
deduplicating source filenames and merging page numbers.
"""

import logging
import re

from app.schemas import Answer, ChallengeResponse, Source

logger = logging.getLogger("lucio.assembler")


def assemble_response(
    questions,
    llm_answers: dict[str, str],
    reranked: dict[str, dict],
) -> ChallengeResponse:
    """Assemble the final ChallengeResponse from LLM outputs.

    Deduplicates sources by filename and merges page numbers.

    Args:
        questions: List of Question objects (for ordering).
        llm_answers: question_id -> answer text from Phase 5.
        reranked: question_id -> {context, sources} from Phase 4.

    Returns:
        ChallengeResponse ready for HTTP response.
    """
    results = []

    for q in questions:
        answer_text = llm_answers.get(q.id, "[ERROR: No answer produced]")
        raw_sources = reranked.get(q.id, {}).get("sources", [])

        # Deduplicate: merge page_nums for same filename
        source_map: dict[str, set[int]] = {}
        for s in raw_sources:
            fn = s["filename"]
            pages = s["page_nums"]
            # Handle both list and nested JSON formats
            if isinstance(pages, list):
                source_map.setdefault(fn, set()).update(pages)
            else:
                source_map.setdefault(fn, set()).add(pages)

        # Extract inline citations: e.g. [Source: filename.pdf]
        cited_files = set()
        for match in re.finditer(r"\[Source:\s*([^\]]+)\]", answer_text, re.IGNORECASE):
            cited_files.add(match.group(1).strip())

        # Filter sources: if LLM cited specific files, only keep those
        # If it failed to cite anything, fallback to returning all raw_sources
        if cited_files:
            filtered_source_map = {}
            for fn, pgs in source_map.items():
                # We do a loose 'in' check passing in case LLM truncated the filename slightly
                if any(
                    cited_fn.lower() in fn.lower() or fn.lower() in cited_fn.lower()
                    for cited_fn in cited_files
                ):
                    filtered_source_map[fn] = pgs
            # Only use the filtered map if it actually matched something
            if filtered_source_map:
                source_map = filtered_source_map

        sources = [
            Source(filename=fn, pages=sorted(pgs)) for fn, pgs in source_map.items()
        ]

        results.append(
            Answer(
                question_id=q.id,
                answer=answer_text,
                sources=sources,
            )
        )

    logger.info(f"Assembled response: {len(results)} answers")

    # TODO: POST to Lucio submission webhook on Challenge Day
    # response_json = ChallengeResponse(results=results).model_dump_json()
    # async with httpx.AsyncClient() as client:
    #     await client.post(SUBMISSION_URL, content=response_json)

    return ChallengeResponse(results=results)
