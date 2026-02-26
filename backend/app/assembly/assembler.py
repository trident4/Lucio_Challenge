"""Phase 6: Payload assembly — format results into ChallengeResponse.

Maps LLM answers + source tracking into the required Pydantic schema,
deduplicating source filenames and merging page numbers.
"""

import logging

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
