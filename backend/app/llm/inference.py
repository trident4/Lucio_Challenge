"""Phase 5: Unified LLM inference via Qwen-30B on Mac Studio.

Fires all question prompts concurrently via asyncio.gather.
Includes a keyword heuristic that injects corpus-level document
metadata for counting/listing questions.
"""

import asyncio
import json
import logging
import re

from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger("lucio.inference")

SYSTEM_PROMPT = (
    "You are an expert analyst. Answer using ONLY the provided context. "
    "If the answer is not in the context, state that clearly. "
    "Be concise and precise."
)

# Heuristic: questions matching these patterns likely need the full
# document index rather than (or in addition to) chunk-level context.
COUNTING_KEYWORDS = re.compile(
    r"\b(how many|count|list all|name them|name all|enumerate|"
    r"how much|total number|which documents|what documents)\b",
    re.IGNORECASE,
)


def _build_user_prompt(
    question_text: str,
    context: str,
    doc_metadata: list[dict],
) -> str:
    """Build the user prompt, optionally injecting document metadata.

    For counting/listing questions, the full corpus document index
    is injected alongside the retrieved chunk context so the LLM
    can count files, list document names, etc.
    """
    parts = [f"CONTEXT:\n{context}"]

    if COUNTING_KEYWORDS.search(question_text):
        meta_json = json.dumps(doc_metadata, indent=2)
        parts.append(f"DOCUMENT INDEX (all files in corpus):\n{meta_json}")

    parts.append(f"QUESTION:\n{question_text}")
    return "\n\n".join(parts)


async def run_inference(
    client: AsyncOpenAI,
    questions,
    reranked: dict[str, dict],
    doc_metadata: list[dict],
    settings: Settings,
) -> dict[str, str]:
    """Fire all LLM prompts concurrently and collect answers.

    Args:
        client: AsyncOpenAI client pointing to Mac Studio.
        questions: List of Question objects.
        reranked: question_id -> {context, sources} from Phase 4.
        doc_metadata: Corpus-level document metadata from Phase 1.
        settings: App settings.

    Returns:
        Dict mapping question_id -> answer string.
    """

    async def _ask(q) -> tuple[str, str]:
        context = reranked.get(q.id, {}).get("context", "")
        user_prompt = _build_user_prompt(q.text, context, doc_metadata)

        try:
            resp = await client.chat.completions.create(
                model=settings.llm_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
            return q.id, resp.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed for question {q.id}: {e}")
            return q.id, f"[ERROR: LLM call failed — {e}]"

    results = await asyncio.gather(*[_ask(q) for q in questions])
    answers = dict(results)

    logger.info(f"LLM inference complete: {len(answers)} answers")
    return answers
