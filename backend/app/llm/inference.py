"""Phase 5: Unified LLM inference.

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

SYSTEM_PROMPT = """You are an expert document analysis AI. Answer the user's question with maximum precision and extreme brevity using ONLY the provided context.

1. EXTREME BREVITY: Answer directly in the first sentence. No filler, no disclaimers, no unnecessary background. Minimum words required.
2. EXACT EVIDENCE: After your answer, cite a short quote or data point from the context that proves it.
3. PRECISION: For legal text, respect defined terms, carve-outs ("except as..."), and conditions ("subject to"). For financial/tabular data, use exact figures from the source — do not round or approximate.
4. TABULAR DATA: Context may contain markdown tables from spreadsheets, labeled with "Sheet: <name>". To answer:
   - Match the sheet name mentioned in the question to the correct table.
   - Use column headers (row 1) to identify the correct column, and row labels (column 1) to identify the correct row.
   - When asked for a total or sum, add ALL matching rows — do not stop at the first match. Show the arithmetic.
   - When asked to list or count distinct values, scan the entire column across all table chunks.
5. PARTIAL/MISSING INFO: If the context only partially answers, give the partial answer and state what is missing. Only say "This information is not available in the provided documents." if the answer is truly absent.
6. CONFLICTS: If sources conflict, state the contradiction in one sentence and cite both.
7. COUNTING/LISTING: When VERIFIED DOCUMENT COUNTS are provided, use those exact counts and names as ground truth. Do not recount or reinterpret the document index.

Remember: Speed and factual density are critical. Prioritize direct facts over exhaustive explanations.
"""

# Heuristic: questions matching these patterns likely need the full
# document index rather than (or in addition to) chunk-level context.
COUNTING_KEYWORDS = re.compile(
    r"\b(how many|count|list all|name them|name all|enumerate|"
    r"how much|total number|which documents|what documents)\b",
    re.IGNORECASE,
)


def _build_type_summary(doc_metadata: list[dict]) -> str:
    """Group documents by type and produce a verified summary with counts.

    This pre-computes counts and names in Python so the LLM doesn't
    have to parse raw JSON to count documents — eliminating
    non-deterministic miscounting.
    """
    from collections import defaultdict

    groups: dict[str, list[str]] = defaultdict(list)
    for m in doc_metadata:
        doc_type = m.get("type", "Document")
        name = re.sub(r"\.(pdf|docx|xlsx)$", "", m["filename"], flags=re.IGNORECASE).strip()
        groups[doc_type].append(name)

    lines = [
        "VERIFIED DOCUMENT COUNTS (computed by system — use as ground truth, do not recount):"
    ]
    lines.append(f"Total files: {len(doc_metadata)}")
    for doc_type, names in sorted(groups.items()):
        names_str = "; ".join(sorted(names))
        lines.append(f"- {doc_type} ({len(names)}): {names_str}")
    return "\n".join(lines)


def _build_user_prompt(
    question_text: str,
    context: str,
    doc_metadata: list[dict],
) -> str:
    """Build the user prompt, optionally injecting document metadata.

    Context is already capped per-chunk by the reranker, so no
    additional truncation is needed here.

    For counting/listing questions, a verified type summary is placed
    FIRST as ground truth, followed by the full JSON for fallback.
    """
    is_counting = bool(COUNTING_KEYWORDS.search(question_text))

    if is_counting:
        type_summary = _build_type_summary(doc_metadata)
        meta_json = json.dumps(doc_metadata, indent=2)
        parts = [
            type_summary,
            f"\nFULL DOCUMENT INDEX (for detailed lookup):\n{meta_json}",
            f"CONTEXT:\n{context}",
        ]
    else:
        parts = [f"CONTEXT:\n{context}"]

    parts.append(f"QUESTION:\n{question_text}")
    return "\n\n".join(parts)


async def run_inference(
    client: AsyncOpenAI,
    questions,
    reranked: dict[str, dict],
    doc_metadata: list[dict],
    settings: Settings,
    model_override: str | None = None,
) -> tuple[dict[str, str], int]:
    """Fire all LLM prompts concurrently and collect answers.

    Returns:
        Tuple of (Dict mapping question_id -> answer, total_token_usage)
    """
    total_tokens = 0
    token_lock = asyncio.Lock()
    model_to_use = model_override or settings.llm_model

    async def _ask(q) -> tuple[str, str]:
        nonlocal total_tokens
        context = reranked.get(q.id, {}).get("context", "")
        user_prompt = _build_user_prompt(q.text, context, doc_metadata)
        try:
            resp = await client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

            usage = resp.usage.total_tokens if resp.usage else 0
            finish_reason = resp.choices[0].finish_reason

            if finish_reason != "stop":
                logger.warning(f"Question {q.id} finished with reason: {finish_reason}")

            async with token_lock:
                total_tokens += usage

            return q.id, resp.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed for question {q.id}: {e}")
            return q.id, f"[ERROR: LLM call failed — {e}]"

    results = await asyncio.gather(*[_ask(q) for q in questions])
    answers = dict(results)

    logger.info(
        f"LLM inference complete: {len(answers)} answers, {total_tokens} tokens"
    )
    return answers, total_tokens
