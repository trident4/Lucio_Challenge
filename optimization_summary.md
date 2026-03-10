# Lucio RAG Pipeline: Debugging & Optimization Summary

_Session Date: March 9, 2026_

This document summarizes the recent debugging, scaling, and prompt engineering steps taken to optimize the Lucio RAG pipeline for the upcoming Hackathon (Target: 200 documents, 15 questions in <30 seconds).

## 1. Ground Truth Corrections & Document Tricky Cases

### The NVCA "B4" Question Bug

- **Issue:** The question B4 asked for Major Investor approval actions listed under Section 5.5 of the NVCA Investors' Rights Agreement. The pipeline was correctly answering "Not available", but failing the hardcoded ground truth assertions.
- **Root Cause:** The question itself was fundamentally flawed. Section 5.5 of the provided `NVCA-Model-Document-Investor-Rights-Agreement.docx` document explicitly requires "Series A Director" approval, not "Major Investors", and it does not mention Mergers/Change of Control.
- **Fix:** We updated the `ground_truth.json` file and frontend `TEST_DATA` to reflect that the LLM successfully identified the false premise of the question, rather than flagging it as a pipeline failure.

## 2. Speed-Optimizing the System Prompt

- **Issue:** To hit the Hackathon goal of processing 15 parallel questions, the LLM output generation phase was identified as the massive >15s bottleneck. Our original system prompt demanded "highly comprehensive and exhaustive" legal reasoning.
- **Root Cause:** Token generation speed (TPS). Making the LLM write 400+ tokens per answer is incredibly slow when multiplied across 15 parallel requests.
- **Fix:** Completely overhauled the `SYSTEM_PROMPT` in `app/llm/inference.py` to demand **"Extreme Brevity"**.
  - The new prompt forces the answer in the first sentence with zero filler.
  - Allowed "Partial Answers" instead of full failure codes if context is incomplete.
  - **Result:** Cut total LLM phase execution time down from **>31s down to 17.1s** for a 7-question batch.

## 3. The Q5 "Socony-Vacuum" Extraction Bug

- **Issue:** After applying the new "Speed-Optimized" prompt, Question 5 ("How many SCOTUS cases are in the set?") dropped from 5 to 4.
- **Root Cause:** Our Python extraction script `app/extraction/workers.py` automatically flags Supreme Court cases using the regex `\d+\s+U\.S\.\s+\d+`. The 5th case filename was: `United States v. Socony-Vacuum Oil Co., Inc., 310 U. S. 150 (1940) .pdf`. The space in **`U. S.`** caused the regex to miss it. Because the new prompt was strictly following only the provided metadata list, the LLM correctly counted that only 4 items had the `SCOTUS case` flag.
- **Fix:**
  1. Updated the Regex to `\d+\s+U\.\s*S\.\s+\d+` to handle the spatial anomaly.
  2. Cleared the global `corpus_cache` memory inside `app/state.py` (via Uvicorn restart/re-run).
  3. Re-ran the evaluation script.
  - **Result:** Q5 correctly hit 6/6 assertions, naming all 5 required SCOTUS cases. The 7 base questions hit a consistent **100% pass rate at 17.1 seconds.**

## 4. Hackathon Performance Strategy (<30 Seconds)

To securely handle scale (200 docs / 15 questions) in under 30 seconds, we formalized the following architectural strategy:

1. **Hybrid JIT (Just-In-Time) RAG (Complete):** Avoids full upfront database vectorization. Uses fast BM25 to get the top 150 chunks, and only embeds those chunks dynamically (~1.5s total embedding time).
2. **LLM Generator Streaming (Pending):** Modify FastAPI backend to stream the LLM response via Server-Sent Events (SSE). This will allow the user to see the first word of the answer in exactly the time it takes to Search (~7s), totally erasing the perceived generation latency.
3. **Extraction Caching to Disk (Pending):** Move the `corpus_cache` from transient RAM to persistent disk (SQLite or Redis) mapped by ZIP SHA hash, dropping the 5.5s parsing phase entirely on repeat document sets.
