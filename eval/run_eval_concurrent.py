#!/usr/bin/env python3
"""RAG Pipeline Evaluation Runner - Fully Concurrent

Simulates the true competition environment by firing 7 independent
HTTP POST requests at the exact same millisecond.

Usage:
    # Server must be running first
    python eval/run_eval_concurrent.py

    # Custom API URL or corpus path
    python eval/run_eval_concurrent.py --api http://localhost:8000 --corpus /path/to/Archive.zip
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Try to import aiohttp, fallback to httpx if it's what the backend uses
try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    import httpx

    HAS_AIOHTTP = False

EVAL_DIR = Path(__file__).parent
GROUND_TRUTH = EVAL_DIR / "ground_truth.json"

# ── Assertion Checkers ──────────────────────────────────────────────────────


def check_contains(answer: str, assertion: dict) -> bool:
    """Check if value appears in answer (case-insensitive)."""
    return assertion["value"].lower() in answer.lower()


def check_contains_any(answer: str, assertion: dict) -> bool:
    """Check if ANY of the values appear in answer (case-insensitive)."""
    lower = answer.lower()
    return any(v.lower() in lower for v in assertion["values"])


CHECKERS = {
    "contains": check_contains,
    "contains_any": check_contains_any,
}

# ── Async Request Handlers ──────────────────────────────────────────────────


async def fetch_question_aiohttp(session, api_url, payload, q_id):
    """Fetch using aiohttp (faster, true parallel multiplexing)."""
    start = time.perf_counter()
    try:
        async with session.post(f"{api_url}/challenge/run", json=payload) as response:
            data = await response.json()
            elapsed = time.perf_counter() - start
            return q_id, data, elapsed, None
    except Exception as e:
        elapsed = time.perf_counter() - start
        return q_id, None, elapsed, str(e)


async def fetch_question_httpx(client, api_url, payload, q_id):
    """Fetch using httpx."""
    start = time.perf_counter()
    try:
        response = await client.post(
            f"{api_url}/challenge/run", json=payload, timeout=300.0
        )
        data = response.json()
        elapsed = time.perf_counter() - start
        return q_id, data, elapsed, None
    except Exception as e:
        elapsed = time.perf_counter() - start
        return q_id, None, elapsed, str(e)


async def run_concurrent_requests(api_url: str, questions: list, corpus_url: str):
    """Fire all questions at the exact same time."""
    tasks = []

    print(
        f"🚀 Firing {len(questions)} INDEPENDENT requests to {api_url}/challenge/run SIMULTANEOUSLY"
    )
    print(f"   Corpus: {corpus_url}\n")

    if HAS_AIOHTTP:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=300)
        ) as session:
            for q in questions:
                payload = {
                    "corpus_url": corpus_url,
                    "questions": [{"id": q["id"], "text": q["text"]}],
                }
                tasks.append(fetch_question_aiohttp(session, api_url, payload, q["id"]))
            return await asyncio.gather(*tasks)
    else:
        async with httpx.AsyncClient(timeout=300.0) as client:
            for q in questions:
                payload = {
                    "corpus_url": corpus_url,
                    "questions": [{"id": q["id"], "text": q["text"]}],
                }
                tasks.append(fetch_question_httpx(client, api_url, payload, q["id"]))
            return await asyncio.gather(*tasks)


# ── Main ────────────────────────────────────────────────────────────────────


def main(api_url: str, corpus_url: str):
    # Load ground truth
    with open(GROUND_TRUTH) as f:
        gt = json.load(f)

    corpus = corpus_url if corpus_url else gt["corpus_url"]
    questions = gt["questions"]

    start_total = time.perf_counter()

    # Run the concurrent blast
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        run_concurrent_requests(api_url, questions, corpus)
    )

    elapsed_total = time.perf_counter() - start_total

    # ── Map Results & Run Assertions ────────────────────────────────────────

    total_pass = 0
    total_fail = 0
    total_assertions = 0
    question_results = []

    # Map q_id -> (data, duration, error)
    response_map = {res[0]: (res[1], res[2], res[3]) for res in results}

    for q in questions:
        q_id = q["id"]
        data, duration, error = response_map.get(
            q_id, (None, 0.0, "Missing from results")
        )

        if error or not data or "results" not in data or not data["results"]:
            print(f"⚠️  {q_id}: API Failed ({duration:.1f}s) -> {error}")
            question_results.append(
                {
                    "id": q_id,
                    "passed": 0,
                    "total": len(q["assertions"]),
                    "status": "💥 CRASH",
                    "failed_labels": [error or "No results returned"],
                    "duration": duration,
                }
            )
            continue

        answer_data = data["results"][0]
        answer_text = answer_data["answer"]
        passed = 0
        failed = 0
        failed_labels = []

        for assertion in q["assertions"]:
            checker = CHECKERS.get(assertion["type"])
            total_assertions += 1
            if checker and checker(answer_text, assertion):
                passed += 1
                total_pass += 1
            else:
                failed += 1
                total_fail += 1
                failed_labels.append(assertion["label"])

        total = passed + failed
        if failed == 0:
            status = "✅ PASS"
        elif passed > 0:
            status = "⚠️  PARTIAL"
        else:
            status = "❌ FAIL"

        question_results.append(
            {
                "id": q_id,
                "passed": passed,
                "total": total,
                "status": status,
                "failed_labels": failed_labels,
                "duration": duration,
            }
        )

    # ── Print Results ───────────────────────────────────────────────────

    print("=" * 80)
    print(f"{'Q':>4}  {'Status':<12}  {'Score':>7}  {'Time':>6}  {'Details'}")
    print("-" * 80)

    for qr in question_results:
        score = f"{qr['passed']}/{qr['total']}"
        time_str = f"{qr['duration']:.1f}s"
        details = ""
        if qr.get("failed_labels"):
            details = "Error/Missing: " + ", ".join(qr["failed_labels"])
        print(
            f"{qr['id']:>4}  {qr['status']:<12}  {score:>7}  {time_str:>6}  {details}"
        )

    print("-" * 80)
    pct = (total_pass / total_assertions * 100) if total_assertions else 0
    time_status = "✓ UNDER 30s" if elapsed_total < 30 else "⚠ OVER 30s!"
    print(
        f"{'':>4}  {'TOTAL':<12}  {total_pass}/{total_assertions} ({pct:.0f}%)   ⏱ {elapsed_total:.1f}s {time_status}"
    )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel RAG Pipeline Eval Runner")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--corpus", default="", help="Override corpus path")
    args = parser.parse_args()

    main(args.api, args.corpus)
