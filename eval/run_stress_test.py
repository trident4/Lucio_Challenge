#!/usr/bin/env python3
"""Stress test runner — validates <30s target at ~1GB / 15 questions.

No assertions on answer quality. Purely timing-focused.

Usage:
    python eval/run_stress_test.py                          # batch mode, stress corpus
    python eval/run_stress_test.py --mode concurrent        # 15 independent requests
    python eval/run_stress_test.py --mode both              # batch then concurrent
    python eval/run_stress_test.py --warmup                 # warm caches first
    python eval/run_stress_test.py --skip-cold              # only test cached path
    python eval/run_stress_test.py --threshold 45           # custom time limit (seconds)
"""

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

import requests

try:
    import aiohttp

    HAS_AIOHTTP = True
except ImportError:
    import httpx

    HAS_AIOHTTP = False

EVAL_DIR = Path(__file__).parent
GROUND_TRUTH = EVAL_DIR / "ground_truth.json"
DEFAULT_STRESS_CORPUS = EVAL_DIR / "stress_corpus.zip"

# 3 generic stress questions appended to existing eval questions
STRESS_QUESTIONS = [
    {
        "id": "s1",
        "text": "Provide a comprehensive summary of the main topics covered across all documents.",
    },
    {
        "id": "s2",
        "text": "Which documents discuss financial performance metrics or revenue figures?",
    },
    {
        "id": "s3",
        "text": "What regulatory frameworks or legal standards are referenced across the corpus?",
    },
]


def load_questions() -> list[dict]:
    """Load 12 existing questions (standard + battle) + 3 stress = 15 total."""
    with open(GROUND_TRUTH) as f:
        gt = json.load(f)

    questions = []
    # Strip assertions — stress test only cares about timing
    for q in gt["questions"]:
        questions.append({"id": q["id"], "text": q["text"]})
    for q in gt.get("battle_test_questions", []):
        questions.append({"id": q["id"], "text": q["text"]})
    questions.extend(STRESS_QUESTIONS)
    return questions


def format_phase_times(phase_times: dict | None) -> str:
    """Format phase_times dict into compact string."""
    if not phase_times:
        return ""
    parts = []
    for key in ["extract", "index", "retrieve_embed", "rerank", "llm"]:
        val = phase_times.get(key)
        if val is not None:
            parts.append(f"{key}={val:.1f}")
    return " | ".join(parts)


def run_batch(
    api_url: str, corpus: str, questions: list[dict], bypass_cache: bool
) -> dict:
    """Single POST with all questions. Returns timing info."""
    payload = {
        "corpus_url": corpus,
        "questions": questions,
        "bypass_cache": bypass_cache,
    }

    start = time.perf_counter()
    try:
        resp = requests.post(f"{api_url}/challenge/run", json=payload, timeout=600)
        resp.raise_for_status()
        data = resp.json()
        elapsed = time.perf_counter() - start
        return {
            "elapsed": elapsed,
            "total_time": data.get("total_time"),
            "phase_times": data.get("phase_times"),
            "num_results": len(data.get("results", [])),
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"elapsed": elapsed, "error": str(e)}


async def _fetch_one(session_or_client, api_url, payload, q_id):
    """Fire a single concurrent request."""
    start = time.perf_counter()
    try:
        if HAS_AIOHTTP:
            async with session_or_client.post(
                f"{api_url}/challenge/run", json=payload
            ) as resp:
                data = await resp.json()
        else:
            resp = await session_or_client.post(
                f"{api_url}/challenge/run", json=payload, timeout=300.0
            )
            data = resp.json()
        elapsed = time.perf_counter() - start
        return {
            "q_id": q_id,
            "elapsed": elapsed,
            "total_time": data.get("total_time"),
            "phase_times": data.get("phase_times"),
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"q_id": q_id, "elapsed": elapsed, "error": str(e)}


async def run_concurrent_async(
    api_url: str, corpus: str, questions: list[dict], bypass_cache: bool
) -> dict:
    """Fire N independent requests concurrently. Returns aggregate timing."""
    tasks = []
    wall_start = time.perf_counter()

    if HAS_AIOHTTP:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as session:
            for q in questions:
                payload = {
                    "corpus_url": corpus,
                    "questions": [q],
                    "bypass_cache": bypass_cache,
                }
                tasks.append(_fetch_one(session, api_url, payload, q["id"]))
            results = await asyncio.gather(*tasks)
    else:
        async with httpx.AsyncClient(timeout=600.0) as client:
            for q in questions:
                payload = {
                    "corpus_url": corpus,
                    "questions": [q],
                    "bypass_cache": bypass_cache,
                }
                tasks.append(_fetch_one(client, api_url, payload, q["id"]))
            results = await asyncio.gather(*tasks)

    wall_elapsed = time.perf_counter() - wall_start
    errors = [r for r in results if r.get("error")]
    times = [r["elapsed"] for r in results if not r.get("error")]

    return {
        "wall_elapsed": wall_elapsed,
        "slowest": max(times) if times else 0,
        "fastest": min(times) if times else 0,
        "errors": len(errors),
        "error_details": [r["error"] for r in errors] if errors else None,
    }


def run_concurrent(
    api_url: str, corpus: str, questions: list[dict], bypass_cache: bool
) -> dict:
    """Sync wrapper for concurrent test."""
    return asyncio.run(
        run_concurrent_async(api_url, corpus, questions, bypass_cache)
    )


def warmup(api_url: str, corpus: str):
    """Send a single-question batch to warm corpus_cache."""
    print("Warming up (single question, bypass_cache=False)...")
    payload = {
        "corpus_url": corpus,
        "questions": [{"id": "warmup", "text": "What is this corpus about?"}],
        "bypass_cache": False,
    }
    start = time.perf_counter()
    try:
        resp = requests.post(f"{api_url}/challenge/run", json=payload, timeout=600)
        resp.raise_for_status()
        elapsed = time.perf_counter() - start
        print(f"Warmup complete in {elapsed:.1f}s\n")
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"Warmup failed after {elapsed:.1f}s: {e}\n")


def estimate_corpus_size(corpus_path: str) -> str:
    """Get corpus file size for display."""
    try:
        size_mb = os.path.getsize(corpus_path) / 1024 / 1024
        return f"~{size_mb:.0f}MB"
    except OSError:
        return "?"


def main():
    parser = argparse.ArgumentParser(description="Lucio Stress Test Runner")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--corpus", default=None, help="Corpus zip path or URL")
    parser.add_argument(
        "--mode",
        choices=["batch", "concurrent", "both"],
        default="batch",
        help="Test mode (default: batch)",
    )
    parser.add_argument(
        "--warmup", action="store_true", help="Warm caches before testing"
    )
    parser.add_argument(
        "--skip-cold", action="store_true", help="Skip cold run (bypass_cache=True)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Pass/fail threshold in seconds (default: 30)",
    )
    args = parser.parse_args()

    # Resolve corpus
    corpus = args.corpus
    if not corpus:
        if DEFAULT_STRESS_CORPUS.exists():
            corpus = str(DEFAULT_STRESS_CORPUS)
        else:
            with open(GROUND_TRUTH) as f:
                gt = json.load(f)
            corpus = gt["corpus_url"]
            print(
                f"Note: stress_corpus.zip not found, using default corpus.\n"
                f"Run: python eval/build_stress_corpus.py\n"
            )

    questions = load_questions()
    corpus_size = estimate_corpus_size(corpus)
    threshold = args.threshold
    modes = (
        ["batch", "concurrent"]
        if args.mode == "both"
        else [args.mode]
    )

    # Header
    print("=" * 64)
    print(f"Lucio Stress Test — {len(questions)} questions, corpus {corpus_size}")
    print(f"Threshold: {threshold:.0f}s | Modes: {', '.join(modes)}")
    print("=" * 64)

    if args.warmup:
        warmup(args.api, corpus)

    results = []

    for mode in modes:
        if not args.skip_cold:
            # Cold run
            label = f"[{'BATCH' if mode == 'batch' else 'CONC'}]  Cold"
            print(f"\n{label}: running...", end="", flush=True)

            if mode == "batch":
                r = run_batch(args.api, corpus, questions, bypass_cache=True)
                elapsed = r["elapsed"]
                status = "PASS" if elapsed <= threshold else "FAIL"
                phases = format_phase_times(r.get("phase_times"))
                err = r.get("error")
                if err:
                    print(f"\r{label}:   ERROR  — {err}")
                    results.append(("cold", mode, None, False, err))
                else:
                    print(f"\r{label}:   {elapsed:5.1f}s  {status}  | {phases}")
                    results.append(("cold", mode, elapsed, elapsed <= threshold, None))
            else:
                r = run_concurrent(args.api, corpus, questions, bypass_cache=True)
                elapsed = r["wall_elapsed"]
                status = "PASS" if elapsed <= threshold else "FAIL"
                errs = r.get("errors", 0)
                err_note = f" ({errs} errors)" if errs else ""
                print(
                    f"\r{label}:   {elapsed:5.1f}s  {status}{err_note}  "
                    f"| slowest={r['slowest']:.1f} fastest={r['fastest']:.1f}"
                )
                results.append(("cold", mode, elapsed, elapsed <= threshold, None))

        # Cached run
        label = f"[{'BATCH' if mode == 'batch' else 'CONC'}]  Cached"
        print(f"\n{label}: running...", end="", flush=True)

        if mode == "batch":
            r = run_batch(args.api, corpus, questions, bypass_cache=False)
            elapsed = r["elapsed"]
            status = "PASS" if elapsed <= threshold else "FAIL"
            phases = format_phase_times(r.get("phase_times"))
            err = r.get("error")
            if err:
                print(f"\r{label}: ERROR  — {err}")
                results.append(("cached", mode, None, False, err))
            else:
                print(f"\r{label}: {elapsed:5.1f}s  {status}  | {phases}")
                results.append(("cached", mode, elapsed, elapsed <= threshold, None))
        else:
            r = run_concurrent(args.api, corpus, questions, bypass_cache=False)
            elapsed = r["wall_elapsed"]
            status = "PASS" if elapsed <= threshold else "FAIL"
            errs = r.get("errors", 0)
            err_note = f" ({errs} errors)" if errs else ""
            print(
                f"\r{label}: {elapsed:5.1f}s  {status}{err_note}  "
                f"| slowest={r['slowest']:.1f} fastest={r['fastest']:.1f}"
            )
            results.append(("cached", mode, elapsed, elapsed <= threshold, None))

    # Summary
    passed = sum(1 for r in results if r[3])
    total = len(results)
    print(f"\n{'=' * 64}")
    print(f"Overall: {passed}/{total} passed (threshold: {threshold:.0f}s)")
    print("=" * 64)


if __name__ == "__main__":
    main()
