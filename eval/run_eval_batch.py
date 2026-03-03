#!/usr/bin/env python3
"""RAG Pipeline Evaluation Runner - Batched (Mimics index.html)

Sends all questions in a single HTTP POST request. This is the most efficient
way to run evaluations as it deduplicates all RAG pipeline stages (Phase 1-3)
and allows the backend to orchestrate LLM inference (Phase 5) more safely.

Usage:
    python eval/run_eval_batch.py
"""

import argparse
import json
import requests
import time
from pathlib import Path

EVAL_DIR = Path(__file__).parent
GROUND_TRUTH = EVAL_DIR / "ground_truth.json"


def check_contains(answer: str, assertion: dict) -> bool:
    return assertion["value"].lower() in answer.lower()


def check_contains_any(answer: str, assertion: dict) -> bool:
    lower = answer.lower()
    return any(v.lower() in lower for v in assertion["values"])


CHECKERS = {
    "contains": check_contains,
    "contains_any": check_contains_any,
}


def main(api_url: str, corpus_url: str, include_battle: bool):
    # Load ground truth
    with open(GROUND_TRUTH) as f:
        gt = json.load(f)

    corpus = corpus_url if corpus_url else gt["corpus_url"]
    questions = gt["questions"]
    if include_battle:
        questions += gt.get("battle_test_questions", [])

    print(
        f"🚀 Firing BATCH request with {len(questions)} questions to {api_url}/challenge/run"
    )
    print(f"   Corpus: {corpus}\n")

    payload = {
        "corpus_url": corpus,
        "questions": [{"id": q["id"], "text": q["text"]} for q in questions],
    }

    start_total = time.perf_counter()
    try:
        response = requests.post(f"{api_url}/challenge/run", json=payload, timeout=600)
        response.raise_for_status()
        data = response.json()
    except Exception as e:
        print(f"❌ Batch Request Failed: {e}")
        return

    elapsed_total = time.perf_counter() - start_total

    # Run Assertions
    total_pass = 0
    total_assertions = 0
    question_results = []

    # Map q_id -> result
    result_map = {res["question_id"]: res for res in data.get("results", [])}

    for q in questions:
        q_id = q["id"]
        res = result_map.get(q_id)

        if not res:
            question_results.append(
                {
                    "id": q_id,
                    "status": "💥 MISSING",
                    "passed": 0,
                    "total": len(q["assertions"]),
                    "details": "No result in batch response",
                }
            )
            continue

        answer_text = res["answer"]
        passed = 0
        failed_labels = []

        for assertion in q["assertions"]:
            total_assertions += 1
            checker = CHECKERS.get(assertion["type"])
            if checker and checker(answer_text, assertion):
                passed += 1
                total_pass += 1
            else:
                failed_labels.append(assertion["label"])

        total = len(q["assertions"])
        status = (
            "✅ PASS"
            if passed == total
            else ("⚠️  PARTIAL" if passed > 0 else "❌ FAIL")
        )

        question_results.append(
            {
                "id": q_id,
                "status": status,
                "passed": passed,
                "total": total,
                "answer": answer_text,
                "details": (
                    "Missing: " + ", ".join(failed_labels) if failed_labels else ""
                ),
            }
        )

    # Print Summary
    print("=" * 100)
    print(f"{'Q':>4}  {'Status':<12}  {'Score':>7}  {'Details'}")
    print("-" * 100)
    for qr in question_results:
        print(
            f"{qr['id']:>4}  {qr['status']:<12}  {qr['passed']}/{qr['total']:>2}  {qr['details']}"
        )
        if qr["status"] != "✅ PASS" and "answer" in qr:
            print(f"      [ANSWER]: {qr['answer']}\n")
    print("-" * 100)

    pct = (total_pass / total_assertions * 100) if total_assertions else 0
    time_status = "✓ UNDER 30s" if elapsed_total < 30 else "⚠ OVER 30s!"
    print(
        f"TOTAL: {total_pass}/{total_assertions} ({pct:.0f}%)   ⏱ {elapsed_total:.1f}s {time_status}"
    )
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--corpus", default="")
    parser.add_argument(
        "--battle", action="store_true", help="Include battle test questions"
    )
    args = parser.parse_args()
    main(args.api, args.corpus, args.battle)
