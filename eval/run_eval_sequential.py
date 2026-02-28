#!/usr/bin/env python3
"""RAG Pipeline Evaluation Runner (Sequential).

Sends ground truth questions to the API ONE BY ONE, checks assertions,
prints a pass/fail table, and appends results to history.
This mimics the frontend UI perfectly by preventing batch-search retrieval interference.

Usage:
    # Server must be running first
    python eval/run_eval_sequential.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

EVAL_DIR = Path(__file__).parent
GROUND_TRUTH = EVAL_DIR / "ground_truth.json"
HISTORY_FILE = EVAL_DIR / "history.jsonl"


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


# ── Main ────────────────────────────────────────────────────────────────────


def run_eval_sequential(api_url: str, corpus_url: str) -> dict:
    """Run evaluation and return results dict."""

    # Load ground truth
    with open(GROUND_TRUTH) as f:
        gt = json.load(f)

    # Override corpus URL if provided
    if corpus_url:
        gt["corpus_url"] = corpus_url

    print(
        f"🚀 Sending {len(gt['questions'])} questions to {api_url}/challenge/run SEQUENTIALLY"
    )
    print(f"   Corpus: {gt['corpus_url']}")
    print()

    total_pass = 0
    total_fail = 0
    total_assertions = 0
    question_results = []

    start_time_global = time.perf_counter()
    all_answers = {}

    for q in gt["battle_test_questions"]:
        q_id = q["id"]

        payload = {
            "corpus_url": gt["corpus_url"],
            "questions": [{"id": q_id, "text": q["text"]}],
        }

        # Call API
        try:
            req = Request(
                f"{api_url}/challenge/run",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(req, timeout=300) as resp:
                response = json.loads(resp.read())
                result = response["results"][0]
                answer_text = result["answer"]
                all_answers[q_id] = result
        except URLError as e:
            print(f"❌ Failed to connect to API on question {q_id}")
            print(f"   Error: {e}")
            sys.exit(1)
        except (KeyError, IndexError):
            print(f"⚠️  {q_id}: No answer returned!")
            question_results.append(
                {
                    "id": q_id,
                    "passed": 0,
                    "total": len(q["assertions"]),
                    "status": "MISSING",
                }
            )
            continue

        # ── Run Assertions ──────────────────────────────────────────────────
        passed = 0
        failed = 0
        failed_labels = []

        for assertion in q["assertions"]:
            checker = CHECKERS.get(assertion["type"])
            if not checker:
                continue

            total_assertions += 1
            if checker(answer_text, assertion):
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
            }
        )
        print(f"[{q_id}] {status} ({passed}/{total})")

    elapsed_global = time.perf_counter() - start_time_global

    # ── Print Results ───────────────────────────────────────────────────

    print("=" * 70)
    print(f"{'Q':>4}  {'Status':<12}  {'Score':>7}  {'Details'}")
    print("-" * 70)

    for qr in question_results:
        score = f"{qr['passed']}/{qr['total']}"
        details = ""
        if qr.get("failed_labels"):
            details = "Missing: " + ", ".join(qr["failed_labels"])
        print(f"{qr['id']:>4}  {qr['status']:<12}  {score:>7}  {details}")

    print("-" * 70)
    pct = (total_pass / total_assertions * 100) if total_assertions else 0
    time_status = "✓ UNDER 30s per batch"
    print(
        f"{'':>4}  {'TOTAL':<12}  {total_pass}/{total_assertions} ({pct:.0f}%)   ⏱ {elapsed_global:.1f}s Total Sequential Time"
    )
    print("=" * 70)

    # ── Save to History ─────────────────────────────────────────────────

    # Get git commit hash if available
    commit = "unknown"
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=EVAL_DIR.parent,
        )
        if result.returncode == 0:
            commit = result.stdout.strip()
    except Exception:
        pass

    # Get current branch
    branch = "unknown"
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            cwd=EVAL_DIR.parent,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
    except Exception:
        pass

    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "commit": commit,
        "branch": branch,
        "score": f"{total_pass}/{total_assertions}",
        "pct": round(pct, 1),
        "time_s": round(elapsed_global, 1),
        "per_question": {
            qr["id"]: {"passed": qr["passed"], "total": qr["total"]}
            for qr in question_results
        },
    }

    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(history_entry) + "\n")

    # ── Generate Markdown Report ────────────────────────────────────────

    report_path = EVAL_DIR / "latest_report.md"
    lines = [
        f"# Eval Report",
        f"",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')} (SEQUENTIAL RUN)  ",
        f"**Branch:** `{branch}` (`{commit}`)  ",
        f"**Score:** {total_pass}/{total_assertions} ({pct:.0f}%)  ",
        f"**Time:** {elapsed_global:.1f}s ",
        f"",
        f"| Q | Status | Score | Missing |",
        f"|---|--------|-------|---------|",
    ]

    for qr in question_results:
        score = f"{qr['passed']}/{qr['total']}"
        missing = ", ".join(qr.get("failed_labels", [])) or "—"
        lines.append(f"| {qr['id']} | {qr['status']} | {score} | {missing} |")

    lines.append(f"| **Total** | | **{total_pass}/{total_assertions}** | |")
    lines.append("")

    # Add raw answers for debugging
    lines.append("## Raw Answers")
    lines.append("")
    for q in gt["battle_test_questions"]:
        q_id = q["id"]
        result = all_answers.get(q_id)
        if result:
            answer_preview = result["answer"][:300].replace("\n", " ")
            lines.append(f"**{q_id}:** {answer_preview}...")
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\n📝 Results saved to {HISTORY_FILE}")
    print(f"📄 Report saved to {report_path}")
    print(f"   Commit: {commit} ({branch})")

    return history_entry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG Pipeline Eval Runner (Sequential)"
    )
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--corpus", default="", help="Override corpus path")
    args = parser.parse_args()

    run_eval_sequential(args.api, args.corpus)
