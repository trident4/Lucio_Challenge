#!/usr/bin/env python3
"""RAG Pipeline Evaluation Runner.

Sends all ground truth questions to the API, checks assertions,
prints a pass/fail table, and appends results to history.

Usage:
    # Server must be running first
    python eval/run_eval.py

    # Custom API URL or corpus path
    python eval/run_eval.py --api http://localhost:8000 --corpus /path/to/Archive.zip
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


def run_eval(api_url: str, corpus_url: str, password: str = "", gt_path: Path = GROUND_TRUTH) -> dict:
    """Run evaluation and return results dict."""

    # Load ground truth
    with open(gt_path) as f:
        gt = json.load(f)

    # Override corpus URL if provided
    if corpus_url:
        gt["corpus_url"] = corpus_url

    # Build API request
    payload = {
        "corpus_url": gt["corpus_url"],
        "questions": [{"id": q["id"], "text": q["text"]} for q in gt["questions"]],
    }
    if password:
        payload["password"] = password

    print(
        f"🚀 Sending {len(payload['questions'])} questions to {api_url}/challenge/run"
    )
    print(f"   Corpus: {gt['corpus_url']}")
    print()

    # Call API
    start = time.perf_counter()
    try:
        req = Request(
            f"{api_url}/challenge/run",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=300) as resp:
            response = json.loads(resp.read())
    except URLError as e:
        print(f"❌ Failed to connect to API at {api_url}")
        print(f"   Error: {e}")
        print(f"   Is the server running? (uvicorn app.main:app --reload)")
        sys.exit(1)

    elapsed = time.perf_counter() - start

    # Build answer lookup
    answers = {r["question_id"]: r for r in response["results"]}

    # ── Run Assertions ──────────────────────────────────────────────────

    total_pass = 0
    total_fail = 0
    total_assertions = 0
    question_results = []

    for q in gt["questions"]:
        q_id = q["id"]
        result = answers.get(q_id)
        if not result:
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

        answer_text = result["answer"]
        passed = 0
        failed = 0
        failed_labels = []

        for assertion in q["assertions"]:
            checker = CHECKERS.get(assertion["type"])
            if not checker:
                print(f"   ⚠️  Unknown assertion type: {assertion['type']}")
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
    time_status = "✓ UNDER 30s" if elapsed < 30 else "⚠ OVER 30s!"
    print(
        f"{'':>4}  {'TOTAL':<12}  {total_pass}/{total_assertions} ({pct:.0f}%)   ⏱ {elapsed:.1f}s {time_status}"
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
        "time_s": round(elapsed, 1),
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
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Branch:** `{branch}` (`{commit}`)  ",
        f"**Score:** {total_pass}/{total_assertions} ({pct:.0f}%)  ",
        f"**Time:** {elapsed:.1f}s {'✓ under 30s' if elapsed < 30 else '⚠ OVER 30s'}",
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
    for q in gt["questions"]:
        q_id = q["id"]
        result = answers.get(q_id)
        if result:
            answer_preview = result["answer"][:300].replace("\n", " ")
            lines.append(f"**{q_id}:** {answer_preview}...")
            lines.append("")

    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    # ── Generate Submission JSON ─────────────────────────────────────

    submission = {
        "answers": [
            {
                "question_id": q["id"],
                "answer": answers[q["id"]]["answer"],
                "citations": answers[q["id"]].get("sources", []),
            }
            for q in gt["questions"]
            if q["id"] in answers
        ]
    }
    submission_path = EVAL_DIR / "submission.json"
    with open(submission_path, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"\n📝 Results saved to {HISTORY_FILE}")
    print(f"📄 Report saved to {report_path}")
    print(f"📤 Submission saved to {submission_path}")
    print(f"   Commit: {commit} ({branch})")

    return history_entry


def submit_answers(submit_url: str) -> None:
    """POST submission.json to the submission API."""
    submission_path = EVAL_DIR / "submission.json"
    if not submission_path.exists():
        print("❌ No submission.json found. Run eval first.")
        sys.exit(1)

    with open(submission_path) as f:
        payload = json.load(f)

    print(f"\n📤 Submitting {len(payload.get('answers', []))} answers to {submit_url}")
    try:
        req = Request(
            submit_url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        print(f"✅ Submission response:")
        print(json.dumps(result, indent=2))
    except URLError as e:
        print(f"❌ Submission failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Pipeline Eval Runner")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--corpus", default="", help="Override corpus path")
    parser.add_argument("--password", default="", help="Password for encrypted zip files")
    parser.add_argument("--ground-truth", default=str(GROUND_TRUTH), help="Path to ground truth JSON (default: ground_truth.json)")
    parser.add_argument("--submit", default="", help="Submission API URL — POST answers after eval")
    args = parser.parse_args()

    run_eval(args.api, args.corpus, args.password, gt_path=Path(args.ground_truth))

    if args.submit:
        submit_answers(args.submit)
