#!/usr/bin/env python3
"""Hackathon runner — reads questions from CSV, sends to API, outputs submission.json.

Usage:
    python eval/run_hackathon.py --questions questions.csv --corpus /path/to/data.zip --password "secret"
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

EVAL_DIR = Path(__file__).parent


def run(api_url: str, questions_csv: str, corpus_path: str, password: str = "") -> None:
    # Read questions from CSV
    questions = []
    with open(questions_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q_id = row.get("ID") or row.get("id")
            q_text = row.get("Questions") or row.get("questions") or row.get("Question") or row.get("question")
            if q_id and q_text:
                questions.append({"id": str(q_id), "text": q_text.strip()})

    if not questions:
        print("❌ No questions found in CSV. Expected columns: ID, Questions")
        sys.exit(1)

    # Build API request
    payload = {
        "corpus_url": corpus_path,
        "questions": questions,
    }
    if password:
        payload["password"] = password

    print(f"🚀 Sending {len(questions)} questions to {api_url}/challenge/run")
    print(f"   Corpus: {corpus_path}")
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
        if hasattr(e, "read"):
            body = e.read().decode(errors="replace")[:500]
            print(f"   Response: {body}")
        print(f"   Is the server running? (uvicorn app.main:app --reload)")
        sys.exit(1)

    elapsed = time.perf_counter() - start
    answers = {r["question_id"]: r for r in response["results"]}

    # Print answers
    print("=" * 70)
    for q in questions:
        result = answers.get(q["id"])
        answer_preview = result["answer"][:200] if result else "NO ANSWER"
        print(f"  Q{q['id']}: {q['text'][:60]}")
        print(f"  A:  {answer_preview}")
        print()
    print("=" * 70)
    time_status = "✓ UNDER 30s" if elapsed < 30 else "⚠ OVER 30s!"
    print(f"  ⏱ {elapsed:.1f}s {time_status}")
    print("=" * 70)

    # Save submission.json — flatten citations to {document_id, page} per page
    def flatten_citations(sources):
        flat = []
        for src in sources:
            for page in src.get("pages", []):
                flat.append({"document_id": src["filename"], "page": page})
        return flat

    submission = {
        "answers": [
            {
                "question_id": q["id"],
                "answer": answers[q["id"]]["answer"],
                "citations": flatten_citations(answers[q["id"]].get("sources", [])),
            }
            for q in questions
            if q["id"] in answers
        ]
    }
    submission_path = EVAL_DIR / "submission.json"
    with open(submission_path, "w") as f:
        json.dump(submission, f, indent=2)

    print(f"\n📤 Submission saved to {submission_path}")
    print(f"   {len(submission['answers'])} answers")

    return submission


def submit(submit_url: str, api_key: str) -> None:
    """POST submission.json to the hackathon submission API."""
    submission_path = EVAL_DIR / "submission.json"
    if not submission_path.exists():
        print("❌ No submission.json found. Run questions first.")
        sys.exit(1)

    with open(submission_path) as f:
        payload = json.load(f)

    print(f"\n📤 Submitting {len(payload.get('answers', []))} answers to {submit_url}")
    try:
        req = Request(
            submit_url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "X-API-Key": api_key,
            },
            method="POST",
        )
        with urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        print(f"✅ Submission response:")
        print(json.dumps(result, indent=2))
    except URLError as e:
        print(f"❌ Submission failed: {e}")
        sys.exit(1)


SUBMIT_URL = "https://luciohackathon.purplewater-eec0a096.centralindia.azurecontainerapps.io/submissions"
API_KEY = "c4ee4023-42c9-462f-a739-d0d158292f86"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hackathon Runner")
    parser.add_argument("--api", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--questions", required=True, help="Path to questions CSV")
    parser.add_argument("--corpus", required=True, help="Path to corpus zip")
    parser.add_argument("--password", default="", help="Password for encrypted zip")
    parser.add_argument("--dry-run", action="store_true", help="Just print questions from CSV and exit")
    parser.add_argument("--submit", action="store_true", help="Submit answers after running")
    args = parser.parse_args()

    if args.dry_run:
        questions = []
        with open(args.questions, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q_id = row.get("ID") or row.get("id")
                q_text = row.get("Questions") or row.get("questions") or row.get("Question") or row.get("question")
                if q_id and q_text:
                    questions.append((q_id, q_text.strip()))
        print(f"📋 {len(questions)} questions from {args.questions}\n")
        for q_id, q_text in questions:
            print(f"  Q{q_id}: {q_text}")
        sys.exit(0)

    run(args.api, args.questions, args.corpus, args.password)

    if args.submit:
        submit(SUBMIT_URL, API_KEY)
