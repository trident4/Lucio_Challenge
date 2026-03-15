#!/usr/bin/env python3
"""Excel/JSON → ground_truth.json converter.

Reads an Excel (.xlsx) or JSON file with Question/Answer pairs and converts
them directly to ground_truth.json for automated evaluation. Each answer
becomes a simple "contains" assertion.

Usage:
    python eval/generate_ground_truth.py /path/to/questions.xlsx
    python eval/generate_ground_truth.py /path/to/questions.json --corpus "https://..."
    python eval/generate_ground_truth.py /path/to/questions.xlsx --dry-run
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import openpyxl

EVAL_DIR = Path(__file__).parent
DEFAULT_OUTPUT = EVAL_DIR / "ground_truth.json"


# ── Input Reading ────────────────────────────────────────────────────────────

QUESTION_VARIANTS = {"question", "questions", "q"}
ANSWER_VARIANTS = {"answer", "answers", "a", "reference answer", "expected answer"}


def _match_column(header: str, variants: set[str]) -> bool:
    return header.strip().lower() in variants


def read_input(path: str) -> list[dict]:
    """Read Excel or JSON file and return list of {question, answer} dicts."""
    if path.lower().endswith(".json"):
        return _read_json(path)
    return _read_excel(path)


def _read_json(path: str) -> list[dict]:
    """Read JSON file — supports multiple formats:

    Format 1: [{"question": "...", "answer": "..."}, ...]
    Format 2: [{"q": "...", "a": "..."}, ...]
    Format 3: [{"text": "...", "expected": "..."}, ...]
    """
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Could not parse JSON — must be an array of objects.")
        sys.exit(1)

    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = item.get("question") or item.get("text") or item.get("q") or ""
        a = item.get("answer") or item.get("expected") or item.get("a") or ""
        q, a = str(q).strip(), str(a).strip()
        if q and a:
            rows.append({"question": q, "answer": a})
    return rows


def _read_excel(path: str) -> list[dict]:
    """Read Excel file and return list of {question, answer} dicts."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.active

    headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    col_map = {}
    for idx, h in enumerate(headers):
        if h is None:
            continue
        h_str = str(h)
        if _match_column(h_str, QUESTION_VARIANTS):
            col_map["question"] = idx
        elif _match_column(h_str, ANSWER_VARIANTS):
            col_map["answer"] = idx

    if "question" not in col_map:
        print(f"Could not find a 'Question' column. Found headers: {headers}")
        sys.exit(1)
    if "answer" not in col_map:
        print(f"Could not find an 'Answer' column. Found headers: {headers}")
        sys.exit(1)

    rows = []
    max_col = max(col_map.values())
    for row in ws.iter_rows(min_row=2, values_only=True):
        if len(row) <= max_col:
            continue
        q = row[col_map["question"]]
        a = row[col_map["answer"]]
        if not q or not a:
            continue
        rows.append({"question": str(q).strip(), "answer": str(a).strip()})

    wb.close()
    return rows


# ── Assertion Generation (no LLM — use answer text directly) ────────────────


def make_assertions(answer: str) -> list[dict]:
    """Convert an answer string into contains assertions.

    Splits on common delimiters (commas, semicolons, newlines) to create
    multiple fine-grained assertions when the answer has multiple facts.
    Single-fact answers get one assertion.
    """
    answer = answer.strip()
    if not answer:
        return []

    # Check for "not available" style answers
    not_available = ["not available", "not provided", "not in the document", "not mentioned"]
    if any(phrase in answer.lower() for phrase in not_available):
        return [{
            "type": "contains_any",
            "values": ["not available", "not provided", "not exist", "not mentioned"],
            "label": "Correctly states information is missing",
        }]

    # For short answers (single fact), use the whole answer
    if len(answer) < 80 and "," not in answer and ";" not in answer:
        return [{"type": "contains", "value": answer, "label": answer}]

    # For longer answers, use the full answer as one assertion
    return [{"type": "contains", "value": answer, "label": answer[:80]}]


# ── Assembly & Output ────────────────────────────────────────────────────────


def build_ground_truth(rows: list[dict], corpus_url: str) -> dict:
    questions = []
    for i, row in enumerate(rows, start=1):
        questions.append({
            "id": f"q{i}",
            "text": row["question"],
            "assertions": make_assertions(row["answer"]),
        })
    return {"corpus_url": corpus_url, "questions": questions}


def print_summary(gt: dict) -> None:
    questions = gt["questions"]
    total_assertions = sum(len(q["assertions"]) for q in questions)
    print(f"\nGenerated ground_truth.json: {len(questions)} questions, "
          f"{total_assertions} assertions\n")

    print(f"{'Q':<5}| {'#':>3} | Question / Assertion")
    print("-" * 70)
    for q in questions:
        a = q["assertions"]
        q_preview = q["text"][:55]
        a_preview = ""
        if a:
            if a[0]["type"] == "contains":
                a_preview = a[0]["value"][:40]
            else:
                a_preview = str(a[0].get("values", []))[:40]
        print(f"{q['id']:<5}| {len(a):>3} | {q_preview}")
        print(f"{'':5}|     | -> {a_preview}")
    print()


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Convert Excel/JSON questions+answers to ground_truth.json"
    )
    parser.add_argument("input_path", help="Path to Excel (.xlsx) or JSON file")
    parser.add_argument(
        "--corpus", default=None,
        help="Corpus URL/path (default: read from existing ground_truth.json)",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT),
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print generated output without writing file",
    )
    args = parser.parse_args()

    # Resolve corpus URL
    corpus_url = args.corpus
    if corpus_url is None:
        if DEFAULT_OUTPUT.exists():
            with open(DEFAULT_OUTPUT) as f:
                corpus_url = json.load(f).get("corpus_url", "")
            print(f"Using corpus URL from existing ground_truth.json: {corpus_url}")
        else:
            corpus_url = ""
            print("No corpus URL provided and no existing ground_truth.json found.")

    # Read input
    print(f"Reading {args.input_path}...")
    rows = read_input(args.input_path)
    if not rows:
        print("No valid questions found.")
        sys.exit(1)
    print(f"Found {len(rows)} questions")

    # Build ground truth
    gt = build_ground_truth(rows, corpus_url)
    print_summary(gt)

    if args.dry_run:
        print("Dry run — full output:\n")
        print(json.dumps(gt, indent=2))
        return

    # Backup existing file
    output_path = Path(args.output)
    if output_path.exists():
        backup_path = output_path.with_suffix(".backup.json")
        shutil.copy2(output_path, backup_path)
        print(f"Backed up {output_path.name} -> {backup_path.name}")

    # Write output
    output_path.write_text(json.dumps(gt, indent=2) + "\n")
    print(f"Written to {output_path}")


if __name__ == "__main__":
    main()
