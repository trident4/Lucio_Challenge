#!/usr/bin/env python3
import json
import requests
import time
from pathlib import Path

# Configuration
API_URL = "http://127.0.0.1:8000/challenge/run"
GROUND_TRUTH = Path(__file__).parent / "ground_truth.json"

MODELS = [
    {
        "name": "GPT-4o-mini",
        "path": "openai/gpt-4o-mini",
        "cost_per_1m": 0.20,
        "top_k": 8,
    },
    {
        "name": "Claude 3 Haiku",
        "path": "anthropic/claude-3-haiku",
        "cost_per_1m": 0.35,
        "top_k": 8,
    },
    {
        "name": "Mistral Nemo",
        "path": "mistralai/mistral-nemo",
        "cost_per_1m": 0.15,
        "top_k": 8,
    },
    {
        "name": "Llama 3.1 8B",
        "path": "meta-llama/llama-3.1-8b-instruct",
        "cost_per_1m": 0.05,
        "top_k": 8,
    },
    {
        "name": "Nemo (Justice-12)",
        "path": "mistralai/mistral-nemo",
        "cost_per_1m": 0.15,
        "top_k": 12,
    },
]


def check_contains(answer: str, assertion: dict) -> bool:
    return assertion["value"].lower() in answer.lower()


def check_contains_any(answer: str, assertion: dict) -> bool:
    lower = answer.lower()
    return any(v.lower() in lower for v in assertion["values"])


CHECKERS = {
    "contains": check_contains,
    "contains_any": check_contains_any,
}


def run_benchmark():
    with open(GROUND_TRUTH) as f:
        gt = json.load(f)

    questions = gt["questions"]
    corpus_url = gt["corpus_url"]

    print(f"🚀 Starting Lucio Model Leaderboard Benchmark ({len(questions)} questions)")
    print("-" * 80)

    results = []

    for model in MODELS:
        print(f"Testing {model['name']} ({model['path']})...")

        payload = {
            "corpus_url": corpus_url,
            "questions": [{"id": q["id"], "text": q["text"]} for q in questions],
            "llm_model": model["path"],
            "rerank_top_k": model["top_k"],
            "bypass_cache": True,
        }

        try:
            start_time = time.perf_counter()
            resp = requests.post(API_URL, json=payload, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            total_time = data.get("total_time", time.perf_counter() - start_time)
            total_tokens = data.get("total_tokens", 0)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            continue

        # Accuracy Check
        correct_questions = 0
        total_assertions = 0
        passed_assertions = 0
        model_failures = []

        result_map = {res["question_id"]: res for res in data.get("results", [])}

        for q in questions:
            res = result_map.get(q["id"])
            if not res:
                model_failures.append({"q_id": q["id"], "error": "No result in batch"})
                continue

            q_passed = True
            q_failures = []
            for assertion in q["assertions"]:
                total_assertions += 1
                checker = CHECKERS.get(assertion["type"])
                if checker and checker(res["answer"], assertion):
                    passed_assertions += 1
                else:
                    q_passed = False
                    q_failures.append(assertion["label"])

            if q_passed:
                correct_questions += 1
            else:
                model_failures.append(
                    {
                        "q_id": q["id"],
                        "failed_assertions": q_failures,
                        "answer": res["answer"],
                    }
                )

        accuracy = (
            (passed_assertions / total_assertions * 100) if total_assertions else 0
        )
        cost = (total_tokens / 1_000_000) * model["cost_per_1m"]

        results.append(
            {
                "Model": model["name"],
                "K": model["top_k"],
                "Accuracy": f"{accuracy:.1f}%",
                "Correct Qs": f"{correct_questions}/{len(questions)}",
                "Latency": f"{total_time:.2f}s",
                "Tokens": f"{total_tokens:,}",
                "Est. Cost": f"${cost:.4f}",
                "Failures": model_failures,
            }
        )

    # Print Report to console
    print("\n" + "=" * 80)
    print(
        f"{'MODEL':<18} | {'K':<2} | {'ACC%':<7} | {'Qs':<7} | {'TIME':<8} | {'TOKENS':<8} | {'COST'}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['Model']:<18} | {r['K']:<2} | {r['Accuracy']:<7} | {r['Correct Qs']:<7} | {r['Latency']:<8} | {r['Tokens']:<8} | {r['Est. Cost']}"
        )
    print("=" * 80)

    # Save Debug Info (Failures)
    debug_path = Path(__file__).parent / "model_debug_failures.json"
    with open(debug_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save to Markdown Report
    report_path = Path(__file__).parent / "model_benchmark_report.md"
    with open(report_path, "w") as f:
        f.write("# 🏆 Lucio Model Leaderboard Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("| MODEL | K | ACC% | Qs | TIME | TOKENS | EST. COST |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            f.write(
                f"| {r['Model']} | {r['K']} | {r['Accuracy']} | {r['Correct Qs']} | {r['Latency']} | {r['Tokens']} | {r['Est. Cost']} |\n"
            )

        f.write("\n\n---\n*Report generated automatically by `eval/compare_models.py`*")

    print(f"\n✅ Detailed report saved to: {report_path}")
    print(f"🐞 Debug failures saved to: {debug_path}")


if __name__ == "__main__":
    run_benchmark()
