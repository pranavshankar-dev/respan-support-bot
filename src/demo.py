"""
demo.py — Full End-to-End Demo
==============================
Runs 5 sample questions through the complete pipeline:
  RAG → Eval (3 judges) → Drift Monitor

Use this to quickly populate your Respan dashboard with real traces and scores.

Run:
    python src/demo.py
"""

import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import evaluate
from monitor import load_scores, detect_drift, print_report, print_score_table

# ── Sample questions covering different doc sections ───────────────────────
DEMO_QUESTIONS = [
    "How do I reset my password?",
    "How do I cancel my subscription?",
    "What are the API rate limits for the Pro plan?",
    "How do I enable two-factor authentication?",
    "How do I get a refund?",
]
# ───────────────────────────────────────────────────────────────────────────


def run_demo():
    print("\n" + "=" * 60)
    print("  Support Bot — Full End-to-End Demo")
    print("  Running 5 questions through RAG + Eval + Monitor")
    print("=" * 60)

    results = []

    for i, question in enumerate(DEMO_QUESTIONS, 1):
        print(f"\n{'─' * 60}")
        print(f"  Question {i}/{len(DEMO_QUESTIONS)}")
        print(f"{'─' * 60}")

        result = evaluate(question)
        results.append(result)

        # Brief pause between calls to be respectful of rate limits
        if i < len(DEMO_QUESTIONS):
            time.sleep(2)

    # Final summary
    print("\n" + "=" * 60)
    print("  Demo Complete — Summary")
    print("=" * 60)

    avg_scores = [r["average"] for r in results]
    overall_avg = sum(avg_scores) / len(avg_scores)

    print(f"\n  Questions evaluated: {len(results)}")
    print(f"  Overall average:     {overall_avg:.2f}")
    print()
    for i, (q, r) in enumerate(zip(DEMO_QUESTIONS, results), 1):
        print(f"  Q{i}: {q[:50]:<52} avg={r['average']:.2f}")

    # Run drift monitor
    print()
    scores = load_scores()
    report = detect_drift(scores)
    print_report(report)

    print("\n  ✓ All traces logged to Respan Dashboard:")
    print("    https://platform.respan.ai/platform/traces")
    print()
    print("  ✓ Scores saved to data/scores.json")
    print("    Re-run python src/monitor.py at any time to check for drift.")


if __name__ == "__main__":
    run_demo()
