"""
monitor.py — Step 4
===================
Drift Detection & Quality Monitoring.

Reads historical eval scores from data/scores.json, computes a rolling
7-day average, and fires an alert if quality drops below the threshold.

This is the MLOps layer — it answers: "Is our bot getting worse over time?"

Run:
    python src/monitor.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────
SCORES_FILE          = Path(__file__).parent.parent / "data" / "scores.json"
DRIFT_THRESHOLD      = 0.70   # Alert if rolling avg drops below this
ROLLING_WINDOW_DAYS  = 7      # Days to include in rolling window
# ───────────────────────────────────────────────────────────────────────────


def load_scores() -> list[dict]:
    """Load all historical eval scores."""
    if not SCORES_FILE.exists():
        return []
    with open(SCORES_FILE) as f:
        return json.load(f)


def parse_timestamp(ts: str) -> datetime:
    """Parse ISO timestamp string to datetime (UTC-aware)."""
    ts = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(ts)


def compute_rolling_average(scores: list[dict], window_days: int) -> dict:
    """Compute rolling average for the last N days."""
    if not scores:
        return {"average": None, "count": 0, "records": []}

    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    recent = [
        s for s in scores
        if parse_timestamp(s["timestamp"]) >= cutoff
    ]

    if not recent:
        return {"average": None, "count": 0, "records": []}

    avg = sum(r["average"] for r in recent) / len(recent)
    return {
        "average": round(avg, 4),
        "count":   len(recent),
        "records": recent,
    }


def compute_trend(scores: list[dict]) -> str:
    """Simple trend: compare last 3 scores to previous 3 scores."""
    if len(scores) < 6:
        return "not enough data for trend"

    recent_avg   = sum(s["average"] for s in scores[-3:]) / 3
    previous_avg = sum(s["average"] for s in scores[-6:-3]) / 3

    delta = recent_avg - previous_avg
    if delta > 0.05:
        return f"↑ improving (+{delta:.2f})"
    elif delta < -0.05:
        return f"↓ declining ({delta:.2f})"
    else:
        return f"→ stable ({delta:+.2f})"


def print_score_table(scores: list[dict], limit: int = 10):
    """Print a formatted table of recent scores."""
    recent = scores[-limit:]
    print(f"\n{'Timestamp':<26} {'Faithfulness':>12} {'Relevance':>10} {'Resolution':>10} {'Average':>8}")
    print("─" * 72)
    for s in recent:
        ts = parse_timestamp(s["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        print(
            f"{ts:<26} "
            f"{s['faithfulness']:>12.2f} "
            f"{s['relevance']:>10.2f} "
            f"{s['resolution']:>10.2f} "
            f"{s['average']:>8.2f}"
        )


def detect_drift(scores: list[dict]) -> dict:
    """Main drift detection logic. Returns a report dict."""
    rolling = compute_rolling_average(scores, ROLLING_WINDOW_DAYS)
    trend   = compute_trend(scores)

    # Overall all-time average
    all_time_avg = (
        sum(s["average"] for s in scores) / len(scores)
        if scores else None
    )

    drift_detected = (
        rolling["average"] is not None
        and rolling["average"] < DRIFT_THRESHOLD
    )

    return {
        "total_evaluations": len(scores),
        "all_time_average":  round(all_time_avg, 4) if all_time_avg else None,
        "rolling_average":   rolling["average"],
        "rolling_count":     rolling["count"],
        "rolling_window":    ROLLING_WINDOW_DAYS,
        "threshold":         DRIFT_THRESHOLD,
        "trend":             trend,
        "drift_detected":    drift_detected,
    }


def print_report(report: dict):
    """Print a formatted drift report to the console."""
    print("\n" + "=" * 55)
    print("  Support Bot — Drift Monitor")
    print("=" * 55)

    print(f"  Total evaluations tracked:  {report['total_evaluations']}")

    if report["all_time_average"] is not None:
        print(f"  All-time average score:     {report['all_time_average']:.2f}")

    if report["rolling_average"] is not None:
        print(
            f"  {report['rolling_window']}-day rolling average:     "
            f"{report['rolling_average']:.2f}  "
            f"(from {report['rolling_count']} evaluations)"
        )
    else:
        print(f"  {report['rolling_window']}-day rolling average:     no data yet")

    print(f"  Threshold:                  {report['threshold']:.2f}")
    print(f"  Trend:                      {report['trend']}")
    print()

    if report["rolling_average"] is None:
        print("  ℹ Status: No evaluations in the rolling window yet.")
        print("           Run python src/evaluate.py \"question\" to add scores.")
    elif report["drift_detected"]:
        print("  ⚠  DRIFT DETECTED")
        print(f"     Rolling average ({report['rolling_average']:.2f}) dropped below")
        print(f"     threshold ({report['threshold']:.2f}).")
        print()
        print("  Recommended actions:")
        print("    1. Check if your docs/ folder is up to date")
        print("    2. Re-run python src/ingest.py to refresh the vector store")
        print("    3. Review failing queries in your Respan traces dashboard")
        print("    4. Revise your system prompt in rag.py if needed")
        print()
        print("  Respan Dashboard:")
        print("    https://platform.respan.ai/platform/traces")
    else:
        print(f"  ✓  Quality looks good (avg {report['rolling_average']:.2f} ≥ threshold {report['threshold']:.2f})")

    print("=" * 55)


def main():
    print("=" * 55)
    print("  Support Bot — Drift Detection & Monitoring")
    print("=" * 55)

    scores = load_scores()

    if not scores:
        print("\nNo evaluation scores found yet.")
        print("Run some evaluations first:")
        print('  python src/evaluate.py "How do I reset my password?"')
        print('  python src/evaluate.py "How do I cancel my subscription?"')
        sys.exit(0)

    # Print score history table
    print(f"\nLast {min(10, len(scores))} evaluation records:")
    print_score_table(scores)

    # Compute and print drift report
    report = detect_drift(scores)
    print_report(report)

    # Exit with error code if drift detected (useful in CI pipelines)
    if report["drift_detected"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
