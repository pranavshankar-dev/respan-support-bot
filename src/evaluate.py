"""
evaluate.py — Step 3
====================
LLM-as-judge evaluation pipeline.
Runs 3 automated evaluators on each RAG response:
  1. Faithfulness  — Is the answer grounded in the retrieved context? (no hallucinations)
  2. Relevance     — Did the retrieved chunks actually match the query?
  3. Resolution    — Did the answer fully resolve the user's question?

Each score is 0.0–1.0. Scores are saved to data/scores.json for drift monitoring.

Run:
    python src/evaluate.py "How do I reset my password?"
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

# Import RAG pipeline to get a response + context to evaluate
sys.path.insert(0, str(Path(__file__).parent))
from rag import answer_question, respan

# ── Config ─────────────────────────────────────────────────────────────────
SCORES_FILE    = Path(__file__).parent.parent / "data" / "scores.json"
EVAL_MODEL     = "llama-3.1-8b-instant"  # Free Groq model for judging
# ───────────────────────────────────────────────────────────────────────────

# Groq client for the judge (free, OpenAI-compatible)
judge_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)


# ── Evaluator Prompts ──────────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are an expert evaluator assessing whether an AI assistant's answer is faithful to the provided context.

CONTEXT (retrieved documents):
{context}

QUESTION:
{question}

AI ANSWER:
{answer}

TASK: Score the FAITHFULNESS of the answer on a scale of 0.0 to 1.0.

Faithfulness means: every claim in the answer is directly supported by the context.
- 1.0 = All claims are grounded in the context. No hallucinations.
- 0.7 = Most claims are grounded. Minor unsupported details.
- 0.4 = Some claims are grounded, but notable hallucinations present.
- 0.0 = Answer is mostly or entirely fabricated, not from context.

Respond with ONLY a JSON object in this exact format:
{{"score": 0.0, "reason": "one sentence explanation"}}"""


RELEVANCE_PROMPT = """You are an expert evaluator assessing whether retrieved document chunks are relevant to a user's question.

QUESTION:
{question}

RETRIEVED CHUNKS:
{context}

TASK: Score the RELEVANCE of the retrieved chunks on a scale of 0.0 to 1.0.

Relevance means: the chunks contain information that would help answer the question.
- 1.0 = All chunks are highly relevant to the question.
- 0.7 = Most chunks are relevant. One or two tangential.
- 0.4 = Mixed relevance. Some chunks are off-topic.
- 0.0 = Chunks are completely unrelated to the question.

Respond with ONLY a JSON object in this exact format:
{{"score": 0.0, "reason": "one sentence explanation"}}"""


RESOLUTION_PROMPT = """You are an expert evaluator assessing whether an AI assistant fully resolved a user's question.

QUESTION:
{question}

AI ANSWER:
{answer}

TASK: Score the RESOLUTION quality on a scale of 0.0 to 1.0.

Resolution means: the answer actually addresses and solves what the user was asking.
- 1.0 = Question is fully and clearly answered.
- 0.7 = Question is mostly answered. Minor gaps.
- 0.4 = Partial answer. Important parts of the question are left unaddressed.
- 0.0 = Answer does not address the question at all.

Respond with ONLY a JSON object in this exact format:
{{"score": 0.0, "reason": "one sentence explanation"}}"""


# ── Evaluator Functions ────────────────────────────────────────────────────

def run_judge(prompt: str, eval_name: str) -> dict:
    """Call the LLM judge and parse the JSON score response."""
    try:
        response = judge_client.chat.completions.create(
            model=EVAL_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        raw = response.choices[0].message.content.strip()

        # Parse JSON score
        parsed = json.loads(raw)
        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "")

        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        return {"score": score, "reason": reason, "status": "ok"}

    except json.JSONDecodeError:
        return {"score": 0.0, "reason": "Failed to parse judge response", "status": "error"}
    except Exception as e:
        return {"score": 0.0, "reason": str(e), "status": "error"}


def evaluate_faithfulness(question: str, answer: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(f"- {c['content']}" for c in chunks)
    prompt = FAITHFULNESS_PROMPT.format(
        context=context, question=question, answer=answer
    )
    return run_judge(prompt, "faithfulness")


def evaluate_relevance(question: str, chunks: list[dict]) -> dict:
    context = "\n\n".join(f"- {c['content']}" for c in chunks)
    prompt = RELEVANCE_PROMPT.format(context=context, question=question)
    return run_judge(prompt, "relevance")


def evaluate_resolution(question: str, answer: str) -> dict:
    prompt = RESOLUTION_PROMPT.format(question=question, answer=answer)
    return run_judge(prompt, "resolution")


# ── Score Persistence ──────────────────────────────────────────────────────

def save_score(query: str, scores: dict):
    """Append score record to data/scores.json for drift monitoring."""
    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Load existing scores
    if SCORES_FILE.exists():
        with open(SCORES_FILE) as f:
            history = json.load(f)
    else:
        history = []

    avg = (
        scores["faithfulness"]["score"]
        + scores["relevance"]["score"]
        + scores["resolution"]["score"]
    ) / 3.0

    record = {
        "timestamp":   datetime.utcnow().isoformat() + "Z",
        "query":       query,
        "faithfulness": scores["faithfulness"]["score"],
        "relevance":    scores["relevance"]["score"],
        "resolution":   scores["resolution"]["score"],
        "average":      round(avg, 4),
    }
    history.append(record)

    with open(SCORES_FILE, "w") as f:
        json.dump(history, f, indent=2)

    return record


# ── Main Evaluation Pipeline ───────────────────────────────────────────────

def evaluate(query: str) -> dict:
    """Run the full RAG + evaluation pipeline for a query."""

    # Step 1: Get RAG response
    print(f"\nQuery: {query}")
    print("Getting RAG response...")
    rag_result = answer_question(query)

    question = rag_result["query"]
    answer   = rag_result["answer"]
    chunks   = rag_result["chunks"]

    # Step 2: Run 3 evaluators
    print("\nRunning 3 LLM-as-judge evaluators...")

    scores = {}

    print("  [1/3] Faithfulness judge...", end=" ", flush=True)
    scores["faithfulness"] = evaluate_faithfulness(question, answer, chunks)
    status = "✓" if scores["faithfulness"]["status"] == "ok" else "✗"
    print(f"{status}  score={scores['faithfulness']['score']:.2f}")

    print("  [2/3] Relevance judge...", end=" ", flush=True)
    scores["relevance"] = evaluate_relevance(question, chunks)
    status = "✓" if scores["relevance"]["status"] == "ok" else "✗"
    print(f"{status}  score={scores['relevance']['score']:.2f}")

    print("  [3/3] Resolution judge...", end=" ", flush=True)
    scores["resolution"] = evaluate_resolution(question, answer)
    status = "✓" if scores["resolution"]["status"] == "ok" else "✗"
    print(f"{status}  score={scores['resolution']['score']:.2f}")

    # Step 3: Compute average & print summary
    avg = (
        scores["faithfulness"]["score"]
        + scores["relevance"]["score"]
        + scores["resolution"]["score"]
    ) / 3.0

    print("\n" + "─" * 50)
    print(f"  Faithfulness:  {scores['faithfulness']['score']:.2f}  — {scores['faithfulness']['reason']}")
    print(f"  Relevance:     {scores['relevance']['score']:.2f}  — {scores['relevance']['reason']}")
    print(f"  Resolution:    {scores['resolution']['score']:.2f}  — {scores['resolution']['reason']}")
    print("─" * 50)
    print(f"  Average score: {avg:.2f}")

    # Step 4: Save to disk for drift monitoring
    record = save_score(query, scores)
    print(f"\n✓ Score saved to {SCORES_FILE}")

    # Step 5: Flush Respan traces
    respan.flush()

    return {
        "rag_result": rag_result,
        "scores":     scores,
        "average":    avg,
        "record":     record,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python src/evaluate.py \"your question here\"")
        print('Example: python src/evaluate.py "How do I cancel my subscription?"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    print("=" * 55)
    print("  Support Bot — LLM-as-Judge Evaluation")
    print("=" * 55)

    result = evaluate(query)

    print("\nNext step: python src/monitor.py")


if __name__ == "__main__":
    main()
