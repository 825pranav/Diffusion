"""
Ragas evaluation pipeline for Diffusion case files.

Scores each agent output on three dimensions:
  retrieval_relevance   — are the retrieved similar cases relevant to the anomaly?
  reasoning_consistency — is the classification grounded in the retrieved evidence?
  answer_relevance      — does the answer directly address the investigation question?

These score the *retrieval and grounding* of an investigation. None of them is a
calibration measure: Ragas answer_relevancy asks whether the response addresses
the question, which says nothing about whether a stated confidence of 0.84 is
right 84% of the time. This field was previously named confidence_calibration,
which claimed exactly that. Real calibration — Brier score, a reliability curve,
and the review threshold derived from it — comes from ml/evaluate.py.

Evaluation is best-effort: if the Ragas pipeline fails (missing API key,
rate limit, empty contexts) the function returns zero scores rather than
crashing the agent loop.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

log = logging.getLogger(__name__)


async def evaluate_case(
    trend: str,
    classification: str,
    signals: list[str],
    similar_cases: list[dict[str, Any]],
    reasoning_steps: list[str],
) -> dict[str, float]:
    """
    Run Ragas evaluation in a thread and return a scores dict.
    Always returns a valid dict — falls back to zeros on any error.
    """
    try:
        return await asyncio.to_thread(
            _run_ragas, trend, classification, signals, similar_cases, reasoning_steps
        )
    except Exception:
        log.warning("ragas evaluation failed — returning zero scores", exc_info=True)
        return {
            "retrieval_relevance": 0.0,
            "reasoning_consistency": 0.0,
            "answer_relevance": 0.0,
        }


def _run_ragas(
    trend: str,
    classification: str,
    signals: list[str],
    similar_cases: list[dict[str, Any]],
    reasoning_steps: list[str],
) -> dict[str, float]:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    question = (
        f"Is the trend '{trend}' spreading organically "
        "or via coordinated amplification?"
    )
    answer = f"Classification: {classification}. Evidence: {'; '.join(signals)}"
    contexts = [str(c) for c in similar_cases] + reasoning_steps

    if not contexts:
        raise ValueError("no contexts — skipping ragas evaluation")

    ds = Dataset.from_dict({
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    })

    result = evaluate(ds, metrics=[context_precision, faithfulness, answer_relevancy])

    return {
        "retrieval_relevance": float(result["context_precision"]),
        "reasoning_consistency": float(result["faithfulness"]),
        "answer_relevance": float(result["answer_relevancy"]),
    }
