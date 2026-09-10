"""
Confidence gate for case file publishing.

Case files below CONFIDENCE_THRESHOLD are flagged for human review
and excluded from the published feed until manually cleared.

The default is read off the classifier's reliability curve by ml/evaluate.py,
not chosen by feel: 0.80 is the lowest gate whose retained predictions hold 90%
accuracy on the held-out split (90.2% at 76.8% coverage), sending the remaining
quarter to a human. It replaces a guessed 0.65, which published a band where the
model is measurably not that reliable.

Lowest rather than safest, because every extra point of threshold buys accuracy
by sending more cases to review. The number is a property of the fitted model —
re-derive it after a retrain with `python -m ml.evaluate`.
"""

from __future__ import annotations

import os

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.80"))


def apply_gate(confidence: float) -> bool:
    """Returns True if the case needs human review (confidence below threshold)."""
    return confidence < CONFIDENCE_THRESHOLD
