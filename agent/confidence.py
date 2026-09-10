"""
Confidence gate for case file publishing.

Case files below CONFIDENCE_THRESHOLD are flagged for human review
and excluded from the published feed until manually cleared.

The default is read off the classifier's reliability curve by ml/evaluate.py,
not chosen by feel: 0.55 is the lowest gate whose retained predictions hold 90%
accuracy on the held-out split (90.7% at 96.8% coverage). It replaces a guessed
0.65.

The value has moved with the model — 0.85, then 0.80, now 0.55 — and the last
drop was not a tuning choice. Fixing drift in the author-reuse features improved
Brier from 0.119 to 0.086, and a better-calibrated model can publish far more at
the same accuracy. That is the point of deriving it rather than picking it.

Lowest rather than safest, because every extra point of threshold buys accuracy
by sending more cases to review. The number is a property of the fitted model —
re-derive it after a retrain with `python -m ml.evaluate`.
"""

from __future__ import annotations

import os

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))


def apply_gate(confidence: float) -> bool:
    """Returns True if the case needs human review (confidence below threshold)."""
    return confidence < CONFIDENCE_THRESHOLD
