"""
Confidence gate for case file publishing.

Case files below CONFIDENCE_THRESHOLD are flagged for human review
and excluded from the published feed until manually cleared.
"""

from __future__ import annotations

CONFIDENCE_THRESHOLD = 0.65


def apply_gate(confidence: float) -> tuple[bool, bool]:
    """
    Evaluate a confidence score against the publishing threshold.

    Returns:
        (should_publish, needs_review)
    """
    needs_review = confidence < CONFIDENCE_THRESHOLD
    return not needs_review, needs_review
