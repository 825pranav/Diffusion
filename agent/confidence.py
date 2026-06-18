"""
Confidence gate for case file publishing.

Case files below CONFIDENCE_THRESHOLD are flagged for human review
and excluded from the published feed until manually cleared.
"""

from __future__ import annotations

import os

CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))


def apply_gate(confidence: float) -> bool:
    """Returns True if the case needs human review (confidence below threshold)."""
    return confidence < CONFIDENCE_THRESHOLD
