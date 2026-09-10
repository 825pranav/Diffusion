"""Confidence gate."""

from __future__ import annotations

from agent.confidence import CONFIDENCE_THRESHOLD, apply_gate


def test_below_threshold_needs_review():
    assert apply_gate(CONFIDENCE_THRESHOLD - 0.01) is True


def test_at_or_above_threshold_publishes():
    assert apply_gate(CONFIDENCE_THRESHOLD) is False
    assert apply_gate(1.0) is False


def test_default_matches_the_derived_gate():
    """
    Read off the reliability curve by ml/evaluate.py, not guessed.

    If a retrain moves it, re-derive rather than editing this test.
    """
    assert CONFIDENCE_THRESHOLD == 0.85
