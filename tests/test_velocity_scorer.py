"""Velocity scorer — rolling-window event rate."""

from __future__ import annotations

import time

from processing.velocity_scorer import VelocityScorer


def test_velocity_is_events_per_minute():
    scorer = VelocityScorer(window_seconds=300)
    for i in range(10):
        score = scorer.record("e", ts=1000.0 + i)
    # 10 events in a 5-minute window is 2 per minute.
    assert score.events_in_window == 10
    assert score.velocity == 2.0


def test_events_outside_the_window_are_evicted():
    scorer = VelocityScorer(window_seconds=300)
    scorer.record("e", ts=1000.0)
    scorer.record("e", ts=1100.0)
    # Far enough ahead that both earlier events have aged out.
    score = scorer.record("e", ts=2000.0)
    assert score.events_in_window == 1


def test_score_accepts_a_simulated_clock():
    """Replaying history is impossible if the window is pinned to wall time."""
    scorer = VelocityScorer(window_seconds=300)
    scorer.record("e", ts=1000.0)
    assert scorer.score("e", ts=1100.0).events_in_window == 1
    assert scorer.score("e", ts=9999.0).events_in_window == 0


def test_score_does_not_record_an_event():
    scorer = VelocityScorer(window_seconds=300)
    scorer.record("e", ts=1000.0)
    scorer.score("e", ts=1000.0)
    assert scorer.score("e", ts=1000.0).events_in_window == 1


def test_entities_are_tracked_independently():
    scorer = VelocityScorer(window_seconds=300)
    scorer.record("a", ts=1000.0)
    scorer.record("a", ts=1001.0)
    scorer.record("b", ts=1000.0)
    assert scorer.score("a", ts=1002.0).events_in_window == 2
    assert scorer.score("b", ts=1002.0).events_in_window == 1


def test_top_k_skips_fully_expired_entities():
    # top_k() reads the wall clock, so the fixture has to sit near real "now".
    now = time.time()
    scorer = VelocityScorer(window_seconds=300)
    scorer.record("stale", ts=now - 10_000)
    scorer.record("live", ts=now - 5)
    top = scorer.top_k(k=5)
    assert [s.entity_id for s in top] == ["live"]


def test_unknown_entity_scores_zero():
    scorer = VelocityScorer(window_seconds=300)
    assert scorer.score("never-seen").velocity == 0.0
