"""Threshold selection and the Pareto front used by the anomaly sweep."""

from __future__ import annotations

from ml.eval_anomaly import choose_threshold, pareto_front


def test_pareto_front_drops_dominated_points():
    # (false alarms, recall): the third is beaten by the first on both axes.
    points = [(1.0, 0.8), (3.0, 0.9), (2.0, 0.7), (0.5, 0.6)]
    assert pareto_front(points) == [True, True, False, True]


def test_ties_are_not_dominated():
    assert pareto_front([(1.0, 0.8), (1.0, 0.8)]) == [True, True]


def test_choose_threshold_holds_recall_and_minimises_noise():
    sweep = [
        {"threshold": 2.0, "coordinated_recall": 0.90, "fp_per_topic_day": 5.0},
        {"threshold": 3.0, "coordinated_recall": 0.80, "fp_per_topic_day": 1.0},
        {"threshold": 4.0, "coordinated_recall": 0.60, "fp_per_topic_day": 0.1},
    ]
    assert choose_threshold(sweep, 0.75)["threshold"] == 3.0
    assert choose_threshold(sweep, 0.95) is None
