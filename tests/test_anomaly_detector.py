"""Anomaly detector — the three baselines, and the guards around them."""

from __future__ import annotations

import pytest

from processing.anomaly_detector import AnomalyDetector, _median
from processing.velocity_scorer import VelocityScore


def _score(velocity: float, entity: str = "e") -> VelocityScore:
    return VelocityScore(
        entity_id=entity,
        events_in_window=int(velocity * 5),
        velocity=velocity,
        window_seconds=300,
    )


def _feed(detector: AnomalyDetector, values: list[float], entity: str = "e"):
    event = None
    for value in values:
        event = detector.detect(_score(value, entity), platform="hn")
    return event


@pytest.mark.parametrize("baseline", ["zscore", "robust", "ewma"])
def test_spike_after_a_quiet_baseline_fires(baseline):
    detector = AnomalyDetector(baseline=baseline, min_samples=5, threshold=2.5)
    assert _feed(detector, [2.0, 2.2, 1.9, 2.1, 2.0, 2.05, 40.0]) is not None


@pytest.mark.parametrize("baseline", ["zscore", "robust", "ewma"])
def test_steady_traffic_does_not_fire(baseline):
    detector = AnomalyDetector(baseline=baseline, min_samples=5, threshold=2.5)
    assert _feed(detector, [2.0, 2.2, 1.9, 2.1, 2.0, 2.05, 1.98, 2.1]) is None


@pytest.mark.parametrize("baseline", ["zscore", "robust", "ewma"])
def test_no_verdict_before_min_samples(baseline):
    detector = AnomalyDetector(baseline=baseline, min_samples=5, threshold=2.5)
    assert _feed(detector, [1.0, 500.0]) is None


@pytest.mark.parametrize("baseline", ["zscore", "robust"])
def test_a_flat_baseline_abstains(baseline):
    """
    Zero spread means a spike is unmeasurable, not infinitely significant.

    Dividing by it would make the first non-identical sample fire at any
    threshold.
    """
    detector = AnomalyDetector(baseline=baseline, min_samples=5, threshold=2.5)
    assert _feed(detector, [3.0] * 8 + [3.5]) is None


def test_robust_baseline_survives_a_contaminated_history():
    """
    A history that already contains bursts should not blind the detector.

    The mean and standard deviation are both dragged up by past spikes, so the
    plain z-score stops seeing a repeat of the same burst; the median does not
    move until half the samples are contaminated.
    """
    # Jittered rather than perfectly flat: identical quiet samples drive MAD to
    # zero, at which point the robust baseline abstains instead of firing.
    history = [2.0, 2.1, 60.0, 1.9, 2.2, 60.0, 2.0, 1.8, 60.0, 2.3]
    spike = 60.0

    plain = AnomalyDetector(baseline="zscore", min_samples=5, threshold=2.5)
    robust = AnomalyDetector(baseline="robust", min_samples=5, threshold=2.5)

    assert _feed(plain, history + [spike]) is None
    assert _feed(robust, history + [spike]) is not None


def test_the_current_sample_is_judged_against_prior_history():
    """A sample must not be folded into the baseline it is compared with."""
    detector = AnomalyDetector(baseline="zscore", min_samples=3, threshold=2.5)
    event = _feed(detector, [1.0, 1.1, 0.9, 1.0, 50.0])
    assert event is not None
    # A baseline that had absorbed the spike would sit far above the quiet level.
    assert event.baseline_mean < 2.0


def test_entities_keep_separate_baselines():
    detector = AnomalyDetector(baseline="zscore", min_samples=5, threshold=2.5)
    _feed(detector, [2.0, 2.2, 1.9, 2.1, 2.0, 2.05], entity="a")
    # "b" has no history of its own, so it cannot yet be judged.
    assert detector.detect(_score(99.0, "b"), platform="hn") is None


def test_unknown_baseline_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown baseline"):
        AnomalyDetector(baseline="nope")


def test_detect_performs_no_io():
    """detect() must stay pure so it can be replayed and unit-tested."""
    detector = AnomalyDetector(baseline="robust", min_samples=2, threshold=1.0)
    assert _feed(detector, [1.0, 1.3, 0.9, 1.1, 90.0]) is not None


@pytest.mark.parametrize(
    "values,expected",
    [([1.0], 1.0), ([1.0, 3.0], 2.0), ([1.0, 2.0, 3.0], 2.0), ([1.0, 2.0, 3.0, 4.0], 2.5)],
)
def test_median_of_sorted_values(values, expected):
    assert _median(values) == expected
