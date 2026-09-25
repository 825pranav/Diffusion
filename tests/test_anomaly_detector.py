"""Anomaly detector — the three baselines, and the guards around them."""

from __future__ import annotations

import pytest

from processing.anomaly_detector import (
    AnomalyDetector,
    _median,
    negbin_upper_tail,
    poisson_upper_tail,
    tail_to_z,
)
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


# ── count-based baselines ─────────────────────────────────────────────────────


def _count_score(count: int, entity: str = "e") -> VelocityScore:
    return VelocityScore(
        entity_id=entity, events_in_window=count, velocity=count / 5.0, window_seconds=300
    )


def _feed_counts(detector: AnomalyDetector, counts: list[int]):
    return [detector.detect(_count_score(c), platform="hn") for c in counts]


@pytest.mark.parametrize(("count", "rate"), [(0, 1.0), (1, 0.3), (4, 1.5), (30, 2.0)])
def test_poisson_upper_tail_matches_scipy(count, rate):
    stats = pytest.importorskip("scipy.stats")
    assert poisson_upper_tail(count, rate) == pytest.approx(
        stats.poisson.sf(count - 1, rate), rel=1e-9, abs=1e-300
    )


def test_tail_to_z_is_the_normal_quantile():
    assert tail_to_z(0.5) == pytest.approx(0.0, abs=1e-12)
    assert tail_to_z(0.02275) == pytest.approx(2.0, abs=1e-3)
    # Probabilities below machine epsilon still map to a finite, large z.
    assert 30 < tail_to_z(1e-250) < 40


def test_poisson_fires_on_sparse_counts_where_robust_abstains():
    """
    A quiet topic's window counts are mostly zero, so the MAD is zero and the
    robust baseline cannot score anything. The count model still can.
    """
    quiet = [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    robust = AnomalyDetector(baseline="robust", min_samples=5, threshold=2.5)
    poisson = AnomalyDetector(baseline="poisson", min_samples=5, threshold=2.5)
    assert _feed_counts(robust, quiet + [12])[-1] is None
    assert _feed_counts(poisson, quiet + [12])[-1] is not None


def test_poisson_ignores_ordinary_fluctuation():
    detector = AnomalyDetector(baseline="poisson", min_samples=5, threshold=3.0)
    events = _feed_counts(detector, [2, 3, 1, 2, 4, 2, 3, 2, 1, 3, 4, 2])
    assert all(e is None for e in events)


def test_cusum_accumulates_sustained_excess_and_resets():
    """Modest excess that no single sample would flag adds up; one alert, then reset."""
    detector = AnomalyDetector(baseline="cusum", min_samples=5, threshold=5.0)
    baseline = [2, 3, 2, 2, 3, 2, 2, 3, 2, 2]
    events = _feed_counts(detector, baseline + [6, 6, 6, 6, 6])
    fired = [i for i, e in enumerate(events) if e is not None]
    assert len(fired) >= 1
    assert fired[0] > len(baseline)  # needed more than one elevated sample
    single = AnomalyDetector(baseline="poisson", min_samples=5, threshold=5.0)
    assert all(e is None for e in _feed_counts(single, baseline + [6, 6, 6, 6, 6]))


@pytest.mark.parametrize(("count", "mean", "var"), [(3, 1.0, 3.0), (10, 2.0, 8.0), (1, 0.4, 0.9)])
def test_negbin_upper_tail_matches_scipy(count, mean, var):
    stats = pytest.importorskip("scipy.stats")
    r = mean * mean / (var - mean)
    expected = stats.nbinom.sf(count - 1, r, r / (r + mean))
    assert negbin_upper_tail(count, mean, var) == pytest.approx(expected, rel=1e-9)


def test_negbin_without_overdispersion_is_poisson():
    assert negbin_upper_tail(5, 1.5, 1.5) == pytest.approx(poisson_upper_tail(5, 1.5))


def test_negbin_is_less_alarmed_by_clumpy_history_than_poisson():
    clumpy = [0, 0, 5, 0, 1, 0, 6, 0, 0, 4, 0, 1, 0, 5, 0]
    z = {}
    for baseline in ("poisson", "negbin"):
        detector = AnomalyDetector(baseline=baseline, min_samples=5, threshold=0.0)
        z[baseline] = _feed_counts(detector, clumpy + [7])[-1].z_score
    assert z["negbin"] < z["poisson"]


def test_every_baseline_has_a_default_threshold():
    from processing.anomaly_detector import _BASELINES, DEFAULT_THRESHOLDS

    assert set(DEFAULT_THRESHOLDS) == set(_BASELINES)


def test_default_threshold_follows_the_baseline(monkeypatch):
    import processing.anomaly_detector as ad

    monkeypatch.setattr(ad, "ANOMALY_THRESHOLD", None)
    assert ad.AnomalyDetector(baseline="cusum", threshold=None)._threshold == 12.0
    assert ad.AnomalyDetector(baseline="robust", threshold=None)._threshold == 2.5
    assert ad.AnomalyDetector(baseline="cusum", threshold=4.0)._threshold == 4.0
