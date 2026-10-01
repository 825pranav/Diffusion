"""
Anomaly detector for stream processing.

Maintains a per-entity rolling history of velocity scores. When a new
sample scores above the threshold, an AnomalyEvent is emitted via Postgres
LISTEN/NOTIFY — this is how the agent process wakes without polling.

Six baselines are available via ANOMALY_BASELINE:

  zscore  mean and standard deviation. Both are dragged upward by the very
          spikes being looked for, so a sustained campaign hides itself.
  robust  median and scaled MAD (default). Half the history must be
          contaminated before the centre moves, which matters because social
          velocity is spiky by nature.
  ewma    exponentially weighted mean and variance. Tracks drift, so a topic
          that is genuinely busier this week stops firing continuously.
  poisson the window's event *count* against a Poisson with a trimmed-mean
          rate. Velocity on a quiet topic is a small integer count, where the
          MAD is usually exactly zero and `robust` abstains; a count model
          stays defined there. The tail probability is reported as the
          equivalent one-sided normal z, so thresholds mean the same thing.
  negbin  as `poisson`, but with a negative binomial whose variance is read
          from the history. Real chatter clumps (a reply draws replies), so
          counts are overdispersed and a pure Poisson baseline overstates how
          surprising an ordinary clump is.
  cusum   one-sided CUSUM over the Poisson z-scores. Accumulates modest,
          sustained excess instead of waiting for a single large sample, and
          resets after firing so one burst raises one alert rather than one
          per tick.

ml/eval_anomaly.py measures all three on replayed simulator traffic.

History depth: ANOMALY_HISTORY_SIZE samples (default 60). Combined with
the velocity scorer's 5-minute window, that covers ~5 hours of baseline
before the oldest samples are evicted.
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime
from statistics import NormalDist

import asyncpg

from processing.velocity_scorer import VelocityScore

HISTORY_SIZE = int(os.getenv("ANOMALY_HISTORY_SIZE", "60"))
MIN_SAMPLES = int(os.getenv("ANOMALY_MIN_SAMPLES", "5"))

# Which baseline a sample is scored against; see the module docstring.
#
# CUSUM is the default because it is the one measured to beat the previous
# default (robust) on every axis that was tested, in the regime production
# actually runs — one sample per arriving event. On 150 replayed topics none of
# its settings were chosen on, it caught more campaigns (0.846 vs 0.832
# recall) with 0.08 false alarms per topic-day against 1.35, and it stayed
# ahead when the background chatter was made overdispersed. It costs about
# twenty seconds of median detection delay. See `python -m ml.eval_anomaly
# --sweep --tick 0` and docs/results.md.
BASELINE = os.getenv("ANOMALY_BASELINE", "cusum")

# Default operating point per baseline, used unless ANOMALY_Z_THRESHOLD is set.
# The thresholds are not interchangeable: CUSUM's is on an accumulated sum,
# and each was chosen separately for per-event scoring. The z-scale baselines
# keep their historical 2.5.
DEFAULT_THRESHOLDS = {
    "zscore": 2.5,
    "robust": 2.5,
    "ewma": 2.5,
    "poisson": 3.0,
    "negbin": 2.5,
    "cusum": 12.0,
}
_THRESHOLD_ENV = os.getenv("ANOMALY_Z_THRESHOLD")
ANOMALY_THRESHOLD: float | None = float(_THRESHOLD_ENV) if _THRESHOLD_ENV else None

# Smoothing factor for the EWMA baseline — roughly a 20-sample memory.
EWMA_ALPHA = float(os.getenv("ANOMALY_EWMA_ALPHA", "0.1"))

# Scale factor making MAD a consistent estimator of the standard deviation for
# normally distributed data, so the robust threshold stays comparable to the
# plain z-score threshold.
MAD_TO_SIGMA = 1.4826

# Share of the highest history samples dropped before estimating the Poisson
# rate — past bursts must not inflate the baseline they are compared against.
POISSON_TRIM = float(os.getenv("ANOMALY_POISSON_TRIM", "0.2"))
# CUSUM reference value: per-sample excess (in z units) that is absorbed as
# noise before anything accumulates.
CUSUM_K = float(os.getenv("ANOMALY_CUSUM_K", "0.5"))

_NORMAL = NormalDist()

log = logging.getLogger(__name__)


_BASELINES = ("zscore", "robust", "ewma", "poisson", "negbin", "cusum")


def poisson_upper_tail(count: int, rate: float) -> float:
    """P(X >= count) for X ~ Poisson(rate), summed directly in log space.

    Computing 1 - CDF instead loses every digit exactly where it matters: a
    burst far above a quiet baseline has a tail probability below machine
    epsilon, which 1 - CDF rounds to zero.
    """
    if count <= 0:
        return 1.0
    log_rate = math.log(rate)
    terms = []
    i = count
    while True:
        term = -rate + i * log_rate - math.lgamma(i + 1)
        terms.append(term)
        # Past the mode the terms shrink geometrically; stop once negligible.
        if i > rate and term < terms[0] - 40:
            break
        i += 1
    peak = max(terms)
    return min(1.0, math.exp(peak) * sum(math.exp(t - peak) for t in terms))


def negbin_upper_tail(count: int, mean: float, variance: float) -> float:
    """
    P(X >= count) for a negative binomial with the given mean and variance.

    Falls back to Poisson when the variance does not exceed the mean — the
    negative binomial cannot be underdispersed.
    """
    if variance <= mean * (1.0 + 1e-9):
        return poisson_upper_tail(count, mean)
    if count <= 0:
        return 1.0
    r = mean * mean / (variance - mean)          # "number of successes"
    q = mean / (r + mean)                        # per-trial failure probability
    log_q, log_1q = math.log(q), math.log1p(-q)
    base = r * log_1q - math.lgamma(r)
    terms = []
    i = count
    while True:
        term = base + math.lgamma(i + r) - math.lgamma(i + 1) + i * log_q
        terms.append(term)
        if i > mean and term < terms[0] - 40:
            break
        i += 1
        if i - count > 100_000:  # pathological tail; the sum has long converged
            break
    peak = max(terms)
    return min(1.0, math.exp(peak) * sum(math.exp(t - peak) for t in terms))


def tail_to_z(p: float) -> float:
    """The one-sided normal z with the same upper-tail probability."""
    p = min(max(p, 1e-300), 1.0 - 1e-16)
    return -_NORMAL.inv_cdf(p)


def _median(ordered: list[float]) -> float:
    """Median of an already-sorted sequence."""
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass
class AnomalyEvent:
    entity_id: str
    platform: str
    z_score: float
    current_velocity: float   # events/min at detection time
    baseline_mean: float
    baseline_std: float
    detected_at: str          # ISO 8601
    window_seconds: int


class AnomalyDetector:
    """
    Velocity spike detector with Postgres NOTIFY integration.

    The caller is responsible for providing a connection on each store() or
    evaluate() call — the detector holds no DB state of its own. detect() is
    the pure half and needs no connection at all; processing/scorer.py calls it
    per mention and stores the batch's events in one transaction.

    Usage:
        detector = AnomalyDetector()
        event = detector.detect(velocity_score, platform="hn")
        if event:
            async with pool.acquire() as conn:
                await detector.store(conn, event)
    """

    def __init__(
        self,
        threshold: float | None = ANOMALY_THRESHOLD,
        history_size: int = HISTORY_SIZE,
        min_samples: int = MIN_SAMPLES,
        baseline: str = BASELINE,
    ) -> None:
        if baseline not in _BASELINES:
            raise ValueError(
                f"unknown baseline {baseline!r}; expected one of {sorted(_BASELINES)}"
            )
        # An explicit threshold wins; otherwise the baseline's own default.
        self._threshold = threshold if threshold is not None else DEFAULT_THRESHOLDS[baseline]
        self._history_size = history_size
        self._min_samples = min_samples
        self._baseline = baseline
        self._history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        # EWMA carries (mean, variance) per entity rather than a window.
        self._ewma: dict[str, tuple[float, float]] = {}
        # Poisson and CUSUM work on raw window counts, not velocities.
        self._counts: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._cusum: dict[str, float] = defaultdict(float)

    async def store(self, conn: asyncpg.Connection, event: AnomalyEvent) -> None:
        """
        Insert an anomaly; the INSERT fires trg_anomaly_notify, which wakes the agent.

        Errors propagate. This runs inside the scorer's batch transaction, and a
        swallowed failure would leave that transaction aborted, failing every
        insert after it while the offsets still committed.
        """
        await conn.execute(
            """
            INSERT INTO anomaly_events (node_id, platform, z_score, velocity)
            VALUES ($1, $2, $3, $4)
            """,
            event.entity_id, event.platform,
            event.z_score, event.current_velocity,
        )
        log.info(
            "anomaly inserted: entity=%s platform=%s z=%.2f velocity=%.2f",
            event.entity_id, event.platform,
            event.z_score, event.current_velocity,
        )

    def _score_zscore(self, buf: deque[float], velocity: float) -> tuple[float, float, float] | None:
        """
        Classic z-score against the buffer's mean and standard deviation.

        Both statistics are pulled by the very spikes we are trying to find: a
        large burst inflates the mean and the standard deviation it is measured
        against, so a sustained campaign progressively hides itself.
        """
        n = len(buf)
        mean = sum(buf) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in buf) / n)
        if std == 0.0:
            return None
        return (velocity - mean) / std, mean, std

    def _score_robust(self, buf: deque[float], velocity: float) -> tuple[float, float, float] | None:
        """
        Median and scaled MAD — the same statistic with a breakdown point of 50%.

        Social velocity is spiky by nature, so the baseline has to survive its own
        history containing bursts. Half the samples must be contaminated before
        the median moves, where a single outlier already shifts the mean.
        """
        ordered = sorted(buf)
        median = _median(ordered)
        mad = _median(sorted(abs(x - median) for x in buf))
        if mad == 0.0:
            return None
        sigma = mad * MAD_TO_SIGMA
        return (velocity - median) / sigma, median, sigma

    def _score_ewma(self, entity_id: str, velocity: float) -> tuple[float, float, float] | None:
        """
        Exponentially weighted mean and variance.

        Unlike a fixed window this tracks drift, so a topic that is genuinely
        busier this week does not fire continuously. It updates *after* scoring,
        so the current sample is judged against the prior baseline.
        """
        state = self._ewma.get(entity_id)
        if state is None:
            return None
        mean, var = state
        std = math.sqrt(var)
        if std == 0.0:
            return None
        return (velocity - mean) / std, mean, std

    def _update_ewma(self, entity_id: str, velocity: float) -> None:
        state = self._ewma.get(entity_id)
        if state is None:
            self._ewma[entity_id] = (velocity, 0.0)
            return
        mean, var = state
        delta = velocity - mean
        new_mean = mean + EWMA_ALPHA * delta
        new_var = (1 - EWMA_ALPHA) * (var + EWMA_ALPHA * delta * delta)
        self._ewma[entity_id] = (new_mean, new_var)

    def _score_poisson(
        self, counts: deque[int], count: int
    ) -> tuple[float, float, float]:
        """
        Tail probability of the current count under a Poisson baseline.

        The rate is a trimmed mean of past counts, with half an event of prior
        mass so an all-zero history gives a small positive rate instead of an
        infinitely surprising first event.
        """
        ordered = sorted(counts)
        keep = ordered[: max(1, math.ceil(len(ordered) * (1.0 - POISSON_TRIM)))]
        rate = (sum(keep) + 0.5) / (len(keep) + 1.0)
        z = tail_to_z(poisson_upper_tail(count, rate))
        return z, rate, math.sqrt(rate)

    def _score_negbin(
        self, counts: deque[int], count: int
    ) -> tuple[float, float, float]:
        """
        Tail probability under a negative binomial fitted to the history.

        The mean is the same trimmed mean the Poisson baseline uses. The
        variance comes from the history winsorised at its 90th percentile, so
        clumpy chatter widens the baseline while a single past burst cannot
        blow it open.
        """
        ordered = sorted(counts)
        keep = ordered[: max(1, math.ceil(len(ordered) * (1.0 - POISSON_TRIM)))]
        mean = (sum(keep) + 0.5) / (len(keep) + 1.0)
        cap = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
        clipped = [min(c, cap) for c in ordered]
        m = sum(clipped) / len(clipped)
        variance = max(sum((c - m) ** 2 for c in clipped) / max(len(clipped) - 1, 1), mean)
        z = tail_to_z(negbin_upper_tail(count, mean, variance))
        return z, mean, math.sqrt(variance)

    def _baseline_score(
        self, entity_id: str, velocity: float, count: int | None = None
    ) -> tuple[float, float, float] | None:
        """
        Score velocity against the entity's baseline.

        Returns (z, centre, scale) or None when there is not yet enough history,
        or when the baseline has no spread at all — identical history values make
        a spike indistinguishable from noise rather than infinitely significant.
        """
        buf = self._history[entity_id]
        if len(buf) < self._min_samples:
            return None
        if self._baseline in ("poisson", "cusum"):
            return self._score_poisson(self._counts[entity_id], int(count or 0))
        if self._baseline == "negbin":
            return self._score_negbin(self._counts[entity_id], int(count or 0))
        if self._baseline == "ewma":
            return self._score_ewma(entity_id, velocity)
        if self._baseline == "robust":
            return self._score_robust(buf, velocity)
        return self._score_zscore(buf, velocity)

    def detect(
        self,
        score: VelocityScore,
        platform: str,
        now: datetime | None = None,
    ) -> AnomalyEvent | None:
        """
        Record a velocity sample and check for a spike. Pure — performs no I/O.

        The score is computed against the baseline *before* the new sample is
        folded in, so the current sample is judged against prior history rather
        than against itself.

        Separated from evaluate() so the detector can be unit-tested and replayed
        over historical data without a database.
        """
        result = self._baseline_score(
            score.entity_id, score.velocity, score.events_in_window
        )
        self._history[score.entity_id].append(score.velocity)
        self._counts[score.entity_id].append(score.events_in_window)
        self._update_ewma(score.entity_id, score.velocity)

        if result is None:
            return None

        z, centre, scale = result
        if self._baseline == "cusum":
            # Accumulate excess over the reference value; fire and reset once
            # the running sum crosses the threshold.
            total = max(0.0, self._cusum[score.entity_id] + z - CUSUM_K)
            if total < self._threshold:
                self._cusum[score.entity_id] = total
                return None
            self._cusum[score.entity_id] = 0.0
            z = total
        elif z < self._threshold:
            return None

        return AnomalyEvent(
            entity_id=score.entity_id,
            platform=platform,
            z_score=round(z, 4),
            current_velocity=round(score.velocity, 4),
            baseline_mean=round(centre, 4),
            baseline_std=round(scale, 4),
            detected_at=(now or datetime.now(UTC)).isoformat(),
            window_seconds=score.window_seconds,
        )

    async def evaluate(
        self,
        conn: asyncpg.Connection,
        score: VelocityScore,
        platform: str,
    ) -> AnomalyEvent | None:
        """
        Detect a spike and, if found, record it so the agent is notified.

        Returns an AnomalyEvent if one fired, otherwise None. The INSERT fires
        trg_anomaly_notify, which is what wakes the agent.
        """
        event = self.detect(score, platform)
        if event is not None:
            await self.store(conn, event)
        return event
