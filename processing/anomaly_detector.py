"""
Anomaly detector for stream processing.

Maintains a per-entity rolling history of velocity scores. When a new
sample scores above the threshold, an AnomalyEvent is emitted via Postgres
LISTEN/NOTIFY — this is how the agent process wakes without polling.

Three baselines are available via ANOMALY_BASELINE:

  zscore  mean and standard deviation. Both are dragged upward by the very
          spikes being looked for, so a sustained campaign hides itself.
  robust  median and scaled MAD (default). Half the history must be
          contaminated before the centre moves, which matters because social
          velocity is spiky by nature.
  ewma    exponentially weighted mean and variance. Tracks drift, so a topic
          that is genuinely busier this week stops firing continuously.

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

import asyncpg

from processing.velocity_scorer import VelocityScore

ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_Z_THRESHOLD", "2.5"))
HISTORY_SIZE = int(os.getenv("ANOMALY_HISTORY_SIZE", "60"))
MIN_SAMPLES = int(os.getenv("ANOMALY_MIN_SAMPLES", "5"))

# Which baseline a sample is scored against; see the module docstring.
BASELINE = os.getenv("ANOMALY_BASELINE", "robust")

# Smoothing factor for the EWMA baseline — roughly a 20-sample memory.
EWMA_ALPHA = float(os.getenv("ANOMALY_EWMA_ALPHA", "0.1"))

# Scale factor making MAD a consistent estimator of the standard deviation for
# normally distributed data, so the robust threshold stays comparable to the
# plain z-score threshold.
MAD_TO_SIGMA = 1.4826

log = logging.getLogger(__name__)


_BASELINES = ("zscore", "robust", "ewma")


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

    The caller is responsible for providing a connection on each evaluate()
    call — the detector holds no DB state of its own. detect() is the pure
    half and needs no connection at all.

    Usage:
        detector = AnomalyDetector()
        async with pool.acquire() as conn:
            event = await detector.evaluate(conn, velocity_score, platform="hn")
    """

    def __init__(
        self,
        threshold: float = ANOMALY_THRESHOLD,
        history_size: int = HISTORY_SIZE,
        min_samples: int = MIN_SAMPLES,
        baseline: str = BASELINE,
    ) -> None:
        if baseline not in _BASELINES:
            raise ValueError(
                f"unknown baseline {baseline!r}; expected one of {sorted(_BASELINES)}"
            )
        self._threshold = threshold
        self._history_size = history_size
        self._min_samples = min_samples
        self._baseline = baseline
        self._history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        # EWMA carries (mean, variance) per entity rather than a window.
        self._ewma: dict[str, tuple[float, float]] = {}

    async def _notify(self, conn: asyncpg.Connection, event: AnomalyEvent) -> None:
        try:
            # INSERT triggers trg_anomaly_notify which fires NOTIFY automatically
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
        except Exception:
            log.exception("anomaly insert failed for entity %s", event.entity_id)

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

    def _baseline_score(
        self, entity_id: str, velocity: float
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
        result = self._baseline_score(score.entity_id, score.velocity)
        self._history[score.entity_id].append(score.velocity)
        self._update_ewma(score.entity_id, score.velocity)

        if result is None:
            return None

        z, centre, scale = result
        if z < self._threshold:
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
            await self._notify(conn, event)
        return event
