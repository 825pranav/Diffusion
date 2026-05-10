"""
Z-score anomaly detector for stream processing.

Maintains a per-entity rolling history of velocity scores. When a new
sample pushes the z-score above the threshold, an AnomalyEvent is emitted
via Postgres LISTEN/NOTIFY — this is how the agent process wakes without
polling.

History depth: ANOMALY_HISTORY_SIZE samples (default 60). Combined with
the velocity scorer's 5-minute window, that covers ~5 hours of baseline
before the oldest samples are evicted.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

import asyncpg

from processing.velocity_scorer import VelocityScore

ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_Z_THRESHOLD", "2.5"))
HISTORY_SIZE = int(os.getenv("ANOMALY_HISTORY_SIZE", "60"))
MIN_SAMPLES = int(os.getenv("ANOMALY_MIN_SAMPLES", "5"))
log = logging.getLogger(__name__)


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
    Z-score spike detector with Postgres NOTIFY integration.

    The caller is responsible for providing a connection on each evaluate()
    call — the detector holds no DB state of its own.

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
    ) -> None:
        self._threshold = threshold
        self._history_size = history_size
        self._min_samples = min_samples
        self._history: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

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

    def _z_score(
        self, entity_id: str, velocity: float
    ) -> tuple[float, float, float] | None:
        """
        Compute z-score of velocity against the entity's history buffer.

        Returns (z, mean, std) or None if there are fewer than min_samples.
        Returns None if std == 0 (all history values identical — no baseline
        variance means we can't distinguish a spike from noise).
        """
        buf = self._history[entity_id]
        if len(buf) < self._min_samples:
            return None
        n = len(buf)
        mean = sum(buf) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in buf) / n)
        if std == 0.0:
            return None
        return (velocity - mean) / std, mean, std

    async def evaluate(
        self,
        conn: asyncpg.Connection,
        score: VelocityScore,
        platform: str,
    ) -> AnomalyEvent | None:
        """
        Record a velocity sample and check for a spike.

        The z-score is computed against the buffer *before* appending the
        new sample — so the current sample is evaluated against the prior
        baseline, not against itself.

        Returns an AnomalyEvent if z > threshold, otherwise None. Anomalies
        are also sent via Postgres NOTIFY regardless of return value handling.
        """
        result = self._z_score(score.entity_id, score.velocity)
        self._history[score.entity_id].append(score.velocity)

        if result is None:
            return None

        z, mean, std = result
        if z < self._threshold:
            return None

        event = AnomalyEvent(
            entity_id=score.entity_id,
            platform=platform,
            z_score=round(z, 4),
            current_velocity=round(score.velocity, 4),
            baseline_mean=round(mean, 4),
            baseline_std=round(std, 4),
            detected_at=datetime.now(timezone.utc).isoformat(),
            window_seconds=score.window_seconds,
        )
        await self._notify(conn, event)
        return event
