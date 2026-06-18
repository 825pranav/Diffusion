"""
Velocity scorer for stream processing.

Tracks event rate-of-change per entity over a rolling time window.
Velocity is expressed as events per minute within the window.

Designed to score named entities (ORG, PRODUCT) and repos — the
signals that feed into anomaly detection. Content nodes (individual
posts) are not tracked; only the topics they mention are.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class VelocityScore:
    entity_id: str
    events_in_window: int
    velocity: float        # events per minute
    window_seconds: int


class VelocityScorer:
    """
    Rolling-window event rate tracker.

    Maintains a timestamp deque per entity. On each record() call,
    stale entries (older than window_seconds) are evicted before
    the count is returned. Memory is bounded by max_entities *
    max events per window — at 60 events/min over 5 min that's
    ~300 timestamps per entity.
    """

    def __init__(self, window_seconds: int = 300) -> None:
        self._window = window_seconds
        self._buckets: dict[str, deque[float]] = defaultdict(deque)

    def record(self, entity_id: str, ts: float | None = None) -> VelocityScore:
        """Record one event for entity_id and return its current velocity."""
        now = ts if ts is not None else time.time()
        bucket = self._buckets[entity_id]
        bucket.append(now)
        self._evict(bucket, now)
        return self._make_score(entity_id, bucket)

    def score(self, entity_id: str) -> VelocityScore:
        """Return the current velocity for entity_id without recording a new event."""
        now = time.time()
        bucket = self._buckets[entity_id]
        self._evict(bucket, now)
        return self._make_score(entity_id, bucket)

    def top_k(self, k: int = 10) -> list[VelocityScore]:
        """Return the k highest-velocity entities at the current moment.

        Entities whose window has fully expired are skipped — they would
        all score zero and pollute the results with stale noise.
        """
        now = time.time()
        scores = []
        for eid in list(self._buckets):
            bucket = self._buckets[eid]
            self._evict(bucket, now)
            if bucket:  # skip fully-expired entities
                scores.append(self._make_score(eid, bucket))
        return sorted(scores, key=lambda s: s.velocity, reverse=True)[:k]

    def active_entities(self) -> list[str]:
        """Return entity IDs that still have events in the current window."""
        now = time.time()
        cutoff = now - self._window
        return [eid for eid, bucket in self._buckets.items() if bucket and bucket[-1] >= cutoff]

    def _evict(self, bucket: deque[float], now: float) -> None:
        cutoff = now - self._window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

    def _make_score(self, entity_id: str, bucket: deque[float]) -> VelocityScore:
        count = len(bucket)
        velocity = count / (self._window / 60.0)
        return VelocityScore(
            entity_id=entity_id,
            events_in_window=count,
            velocity=round(velocity, 4),
            window_seconds=self._window,
        )
