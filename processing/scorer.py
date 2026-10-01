"""
Velocity scoring and anomaly detection: entity-mentions -> anomaly_events.

The processor publishes one message per mention of a scored entity, keyed by
entity id. Kafka hashes the key to a partition, so every mention of one entity
lands on the same partition and therefore on the same scorer process. That is
what makes the per-entity state here (a rolling window of timestamps and a
baseline history) complete when several processors and scorers run at once.

Scoring uses each mention's event time rather than this process's clock, so a
scorer catching up on a backlog sees the rates that actually happened.

State is in memory, so a scorer that starts, or is handed a partition in a
rebalance, begins with empty windows. A window that has not yet seen a full
window's worth of time undercounts, and while it fills every entity's count can
only climb — which the detector, comparing each sample with the ones before it,
reads as a spike in everything. So each partition warms up first: mentions are
counted from the first one seen, but nothing is judged until a whole window has
passed in event time. Anomalies are missed for that one window after a start or
a rebalance rather than invented.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import UTC, datetime

import asyncpg
from aiokafka import AIOKafkaConsumer, ConsumerRebalanceListener

from config import DB_URL, KAFKA_BROKER, MENTIONS_TOPIC, configure_logging
from graph.models import ensure_schema
from processing.anomaly_detector import AnomalyDetector, AnomalyEvent
from processing.velocity_scorer import VelocityScorer

GROUP_ID = os.getenv("SCORER_GROUP_ID", "diffusion-scorer")
BATCH_MAX = int(os.getenv("SCORER_BATCH_MAX", "1000"))

log = logging.getLogger(__name__)


class MentionScorer:
    """Rolling velocity and anomaly state for the partitions one scorer owns. Pure — no I/O."""

    def __init__(self, window_seconds: int = 300, detector: AnomalyDetector | None = None) -> None:
        self.window = window_seconds
        self.velocity = VelocityScorer(window_seconds=window_seconds)
        self.detector = detector or AnomalyDetector()
        # Event time of the first mention seen on each partition since it was assigned.
        self._since: dict[int, float] = {}

    def reset(self, partitions) -> None:
        """Restart warm-up for partitions just (re)assigned to this scorer."""
        for p in partitions:
            self._since.pop(p, None)

    def warm(self, partition: int, ts: float) -> bool:
        since = self._since.setdefault(partition, ts)
        return ts - since >= self.window

    def score(self, mentions: list[tuple[int, dict]]) -> list[AnomalyEvent]:
        """Fold (partition, mention) pairs into the state; return the anomalies raised."""
        events = []
        for partition, m in mentions:
            score = self.velocity.record(m["entity_id"], ts=m["ts"])
            if not self.warm(partition, m["ts"]):
                continue
            event = self.detector.detect(score, m["platform"], now=datetime.fromtimestamp(m["ts"], UTC))
            if event is not None:
                events.append(event)
        return events


class _ResetOnAssign(ConsumerRebalanceListener):
    def __init__(self, scorer: MentionScorer) -> None:
        self._scorer = scorer

    async def on_partitions_revoked(self, revoked) -> None:
        pass

    async def on_partitions_assigned(self, assigned) -> None:
        self._scorer.reset(tp.partition for tp in assigned)


async def main() -> None:
    scorer = MentionScorer()
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)
    await ensure_schema(pool)
    consumer = AIOKafkaConsumer(
        bootstrap_servers=KAFKA_BROKER,
        group_id=GROUP_ID,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        # Mentions are written transactionally; skip those from aborted batches.
        isolation_level="read_committed",
        value_deserializer=lambda b: json.loads(b),
    )
    consumer.subscribe([MENTIONS_TOPIC], listener=_ResetOnAssign(scorer))
    await consumer.start()
    log.info("scorer started on %s", MENTIONS_TOPIC)
    try:
        while True:
            batch = await consumer.getmany(timeout_ms=1000, max_records=BATCH_MAX)
            if not batch:
                continue
            mentions = [(tp.partition, m.value) for tp, part in batch.items() for m in part]
            events = scorer.score(mentions)
            if events:
                async with pool.acquire() as conn, conn.transaction():
                    for event in events:
                        await scorer.detector.store(conn, event)
            # After the inserts, so a crash replays the batch instead of losing it.
            await consumer.commit()
            log.info("scored %d mentions, %d anomalies", len(mentions), len(events))
    finally:
        await consumer.stop()
        await pool.close()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
