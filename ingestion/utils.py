"""Shared utilities for Kafka producers."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque

from aiokafka import AIOKafkaProducer

from config import KAFKA_BROKER

log = logging.getLogger(__name__)


def record_key(record: dict) -> bytes:
    """
    Partition key for a raw record: platform plus its own id.

    Keying by id sends every copy of one record to the same partition, and so
    to the same processor. A Jetstream reconnect or an HN re-poll after the
    seen-set evicts can publish a record twice; with no key the copies spread
    across partitions and each processor's in-memory dedup sees only one of
    them, so the duplicate is extracted and written again. Ids are effectively
    uniform, so the key costs no partition balance.
    """
    return f"{record.get('platform', 'unknown')}:{record['id']}".encode()


def make_producer() -> AIOKafkaProducer:
    """
    Producer shared by every platform.

    Idempotence (which implies acks=all) stops a retried send from writing the
    same message twice. Sends are batched for up to `linger_ms` rather than
    awaited one at a time, which a firehose needs: a blocking round trip per
    message caps throughput at the broker's latency.
    """
    return AIOKafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        enable_idempotence=True,
        linger_ms=20,
        value_serializer=lambda record: json.dumps(record).encode(),
    )


def _log_failure(fut: asyncio.Future) -> None:
    if not fut.cancelled() and fut.exception() is not None:
        log.error("kafka delivery failed: %s", fut.exception())


async def publish(producer: AIOKafkaProducer, topic: str, record: dict) -> None:
    """Queue one record for delivery. Blocks only when the send buffer is full."""
    fut = await producer.send(topic, record, key=record_key(record))
    fut.add_done_callback(_log_failure)


class BoundedSeenSet:
    """
    In-memory seen-set with a soft size cap.

    When the set exceeds max_size, the oldest max_size // 2 entries are
    evicted to keep memory usage predictable.  Deduplication is best-effort:
    evicted entries may be re-processed, which is acceptable for idempotent
    Kafka producers.

    Supports `in` and `.add()` for drop-in use in polling loops.
    """

    def __init__(self, max_size: int = 10_000) -> None:
        self._seen: set = set()
        self._order: deque = deque()
        self._max = max_size

    def __contains__(self, item: object) -> bool:
        return item in self._seen

    def add(self, item: object) -> None:
        if item in self._seen:
            return
        self._seen.add(item)
        self._order.append(item)
        if len(self._seen) > self._max:
            for _ in range(self._max // 2):
                self._seen.discard(self._order.popleft())
