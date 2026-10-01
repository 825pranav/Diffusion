"""
Stream processor: raw platform topics -> propagation graph.

    producers -> {bluesky,mastodon,hn,gh}-raw -> processor -> graph_nodes/edges
                                                     |
                                                     +-> entity-mentions -> scorer
                                                     +-> dead-letter

Each poll takes a batch, extracts entities from every record, and writes the
whole batch in one Postgres transaction. Then, in one Kafka transaction, it
publishes the batch's entity mentions, dead-letters anything that could not be
written, and commits the input offsets.

Delivery. Offsets move only after the graph write has committed, so a crash
re-reads the batch rather than skipping it. Re-reading is safe because every
graph write is idempotent (ON CONFLICT on the node key and on the edge key).
Mentions and offsets commit atomically, so a re-read batch never publishes its
mentions twice: the aborted attempt's messages are invisible to the scorer,
which reads with isolation_level=read_committed.

Scaling. Run one process per partition, up to the raw topics' partition count.
Velocity scoring used to live here, which made it wrong as soon as a second
process started: each one counted only the mentions that happened to reach it.
It now lives in processing/scorer.py, fed by a topic keyed by entity id.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import asyncpg
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer, ConsumerRecord

from config import (
    DB_URL,
    DEAD_LETTER_TOPIC,
    KAFKA_BROKER,
    MENTIONS_TOPIC,
    RAW_TOPICS,
    configure_logging,
)
from graph.models import ensure_schema
from graph.queries import insert_edges, upsert_nodes
from processing.dedup import DedupFilter
from processing.entity_extractor import EntitySet, extract_entity_set

GROUP_ID = os.getenv("KAFKA_GROUP_ID", "diffusion-processor")
BATCH_MAX = int(os.getenv("PROCESSOR_BATCH_MAX", "500"))
# Entity types whose mention rate is scored. Content nodes (individual posts)
# are not tracked; only the topics they mention are.
VELOCITY_TYPES = {"named_entity", "repo"}
_RETRIABLE = (asyncpg.exceptions.DeadlockDetectedError, asyncpg.exceptions.SerializationError)

log = logging.getLogger(__name__)


def strip_nul(value: Any) -> Any:
    """
    Remove NUL characters from every string in a decoded record.

    Postgres stores neither U+0000 in TEXT nor the \\u0000 escape in JSONB, and
    the Bluesky firehose does carry it occasionally: about 1 record in 2,000 on
    the first Kafka run, each of which failed its whole write.
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {strip_nul(k): strip_nul(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_nul(v) for v in value]
    return value


@dataclass
class Item:
    msg: ConsumerRecord
    record: dict
    entities: EntitySet


def mentions(item: Item) -> list[dict]:
    """
    One mention per scored entity in a record, stamped with the Kafka timestamp.

    The timestamp is the producer's CreateTime, i.e. when the record left the
    platform's stream. Scoring on it rather than on the processor's clock keeps
    velocity independent of consumer lag: a backlog drained at 3x speed would
    otherwise look like a 3x spike in everything.
    """
    platform = item.record.get("platform", "unknown")
    return [
        {"entity_id": n.id, "platform": platform, "ts": item.msg.timestamp / 1000}
        for n in item.entities.nodes
        if n.type in VELOCITY_TYPES
    ]


def dead_letter(msg: ConsumerRecord, error: BaseException) -> dict:
    raw = msg.value.decode("utf-8", errors="replace") if isinstance(msg.value, bytes) else msg.value
    return {
        "topic": msg.topic,
        "partition": msg.partition,
        "offset": msg.offset,
        "error": f"{type(error).__name__}: {error}",
        "value": raw,
    }


def prepare(msgs: list[ConsumerRecord], dedup: DedupFilter) -> tuple[list[Item], list[dict]]:
    """Decode, clean and extract a batch. Records that fail here are dead-lettered."""
    items, dead = [], []
    for msg in msgs:
        try:
            record = strip_nul(json.loads(msg.value))
            if not dedup.is_new(record):
                continue
            entities = extract_entity_set(record)
        except Exception as exc:
            log.warning("cannot extract %s[%d]@%d: %s", msg.topic, msg.partition, msg.offset, exc)
            dead.append(dead_letter(msg, exc))
            continue
        if entities.nodes:
            items.append(Item(msg, record, entities))
    return items, dead


async def _write(conn: asyncpg.Connection, items: list[Item]) -> None:
    nodes = [(n.id, n.type, n.platform, n.label, n.metadata) for it in items for n in it.entities.nodes]
    edges = [
        (e.source_id, e.target_id, e.edge_type, e.platform, e.timestamp, e.weight)
        for it in items
        for e in it.entities.edges
    ]
    await upsert_nodes(conn, nodes)
    await insert_edges(conn, edges)


async def write_batch(pool: asyncpg.Pool, items: list[Item]) -> list[tuple[Item, Exception]]:
    """
    Write a batch in one transaction; return the items that could not be written.

    One transaction per batch rather than per statement is the throughput fix:
    autocommit made every one of a record's ~5 statements wait on its own WAL
    flush. If the batch fails, it is retried record by record under savepoints
    so one bad record costs only itself.
    """
    for attempt in range(3):
        try:
            async with pool.acquire() as conn, conn.transaction():
                await _write(conn, items)
            return []
        except _RETRIABLE:
            await asyncio.sleep(0.05 * (attempt + 1))
        except Exception:
            break

    failed: list[tuple[Item, Exception]] = []
    async with pool.acquire() as conn, conn.transaction():
        for item in items:
            try:
                async with conn.transaction():  # savepoint
                    await _write(conn, [item])
            except Exception as exc:
                log.warning("cannot write %s: %s", item.record.get("id"), exc)
                failed.append((item, exc))
    return failed


async def process(
    batch: dict,
    dedup: DedupFilter,
    pool: asyncpg.Pool,
    producer: AIOKafkaProducer,
) -> tuple[int, int, int]:
    msgs = [m for part in batch.values() for m in part]
    items, dead = prepare(msgs, dedup)
    failed = await write_batch(pool, items)
    dead += [dead_letter(it.msg, exc) for it, exc in failed]
    failed_ids = {id(it) for it, _ in failed}
    written = [it for it in items if id(it) not in failed_ids]
    out = [m for it in written for m in mentions(it)]

    offsets = {tp: part[-1].offset + 1 for tp, part in batch.items()}
    async with producer.transaction():
        for m in out:
            await producer.send(MENTIONS_TOPIC, m, key=m["entity_id"].encode())
        for d in dead:
            await producer.send(DEAD_LETTER_TOPIC, d)
        await producer.send_offsets_to_transaction(offsets, GROUP_ID)
    return len(written), len(out), len(dead)


async def main() -> None:
    dedup = DedupFilter()
    pool = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)
    await ensure_schema(pool)
    consumer = AIOKafkaConsumer(
        *RAW_TOPICS,
        bootstrap_servers=KAFKA_BROKER,
        group_id=GROUP_ID,
        enable_auto_commit=False,
        # A new group starts from the beginning, so records produced before the
        # first processor came up are not silently skipped.
        auto_offset_reset="earliest",
        # Let the broker hold a fetch for up to 250 ms while it fills. Answered
        # at the first byte, a live stream arrives as batches of two or three
        # records, each paying for a Postgres and a Kafka transaction. Under a
        # backlog the bytes are already there and nothing waits.
        fetch_min_bytes=1 << 20,
        fetch_max_wait_ms=250,
    )
    producer = AIOKafkaProducer(
        bootstrap_servers=KAFKA_BROKER,
        # Unique per process: each instance owns its own transactions.
        transactional_id=f"{GROUP_ID}-{uuid.uuid4().hex[:12]}",
        value_serializer=lambda v: json.dumps(v).encode(),
    )
    await consumer.start()
    await producer.start()
    log.info("processor started on %s (batch %d)", RAW_TOPICS, BATCH_MAX)
    try:
        while True:
            batch = await consumer.getmany(timeout_ms=1000, max_records=BATCH_MAX)
            if not batch:
                continue
            t0 = time.perf_counter()
            written, sent, dead = await process(batch, dedup, pool, producer)
            log.info(
                "batch: %d in, %d written, %d mentions, %d dead-lettered, %.0f ms",
                sum(len(p) for p in batch.values()), written, sent, dead,
                (time.perf_counter() - t0) * 1000,
            )
    finally:
        await consumer.stop()
        await producer.stop()
        await pool.close()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
