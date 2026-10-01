"""
Sample the Kafka hops while the pipeline runs: processor and scorer lag,
produce and consume rates, how long the oldest unprocessed raw message has been
waiting, graph size and anomalies raised.

Usage:
    python -m scripts.measure_kafka --minutes 10 --interval 15 --out run.jsonl

Lag is log-end offset minus the consumer group's committed offset, summed over
partitions. `behind_s` reads the message sitting at each committed offset and
compares its producer timestamp to now, so it is the age of the oldest record the
group has not yet processed: a direct latency figure, not one inferred from lag.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys
import time

import asyncpg
from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.admin import AIOKafkaAdminClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import DB_URL, KAFKA_BROKER, MENTIONS_TOPIC, RAW_TOPICS, configure_logging  # noqa: E402
from processing.consumer import GROUP_ID  # noqa: E402
from processing.scorer import GROUP_ID as SCORER_GROUP_ID  # noqa: E402

log = logging.getLogger(__name__)


async def oldest_unprocessed_age(reader: AIOKafkaConsumer, committed: dict, ends: dict) -> float:
    waiting = [tp for tp in ends if ends[tp] > committed.get(tp, 0)]
    if not waiting:
        return 0.0
    reader.assign(waiting)
    oldest = None
    for tp in waiting:
        reader.seek(tp, committed.get(tp, 0))
        batch = await reader.getmany(tp, timeout_ms=2000, max_records=1)
        for msgs in batch.values():
            if msgs:
                ts = msgs[0].timestamp / 1000
                oldest = ts if oldest is None else min(oldest, ts)
    return round(time.time() - oldest, 1) if oldest else 0.0


async def committed_offsets(admin, group: str) -> dict:
    offsets = await admin.list_consumer_group_offsets(group)
    return {tp: om.offset for tp, om in offsets.items() if om.offset >= 0}


async def sample(admin, reader, db, parts: list[TopicPartition], mention_parts: list[TopicPartition]) -> dict:
    ends = await reader.end_offsets(parts)
    committed = await committed_offsets(admin, GROUP_ID)
    behind = await oldest_unprocessed_age(reader, committed, ends)
    mention_ends = await reader.end_offsets(mention_parts) if mention_parts else {}
    scored = await committed_offsets(admin, SCORER_GROUP_ID)
    async with db.acquire() as conn:
        nodes = await conn.fetchval("SELECT COUNT(*) FROM graph_nodes")
        edges = await conn.fetchval("SELECT COUNT(*) FROM graph_edges")
        anomalies = await conn.fetchval("SELECT COUNT(*) FROM anomaly_events")
    return {
        "t": round(time.time(), 1),
        "produced": sum(ends.values()),
        "consumed": sum(committed.get(tp, 0) for tp in ends),
        "lag": sum(max(0, ends[tp] - committed.get(tp, 0)) for tp in ends),
        "behind_s": behind,
        # Includes transaction markers, so it reads a few offsets high per batch.
        "scorer_lag": sum(max(0, mention_ends[tp] - scored.get(tp, 0)) for tp in mention_ends),
        "nodes": nodes,
        "edges": edges,
        "anomalies": anomalies,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Sample Kafka lag and throughput")
    parser.add_argument("--minutes", type=float, default=10)
    parser.add_argument("--interval", type=float, default=15)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    admin = AIOKafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
    reader = AIOKafkaConsumer(bootstrap_servers=KAFKA_BROKER, enable_auto_commit=False)
    await admin.start()
    await reader.start()
    db = await asyncpg.create_pool(DB_URL, min_size=1, max_size=2)
    # A consumer with no subscription never tracks topic metadata, so
    # partitions_for_topic() returns None; ask the admin client instead.
    async def partitions(topics: list[str]) -> list[TopicPartition]:
        return [
            TopicPartition(t["topic"], p["partition"])
            for t in await admin.describe_topics(topics)
            if not t["error_code"]
            for p in t["partitions"]
        ]

    parts = await partitions(RAW_TOPICS)
    mention_parts = await partitions([MENTIONS_TOPIC])
    deadline = time.monotonic() + args.minutes * 60
    prev = None
    try:
        with args.out.open("a") as fh:
            while time.monotonic() < deadline:
                row = await sample(admin, reader, db, parts, mention_parts)
                if prev:
                    dt = row["t"] - prev["t"]
                    row["produce_rate"] = round((row["produced"] - prev["produced"]) / dt, 1)
                    row["consume_rate"] = round((row["consumed"] - prev["consumed"]) / dt, 1)
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                log.info(
                    "lag=%d behind=%.0fs produce=%s/s consume=%s/s edges=%d scorer_lag=%d anomalies=%d",
                    row["lag"], row["behind_s"], row.get("produce_rate", "-"),
                    row.get("consume_rate", "-"), row["edges"], row["scorer_lag"], row["anomalies"],
                )
                prev = row
                await asyncio.sleep(args.interval)
    finally:
        await reader.stop()
        await admin.close()
        await db.close()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
