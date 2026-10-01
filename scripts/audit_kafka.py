"""
Check that every record on the raw topics reached the graph.

Usage:
    python -m scripts.audit_kafka
    python -m scripts.audit_kafka --since 1790863068   # only messages produced after this epoch

Reads each raw topic from the beginning to its current end, derives the content
node every Bluesky, Mastodon and HN record must produce, and looks each one up.
A node that exists only as a placeholder does not count: a placeholder is what a
reply creates for a parent it has not seen, so its presence says nothing about
whether the parent's own record was written. GitHub events carry no content node
of their own and are left out.

Run it after the processors have drained (lag 0); anything still in flight is
reported as missing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys

import asyncpg
from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.admin import AIOKafkaAdminClient

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import (  # noqa: E402
    DB_URL,
    DEAD_LETTER_TOPIC,
    KAFKA_BROKER,
    RAW_TOPICS,
    configure_logging,
)
from graph.ids import (  # noqa: E402
    BLUESKY_CONTENT_PREFIX,
    HN_CONTENT_PREFIX,
    MASTODON_CONTENT_PREFIX,
)

CONTENT_PREFIX = {"bluesky": BLUESKY_CONTENT_PREFIX, "hn": HN_CONTENT_PREFIX, "mastodon": MASTODON_CONTENT_PREFIX}

log = logging.getLogger(__name__)


async def read_all(topics: list[str], since: float = 0.0) -> list[bytes]:
    admin = AIOKafkaAdminClient(bootstrap_servers=KAFKA_BROKER)
    await admin.start()
    parts = [
        TopicPartition(t["topic"], p["partition"])
        for t in await admin.describe_topics(topics)
        if not t["error_code"]  # a topic that does not exist yet holds nothing
        for p in t["partitions"]
    ]
    await admin.close()
    if not parts:
        return []
    reader = AIOKafkaConsumer(
        bootstrap_servers=KAFKA_BROKER, enable_auto_commit=False, isolation_level="read_committed"
    )
    await reader.start()
    try:
        ends = await reader.end_offsets(parts)
        reader.assign(parts)
        await reader.seek_to_beginning(*parts)
        values: list[bytes] = []
        remaining = {tp for tp in parts if ends[tp] > 0}
        while remaining:
            batch = await reader.getmany(*remaining, timeout_ms=2000, max_records=5000)
            for tp, msgs in batch.items():
                values += [m.value for m in msgs if m.offset < ends[tp] and m.timestamp >= since * 1000]
            # Position, not the last message's offset: transaction markers take
            # offsets of their own and are never returned as messages.
            for tp in list(remaining):
                if await reader.position(tp) >= ends[tp]:
                    remaining.discard(tp)
        return values
    finally:
        await reader.stop()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--since", type=float, default=0.0, help="epoch seconds; older messages are skipped")
    args = parser.parse_args()
    values = await read_all(RAW_TOPICS, args.since)
    expected: dict[tuple[str, str], int] = {}
    for raw in values:
        record = json.loads(raw)
        prefix = CONTENT_PREFIX.get(record.get("platform"))
        if prefix:
            key = (prefix + str(record["id"]), record["platform"])
            expected[key] = expected.get(key, 0) + 1

    ids = [i for i, _ in expected]
    platforms = [p for _, p in expected]
    conn = await asyncpg.connect(DB_URL)
    try:
        rows = await conn.fetch(
            """
            SELECT e.id, e.platform
            FROM unnest($1::text[], $2::text[]) AS e(id, platform)
            LEFT JOIN graph_nodes n ON n.id = e.id AND n.platform = e.platform
            WHERE n.id IS NULL OR n.metadata ? 'placeholder'
            """,
            ids, platforms,
        )
    finally:
        await conn.close()

    dead = len(await read_all([DEAD_LETTER_TOPIC], args.since))
    copies = sum(expected.values()) - len(expected)
    log.info("messages on raw topics: %d", len(values))
    log.info("distinct content records: %d (%d duplicate copies)", len(expected), copies)
    log.info("written to the graph:     %d", len(expected) - len(rows))
    log.info("missing:                  %d", len(rows))
    log.info("dead-lettered:            %d", dead)
    for r in rows[:10]:
        log.info("  missing %s (%s)", r["id"], r["platform"])


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
