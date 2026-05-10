"""
Seed script — inserts synthetic graph data and fires an anomaly event.

Usage:
    python scripts/seed.py [--nodes 30] [--edges 60] [--anomalies 2]

The anomaly INSERT fires the Postgres NOTIFY trigger, waking the agent
if it is running. Useful for local development and end-to-end testing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import string
from datetime import datetime, timedelta, timezone

import asyncpg

from config import DB_URL

PLATFORMS = ["bluesky", "mastodon", "hn", "github"]
EDGE_TYPES = ["repost", "comment", "reference", "share"]

log = logging.getLogger(__name__)


def rand_id(prefix: str = "node") -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"{prefix}_{suffix}"


def rand_ts(minutes_ago_max: int = 120) -> datetime:
    offset = random.randint(0, minutes_ago_max * 60)
    return datetime.now(timezone.utc) - timedelta(seconds=offset)


async def seed(conn: asyncpg.Connection, n_nodes: int, n_edges: int, n_anomalies: int) -> None:
    node_ids: list[str] = []

    # nodes
    for _ in range(n_nodes):
        platform = random.choice(PLATFORMS)
        nid = rand_id(platform[:2])
        label = f"Synthetic {platform} post {nid[-4:]}"
        await conn.execute(
            """
            INSERT INTO graph_nodes (id, type, platform, label, metadata)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (id, platform) DO NOTHING
            """,
            nid, "post", platform, label, json.dumps({"synthetic": True}),
        )
        node_ids.append(nid)

    log.info("inserted %d nodes", n_nodes)

    # edges
    inserted = 0
    for _ in range(n_edges):
        if len(node_ids) < 2:
            break
        src, tgt = random.sample(node_ids, 2)
        platform = random.choice(PLATFORMS)
        ts = rand_ts()
        try:
            await conn.execute(
                """
                INSERT INTO graph_edges (source_id, target_id, edge_type, platform, ts, weight)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (source_id, target_id, platform, ts) DO NOTHING
                """,
                src, tgt, random.choice(EDGE_TYPES), platform, ts, round(random.uniform(0.5, 2.0), 2),
            )
            inserted += 1
        except Exception:
            pass

    log.info("inserted %d edges", inserted)

    # anomaly events — INSERT fires NOTIFY, waking the agent
    for _ in range(n_anomalies):
        node_id = random.choice(node_ids)
        platform = random.choice(PLATFORMS)
        z_score = round(random.uniform(2.6, 5.5), 2)
        velocity = round(random.uniform(10.0, 80.0), 1)
        row = await conn.fetchrow(
            """
            INSERT INTO anomaly_events (node_id, platform, z_score, velocity)
            VALUES ($1, $2, $3, $4)
            RETURNING id
            """,
            node_id, platform, z_score, velocity,
        )
        log.info(
            "anomaly event %d — node=%s platform=%s z=%.2f vel=%.1f",
            row["id"], node_id, platform, z_score, velocity,
        )

    log.info("seed complete")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Seed synthetic diffusion data")
    parser.add_argument("--nodes", type=int, default=30)
    parser.add_argument("--edges", type=int, default=60)
    parser.add_argument("--anomalies", type=int, default=2)
    args = parser.parse_args()

    conn = await asyncpg.connect(DB_URL)
    try:
        await seed(conn, args.nodes, args.edges, args.anomalies)
    finally:
        await conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
