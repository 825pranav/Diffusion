"""
Ingest live Bluesky and Hacker News traffic straight into the propagation graph.

Usage:
    python -m scripts.ingest_live --minutes 20
    python -m scripts.ingest_live --minutes 30 --sources bluesky,hn --max-per-sec 40

Normally records reach the graph as producer -> Kafka -> consumer. This runs the
same producer serialisation and the same `processing.consumer.handle()` against
Postgres directly, with the broker left out. It exists so a real ingestion run is
possible on a machine without Docker; Kafka is the transport, not the logic, and
every record still passes through identical dedup, extraction, velocity and
anomaly-detection code.

Volume is the reason for `--max-per-sec`. The Bluesky firehose runs at hundreds
of events per second and every text record costs a spaCy NER pass, so an
uncapped run would spend all its time on standalone posts. Replies and reposts
are never dropped — they are the only propagation signal in the stream, and the
whole point of the run is to observe cascades rather than volume.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import sys
import time
from collections import Counter

import aiohttp
import asyncpg

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from config import DB_URL, configure_logging  # noqa: E402
from ingestion.bluesky_producer import JETSTREAM_URL, _serialize  # noqa: E402
from ingestion.hn_producer import HN_BASE  # noqa: E402
from processing.anomaly_detector import AnomalyDetector  # noqa: E402
from processing.consumer import handle  # noqa: E402
from processing.dedup import DedupFilter  # noqa: E402
from processing.velocity_scorer import VelocityScorer  # noqa: E402

DEFAULT_MINUTES = 20
DEFAULT_MAX_PER_SEC = 40
# How many HN items back from the newest to sweep per pass. Comments carry a
# `parent`, which is what makes an HN thread a cascade rather than a list.
HN_BATCH = 120
HN_INTERVAL_SECONDS = 45

log = logging.getLogger(__name__)


class Stats(Counter):
    def line(self) -> str:
        return (
            f"posts={self['post']} reposts={self['repost']} replies={self['reply']} "
            f"hn={self['hn']} dropped={self['rate_limited']} dupes={self['duplicate']}"
        )


async def _write(record: dict, pool, dedup, scorer, detector, stats: Stats) -> None:
    if not dedup.is_new(record):
        stats["duplicate"] += 1
        return
    try:
        async with pool.acquire() as conn:
            await handle(record, conn, scorer, detector)
    except Exception:
        stats["error"] += 1
        log.debug("failed to handle %s", record.get("id"), exc_info=True)


async def stream_bluesky(pool, dedup, scorer, detector, stats: Stats, deadline: float, max_per_sec: int) -> None:
    """Consume Jetstream until the deadline, capping standalone posts only."""
    window_start = time.monotonic()
    in_window = 0

    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.ws_connect(JETSTREAM_URL, heartbeat=30) as ws:
                    log.info("Jetstream connected")
                    async for msg in ws:
                        if time.monotonic() >= deadline:
                            return
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            record = _serialize(json.loads(msg.data))
                        except json.JSONDecodeError:
                            continue
                        if not record:
                            continue

                        propagating = bool(record.get("parent_id"))
                        if propagating:
                            stats["repost" if record["type"] == "repost" else "reply"] += 1
                        else:
                            # Rate-cap standalone posts only; propagation events
                            # are rare and are the entire point of the run.
                            now = time.monotonic()
                            if now - window_start >= 1.0:
                                window_start, in_window = now, 0
                            if in_window >= max_per_sec:
                                stats["rate_limited"] += 1
                                continue
                            in_window += 1
                            stats["post"] += 1

                        await _write(record, pool, dedup, scorer, detector, stats)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("Jetstream dropped, reconnecting", exc_info=True)
                await asyncio.sleep(3)


async def _hn_item(session: aiohttp.ClientSession, item_id: int) -> dict | None:
    try:
        async with session.get(f"{HN_BASE}/item/{item_id}.json", timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
    except Exception:
        return None
    if not data or data.get("deleted") or data.get("dead"):
        return None

    kind = data.get("type")
    if kind not in ("story", "comment"):
        return None
    return {
        "id": str(item_id),
        "platform": "hn",
        "type": kind,
        "by": data.get("by"),
        "title": data.get("title") or (data.get("text") or "")[:200],
        "url": data.get("url"),
        "score": data.get("score", 0),
        "descendants": data.get("descendants", 0),
        # A comment's parent is the story or comment it replies to — the HN
        # equivalent of a Bluesky reply, and the same reshare edge downstream.
        "parent_id": str(data["parent"]) if kind == "comment" and data.get("parent") else None,
        "created_utc": data.get("time"),
        "ingested_at": None,
    }


async def stream_hn(pool, dedup, scorer, detector, stats: Stats, deadline: float) -> None:
    """Sweep the newest HN items repeatedly; comments give the thread structure."""
    async with aiohttp.ClientSession() as session:
        while time.monotonic() < deadline:
            try:
                async with session.get(f"{HN_BASE}/maxitem.json", timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    newest = int(await resp.text())
            except Exception:
                await asyncio.sleep(HN_INTERVAL_SECONDS)
                continue

            items = await asyncio.gather(
                *(_hn_item(session, i) for i in range(newest - HN_BATCH, newest))
            )
            for record in items:
                if record is None:
                    continue
                from datetime import UTC, datetime

                record["ingested_at"] = datetime.now(UTC).isoformat()
                stats["hn"] += 1
                await _write(record, pool, dedup, scorer, detector, stats)

            await asyncio.sleep(HN_INTERVAL_SECONDS)


async def report(stats: Stats, pool, deadline: float) -> None:
    while time.monotonic() < deadline:
        await asyncio.sleep(30)
        async with pool.acquire() as conn:
            reshares = await conn.fetchval(
                "SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'reshare' AND source_id NOT LIKE 'sim:%'"
            )
        log.info("%s | reshare edges=%s", stats.line(), reshares)


async def main_async(minutes: float, sources: set[str], max_per_sec: int) -> None:
    pool = await asyncpg.create_pool(DB_URL, min_size=2, max_size=8)
    dedup = DedupFilter(max_size=200_000)
    scorer = VelocityScorer()
    detector = AnomalyDetector()
    stats = Stats()
    deadline = time.monotonic() + minutes * 60

    tasks = [asyncio.create_task(report(stats, pool, deadline))]
    if "bluesky" in sources:
        tasks.append(
            asyncio.create_task(
                stream_bluesky(pool, dedup, scorer, detector, stats, deadline, max_per_sec)
            )
        )
    if "hn" in sources:
        tasks.append(asyncio.create_task(stream_hn(pool, dedup, scorer, detector, stats, deadline)))

    log.info("ingesting %s for %.0f minutes", ", ".join(sorted(sources)), minutes)
    try:
        await asyncio.sleep(minutes * 60)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        async with pool.acquire() as conn:
            reshares = await conn.fetchval(
                "SELECT COUNT(*) FROM graph_edges WHERE edge_type = 'reshare' AND source_id NOT LIKE 'sim:%'"
            )
            nodes = await conn.fetchval(
                "SELECT COUNT(*) FROM graph_nodes WHERE id NOT LIKE 'sim:%'"
            )
        await pool.close()
        log.info("done — %s", stats.line())
        log.info("real graph now holds %s nodes and %s reshare edges", nodes, reshares)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest live traffic into the graph")
    parser.add_argument("--minutes", type=float, default=DEFAULT_MINUTES)
    parser.add_argument("--sources", default="bluesky,hn")
    parser.add_argument("--max-per-sec", type=int, default=DEFAULT_MAX_PER_SEC)
    args = parser.parse_args()

    configure_logging()
    asyncio.run(
        main_async(args.minutes, set(args.sources.split(",")), args.max_per_sec)
    )


if __name__ == "__main__":
    main()
