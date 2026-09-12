"""
Hacker News async Kafka producer.

Polls the HN Firebase REST API for new stories and publishes
them to the `hn-raw` Kafka topic.
"""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime

import aiohttp
from aiokafka import AIOKafkaProducer

from config import KAFKA_BROKER, configure_logging
from ingestion.utils import BoundedSeenSet

TOPIC = "hn-raw"
HN_BASE = "https://hacker-news.firebaseio.com/v0"
POLL_INTERVAL = int(os.getenv("HN_POLL_INTERVAL", "60"))  # seconds

# Items to sweep back from `maxitem` each pass. Covers stories and comments
# together, which is the only way the reply structure is captured.
SWEEP_SIZE = int(os.getenv("HN_SWEEP_SIZE", "150"))

log = logging.getLogger(__name__)


async def _fetch_json(session: aiohttp.ClientSession, url: str) -> dict | list:
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _fetch_item(session: aiohttp.ClientSession, item_id: int) -> dict | None:
    """
    Fetch one HN item — story or comment.

    Comments matter as much as stories here: a comment carries `parent`, which
    is the only propagation signal HN offers, and it becomes a `reshare` edge
    downstream. Fetching stories alone (as this did) meant HN contributed
    authorship and topic mentions but no cascade structure whatsoever.

    The type is read from the item rather than assumed. Everything under
    `maxitem` is a mixture, and labelling a comment "story" made the two
    indistinguishable further down.
    """
    data = await _fetch_json(session, f"{HN_BASE}/item/{item_id}.json")
    if not data or data.get("deleted") or data.get("dead"):
        return None

    kind = data.get("type")
    if kind not in ("story", "comment"):
        return None

    return {
        "id": str(item_id),
        "type": kind,
        "platform": "hn",
        # Comments have no title; their text is what identifies them.
        "title": data.get("title") or (data.get("text") or "")[:200],
        "url": data.get("url"),
        "score": data.get("score", 0),
        "by": data.get("by"),
        "descendants": data.get("descendants", 0),
        "parent_id": str(data["parent"]) if kind == "comment" and data.get("parent") else None,
        "created_utc": data.get("time"),
        "ingested_at": datetime.now(UTC).isoformat(),
    }


async def produce(session: aiohttp.ClientSession, producer: AIOKafkaProducer, seen: BoundedSeenSet) -> None:
    """
    Sweep the newest items, stories and comments alike.

    `maxitem` is used rather than `newstories` because the latter returns only
    stories, and stories alone are isolated nodes — the reply structure that
    makes an HN thread a cascade lives entirely in its comments.
    """
    try:
        newest = int(await _fetch_json(session, f"{HN_BASE}/maxitem.json"))
    except Exception:
        log.exception("failed to fetch HN max item")
        return

    new_ids = list(range(newest - SWEEP_SIZE, newest))
    fresh = [i for i in new_ids if i not in seen]
    published = 0
    for item_id in fresh:
        try:
            item = await _fetch_item(session, item_id)
        except Exception:
            log.warning("failed to fetch HN item %s", item_id)
            seen.add(item_id)
            continue
        if item:
            await producer.send_and_wait(TOPIC, json.dumps(item).encode())
            published += 1
        seen.add(item_id)

    if published:
        log.info("published %d new HN items (stories and comments)", published)

async def main() -> None:
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)
    await producer.start()
    seen: BoundedSeenSet = BoundedSeenSet()
    try:
        async with aiohttp.ClientSession() as session:
            while True:
                await produce(session, producer, seen)
                await asyncio.sleep(POLL_INTERVAL)
    finally:
        await producer.stop()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
