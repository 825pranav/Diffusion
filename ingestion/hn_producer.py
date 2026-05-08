"""
Hacker News async Kafka producer.

Polls the HN Firebase REST API for new stories and publishes
them to the `hn-raw` Kafka topic.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()

import aiohttp
from aiokafka import AIOKafkaProducer

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = "hn-raw"
HN_BASE = "https://hacker-news.firebaseio.com/v0"
POLL_INTERVAL = int(os.getenv("HN_POLL_INTERVAL", "60"))  # seconds

log = logging.getLogger(__name__)


async def _fetch_json(session: aiohttp.ClientSession, url: str) -> dict | list:
    async with session.get(url) as resp:
        resp.raise_for_status()
        return await resp.json()


async def _fetch_item(session: aiohttp.ClientSession, item_id: int) -> dict | None:
    data = await _fetch_json(session, f"{HN_BASE}/item/{item_id}.json")
    if not data or data.get("deleted") or data.get("dead"):
        return None
    return {
        "id": str(item_id),
        "type": "story",
        "platform": "hn",
        "title": data.get("title"),
        "url": data.get("url"),
        "score": data.get("score", 0),
        "by": data.get("by"),
        "descendants": data.get("descendants", 0),
        "created_utc": data.get("time"),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


async def produce(session: aiohttp.ClientSession, producer: AIOKafkaProducer, seen: set) -> None:
    try:
        new_ids: list[int] = await _fetch_json(session, f"{HN_BASE}/newstories.json")
    except Exception:
        log.exception("failed to fetch HN new stories")
        return

    fresh = [i for i in new_ids[:200] if i not in seen]
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
        log.info("published %d new stories from HN", published)

    if len(seen) > 5000:
        seen.difference_update(list(seen)[:2000])


async def main() -> None:
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)
    await producer.start()
    seen: set[int] = set()
    try:
        async with aiohttp.ClientSession() as session:
            while True:
                await produce(session, producer, seen)
                await asyncio.sleep(POLL_INTERVAL)
    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
