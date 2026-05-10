"""
GitHub async Kafka producer.

Polls the GitHub Events API for public events (WatchEvent, ForkEvent,
PushEvent) and publishes them to the `gh-raw` Kafka topic.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import aiohttp
from aiokafka import AIOKafkaProducer

from config import KAFKA_BROKER, configure_logging
from ingestion.utils import BoundedSeenSet

TOPIC = "gh-raw"
GH_TOKEN = os.getenv("GITHUB_TOKEN", "")
POLL_INTERVAL = int(os.getenv("GH_POLL_INTERVAL", "60"))  # seconds

TRACKED_EVENT_TYPES = {"WatchEvent", "ForkEvent", "PushEvent", "CreateEvent"}

log = logging.getLogger(__name__)

_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
    **({"Authorization": f"Bearer {GH_TOKEN}"} if GH_TOKEN else {}),
}


def _serialize_event(event: dict) -> dict:
    return {
        "id": event["id"],
        "type": event["type"],
        "platform": "github",
        "repo": event["repo"]["name"],
        "actor": event["actor"]["login"],
        "created_at": event.get("created_at"),
        "payload_action": event.get("payload", {}).get("action"),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


async def produce(
    session: aiohttp.ClientSession,
    producer: AIOKafkaProducer,
    seen: BoundedSeenSet,
) -> None:
    url = "https://api.github.com/events?per_page=100"
    try:
        async with session.get(url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            events: list[dict] = await resp.json()
    except Exception:
        log.exception("failed to fetch GitHub events")
        return

    published = 0
    for event in events:
        if event["type"] not in TRACKED_EVENT_TYPES:
            continue
        if event["id"] in seen:
            continue
        payload = json.dumps(_serialize_event(event)).encode()
        await producer.send_and_wait(TOPIC, payload)
        seen.add(event["id"])
        published += 1

    if published:
        log.info("published %d new GitHub events", published)

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
