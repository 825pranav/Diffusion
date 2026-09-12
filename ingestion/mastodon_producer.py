"""
Mastodon async Kafka producer.

Polls the public timeline of a Mastodon instance and publishes new
statuses to the `mastodon-raw` Kafka topic. No auth required.

Default instance: mastodon.social — override with MASTODON_INSTANCE env var.
"""

import asyncio
import json
import logging
import os
import re
from datetime import UTC, datetime

import aiohttp
from aiokafka import AIOKafkaProducer

from config import KAFKA_BROKER, configure_logging
from ingestion.utils import BoundedSeenSet

TOPIC = "mastodon-raw"
MASTODON_INSTANCE = os.getenv("MASTODON_INSTANCE", "https://mastodon.social")
POLL_INTERVAL = int(os.getenv("MASTODON_POLL_INTERVAL", "30"))  # seconds
BATCH_SIZE = int(os.getenv("MASTODON_BATCH_SIZE", "40"))        # max per poll

_TAG_RE = re.compile(r"<[^>]+>")

log = logging.getLogger(__name__)


def _strip_html(html: str) -> str:
    return _TAG_RE.sub("", html).strip()


def _serialize(status: dict) -> dict:
    """
    Flatten one status, keeping whatever it propagated from.

    Mastodon exposes two propagation signals and both were being dropped:
    `in_reply_to_id` for replies, and `reblog` — the original status embedded
    whole — for boosts. Without them Mastodon contributes authorship and
    hashtags but no cascade structure, exactly as HN did.

    A boost carries no text of its own, so its language tag is taken from the
    status it boosts.
    """
    account = status.get("account", {})
    tags = [t["name"] for t in status.get("tags", [])]
    reblog = status.get("reblog") or {}
    # A boost propagates the embedded original; a reply propagates its parent.
    parent_id = reblog.get("id") or status.get("in_reply_to_id")

    return {
        "id": status["id"],
        "type": "boost" if reblog else "status",
        "platform": "mastodon",
        "text": _strip_html(status.get("content", "")),
        "author": account.get("acct"),
        "tags": tags,
        "parent_id": str(parent_id) if parent_id else None,
        "langs": [status.get("language") or reblog.get("language") or ""],
        "reblogs": status.get("reblogs_count", 0),
        "favourites": status.get("favourites_count", 0),
        "url": status.get("url"),
        "created_utc": status.get("created_at"),
        "ingested_at": datetime.now(UTC).isoformat(),
    }


async def produce(session: aiohttp.ClientSession, producer: AIOKafkaProducer, seen: BoundedSeenSet) -> None:
    url = f"{MASTODON_INSTANCE}/api/v1/timelines/public"
    params = {"limit": BATCH_SIZE, "local": "false"}
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            resp.raise_for_status()
            statuses = await resp.json()
    except Exception:
        log.exception("failed to fetch Mastodon public timeline")
        return

    fresh = [s for s in statuses if s["id"] not in seen]
    for status in fresh:
        record = _serialize(status)
        if record["text"]:
            await producer.send_and_wait(TOPIC, json.dumps(record).encode())
        seen.add(status["id"])

    log.info("published %d new statuses from %s", len(fresh), MASTODON_INSTANCE)


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
