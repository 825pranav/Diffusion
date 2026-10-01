"""
Mastodon async Kafka producer.

Polls the public timeline of a Mastodon instance and publishes new
statuses to the `mastodon-raw` Kafka topic.

Instances are closing their public timelines to anonymous readers one by one,
so the default is whichever still serves it, and MASTODON_ACCESS_TOKEN (any
account's token with read:statuses) works on instances that do not.
"""

import asyncio
import logging
import os
import re
from datetime import UTC, datetime

import aiohttp
from aiokafka import AIOKafkaProducer

from config import configure_logging
from ingestion.utils import BoundedSeenSet, make_producer, publish

TOPIC = "mastodon-raw"
# mastodon.social returns 422 "This method requires an authenticated user" on
# the public timeline, and since 2026-10 so does mstdn.social, the previous
# default. hachyderm.io and fosstodon.org still serve it anonymously.
MASTODON_INSTANCE = os.getenv("MASTODON_INSTANCE", "https://hachyderm.io")
MASTODON_ACCESS_TOKEN = os.getenv("MASTODON_ACCESS_TOKEN", "")
POLL_INTERVAL = int(os.getenv("MASTODON_POLL_INTERVAL", "30"))  # seconds
BATCH_SIZE = int(os.getenv("MASTODON_BATCH_SIZE", "40"))        # max per poll

_HEADERS = {"Authorization": f"Bearer {MASTODON_ACCESS_TOKEN}"} if MASTODON_ACCESS_TOKEN else {}

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

    The boost branch is defensive rather than load-bearing: /timelines/public
    serves no reblogs (0 across 120 statuses sampled from three instances), so
    replies are the only propagation signal this endpoint actually yields.
    Boosts would arrive from the streaming API or an authenticated home
    timeline, and the branch is here so they are handled when they do.
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


def _should_publish(record: dict) -> bool:
    """
    Whether a serialized status is worth putting on the topic.

    Text alone is not the right test: a boost carries none of its own, so
    requiring it would discard the propagation edge along with the empty body.
    A record earns its place if it has something to say or somewhere it came
    from.
    """
    return bool(record.get("text") or record.get("parent_id"))


async def produce(session: aiohttp.ClientSession, producer: AIOKafkaProducer, seen: BoundedSeenSet) -> None:
    url = f"{MASTODON_INSTANCE}/api/v1/timelines/public"
    params = {"limit": BATCH_SIZE, "local": "false"}
    try:
        async with session.get(url, params=params, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status in (401, 422):
                log.error(
                    "%s requires authentication for its public timeline: set "
                    "MASTODON_ACCESS_TOKEN or point MASTODON_INSTANCE elsewhere",
                    MASTODON_INSTANCE,
                )
                return
            resp.raise_for_status()
            statuses = await resp.json()
    except Exception:
        log.exception("failed to fetch Mastodon public timeline")
        return

    published = 0
    for status in statuses:
        if status["id"] in seen:
            continue
        record = _serialize(status)
        if _should_publish(record):
            await publish(producer, TOPIC, record)
            published += 1
        seen.add(status["id"])

    # Counted on publish, not on arrival: textless, parentless statuses are
    # dropped above and used to be reported as published anyway.
    log.info("published %d new statuses from %s", published, MASTODON_INSTANCE)


async def main() -> None:
    producer = make_producer()
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
