"""
Bluesky async Kafka producer.

Connects to the Bluesky Jetstream WebSocket and streams real-time posts
to the `bluesky-raw` Kafka topic. No auth required — Jetstream is public.

Jetstream docs: https://docs.bsky.app/docs/advanced-guides/firehose
"""

import asyncio
import json
import logging
import os
from datetime import UTC, datetime

import aiohttp
from aiokafka import AIOKafkaProducer

from config import KAFKA_BROKER, configure_logging

TOPIC = "bluesky-raw"

POST_COLLECTION = "app.bsky.feed.post"
REPOST_COLLECTION = "app.bsky.feed.repost"

# Subscribing to reposts as well as posts is what makes propagation observable:
# posts carry replies, reposts carry the post they amplify.
JETSTREAM_URL = os.getenv(
    "BLUESKY_JETSTREAM_URL",
    "wss://jetstream2.us-east.bsky.network/subscribe"
    f"?wantedCollections={POST_COLLECTION}&wantedCollections={REPOST_COLLECTION}",
)
RECONNECT_DELAY = int(os.getenv("BLUESKY_RECONNECT_DELAY", "5"))

log = logging.getLogger(__name__)


def _uri_to_id(uri: str | None) -> str | None:
    """
    Turn an AT-URI into the same id shape this producer emits.

    "at://did:plc:abc/app.bsky.feed.post/3kxyz" -> "did:plc:abc_3kxyz", so a
    reply or repost points at exactly the id its parent post was stored under.
    """
    if not uri or not uri.startswith("at://"):
        return None
    parts = uri.removeprefix("at://").split("/")
    if len(parts) != 3:
        return None
    return f"{parts[0]}_{parts[2]}"


def _serialize(msg: dict) -> dict | None:
    """
    Extract a flat record from a Jetstream commit event.

    Handles posts and reposts. Replies carry `reply.parent`, reposts carry
    `subject` — both name the post being propagated from, and both become a
    `reshare` edge downstream. Without them the graph has authorship and topic
    mentions but no propagation at all, so every cascade is a single node and
    the whole diffusion model has nothing to traverse.
    """
    if msg.get("kind") != "commit":
        return None
    commit = msg.get("commit", {})
    if commit.get("operation") != "create":
        return None

    collection = commit.get("collection", "")
    record = commit.get("record", {})
    did = msg.get("did", "")
    rkey = commit.get("rkey", "")

    base = {
        "id": f"{did}_{rkey}",
        "platform": "bluesky",
        "author": did,
        "created_utc": record.get("createdAt"),
        "ingested_at": datetime.now(UTC).isoformat(),
    }

    if collection == REPOST_COLLECTION:
        # A repost has no text of its own; it exists only to propagate.
        subject = _uri_to_id((record.get("subject") or {}).get("uri"))
        if not subject:
            return None
        return {**base, "type": "repost", "text": "", "langs": [], "parent_id": subject}

    text = record.get("text", "").strip()
    if not text:
        return None

    reply = record.get("reply") or {}
    return {
        **base,
        "type": "post",
        "text": text,
        "langs": record.get("langs") or [],
        "parent_id": _uri_to_id((reply.get("parent") or {}).get("uri")),
        "root_id": _uri_to_id((reply.get("root") or {}).get("uri")),
    }


async def stream(producer: AIOKafkaProducer) -> None:
    """Open Jetstream WebSocket and forward posts to Kafka. Reconnects on drop."""
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                log.info("connecting to Jetstream at %s", JETSTREAM_URL)
                async with session.ws_connect(JETSTREAM_URL) as ws:
                    log.info("Jetstream connected")
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        try:
                            data = json.loads(msg.data)
                        except json.JSONDecodeError:
                            continue
                        record = _serialize(data)
                        if record:
                            await producer.send_and_wait(TOPIC, json.dumps(record).encode())
            except Exception:
                log.exception("Jetstream disconnected, retrying in %ds", RECONNECT_DELAY)
                await asyncio.sleep(RECONNECT_DELAY)


async def main() -> None:
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)
    await producer.start()
    try:
        await stream(producer)
    finally:
        await producer.stop()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
