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
JETSTREAM_URL = os.getenv(
    "BLUESKY_JETSTREAM_URL",
    "wss://jetstream2.us-east.bsky.network/subscribe?wantedCollections=app.bsky.feed.post",
)
RECONNECT_DELAY = int(os.getenv("BLUESKY_RECONNECT_DELAY", "5"))

log = logging.getLogger(__name__)


def _serialize(msg: dict) -> dict | None:
    """Extract a flat record from a Jetstream commit event."""
    if msg.get("kind") != "commit":
        return None
    commit = msg.get("commit", {})
    if commit.get("operation") != "create":
        return None
    record = commit.get("record", {})
    text = record.get("text", "").strip()
    if not text:
        return None

    did = msg.get("did", "")
    rkey = commit.get("rkey", "")
    langs = record.get("langs") or []

    return {
        "id": f"{did}_{rkey}",
        "type": "post",
        "platform": "bluesky",
        "text": text,
        "author": did,
        "langs": langs,
        "created_utc": record.get("createdAt"),
        "ingested_at": datetime.now(UTC).isoformat(),
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
