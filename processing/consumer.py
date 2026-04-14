import asyncio
import json
import logging
import os

from aiokafka import AIOKafkaConsumer

from processing.dedup import DedupFilter
from processing.entity_extractor import extract_entities

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPICS = ["reddit-raw", "hn-raw", "gh-raw"]
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "diffusion-processor")

log = logging.getLogger(__name__)


async def handle(record: dict) -> None:
    entities = extract_entities(record)
    if entities:
        log.info("extracted %d entities from %s/%s", len(entities), record.get("platform"), record.get("id"))
    # downstream: graph writes + velocity scoring go here


async def main() -> None:
    dedup = DedupFilter()
    consumer = AIOKafkaConsumer(
        *TOPICS,
        bootstrap_servers=KAFKA_BROKER,
        group_id=GROUP_ID,
        value_deserializer=lambda b: json.loads(b.decode()),
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )
    await consumer.start()
    log.info("consumer started, listening on %s", TOPICS)
    try:
        async for msg in consumer:
            record = msg.value
            if not dedup.is_new(record):
                continue
            try:
                await handle(record)
            except Exception:
                log.exception("error processing record %s", record.get("id"))
    finally:
        await consumer.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
