import asyncio
import json
import logging
import os

from aiokafka import AIOKafkaConsumer

from processing.anomaly_detector import AnomalyDetector
from processing.dedup import DedupFilter
from processing.entity_extractor import Node, extract_entity_set
from processing.velocity_scorer import VelocityScorer

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPICS = ["reddit-raw", "hn-raw", "gh-raw"]
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "diffusion-processor")

# Entity types whose activity is meaningful to track for trend velocity.
_VELOCITY_TYPES = {"named_entity", "repo"}

log = logging.getLogger(__name__)


async def handle(
    record: dict,
    scorer: VelocityScorer,
    detector: AnomalyDetector,
) -> None:
    platform = record.get("platform", "unknown")
    es = extract_entity_set(record)
    if not es.nodes:
        return

    log.info(
        "extracted %d nodes %d edges from %s/%s",
        len(es.nodes),
        len(es.edges),
        platform,
        record.get("id"),
    )

    # Score velocity for trackable entity types and check for anomalies.
    for node in es.nodes:
        if node.type not in _VELOCITY_TYPES:
            continue
        vscore = scorer.record(node.id)
        await detector.evaluate(vscore, platform=platform)


async def main() -> None:
    dedup = DedupFilter()
    scorer = VelocityScorer()
    detector = AnomalyDetector()
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
                await handle(record, scorer, detector)
            except Exception:
                log.exception("error processing record %s", record.get("id"))
    finally:
        await consumer.stop()
        await detector.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
