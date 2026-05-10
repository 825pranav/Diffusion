import asyncio
import json
import logging
import os

from config import DB_URL, KAFKA_BROKER
import asyncpg
from aiokafka import AIOKafkaConsumer

from processing.anomaly_detector import AnomalyDetector
from processing.dedup import DedupFilter
from processing.entity_extractor import Node, Edge, extract_entity_set
from processing.velocity_scorer import VelocityScorer
from graph.queries import upsert_node, insert_edge

TOPICS = ["bluesky-raw", "mastodon-raw", "hn-raw", "gh-raw"]
GROUP_ID = os.getenv("KAFKA_GROUP_ID", "diffusion-processor")

_VELOCITY_TYPES = {"named_entity", "repo"}

log = logging.getLogger(__name__)


async def handle(
    record: dict,
    conn: asyncpg.Connection,
    scorer: VelocityScorer,
    detector: AnomalyDetector,
) -> None:
    platform = record.get("platform", "unknown")
    es = extract_entity_set(record)
    if not es.nodes:
        return

    # Write nodes to DB
    for node in es.nodes:
        await upsert_node(conn, node.id, node.type, node.platform, node.label, node.metadata)

    # Write edges to DB (triggers graph_delta NOTIFY → WebSocket → frontend)
    for edge in es.edges:
        await insert_edge(
            conn,
            source_id=edge.source_id,
            target_id=edge.target_id,
            edge_type=edge.edge_type,
            platform=edge.platform,
            weight=edge.weight,
            ts=edge.timestamp,
        )

    log.info("wrote %d nodes %d edges from %s/%s", len(es.nodes), len(es.edges), platform, record.get("id"))

    # Score velocity and detect anomalies
    for node in es.nodes:
        if node.type not in _VELOCITY_TYPES:
            continue
        vscore = scorer.record(node.id)
        await detector.evaluate(vscore, platform=platform)


async def main() -> None:
    dedup = DedupFilter()
    scorer = VelocityScorer()
    detector = AnomalyDetector()
    db = await asyncpg.create_pool(DB_URL, min_size=2, max_size=5)
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
                async with db.acquire() as conn:
                    await handle(record, conn, scorer, detector)
            except Exception:
                log.exception("error processing record %s", record.get("id"))
    finally:
        await consumer.stop()
        await detector.close()
        await db.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
