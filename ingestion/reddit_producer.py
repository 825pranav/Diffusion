"""
Reddit async Kafka producer.

Streams new submissions and comments from subreddits via PRAW,
serializes them, and publishes to the `reddit-raw` Kafka topic.
"""

import asyncio
import json
import os
from datetime import datetime, timezone

import praw
from aiokafka import AIOKafkaProducer

KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC = "reddit-raw"
SUBREDDITS = os.getenv("REDDIT_SUBREDDITS", "programming,technology,MachineLearning").split(",")


def _build_reddit_client() -> praw.Reddit:
    return praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.environ.get("REDDIT_USER_AGENT", "diffusion/0.1"),
    )


def _serialize_submission(submission: praw.models.Submission) -> dict:
    return {
        "id": submission.id,
        "type": "submission",
        "platform": "reddit",
        "subreddit": submission.subreddit.display_name,
        "title": submission.title,
        "url": submission.url,
        "score": submission.score,
        "num_comments": submission.num_comments,
        "author": str(submission.author) if submission.author else None,
        "created_utc": submission.created_utc,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


async def produce(producer: AIOKafkaProducer, reddit: praw.Reddit) -> None:
    subreddit = reddit.subreddit("+".join(SUBREDDITS))
    loop = asyncio.get_event_loop()

    for submission in await loop.run_in_executor(None, lambda: list(subreddit.new(limit=None))):
        payload = json.dumps(_serialize_submission(submission)).encode()
        await producer.send_and_wait(TOPIC, payload)


async def main() -> None:
    reddit = _build_reddit_client()
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)
    await producer.start()
    try:
        while True:
            await produce(producer, reddit)
            await asyncio.sleep(30)
    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
