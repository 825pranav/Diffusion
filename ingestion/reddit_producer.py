"""
Reddit async Kafka producer.

Streams new submissions from subreddits via PRAW's built-in stream API,
serializes them, and publishes to the `reddit-raw` Kafka topic.

PRAW's stream handles deduplication internally — only genuinely new
submissions are yielded, so no seen-set is required here.
"""

import asyncio
import json
import os
from datetime import datetime, timezone

import praw
import praw.models
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


def _stream_submissions(reddit: praw.Reddit):
    """Blocking generator — runs in a thread via run_in_executor."""
    subreddit = reddit.subreddit("+".join(SUBREDDITS))
    # skip_existing=True skips the backlog of already-published posts on startup
    yield from subreddit.stream.submissions(skip_existing=True, pause_after=None)


async def main() -> None:
    reddit = _build_reddit_client()
    producer = AIOKafkaProducer(bootstrap_servers=KAFKA_BROKER)
    await producer.start()

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    def _fill_queue():
        """Thread: push serialized payloads into the async queue."""
        for submission in _stream_submissions(reddit):
            if submission is None:
                continue
            payload = json.dumps(_serialize_submission(submission)).encode()
            # Block the feeder thread if the queue is full (back-pressure).
            asyncio.run_coroutine_threadsafe(queue.put(payload), loop).result()

    # Run the blocking PRAW stream in a dedicated thread.
    loop.run_in_executor(None, _fill_queue)

    try:
        while True:
            payload = await queue.get()
            await producer.send_and_wait(TOPIC, payload)
    finally:
        await producer.stop()


if __name__ == "__main__":
    asyncio.run(main())
