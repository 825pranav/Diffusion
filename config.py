"""
Shared configuration — read from environment / .env file.

Import constants from here instead of calling os.getenv directly in each
module.  Only values used by two or more modules live here; module-specific
settings (e.g. MASTODON_INSTANCE, BLUESKY_JETSTREAM_URL) stay in their
own files.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────────

# asyncpg requires the plain postgresql:// scheme; SQLAlchemy async uses
# postgresql+asyncpg://.  Normalise once here so callers never need to.
DB_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://diffusion:diffusion@localhost:5432/diffusion",
).replace("postgresql+asyncpg://", "postgresql://")

# ── Kafka ─────────────────────────────────────────────────────────────────────

KAFKA_BROKER: str = os.getenv("KAFKA_BROKER", "localhost:9092")

# One raw topic per platform, written by the producers in ingestion/.
RAW_TOPICS: list[str] = ["bluesky-raw", "mastodon-raw", "hn-raw", "gh-raw"]
# Entity mentions re-keyed by entity id, so every mention of one entity reaches
# the same scorer instance however many processors run in front of it.
MENTIONS_TOPIC = "entity-mentions"
# Records the processor could not write, with the error and their source offset.
DEAD_LETTER_TOPIC = "dead-letter"

# ── Logging ───────────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root logger. Call once per process entry-point."""
    logging.basicConfig(level=level, format=_LOG_FORMAT)
