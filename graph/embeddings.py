"""
pgvector embedding layer.

Uses Ollama (nomic-embed-text) — already in the stack as the LLM fallback,
so no new dependencies or accounts needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import aiohttp

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_VERSION = os.getenv("EMBEDDING_VERSION", "v1")
EMBEDDING_DIMS = 768

# How often the background indexer looks for nodes that still need embedding.
INDEXER_INTERVAL_SECONDS = int(os.getenv("EMBEDDING_INDEXER_INTERVAL", "30"))
INDEXER_BATCH_SIZE = int(os.getenv("EMBEDDING_INDEXER_BATCH", "50"))

log = logging.getLogger(__name__)


async def embed(text: str, session: aiohttp.ClientSession) -> list[float]:
    async with session.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    return data["embedding"]


async def store_embedding(
    conn,
    node_id: str,
    platform: str,
    text: str,
    session: aiohttp.ClientSession,
) -> None:
    vector = await embed(text, session)
    await conn.execute(
        """
        INSERT INTO trend_embeddings (node_id, platform, embedding, model_name, model_version)
        VALUES ($1, $2, $3::vector, $4, $5)
        ON CONFLICT (node_id, platform, model_name, model_version) DO NOTHING
        """,
        node_id, platform, str(vector), EMBEDDING_MODEL, EMBEDDING_VERSION,
    )


async def search_similar(
    conn,
    text: str,
    session: aiohttp.ClientSession,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Return the closest stored trends by cosine similarity."""
    vector = await embed(text, session)
    rows = await conn.fetch(
        """
        SELECT
            te.node_id,
            te.platform,
            n.label,
            1 - (te.embedding <=> $1::vector) AS similarity
        FROM trend_embeddings te
        JOIN graph_nodes n ON n.id = te.node_id AND n.platform = te.platform
        WHERE te.model_name    = $2
          AND te.model_version = $3
        ORDER BY te.embedding <=> $1::vector
        LIMIT $4
        """,
        str(vector), EMBEDDING_MODEL, EMBEDDING_VERSION, limit,
    )
    return [dict(r) for r in rows]


async def embed_unindexed_nodes(
    conn,
    session: aiohttp.ClientSession,
    node_types: tuple[str, ...] = ("named_entity", "repo"),
    batch_size: int = 50,
) -> int:
    """
    Embed any graph_nodes that don't yet have a trend_embedding row.
    Returns the number of nodes embedded.

    Only *trend* nodes are indexed. Individual posts are content, not trends, and
    embedding them would mean one Ollama round-trip per ingested item — the table
    is searched to answer "what past trend does this resemble", which topics and
    repos answer and individual posts do not.
    """
    rows = await conn.fetch(
        """
        SELECT n.id, n.platform, n.label
        FROM graph_nodes n
        WHERE n.type = ANY($1::text[])
          AND NOT EXISTS (
              SELECT 1 FROM trend_embeddings te
              WHERE te.node_id = n.id
                AND te.platform = n.platform
                AND te.model_name = $2
                AND te.model_version = $3
          )
        LIMIT $4
        """,
        list(node_types), EMBEDDING_MODEL, EMBEDDING_VERSION, batch_size,
    )

    for row in rows:
        await store_embedding(conn, row["id"], row["platform"], row["label"], session)

    return len(rows)


async def indexer_loop(
    pool,
    session: aiohttp.ClientSession,
    interval_seconds: int = INDEXER_INTERVAL_SECONDS,
    batch_size: int = INDEXER_BATCH_SIZE,
) -> None:
    """
    Background task — keeps trend_embeddings caught up with graph_nodes.

    Embedding runs here rather than inline in the Kafka consumer so that a slow
    or unavailable Ollama cannot stall ingestion.  Without this loop nothing ever
    calls embed_unindexed_nodes, trend_embeddings stays empty, and the agent's
    search_similar_trends tool silently returns no results.

    Failures are logged and retried on the next tick — an unreachable embedding
    backend degrades similarity search, it does not stop the pipeline.
    """
    while True:
        try:
            async with pool.acquire() as conn:
                embedded = await embed_unindexed_nodes(
                    conn, session, batch_size=batch_size
                )
            if embedded:
                log.info("embedded %d new nodes", embedded)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.warning("embedding indexer pass failed — retrying", exc_info=True)
        await asyncio.sleep(interval_seconds)
