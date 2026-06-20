"""
pgvector embedding layer.

Uses Ollama (nomic-embed-text) — already in the stack as the LLM fallback,
so no new dependencies or accounts needed.
"""

from __future__ import annotations

import os
from typing import Any

import aiohttp

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_VERSION = os.getenv("EMBEDDING_VERSION", "v1")
EMBEDDING_DIMS = 768


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
    node_types: tuple[str, ...] = ("named_entity", "repo", "content"),
    batch_size: int = 50,
) -> int:
    """
    Embed any graph_nodes that don't yet have a trend_embedding row.
    Returns the number of nodes embedded.
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
