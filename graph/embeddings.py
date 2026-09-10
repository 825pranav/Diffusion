"""
pgvector embedding layer.

Uses Ollama (nomic-embed-text) — already in the stack as the LLM fallback,
so no new dependencies or accounts needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
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


_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9._:-]+$")


def _index_name(model_name: str, model_version: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", f"{model_name}_{model_version}".lower()).strip("_")
    return f"idx_trend_emb_hnsw_{slug}"


async def ensure_partial_index(
    conn,
    model_name: str = EMBEDDING_MODEL,
    model_version: str = EMBEDDING_VERSION,
) -> str:
    """
    Create the HNSW index covering exactly one embedding model version.

    search_similar filters on (model_name, model_version) and orders by distance.
    A single HNSW index over the whole table cannot serve that: the index walk
    returns the globally nearest vectors and the filter is applied *afterwards*,
    so neighbours belonging to other model versions consume the result budget and
    the query silently returns fewer — or worse — rows than asked for. The effect
    grows with the share of the table the filter excludes.

    pgvector's `iterative_scan`, which fixes this in general, needs 0.8+; the
    bundled build is 0.6.2. A partial index does the job here because the filter
    is low-cardinality and known in advance: every row in the index already
    satisfies the predicate, so the whole walk is usable.

    ml/eval_retrieval.py measures the difference.
    """
    if not (_SAFE_IDENTIFIER.match(model_name) and _SAFE_IDENTIFIER.match(model_version)):
        raise ValueError(
            f"unsafe embedding identifiers for DDL: {model_name!r}, {model_version!r}"
        )
    name = _index_name(model_name, model_version)
    await conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS {name}
        ON trend_embeddings USING hnsw (embedding vector_cosine_ops)
        WHERE model_name = '{model_name}' AND model_version = '{model_version}'
        """
    )
    return name


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
    index_ready = False
    while True:
        try:
            async with pool.acquire() as conn:
                if not index_ready:
                    # Cheap and idempotent, but needs a connection, so it happens
                    # here rather than at import time.
                    await ensure_partial_index(conn)
                    index_ready = True
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
