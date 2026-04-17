"""
pgvector embedding layer.

Handles generating and storing embeddings for graph nodes,
and semantic similarity search against historical trends.
"""

from __future__ import annotations

import os
from typing import Any

from openai import AsyncOpenAI

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_VERSION = os.getenv("EMBEDDING_VERSION", "v1")
EMBEDDING_DIMS = 1536

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


async def embed(text: str) -> list[float]:
    resp = await _get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMS,
    )
    return resp.data[0].embedding
