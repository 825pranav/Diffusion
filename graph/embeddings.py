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
