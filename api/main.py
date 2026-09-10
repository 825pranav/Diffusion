from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

import aiohttp
import asyncpg
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from agent.agent import agent_listener
from api.routes import router
from api.sse import publish as sse_publish
from api.sse import router as sse_router
from api.websocket import graph_delta_listener
from api.websocket import router as ws_router
from config import DB_URL, configure_logging
from graph import embeddings

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

configure_logging()


def _make_emit(anomaly_id: int):
    async def emit(event):
        await sse_publish(anomaly_id, event)
    return emit


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = await asyncpg.create_pool(DB_URL, min_size=2, max_size=10)
    app.state.http = aiohttp.ClientSession()
    delta_task = asyncio.create_task(graph_delta_listener())
    agent_task = asyncio.create_task(agent_listener(emit_factory=_make_emit))
    # Keeps trend_embeddings populated; without it the agent's similarity search
    # has nothing to retrieve.
    indexer_task = asyncio.create_task(
        embeddings.indexer_loop(app.state.db, app.state.http)
    )
    yield
    for task in (agent_task, delta_task, indexer_task):
        task.cancel()
    await asyncio.gather(agent_task, delta_task, indexer_task, return_exceptions=True)
    await app.state.http.close()
    await app.state.db.close()


app = FastAPI(
    title="Diffusion API",
    description="Distributed Trend Propagation Engine",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health(request: Request):
    try:
        await request.app.state.db.fetchval("SELECT 1")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"db unavailable: {exc}") from exc
    return {"status": "ok"}


app.include_router(router)
app.include_router(ws_router)
app.include_router(sse_router)
