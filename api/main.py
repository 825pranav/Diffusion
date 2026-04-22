import logging
import os
from contextlib import asynccontextmanager

import asyncio

import aiohttp
import asyncpg
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.websocket import graph_delta_listener, router as ws_router

DB_URL = os.getenv("DATABASE_URL", "postgresql://diffusion:diffusion@localhost:5432/diffusion")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = await asyncpg.create_pool(DB_URL, min_size=2, max_size=10)
    app.state.http = aiohttp.ClientSession()
    listener_task = asyncio.create_task(graph_delta_listener())
    yield
    listener_task.cancel()
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
async def health():
    return {"status": "ok"}


app.include_router(router)
app.include_router(ws_router)
