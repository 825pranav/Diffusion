import os
from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI

DB_URL = os.getenv("DATABASE_URL", "postgresql://diffusion:diffusion@localhost:5432/diffusion")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db = await asyncpg.create_pool(DB_URL, min_size=2, max_size=10)
    yield
    await app.state.db.close()


app = FastAPI(
    title="Diffusion API",
    description="Distributed Trend Propagation Engine",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {"status": "ok"}
