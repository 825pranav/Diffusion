import os

from fastapi import FastAPI

app = FastAPI(
    title="Diffusion API",
    description="Distributed Trend Propagation Engine",
    version="0.1.0",
)


@app.get("/health")
async def health():
    return {"status": "ok"}
