"""
SSE endpoint for real-time agent thought step streaming.

Clients connect to GET /sse/agent/{anomaly_id} and receive a stream of
server-sent events until the investigation completes or the connection times out.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

log = logging.getLogger(__name__)

router = APIRouter()

SSE_STREAM_TIMEOUT_SECONDS = 120.0

# anomaly_id → active subscriber queues; None sentinel signals end-of-stream
_bus: dict[int, list[asyncio.Queue]] = defaultdict(list)


def subscribe(anomaly_id: int) -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue()
    _bus[anomaly_id].append(q)
    return q


def unsubscribe(anomaly_id: int, q: asyncio.Queue) -> None:
    try:
        _bus[anomaly_id].remove(q)
    except ValueError:
        pass
    if not _bus[anomaly_id]:
        _bus.pop(anomaly_id, None)


async def publish(anomaly_id: int, event: dict | None) -> None:
    """Publish a step event to all subscribers. Pass None to signal completion."""
    payload = json.dumps(event) if event is not None else None
    for q in list(_bus.get(anomaly_id, [])):
        await q.put(payload)


async def _event_stream(anomaly_id: int) -> AsyncGenerator[str, None]:
    q = subscribe(anomaly_id)
    try:
        while True:
            try:
                payload = await asyncio.wait_for(q.get(), timeout=SSE_STREAM_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                yield "event: timeout\ndata: {}\n\n"
                break
            if payload is None:
                yield "event: done\ndata: {}\n\n"
                break
            yield f"data: {payload}\n\n"
    finally:
        unsubscribe(anomaly_id, q)


@router.get("/sse/agent/{anomaly_id}")
async def sse_agent_steps(anomaly_id: int):
    """Stream agent investigation steps as SSE for a given anomaly."""
    return StreamingResponse(
        _event_stream(anomaly_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
