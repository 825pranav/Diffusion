"""
WebSocket endpoint for live graph delta streaming.

Clients connect to /ws/graph and receive edge-insert events in real time.
A background listener subscribes to the 'graph_delta' Postgres NOTIFY channel
and fans out each payload to all connected WebSocket clients.
"""

from __future__ import annotations

import asyncio
import logging
import os

import asyncpg
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

DB_URL = os.getenv("DATABASE_URL", "postgresql://diffusion:diffusion@localhost:5432/diffusion")

log = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.add(ws)
        log.info("ws client connected — total=%d", len(self._connections))

    def disconnect(self, ws: WebSocket) -> None:
        self._connections.discard(ws)
        log.info("ws client disconnected — total=%d", len(self._connections))

    async def broadcast(self, payload: str) -> None:
        dead: list[WebSocket] = []
        for ws in self._connections:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._connections.discard(ws)


manager = ConnectionManager()


async def graph_delta_listener() -> None:
    """
    Background coroutine — LISTENs on the 'graph_delta' Postgres channel
    and broadcasts each edge-insert payload to all connected WebSocket clients.
    Reconnects automatically on connection loss.
    """
    from graph.models import GRAPH_DELTA_CHANNEL

    while True:
        try:
            conn = await asyncpg.connect(DB_URL)

            def _on_notify(_conn, _pid, _channel, payload: str) -> None:
                asyncio.get_running_loop().call_soon_threadsafe(
                    asyncio.create_task, manager.broadcast(payload)
                )

            await conn.add_listener(GRAPH_DELTA_CHANNEL, _on_notify)
            log.info("graph delta listener active on channel '%s'", GRAPH_DELTA_CHANNEL)

            while not conn.is_closed():
                await asyncio.sleep(5)

            await conn.close()
        except Exception:
            log.warning("graph delta listener lost connection — retrying in 5s", exc_info=True)
            await asyncio.sleep(5)


@router.websocket("/ws/graph")
async def ws_graph(websocket: WebSocket) -> None:
    """Stream live graph edge deltas to the client."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
