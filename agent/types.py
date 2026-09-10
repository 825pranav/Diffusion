"""Shared type aliases for the agent package."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

# Async callable that accepts one event dict (or None to signal completion)
# and emits it to the SSE bus.
Emitter = Callable[[dict | None], Awaitable[None]]
