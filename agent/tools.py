"""
LlamaIndex FunctionTools for the Diffusion ReAct agent.

Three tools:
  get_propagation_path   — BFS traversal from a root node, up to max_depth hops
  search_similar_trends  — pgvector cosine similarity over stored trend embeddings
  classify_virality      — cascade size + in/out degree features for a node
"""

from __future__ import annotations

import asyncio
import json
from typing import Awaitable, Callable

import aiohttp
from llama_index.core.tools import FunctionTool

from graph import embeddings as emb
from graph import queries

Emitter = Callable[[dict | None], Awaitable[None]]


def build_tools(conn, session: aiohttp.ClientSession, emit: Emitter | None = None) -> list[FunctionTool]:
    """
    Return the three agent tools bound to an open DB connection and HTTP session.
    Call once per investigation — do not share across concurrent runs.
    If emit is provided it will be called with tool_call/tool_result events.
    """

    async def _call(tool: str, **args) -> None:
        if emit:
            await emit({"type": "tool_call", "tool": tool, "args": args})

    async def _result(tool: str, summary: str) -> None:
        if emit:
            await emit({"type": "tool_result", "tool": tool, "summary": summary})

    async def get_propagation_path(node_id: str, max_depth: int = 6) -> str:
        """
        BFS over the propagation graph from node_id up to max_depth hops.
        Returns a JSON array of edges: [{source_id, target_id, edge_type, platform, ts, depth}].
        """
        max_depth = min(max_depth, 10)
        await _call("get_propagation_path", node_id=node_id, max_depth=max_depth)
        edges = await queries.get_propagation_path(conn, node_id, max_depth)
        await _result("get_propagation_path", f"{len(edges)} edges found")
        return json.dumps(edges, default=str)

    async def search_similar_trends(query: str, limit: int = 5) -> str:
        """
        Semantic similarity search over historical trend embeddings.
        Returns top-k results as JSON: [{node_id, platform, label, similarity}].
        """
        await _call("search_similar_trends", query=query, limit=limit)
        results = await emb.search_similar(conn, query, session, limit)
        await _result("search_similar_trends", f"{len(results)} similar trends")
        return json.dumps(results, default=str)

    async def classify_virality(node_id: str) -> str:
        """
        Compute cascade size (total reachable nodes) and in/out degree for node_id.
        Returns JSON: {cascade_size, in_degree, out_degree}.
        Use these features to reason about whether the spread was organic or coordinated.
        """
        await _call("classify_virality", node_id=node_id)
        cascade, degree = await asyncio.gather(
            queries.get_cascade_size(conn, node_id),
            queries.get_node_degree(conn, node_id),
        )
        await _result(
            "classify_virality",
            f"cascade={cascade} in={degree['in_degree']} out={degree['out_degree']}",
        )
        return json.dumps({
            "cascade_size": cascade,
            "in_degree": degree["in_degree"],
            "out_degree": degree["out_degree"],
        })

    def _sync_stub(*args, **kwargs):
        """Sync stub — agent always uses the async variant."""
        raise NotImplementedError

    return [
        FunctionTool.from_defaults(
            fn=_sync_stub,
            async_fn=get_propagation_path,
            name="get_propagation_path",
            description=(
                "BFS traversal of the propagation graph from a given node. "
                "Use to trace how a trend spread across the graph."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_sync_stub,
            async_fn=search_similar_trends,
            name="search_similar_trends",
            description=(
                "Semantic similarity search over historical trend embeddings. "
                "Use to find past cases that resemble the current anomaly."
            ),
        ),
        FunctionTool.from_defaults(
            fn=_sync_stub,
            async_fn=classify_virality,
            name="classify_virality",
            description=(
                "Compute cascade size and degree metrics for a node. "
                "Use to quantify spread velocity and network centrality."
            ),
        ),
    ]
