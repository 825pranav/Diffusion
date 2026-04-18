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

import aiohttp
from llama_index.core.tools import FunctionTool

from graph import embeddings as emb
from graph import queries


def build_tools(conn, session: aiohttp.ClientSession) -> list[FunctionTool]:
    """
    Return the three agent tools bound to an open DB connection and HTTP session.
    Call once per investigation — do not share across concurrent runs.
    """

    async def get_propagation_path(node_id: str, max_depth: int = 6) -> str:
        """
        BFS over the propagation graph from node_id up to max_depth hops.
        Returns a JSON array of edges: [{source_id, target_id, edge_type, platform, ts, depth}].
        """
        edges = await queries.get_propagation_path(conn, node_id, max_depth)
        return json.dumps(edges, default=str)

    async def search_similar_trends(query: str, limit: int = 5) -> str:
        """
        Semantic similarity search over historical trend embeddings.
        Returns top-k results as JSON: [{node_id, platform, label, similarity}].
        """
        results = await emb.search_similar(conn, query, session, limit)
        return json.dumps(results, default=str)

    async def classify_virality(node_id: str) -> str:
        """
        Compute cascade size (total reachable nodes) and in/out degree for node_id.
        Returns JSON: {cascade_size, in_degree, out_degree}.
        Use these features to reason about whether the spread was organic or coordinated.
        """
        cascade, degree = await asyncio.gather(
            queries.get_cascade_size(conn, node_id),
            queries.get_node_degree(conn, node_id),
        )
        return json.dumps({
            "cascade_size": cascade,
            "in_degree": degree["in_degree"],
            "out_degree": degree["out_degree"],
        })

    return [
        FunctionTool.from_defaults(
            async_fn=get_propagation_path,
            name="get_propagation_path",
            description=(
                "BFS traversal of the propagation graph from a given node. "
                "Use to trace how a trend spread across the graph."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=search_similar_trends,
            name="search_similar_trends",
            description=(
                "Semantic similarity search over historical trend embeddings. "
                "Use to find past cases that resemble the current anomaly."
            ),
        ),
        FunctionTool.from_defaults(
            async_fn=classify_virality,
            name="classify_virality",
            description=(
                "Compute cascade size and degree metrics for a node. "
                "Use to quantify spread velocity and network centrality."
            ),
        ),
    ]
