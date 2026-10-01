"""
LlamaIndex FunctionTools for the Diffusion ReAct agent.

Four tools:
  get_propagation_path     — BFS traversal from a root node, up to max_depth hops
  search_similar_trends    — pgvector cosine similarity over stored trend embeddings
  classify_virality        — cascade size + in/out degree features for a node
  classify_virality_model  — trained LightGBM probability the cascade was coordinated

classify_virality is kept alongside the model: it needs no training artefact and
still works on nodes that have no reshare cascade, which is where the model
declines to answer.
"""

from __future__ import annotations

import json

import aiohttp
from llama_index.core.tools import FunctionTool
from llama_index.core.tools.utils import create_schema_from_function

from agent.types import Emitter
from graph import embeddings as emb
from graph import queries
from ml.predict import score_cascade

# Edges shown verbatim by get_propagation_path; the rest are summarised.
PATH_SAMPLE = 25
# Cascades scored when classify_virality_model is asked about a topic.
TOPIC_CASCADES = 5
_TOPIC_PREFIXES = ("entity:", "github:repo:")


def is_topic(node_id: str) -> bool:
    """Entities and repos: the node types the anomaly detector scores."""
    return node_id.startswith(_TOPIC_PREFIXES)


def build_tools(
    conn,
    session: aiohttp.ClientSession,
    emit: Emitter | None = None,
    include_model_tool: bool = True,
) -> list[FunctionTool]:
    """
    Return the agent tools bound to an open DB connection and HTTP session.
    Call once per investigation — do not share across concurrent runs.
    If emit is provided it will be called with tool_call/tool_result events.

    include_model_tool=False withholds classify_virality_model, which is what
    ml/evaluate.py needs to measure the LLM on its own rather than measuring the
    classifier through it.
    """

    async def _call(tool: str, **args) -> None:
        if emit:
            await emit({"type": "tool_call", "tool": tool, "args": args})

    async def _result(tool: str, summary: str, empty: bool = False) -> None:
        # `empty` marks a result that found nothing about this node, so the
        # caller can tell a verdict built on evidence from one built on none.
        if emit:
            await emit({"type": "tool_result", "tool": tool, "summary": summary, "empty": empty})

    async def get_propagation_path(node_id: str, max_depth: int = 6) -> str:
        """
        BFS over the propagation graph from node_id up to max_depth hops.
        Returns JSON: {edges_total, edges_per_depth, distinct_sources, sample}, where
        sample holds the first edges as {source_id, target_id, edge_type, platform, ts, depth}.
        """
        max_depth = min(max_depth, 10)
        await _call("get_propagation_path", node_id=node_id, max_depth=max_depth)
        edges = await queries.get_propagation_path(conn, node_id, max_depth)
        await _result("get_propagation_path", f"{len(edges)} edges found", empty=not edges)
        # A summary plus a sample, not every edge: a 350-edge cascade as raw
        # JSON overran Groq's request limit (HTTP 413) and would crowd a local
        # model's context with rows it cannot use anyway.
        by_depth: dict[int, int] = {}
        for e in edges:
            by_depth[e["depth"]] = by_depth.get(e["depth"], 0) + 1
        return json.dumps({
            "edges_total": len(edges),
            "edges_per_depth": by_depth,
            "distinct_sources": len({e["source_id"] for e in edges}),
            "sample": edges[:PATH_SAMPLE],
        }, default=str)

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
        # Sequential: one asyncpg connection runs one query at a time, and the
        # asyncio.gather this used failed every call with "another operation
        # is in progress".
        cascade = await queries.get_cascade_size(conn, node_id)
        degree = await queries.get_node_degree(conn, node_id)
        await _result(
            "classify_virality",
            f"cascade={cascade} in={degree['in_degree']} out={degree['out_degree']}",
            empty=not (cascade or degree["in_degree"] or degree["out_degree"]),
        )
        return json.dumps({
            "cascade_size": cascade,
            "in_degree": degree["in_degree"],
            "out_degree": degree["out_degree"],
        })

    async def _score_topic(node_id: str) -> str:
        """
        Score the cascades a topic's recent mentions belong to.

        An entity is not the root of a cascade, so the classifier has nothing to
        say about it directly; what it can judge are the reshare trees the
        mentioning posts sit in. Those are scored one by one and combined,
        weighted by how many of the topic's mentions each one carries.
        """
        activity = await queries.get_entity_activity(conn, node_id, top=TOPIC_CASCADES)
        scored = []
        for c in activity["top_cascades"]:
            r = await score_cascade(conn, c["root_id"])
            if r is not None:
                scored.append({"root_id": c["root_id"], "mentions": c["mentions"],
                               "cascade_size": r["cascade_size"], "p_coordinated": r["p_coordinated"]})
        if not scored:
            await _result("classify_virality_model", "no scorable cascade behind the topic", empty=True)
            return json.dumps({
                "available": False,
                "reason": (
                    f"{activity['mentions']} recent mentions, but none of the posts belongs "
                    "to a reshare cascade the model can score: the spike is standalone posts."
                ),
            })
        weight = sum(c["mentions"] for c in scored)
        p = sum(c["p_coordinated"] * c["mentions"] for c in scored) / weight
        share = weight / activity["mentions"]
        await _result("classify_virality_model", f"topic: p_coordinated={p:.3f} over {len(scored)} cascades")
        return json.dumps({
            "topic": node_id,
            "p_coordinated_weighted": round(p, 4),
            "share_of_mentions_scored": round(share, 3),
            "cascades": scored,
            # Stated, not left for the model to infer. On a live topic the
            # scored cascades held a handful of 237 mentions, and the agent
            # still called the whole topic coordinated on that probability.
            "caution": (
                f"These cascades carry {share:.0%} of the topic's mentions, and live "
                "cascades are far smaller and flatter than the ones the model was "
                "trained on. Treat the probability as weak evidence about the topic "
                "as a whole; weigh get_entity_activity alongside it."
            ),
        })

    async def get_entity_activity(node_id: str) -> str:
        """
        Recent activity behind a topic node (an entity or a repo).
        Returns JSON: {mentions, distinct_authors, top_author_share, distinct_cascades, top_cascades}.
        """
        await _call("get_entity_activity", node_id=node_id)
        activity = await queries.get_entity_activity(conn, node_id)
        await _result(
            "get_entity_activity",
            f"{activity['mentions']} mentions by {activity['distinct_authors']} authors",
            empty=not activity["mentions"],
        )
        return json.dumps(activity, default=str)

    async def classify_virality_model(node_id: str) -> str:
        """
        Score the cascade with the trained LightGBM classifier.
        Returns JSON: {p_coordinated, cascade_size, top_features}.
        p_coordinated near 1.0 means coordinated amplification, near 0.0 organic.
        """
        await _call("classify_virality_model", node_id=node_id)
        result = await score_cascade(conn, node_id)
        if result is None and is_topic(node_id):
            return await _score_topic(node_id)
        if result is None:
            await _result("classify_virality_model", "unavailable", empty=True)
            return json.dumps(
                {
                    "available": False,
                    "reason": (
                        "no trained model, or this node has no reshare cascade. "
                        "Fall back to classify_virality."
                    ),
                }
            )
        await _result(
            "classify_virality_model",
            f"p_coordinated={result['p_coordinated']:.3f}",
        )
        return json.dumps(result)

    def _sync_stub(*args, **kwargs):
        """Sync stub — agent always uses the async variant."""
        raise NotImplementedError

    def _tool(async_fn, name: str, description: str) -> FunctionTool:
        # The schema has to come from the async function. Left to default,
        # FunctionTool reads it off the sync stub and advertises every tool as
        # taking `args` and `kwargs`, so no model was ever told the parameter is
        # `node_id` — the local models guessed `node` and every call failed.
        return FunctionTool.from_defaults(
            fn=_sync_stub,
            async_fn=async_fn,
            name=name,
            description=description,
            fn_schema=create_schema_from_function(name, async_fn),
        )

    tools = [
        _tool(
            get_propagation_path,
            "get_propagation_path",
            (
                "BFS traversal of the propagation graph from a given node. "
                "Use to trace how a trend spread across the graph."
            ),
        ),
        _tool(
            search_similar_trends,
            "search_similar_trends",
            (
                "Semantic similarity search over historical trend embeddings. "
                "Use to find past cases that resemble the current anomaly."
            ),
        ),
        _tool(
            classify_virality,
            "classify_virality",
            (
                "Compute cascade size and degree metrics for a node. "
                "Use to quantify spread velocity and network centrality."
            ),
        ),
    ]

    # Anomalies fire on topics, so the topic tool is always offered.
    tools.append(
        _tool(
            get_entity_activity,
            "get_entity_activity",
            (
                "For a topic node — an entity such as 'entity:person:...' or a repo, "
                "which is what anomalies fire on — the posts that mentioned it in the "
                "last hour: how many, how many distinct accounts wrote them, the share "
                "from the single most active account, and which cascades they sit in. "
                "Many independent accounts suggests organic interest; a few accounts "
                "producing most of the volume suggests coordination."
            ),
        )
    )

    if not include_model_tool:
        return tools

    tools.append(
        _tool(
            classify_virality_model,
            "classify_virality_model",
            (
                "Score the cascade with a trained classifier and get the "
                "probability it was coordinated, plus the features that drove "
                "that score. This is the most reliable evidence available — "
                "prefer it over reasoning from raw counts, but state the "
                "probability alongside your own reading of the other tools."
            ),
        )
    )
    return tools
