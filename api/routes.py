from fastapi import APIRouter, HTTPException, Query, Request

from graph.queries import (
    get_cascade_size,
    get_node_degree,
    get_propagation_path,
    get_top_nodes_by_degree,
)

router = APIRouter()


# ── nodes ─────────────────────────────────────────────────────────────────────

@router.get("/nodes")
async def list_nodes(
    request: Request,
    platform: str | None = None,
    since_minutes: int = 60,
    limit: int = Query(20, le=100),
):
    """Top nodes by in-degree within the last `since_minutes` minutes."""
    async with request.app.state.db.acquire() as conn:
        rows = await get_top_nodes_by_degree(
            conn, platform=platform, since_minutes=since_minutes, limit=limit
        )
    return rows


@router.get("/nodes/{node_id}/degree")
async def node_degree(node_id: str, request: Request):
    """In/out degree for a single node."""
    async with request.app.state.db.acquire() as conn:
        return await get_node_degree(conn, node_id)


@router.get("/nodes/{node_id}/cascade")
async def node_cascade(
    node_id: str,
    request: Request,
    max_depth: int = Query(10, le=15),
):
    """Total nodes reachable from `node_id`."""
    async with request.app.state.db.acquire() as conn:
        size = await get_cascade_size(conn, node_id, max_depth=max_depth)
    return {"node_id": node_id, "cascade_size": size}


@router.get("/nodes/{node_id}/propagation")
async def node_propagation(
    node_id: str,
    request: Request,
    max_depth: int = Query(6, le=10),
):
    """BFS propagation path from `node_id` up to `max_depth` hops."""
    async with request.app.state.db.acquire() as conn:
        edges = await get_propagation_path(conn, node_id, max_depth=max_depth)
    if not edges:
        raise HTTPException(status_code=404, detail="node not found or no edges")
    return {"node_id": node_id, "edges": edges}
