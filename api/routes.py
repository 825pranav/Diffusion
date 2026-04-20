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


# ── anomalies ─────────────────────────────────────────────────────────────────

@router.get("/anomalies")
async def list_anomalies(
    request: Request,
    investigated: bool | None = None,
    platform: str | None = None,
    limit: int = Query(20, le=100),
    offset: int = 0,
):
    """List anomaly events with optional filters."""
    filters, args = [], []

    if investigated is not None:
        args.append(investigated)
        filters.append(f"investigated = ${len(args)}")
    if platform is not None:
        args.append(platform)
        filters.append(f"platform = ${len(args)}")

    where = ("WHERE " + " AND ".join(filters)) if filters else ""
    args += [limit, offset]

    async with request.app.state.db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT id, node_id, platform, z_score, velocity, detected_at, investigated
            FROM anomaly_events
            {where}
            ORDER BY detected_at DESC
            LIMIT ${len(args) - 1} OFFSET ${len(args)}
            """,
            *args,
        )
    return [dict(r) for r in rows]


@router.get("/anomalies/{anomaly_id}")
async def get_anomaly(anomaly_id: int, request: Request):
    """Single anomaly event by id."""
    async with request.app.state.db.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM anomaly_events WHERE id = $1", anomaly_id
        )
    if not row:
        raise HTTPException(status_code=404, detail="anomaly not found")
    return dict(row)
