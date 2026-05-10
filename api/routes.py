import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Query, Request

from graph.queries import (
    get_cascade_size,
    get_node_degree,
    get_propagation_path,
    get_top_nodes_by_degree,
)

router = APIRouter()


# ── trending ──────────────────────────────────────────────────────────────────

# Frontend uses short codes; DB stores full names
_PLATFORM_DB = {"gh": "github", "hn": "hn", "bsky": "bluesky", "mdn": "mastodon"}

# DB platform names → frontend short codes
_PLATFORM_SHORT = {
    "hn": "hn",
    "github": "gh", "gh": "gh",
    "bluesky": "bsky", "bsky": "bsky",
    "mastodon": "mdn", "mdn": "mdn",
}


BREAKING_ANOMALY_WINDOW_MINUTES = 15
SPARKLINE_BUCKET_COUNT = 12
TRAJECTORY_RISE_THRESHOLD = 10
TRAJECTORY_COOL_THRESHOLD = -10


def _norm_platform(p: str) -> str:
    return _PLATFORM_SHORT.get(p.lower(), p[:4])


def _window_minutes(window: str) -> int:
    if window == "30m":
        return 30
    if window == "2h":
        return 120
    if window == "today":
        now = datetime.now(timezone.utc)
        return int((now.hour * 60 + now.minute) or 1)
    raise HTTPException(status_code=400, detail=f"invalid window: {window!r}; expected 30m, 2h, or today")


@router.get("/trending")
async def get_trending(
    request: Request,
    window: str = "30m",
    platform: str = "all",
    sort: str = "velocity",
    limit: int = Query(20, le=50),
):
    wm = _window_minutes(window)
    platform_filter = None if platform == "all" else _PLATFORM_DB.get(platform, platform)

    async with request.app.state.db.acquire() as conn:
        plat_args = [platform_filter] if platform_filter else []

        rows = await conn.fetch(
            f"""
            WITH recent AS (
                SELECT e.source_id, e.target_id, e.platform, e.ingested_at, e.ts
                FROM graph_edges e
                WHERE e.ingested_at >= now() - $1 * interval '1 minute'
                  {("AND e.platform = $3" if platform_filter else "")}
            ),
            prev AS (
                SELECT e.source_id, e.target_id
                FROM graph_edges e
                WHERE e.ingested_at >= now() - $2 * interval '1 minute'
                  AND e.ingested_at < now() - $1 * interval '1 minute'
                  {("AND e.platform = $3" if platform_filter else "")}
            ),
            node_curr AS (
                SELECT n.id, n.label, n.type AS node_type, n.platform AS node_platform,
                       n.metadata,
                       COUNT(r.source_id) AS curr_count
                FROM graph_nodes n
                JOIN recent r ON r.target_id = n.id
                WHERE n.type IN ('named_entity', 'repo', 'content')
                GROUP BY n.id, n.label, n.type, n.platform, n.metadata
            ),
            node_prev AS (
                SELECT n.id, COUNT(p.source_id) AS prev_count
                FROM graph_nodes n
                JOIN prev p ON p.target_id = n.id
                WHERE n.type IN ('named_entity', 'repo', 'content')
                GROUP BY n.id
            ),
            node_platforms AS (
                SELECT r.target_id AS node_id,
                       array_agg(DISTINCT r.platform) AS platforms,
                       MIN(r.ts) AS first_seen,
                       MAX(r.ingested_at) AS last_seen
                FROM recent r
                GROUP BY r.target_id
            ),
            breaking AS (
                SELECT DISTINCT node_id
                FROM anomaly_events
                WHERE detected_at >= now() - BREAKING_ANOMALY_WINDOW_MINUTES * interval '1 minute'
            )
            SELECT
                nc.id,
                nc.label,
                nc.node_type,
                nc.node_platform,
                nc.metadata,
                nc.curr_count,
                COALESCE(np2.prev_count, 0) AS prev_count,
                COALESCE(np.platforms, ARRAY[nc.node_platform]) AS platforms,
                np.first_seen,
                np.last_seen,
                (b.node_id IS NOT NULL) AS breaking
            FROM node_curr nc
            LEFT JOIN node_prev  np2 ON np2.id = nc.id
            LEFT JOIN node_platforms np ON np.node_id = nc.id
            LEFT JOIN breaking b ON b.node_id = nc.id
            ORDER BY nc.curr_count DESC
            LIMIT 50
            """,
            wm, wm * 2, *plat_args,
        )

        if not rows:
            return {"topics": [], "stats": {
                "trendingTopics": 0, "trendingDelta": 0,
                "fastestVelocity": 0, "fastestName": "—",
                "crossPlatform": 0, "breakingNow": 0,
            }}

        # ── Sparkline: SPARKLINE_BUCKET_COUNT buckets over the window per node ─
        node_ids = [r["id"] for r in rows]
        bucket_rows = await conn.fetch(
            """
            SELECT
                target_id AS node_id,
                width_bucket(
                    EXTRACT(EPOCH FROM ingested_at),
                    EXTRACT(EPOCH FROM now() - $2 * interval '1 minute'),
                    EXTRACT(EPOCH FROM now()),
                    $3
                ) AS bucket,
                COUNT(*) AS cnt
            FROM graph_edges
            WHERE target_id = ANY($1)
              AND ingested_at >= now() - $2 * interval '1 minute'
            GROUP BY target_id, bucket
            ORDER BY target_id, bucket
            """,
            node_ids, wm, SPARKLINE_BUCKET_COUNT,
        )

        # Build sparkline map: node_id → [12 values]
        sparkline_map: dict[str, list[float]] = {}
        for br in bucket_rows:
            nid = br["node_id"]
            if nid not in sparkline_map:
                sparkline_map[nid] = [0.0] * SPARKLINE_BUCKET_COUNT
            b = min(max(int(br["bucket"]) - 1, 0), 11)
            sparkline_map[nid][b] = float(br["cnt"])

        # Normalise 0-1
        for nid, vals in sparkline_map.items():
            mx = max(vals) or 1
            sparkline_map[nid] = [round(v / mx, 3) for v in vals]

        # ── Propagation: first edge per platform per node ─────────────────────
        prop_rows = await conn.fetch(
            """
            SELECT DISTINCT ON (target_id, platform)
                target_id AS node_id,
                platform,
                EXTRACT(EPOCH FROM (now() - ingested_at)) / 60 AS minutes_ago
            FROM graph_edges
            WHERE target_id = ANY($1)
              AND ingested_at >= now() - $2 * interval '1 minute'
            ORDER BY target_id, platform, ingested_at ASC
            """,
            node_ids, wm,
        )

        prop_map: dict[str, list[dict]] = {}
        for pr in prop_rows:
            nid = pr["node_id"]
            if nid not in prop_map:
                prop_map[nid] = []
            prop_map[nid].append({
                "platform": pr["platform"],
                "minutesAgo": int(pr["minutes_ago"]),
            })
        # Sort each by minutesAgo ascending (earliest first = origin)
        for nid in prop_map:
            prop_map[nid].sort(key=lambda x: x["minutesAgo"], reverse=True)

        # ── Assemble topics ───────────────────────────────────────────────────
        topics = []
        for r in rows:
            curr = float(r["curr_count"])
            prev = float(r["prev_count"])
            velocity = round(curr / wm, 2)
            vel_delta = 0
            if prev > 0:
                vel_delta = int(round((curr - prev) / prev * 100))
            elif curr > 0:
                vel_delta = 100

            trajectory = "plateau"
            if vel_delta > TRAJECTORY_RISE_THRESHOLD:
                trajectory = "rising"
            elif vel_delta < TRAJECTORY_COOL_THRESHOLD:
                trajectory = "cooling"

            raw_platforms = r["platforms"] or [r["node_platform"]]
            platforms = list(dict.fromkeys(
                _norm_platform(p) for p in raw_platforms
                if _norm_platform(p) in ("hn", "gh", "bsky", "mdn")
            ))
            if not platforms:
                platforms = [_norm_platform(r["node_platform"])]

            sparkline = sparkline_map.get(r["id"], [0.0] * SPARKLINE_BUCKET_COUNT)

            prop_steps = [
                {"platform": _norm_platform(s["platform"]), "minutesAgo": s["minutesAgo"]}
                for s in prop_map.get(r["id"], [])
                if _norm_platform(s["platform"]) in ("hn", "gh", "bsky", "mdn")
            ]
            if not prop_steps:
                prop_steps = [{"platform": platforms[0], "minutesAgo": 0}]

            # Derive source URL
            node_id = r["id"]
            node_type = r["node_type"]
            meta = r["metadata"] or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            url = None
            if node_type == "repo" and node_id.startswith("github:repo:"):
                repo_name = node_id.removeprefix("github:repo:")
                url = f"https://github.com/{repo_name}"
            elif node_type == "content":
                url = meta.get("url")
                if not url and node_id.startswith("hn:"):
                    hn_id = node_id.removeprefix("hn:")
                    url = f"https://news.ycombinator.com/item?id={hn_id}"
            elif node_type == "named_entity":
                label = r["label"] or ""
                url = f"https://hn.algolia.com/?q={label.replace(' ', '+')}"

            topics.append({
                "id": node_id,
                "name": r["label"] or node_id,
                "url": url,
                "platforms": platforms,
                "velocity": velocity,
                "velocityDelta": vel_delta,
                "trajectory": trajectory,
                "propagation": prop_steps,
                "sparkline": sparkline,
                "breaking": bool(r["breaking"]),
            })

        # ── Sort ──────────────────────────────────────────────────────────────
        if sort == "spread":
            topics.sort(key=lambda t: len(t["platforms"]), reverse=True)
        elif sort == "newest":
            # rows already ordered by activity; keep as-is
            pass
        # default (velocity) already sorted by curr_count DESC

        topics = topics[:limit]

        # ── Stats ─────────────────────────────────────────────────────────────
        total = len(rows)
        cross = sum(1 for t in topics if len(t["platforms"]) >= 2)
        breaking_count = sum(1 for t in topics if t["breaking"])
        fastest = topics[0] if topics else None

        # delta: compare total nodes active now vs prev window
        prev_total = await conn.fetchval(
            """
            SELECT COUNT(DISTINCT target_id)
            FROM graph_edges
            WHERE ingested_at >= now() - $1 * interval '1 minute'
              AND ingested_at < now() - $2 * interval '1 minute'
            """,
            wm * 2, wm,
        )

        stats = {
            "trendingTopics": total,
            "trendingDelta": max(0, total - (prev_total or 0)),
            "fastestVelocity": fastest["velocity"] if fastest else 0,
            "fastestName": (fastest["name"][:30] + "…" if fastest and len(fastest["name"]) > 30
                            else (fastest["name"] if fastest else "—")),
            "crossPlatform": cross,
            "breakingNow": breaking_count,
        }

    return {"topics": topics, "stats": stats}


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


# ── edges ─────────────────────────────────────────────────────────────────────

@router.get("/edges")
async def list_edges(
    request: Request,
    since_minutes: int = 1440,
    limit: int = Query(200, le=500),
):
    """Recent edges for graph bootstrap."""
    async with request.app.state.db.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT source_id, target_id, platform, ts
            FROM graph_edges
            WHERE ingested_at >= now() - $1 * interval '1 minute'
            ORDER BY ingested_at DESC
            LIMIT $2
            """,
            since_minutes, limit,
        )
    return [dict(r) for r in rows]


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


# ── case files ────────────────────────────────────────────────────────────────

@router.get("/case-files")
async def list_case_files(
    request: Request,
    needs_review: bool | None = None,
    classification: str | None = None,
    limit: int = Query(20, le=100),
    offset: int = 0,
):
    """List case files. Filter by needs_review or classification."""
    filters, args = [], []

    if needs_review is not None:
        args.append(needs_review)
        filters.append(f"needs_review = ${len(args)}")
    if classification is not None:
        args.append(classification)
        filters.append(f"classification = ${len(args)}")

    where_parts = ["cf." + f for f in filters]
    where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    args += [limit, offset]

    async with request.app.state.db.acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT cf.id, cf.anomaly_event_id, cf.trend, cf.platform_origin,
                   cf.detected_at, cf.classification, cf.confidence,
                   cf.needs_review, cf.created_at,
                   COALESCE(gn.label, cf.trend) AS label,
                   ae.z_score, ae.velocity
            FROM case_files cf
            LEFT JOIN graph_nodes gn ON gn.id = cf.trend AND gn.platform = cf.platform_origin
            LEFT JOIN anomaly_events ae ON ae.id = cf.anomaly_event_id
            {where}
            ORDER BY cf.created_at DESC
            LIMIT ${len(args) - 1} OFFSET ${len(args)}
            """,
            *args,
        )
    return [dict(r) for r in rows]


@router.get("/case-files/{case_id}")
async def get_case_file(case_id: int, request: Request):
    """Full case file including signals, ragas scores, similar past cases, and reasoning steps."""
    async with request.app.state.db.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT cf.*,
                   COALESCE(gn.label, cf.trend) AS label,
                   ae.z_score, ae.velocity
            FROM case_files cf
            LEFT JOIN graph_nodes gn ON gn.id = cf.trend AND gn.platform = cf.platform_origin
            LEFT JOIN anomaly_events ae ON ae.id = cf.anomaly_event_id
            WHERE cf.id = $1
            """,
            case_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail="case file not found")
    return dict(row)
