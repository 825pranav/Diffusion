"""
Per-cascade feature extraction for the virality classifier.

A cascade is the reshare tree rooted at a seed post.  Its structure lives in
`graph_edges` rows with edge_type='reshare'; authorship comes from 'authored'
edges.  Everything the classifier sees is derived from graph structure and
timing — never from a node id, label, or metadata field, any of which would leak
the simulator's ground truth.

Extraction is two-stage:

  1. One recursive CTE expands the reshare tree for *every* root at once and
     returns each edge tagged with its root and depth.  Running a per-cascade
     query instead would mean thousands of round-trips for the same result.
  2. Aggregation into features happens in pandas, where the windowed and
     order-dependent statistics are far easier to express than in SQL.

Author-reuse features are computed **causally**: a cascade only sees authors from
cascades that started strictly before it.  Counting reuse across the whole
dataset would leak information from the future and inflate the score.
"""

from __future__ import annotations

import logging
from collections import Counter

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Depth guard for the recursive expansion. Simulated cascades are far shallower;
# this only stops a cycle in real ingested data from looping forever.
MAX_TREE_DEPTH = 40

# Window used by the peak-rate feature.
PEAK_WINDOW_SECONDS = 60.0

FEATURE_COLUMNS = [
    # structure
    "size",
    "max_depth",
    "root_fanout",
    "root_fanout_share",
    "mean_children",
    "max_children",
    "depth_over_size",
    "leaf_frac",
    # timing
    "delay_median_s",
    "delay_mean_s",
    "delay_std_s",
    "delay_cv",
    "duration_s",
    "time_to_half_s",
    "peak_rate_per_min",
    # authors
    "unique_author_ratio",
    "max_author_share",
    "prior_author_mean",
    "prior_author_max",
    "prior_author_frac",
    # platform
    "n_platforms",
    "cross_platform_edge_frac",
    "first_hop_lag_s",
]

_TREE_SQL = """
WITH RECURSIVE roots AS (
    SELECT unnest($1::text[]) AS root_id
),
tree AS (
    SELECT
        r.root_id,
        e.source_id,
        e.target_id,
        e.ts,
        e.platform,
        1 AS depth
    FROM roots r
    JOIN graph_edges e
      ON e.source_id = r.root_id
     AND e.edge_type = 'reshare'

    UNION ALL

    SELECT
        t.root_id,
        e.source_id,
        e.target_id,
        e.ts,
        e.platform,
        t.depth + 1
    FROM tree t
    JOIN graph_edges e
      ON e.source_id = t.target_id
     AND e.edge_type = 'reshare'
    WHERE t.depth < $2
)
SELECT root_id, source_id, target_id, ts, platform, depth FROM tree
"""

# Authorship for every post in the selected trees, plus the roots themselves.
_AUTHORS_SQL = """
WITH RECURSIVE roots AS (
    SELECT unnest($1::text[]) AS root_id
),
tree AS (
    SELECT r.root_id, r.root_id AS content_id, 0 AS depth
    FROM roots r

    UNION ALL

    SELECT t.root_id, e.target_id, t.depth + 1
    FROM tree t
    JOIN graph_edges e
      ON e.source_id = t.content_id
     AND e.edge_type = 'reshare'
    WHERE t.depth < $2
)
SELECT
    t.root_id,
    t.content_id,
    e.source_id AS author_id,
    e.ts,
    e.platform
FROM tree t
JOIN graph_edges e
  ON e.target_id = t.content_id
 AND e.edge_type = 'authored'
"""


async def fetch_frames(conn, root_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull the reshare trees and authorship for the given roots."""
    tree_rows = await conn.fetch(_TREE_SQL, root_ids, MAX_TREE_DEPTH)
    author_rows = await conn.fetch(_AUTHORS_SQL, root_ids, MAX_TREE_DEPTH)

    tree = pd.DataFrame(
        tree_rows, columns=["root_id", "source_id", "target_id", "ts", "platform", "depth"]
    )
    authors = pd.DataFrame(
        author_rows, columns=["root_id", "content_id", "author_id", "ts", "platform"]
    )
    for frame in (tree, authors):
        if not frame.empty:
            frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    return tree, authors


def _peak_rate_per_min(sorted_epochs: np.ndarray) -> float:
    """Largest number of posts falling inside any PEAK_WINDOW_SECONDS window."""
    if sorted_epochs.size == 0:
        return 0.0
    right = np.searchsorted(sorted_epochs, sorted_epochs + PEAK_WINDOW_SECONDS, side="right")
    counts = right - np.arange(sorted_epochs.size)
    return float(counts.max()) * (60.0 / PEAK_WINDOW_SECONDS)


def _structure_and_timing(group: pd.DataFrame, root_id: str) -> dict[str, float]:
    """Features derived from one cascade's reshare edges."""
    n_edges = len(group)
    size = n_edges + 1  # edges + the root post

    # Parent timestamp per edge, so the delay is the true parent -> child gap
    # rather than the gap between consecutive events anywhere in the cascade.
    ts_by_node = dict(zip(group["target_id"], group["ts"]))
    root_ts = group["ts"].min()
    ts_by_node[root_id] = root_ts

    parent_ts = group["source_id"].map(ts_by_node)
    delays = (group["ts"] - parent_ts).dt.total_seconds().to_numpy(dtype=float)
    delays = delays[np.isfinite(delays)]
    if delays.size == 0:
        delays = np.array([0.0])

    children = group["source_id"].value_counts()
    root_fanout = float(children.get(root_id, 0))

    epochs = np.sort(group["ts"].astype("int64").to_numpy() / 1e9)
    t0 = float(root_ts.timestamp())
    duration = float(epochs.max() - t0) if epochs.size else 0.0
    half_idx = max(int(np.ceil(epochs.size / 2)) - 1, 0)
    time_to_half = float(epochs[half_idx] - t0) if epochs.size else 0.0

    delay_mean = float(delays.mean())
    delay_std = float(delays.std())

    # A post that never appears as a parent is a leaf.
    internal = set(group["source_id"])
    leaves = size - len(internal)

    platforms = group["platform"].nunique()
    platform_of = dict(zip(group["target_id"], group["platform"]))
    platform_of[root_id] = group.loc[group["ts"].idxmin(), "platform"]
    parent_platform = group["source_id"].map(platform_of)
    cross = parent_platform.ne(group["platform"])
    cross_frac = float(cross.mean()) if n_edges else 0.0
    if cross.any():
        first_hop_lag = float(group.loc[cross, "ts"].min().timestamp() - t0)
    else:
        # No cross-platform hop occurred. -1 is a sentinel the tree model can
        # split on; NaN would be imputed and confused with a genuine short lag.
        first_hop_lag = -1.0

    return {
        "size": float(size),
        "max_depth": float(group["depth"].max()),
        "root_fanout": root_fanout,
        "root_fanout_share": root_fanout / n_edges if n_edges else 0.0,
        "mean_children": float(children.mean()) if len(children) else 0.0,
        "max_children": float(children.max()) if len(children) else 0.0,
        "depth_over_size": float(group["depth"].max()) / size,
        "leaf_frac": leaves / size,
        "delay_median_s": float(np.median(delays)),
        "delay_mean_s": delay_mean,
        "delay_std_s": delay_std,
        "delay_cv": delay_std / delay_mean if delay_mean > 0 else 0.0,
        "duration_s": duration,
        "time_to_half_s": time_to_half,
        "peak_rate_per_min": _peak_rate_per_min(epochs),
        "n_platforms": float(platforms),
        "cross_platform_edge_frac": cross_frac,
        "first_hop_lag_s": first_hop_lag,
    }


def _author_features(authors: pd.DataFrame, order: list[str]) -> dict[str, dict[str, float]]:
    """
    Author-reuse features, computed in cascade start order.

    Walking the cascades chronologically and only counting authors already seen
    keeps the feature causal.  Counting over the entire dataset would let a
    cascade "know" about campaigns that ran after it — the classic form of
    leakage in this kind of feature, and one that would quietly inflate F1.
    """
    seen: Counter[str] = Counter()
    out: dict[str, dict[str, float]] = {}
    grouped = {root: frame for root, frame in authors.groupby("root_id", sort=False)}

    for root_id in order:
        frame = grouped.get(root_id)
        if frame is None or frame.empty:
            out[root_id] = {
                "unique_author_ratio": 0.0,
                "max_author_share": 0.0,
                "prior_author_mean": 0.0,
                "prior_author_max": 0.0,
                "prior_author_frac": 0.0,
            }
            continue

        author_ids = frame["author_id"].tolist()
        unique_authors = set(author_ids)
        counts = frame["author_id"].value_counts()
        prior = np.array([seen[a] for a in unique_authors], dtype=float)

        out[root_id] = {
            "unique_author_ratio": len(unique_authors) / len(author_ids),
            "max_author_share": float(counts.max()) / len(author_ids),
            "prior_author_mean": float(prior.mean()),
            "prior_author_max": float(prior.max()),
            "prior_author_frac": float((prior > 0).mean()),
        }
        for author in unique_authors:
            seen[author] += 1

    return out


def compute_features(
    tree: pd.DataFrame,
    authors: pd.DataFrame,
    order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Turn raw cascade edges into one feature row per root.

    `order` is the chronological cascade ordering used for the causal author
    features; when omitted it is derived from each cascade's earliest timestamp.
    Roots with no reshare edges are dropped — a single-post "cascade" has no
    structure or timing to describe.
    """
    if tree.empty:
        return pd.DataFrame(columns=["root_node_id", *FEATURE_COLUMNS])

    if order is None:
        starts = tree.groupby("root_id")["ts"].min().sort_values()
        order = starts.index.tolist()

    author_feats = _author_features(authors, order)

    rows = []
    for root_id, group in tree.groupby("root_id", sort=False):
        row = {"root_node_id": root_id}
        row.update(_structure_and_timing(group, root_id))
        row.update(author_feats.get(root_id, {}))
        rows.append(row)

    frame = pd.DataFrame(rows)
    # Guarantee a stable column set even if a feature was never populated.
    for column in FEATURE_COLUMNS:
        if column not in frame:
            frame[column] = 0.0
    return frame[["root_node_id", *FEATURE_COLUMNS]]


async def build_feature_table(conn, root_ids: list[str]) -> pd.DataFrame:
    """Fetch and featurise the given cascades in one pass."""
    tree, authors = await fetch_frames(conn, root_ids)
    log.info(
        "fetched %d reshare edges and %d authorship edges for %d roots",
        len(tree), len(authors), len(root_ids),
    )
    features = compute_features(tree, authors)
    log.info("computed %d feature rows", len(features))
    return features
