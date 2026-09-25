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
from collections import Counter, deque

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Depth guard for the recursive expansion. Simulated cascades are far shallower;
# this only stops a cycle in real ingested data from looping forever.
MAX_TREE_DEPTH = 40

# Window used by the peak-rate feature.
PEAK_WINDOW_SECONDS = 60.0

# Trailing window for author-reuse features. Cumulative counts grow without
# bound and stop meaning the same thing across a temporal split; a window keeps
# them stationary and matches what production could actually compute.
AUTHOR_WINDOW_SECONDS = 6 * 3600.0

# The original feature set. Kept as its own list so experiments can compare
# against it directly (ml/experiment.py); the model reads FEATURE_COLUMNS.
BASE_FEATURE_COLUMNS = [
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

# Second-generation features. Each targets a signal the base set only reaches
# indirectly; see _extended_structure_and_timing and _author_features.
EXTENDED_FEATURE_COLUMNS = [
    # timing, on a log scale: inter-arrival delays are heavy-tailed, so their
    # raw-scale mean and std are dominated by one or two slow replies. The spread
    # of log-delays is what actually measures synchrony.
    "log_delay_mean",
    "log_delay_std",
    "log_delay_iqr",
    "arrival_burstiness",
    "arrival_memory",
    # structure
    "structural_virality",
    "mean_depth",
    "max_width_frac",
    "branching_frac",
    "root_attach_mle",
    # coordination across cascades (causal, windowed like the reuse features)
    "prior_author_p75",
    "prior_author_frac_ge3",
    "coauthor_linked_frac",
    "coauthor_overlap_max",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + EXTENDED_FEATURE_COLUMNS

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


def to_epoch_seconds(values: pd.Series) -> np.ndarray:
    """
    Convert a datetime column to epoch seconds.

    asyncpg hands back timestamptz as datetime64[**us**], so the obvious
    `.astype("int64") / 1e9` is a thousand times too small and silently rescales
    every duration. Normalising the unit first makes the conversion independent
    of whatever resolution the driver chose.
    """
    return pd.to_datetime(values, utc=True).dt.as_unit("ns").astype("int64").to_numpy() / 1e9


def _peak_rate_per_min(sorted_epochs: np.ndarray) -> float:
    """Largest number of posts falling inside any PEAK_WINDOW_SECONDS window."""
    if sorted_epochs.size == 0:
        return 0.0
    right = np.searchsorted(sorted_epochs, sorted_epochs + PEAK_WINDOW_SECONDS, side="right")
    counts = right - np.arange(sorted_epochs.size)
    return float(counts.max()) * (60.0 / PEAK_WINDOW_SECONDS)


def _structure_and_timing(
    group: pd.DataFrame,
    root_id: str,
    root_ts: pd.Timestamp | None = None,
    root_platform: str | None = None,
) -> dict[str, float]:
    """
    Features derived from one cascade's reshare edges.

    `root_ts` is when the seed post was published, which comes from its authored
    edge — the reshare edges only carry child timestamps. Falling back to the
    earliest child would put the origin *after* the root, forcing the first
    root-attached child to a delay of zero and measuring every sibling from it
    instead. Since root attachment is exactly what separates the two classes,
    that error would land unevenly on them.
    """
    n_edges = len(group)
    size = n_edges + 1  # edges + the root post

    # Parent timestamp per edge, so the delay is the true parent -> child gap
    # rather than the gap between consecutive events anywhere in the cascade.
    ts_by_node = dict(zip(group["target_id"], group["ts"], strict=True))
    if root_ts is None:
        root_ts = group["ts"].min()
    ts_by_node[root_id] = root_ts

    parent_ts = group["source_id"].map(ts_by_node)
    delays = (group["ts"] - parent_ts).dt.total_seconds().to_numpy(dtype=float)
    delays = delays[np.isfinite(delays)]
    if delays.size == 0:
        delays = np.array([0.0])

    children = group["source_id"].value_counts()
    root_fanout = float(children.get(root_id, 0))

    epochs = np.sort(to_epoch_seconds(group["ts"]))
    t0 = float(pd.Timestamp(root_ts).timestamp())
    duration = float(epochs.max() - t0) if epochs.size else 0.0
    half_idx = max(int(np.ceil(epochs.size / 2)) - 1, 0)
    time_to_half = float(epochs[half_idx] - t0) if epochs.size else 0.0

    delay_mean = float(delays.mean())
    delay_std = float(delays.std())

    # A post that never appears as a parent is a leaf.
    internal = set(group["source_id"])
    leaves = size - len(internal)

    platforms = group["platform"].nunique()
    platform_of = dict(zip(group["target_id"], group["platform"], strict=True))
    # The seed's platform comes from its authored edge. Falling back to the
    # earliest reshare edge reads the *first child's* platform instead, which
    # flips the cross-platform flag for every root-attached child whenever that
    # child happened to hop. Measured at 0.4% of cascades — a hop adds lag, so
    # the earliest child is nearly always same-platform — but the error would
    # land unevenly, since root attachment is what separates the classes.
    platform_of[root_id] = (
        root_platform
        if root_platform is not None
        else group.loc[group["ts"].idxmin(), "platform"]
    )
    parent_platform = group["source_id"].map(platform_of)
    cross = parent_platform.ne(group["platform"])
    cross_frac = float(cross.mean()) if n_edges else 0.0
    if cross.any():
        first_hop_lag = float(group.loc[cross, "ts"].min().timestamp() - t0)
    else:
        # No cross-platform hop occurred. -1 is a sentinel the tree model can
        # split on; NaN would be imputed and confused with a genuine short lag.
        first_hop_lag = -1.0

    same_platform_delays = (
        (group["ts"] - parent_ts).dt.total_seconds().to_numpy(dtype=float)[~cross.to_numpy()]
    )
    extended = _extended_structure_and_timing(
        group, root_id, t0, epochs, same_platform_delays, delays, size
    )

    return {
        **extended,
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


def _root_attach_mle(parents: np.ndarray, root_id: str) -> float:
    """
    Maximum-likelihood share of new posts that attach straight to the seed.

    Children are taken in arrival order. The i-th child (0-based) chooses among
    the i + 1 posts already published: with probability `a` it takes the seed,
    otherwise a uniformly random earlier post (which may also be the seed). The
    log-likelihood is concave in `a`, so a fine grid is exact enough.

    `root_fanout_share` counts root attachments but ignores *when* they happen —
    a seed reply is unremarkable when the cascade is two posts old and telling
    when it is fifty. The likelihood weights each one by how unlikely it was
    under uniform attachment.
    """
    if parents.size == 0:
        return 0.0
    is_root = (parents == root_id).astype(float)
    inv_k = 1.0 / np.arange(1, parents.size + 1, dtype=float)
    grid = np.linspace(0.0, 1.0, 201)[:, None]
    with np.errstate(divide="ignore"):
        ll = np.log(grid * is_root + (1.0 - grid) * inv_k).sum(axis=1)
    return float(grid[int(np.argmax(ll)), 0])


def _structural_virality(sources: np.ndarray, targets: np.ndarray, root_id: str) -> float:
    """
    Mean shortest-path distance between all pairs of posts (Goel et al., 2016).

    For a tree the Wiener index is the sum over edges of s * (n - s), where s is
    the size of the subtree below the edge, so this is linear rather than an
    all-pairs search. Broadcast-like stars score near 2; person-to-person chains
    score high.
    """
    n = targets.size + 1
    if n < 2:
        return 0.0
    children: dict[str, list[str]] = {}
    for s, t in zip(sources, targets, strict=True):
        children.setdefault(s, []).append(t)
    subtree: dict[str, int] = {}
    # Iterative post-order walk; real threads can be deep enough to hurt recursion.
    stack: list[tuple[str, bool]] = [(root_id, False)]
    while stack:
        node, done = stack.pop()
        if done:
            subtree[node] = 1 + sum(subtree.get(c, 0) for c in children.get(node, []))
            continue
        stack.append((node, True))
        for c in children.get(node, []):
            stack.append((c, False))
    wiener = sum(subtree.get(t, 1) * (n - subtree.get(t, 1)) for t in targets)
    return 2.0 * wiener / (n * (n - 1))


def _extended_structure_and_timing(
    group: pd.DataFrame,
    root_id: str,
    t0: float,
    epochs: np.ndarray,
    same_platform_delays: np.ndarray,
    all_delays: np.ndarray,
    size: int,
) -> dict[str, float]:
    """The second-generation structure and timing features for one cascade."""
    # Log-delays on same-platform edges: a platform hop adds its own lag, which
    # would otherwise masquerade as dispersion in response time.
    base = same_platform_delays[np.isfinite(same_platform_delays)]
    if base.size == 0:
        base = all_delays
    log_delays = np.log(np.maximum(base, 1.0))
    q75, q25 = np.percentile(log_delays, [75, 25])

    # Inter-arrival gaps across the whole cascade, seed included.
    arrivals = np.concatenate([[t0], epochs])
    gaps = np.diff(np.sort(arrivals))
    mu, sd = float(gaps.mean()), float(gaps.std())
    # Burstiness (Goh & Barabasi, 2008): -1 periodic, 0 Poisson, 1 maximally bursty.
    burstiness = (sd - mu) / (sd + mu) if (sd + mu) > 0 else 0.0
    memory = 0.0
    if gaps.size >= 3:
        a, b = gaps[:-1], gaps[1:]
        if a.std() > 0 and b.std() > 0:
            memory = float(np.corrcoef(a, b)[0, 1])

    ordered = group.sort_values("ts")
    sources = ordered["source_id"].to_numpy()
    targets = ordered["target_id"].to_numpy()

    width = group["depth"].value_counts()
    children = group["source_id"].value_counts()

    return {
        "log_delay_mean": float(log_delays.mean()),
        "log_delay_std": float(log_delays.std()),
        "log_delay_iqr": float(q75 - q25),
        "arrival_burstiness": float(burstiness),
        "arrival_memory": memory,
        "structural_virality": _structural_virality(sources, targets, root_id),
        "mean_depth": float(group["depth"].mean()),
        "max_width_frac": float(width.max()) / size if len(width) else 0.0,
        "branching_frac": float((children >= 2).mean()) if len(children) else 0.0,
        "root_attach_mle": _root_attach_mle(sources, root_id),
    }


def _coauthor_features(
    unique_authors: set[str], member_of: dict[str, set[int]]
) -> dict[str, float]:
    """
    Coordination structure *across* cascades.

    Reuse features ask whether each account has been seen before, one account
    at a time. A campaign leaves a stronger trace: the same *group* of accounts
    turns up together. Two strangers each posting in some earlier cascade is
    ordinary; two of this cascade's accounts having appeared in the same
    earlier cascade is not, and a pool of accounts rotated across campaigns
    produces it constantly even when each one is used sparingly.

      coauthor_linked_frac   share of this cascade's accounts that co-appeared
                             with another of its accounts in a recent cascade
      coauthor_overlap_max   largest number of this cascade's accounts found in
                             any single recent cascade, as a share of its accounts
    """
    shared: Counter[int] = Counter()
    for author in unique_authors:
        for cascade in member_of.get(author, ()):
            shared[cascade] += 1
    if not shared:
        return {"coauthor_linked_frac": 0.0, "coauthor_overlap_max": 0.0}
    linked = sum(
        1
        for author in unique_authors
        if any(shared[c] >= 2 for c in member_of.get(author, ()))
    )
    n = len(unique_authors)
    return {
        "coauthor_linked_frac": linked / n,
        "coauthor_overlap_max": max(shared.values()) / n,
    }


def _author_features(
    authors: pd.DataFrame,
    order: list[str],
    start_times: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Author-reuse features over a trailing time window, in cascade start order.

    Walking the cascades chronologically and only counting authors already seen
    keeps the feature causal.  Counting over the entire dataset would let a
    cascade "know" about campaigns that ran after it — the classic form of
    leakage in this kind of feature, and one that would quietly inflate F1.

    The count is windowed rather than cumulative, which matters as much as the
    causality.  A running total since the dataset began grows without bound, so
    with a temporal split the model learns thresholds on early, small counts and
    is then tested on late, large ones: measured here, mean reuse nearly trebled
    between train and holdout (1.7 sd) and "fraction of authors seen before"
    saturated at 1.000, carrying no information at all.  A trailing window is
    stationary, and it is also the only version computable in production, where
    history cannot be accumulated forever.
    """
    window = AUTHOR_WINDOW_SECONDS
    # `recent` is reuse within the trailing window; `counts` below is per-cascade
    # author frequency. Two different things — do not merge the names.
    recent: Counter[str] = Counter()
    history: deque[tuple[float, list[str], int]] = deque()
    # Which recent cascades each author took part in, for the co-occurrence
    # features. Keyed by position in `order`, evicted with `recent`.
    member_of: dict[str, set[int]] = {}
    out: dict[str, dict[str, float]] = {}
    grouped = {root: frame for root, frame in authors.groupby("root_id", sort=False)}

    for index, root_id in enumerate(order):
        now = (start_times or {}).get(root_id)
        if now is not None:
            # Drop cascades that have aged out of the window.
            while history and history[0][0] < now - window:
                _, expired, expired_index = history.popleft()
                for author in expired:
                    recent[author] -= 1
                    if recent[author] <= 0:
                        del recent[author]
                    cascades = member_of.get(author)
                    if cascades is not None:
                        cascades.discard(expired_index)
                        if not cascades:
                            del member_of[author]

        frame = grouped.get(root_id)
        if frame is None or frame.empty:
            out[root_id] = {
                "unique_author_ratio": 0.0,
                "max_author_share": 0.0,
                "prior_author_mean": 0.0,
                "prior_author_max": 0.0,
                "prior_author_frac": 0.0,
                "prior_author_p75": 0.0,
                "prior_author_frac_ge3": 0.0,
                "coauthor_linked_frac": 0.0,
                "coauthor_overlap_max": 0.0,
            }
            continue

        author_ids = frame["author_id"].tolist()
        unique_authors = set(author_ids)
        counts = frame["author_id"].value_counts()
        prior = np.array([recent[a] for a in unique_authors], dtype=float)

        out[root_id] = {
            "unique_author_ratio": len(unique_authors) / len(author_ids),
            "max_author_share": float(counts.max()) / len(author_ids),
            "prior_author_mean": float(prior.mean()),
            "prior_author_max": float(prior.max()),
            "prior_author_frac": float((prior > 0).mean()),
            # The mean is pulled around by one very active account; the upper
            # quartile and the share of repeat accounts estimate how much of
            # the cascade comes from a reused pool.
            "prior_author_p75": float(np.percentile(prior, 75)),
            "prior_author_frac_ge3": float((prior >= 3).mean()),
            **_coauthor_features(unique_authors, member_of),
        }
        for author in unique_authors:
            recent[author] += 1
            member_of.setdefault(author, set()).add(index)
        if now is not None:
            history.append((now, list(unique_authors), index))

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

    start_seconds = {
        root: float(ts.timestamp())
        for root, ts in tree.groupby("root_id")["ts"].min().items()
    }
    author_feats = _author_features(authors, order, start_seconds)

    # When the seed post was published, taken from its own authored edge.
    root_ts_map: dict[str, pd.Timestamp] = {}
    root_platform_map: dict[str, str] = {}
    if not authors.empty:
        seeds = authors[authors["content_id"] == authors["root_id"]]
        if not seeds.empty:
            root_ts_map = seeds.groupby("root_id")["ts"].min().to_dict()
            root_platform_map = (
                seeds.sort_values("ts").groupby("root_id")["platform"].first().to_dict()
            )

    rows = []
    for root_id, group in tree.groupby("root_id", sort=False):
        row = {"root_node_id": root_id}
        row.update(
            _structure_and_timing(
                group,
                root_id,
                root_ts_map.get(root_id),
                root_platform_map.get(root_id),
            )
        )
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
