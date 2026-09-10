"""
Measure the recall cost of filtered vector search, and the partial-index fix.

Usage:
    python -m ml.eval_retrieval
    python -m ml.eval_retrieval --corpus 10000 --queries 100

`search_similar` filters on (model_name, model_version) and orders by cosine
distance.  A single HNSW index spanning the whole table walks the graph without
knowing about that filter, so neighbours belonging to other model versions are
found first and discarded afterwards — spending the result budget and returning
worse rows, or fewer than requested.  Nothing errors; recall just quietly drops,
which is why the problem survives until it is measured.

The benchmark loads clustered vectors under a dedicated `benchmark` model name so
it never touches real embeddings, then compares each configuration against exact
brute-force results:

  shared index    the pre-existing table-wide HNSW index
  partial index   one HNSW index per (model_name, model_version)

Recall@k is measured against a sequential scan, which is exact by construction.
The query hits trend_embeddings directly rather than going through
search_similar's join to graph_nodes, so what is measured is index behaviour
alone and not the planner's join choices.
"""

from __future__ import annotations

import argparse
import asyncio
import logging

import asyncpg
import numpy as np

from config import DB_URL, configure_logging
from graph.embeddings import EMBEDDING_DIMS, _index_name, ensure_partial_index
from ml.report import markdown_table, upsert_section

BENCHMARK_MODEL = "benchmark"
TARGET_VERSION = "v1"
DECOY_VERSION = "v0"

DEFAULT_CORPUS = 10000
DEFAULT_QUERIES = 100
DEFAULT_K = 5
# Cluster count for the synthetic corpus. Uniform random vectors in 768
# dimensions are all nearly equidistant, which makes "nearest neighbour"
# meaningless; clusters give the ranking something real to recover.
N_CLUSTERS = 60
CLUSTER_SPREAD = 0.35

INSERT_BATCH = 1000

log = logging.getLogger(__name__)


def _vector_literal(vec: np.ndarray) -> str:
    return "[" + ",".join(f"{x:.5f}" for x in vec) + "]"


def _sample_clustered(rng: np.random.Generator, n: int, centroids: np.ndarray) -> np.ndarray:
    idx = rng.integers(0, centroids.shape[0], size=n)
    noise = rng.normal(0, CLUSTER_SPREAD, size=(n, centroids.shape[1]))
    vecs = centroids[idx] + noise
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


async def _clear_benchmark(conn: asyncpg.Connection) -> None:
    await conn.execute("DELETE FROM trend_embeddings WHERE model_name = $1", BENCHMARK_MODEL)
    for version in (TARGET_VERSION, DECOY_VERSION):
        await conn.execute(f"DROP INDEX IF EXISTS {_index_name(BENCHMARK_MODEL, version)}")


async def _load(
    conn: asyncpg.Connection, vecs: np.ndarray, version: str, prefix: str
) -> None:
    rows = [
        (f"{prefix}{i}", "bench", _vector_literal(v), BENCHMARK_MODEL, version)
        for i, v in enumerate(vecs)
    ]
    for start in range(0, len(rows), INSERT_BATCH):
        await conn.executemany(
            """
            INSERT INTO trend_embeddings (node_id, platform, embedding, model_name, model_version)
            VALUES ($1, $2, $3::vector, $4, $5)
            ON CONFLICT (node_id, platform, model_name, model_version) DO NOTHING
            """,
            rows[start : start + INSERT_BATCH],
        )


_SEARCH_SQL = """
SELECT node_id
FROM trend_embeddings
WHERE model_name = $2 AND model_version = $3
ORDER BY embedding <=> $1::vector
LIMIT $4
"""


async def _topk(
    conn: asyncpg.Connection, qvec: str, k: int, exact: bool, ef_search: int = 40
) -> list[str]:
    async with conn.transaction():
        # Pinned so both configurations get the same candidate-list budget and the
        # comparison is about the filter, not about search effort.
        await conn.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
        if exact:
            # Forcing a sequential scan makes the result exact by definition —
            # the ground truth every approximate configuration is scored against.
            await conn.execute("SET LOCAL enable_indexscan = off")
            await conn.execute("SET LOCAL enable_bitmapscan = off")
        rows = await conn.fetch(_SEARCH_SQL, qvec, BENCHMARK_MODEL, TARGET_VERSION, k)
    return [r["node_id"] for r in rows]


async def _scan_type(conn: asyncpg.Connection, qvec: str, k: int) -> str:
    """Report which access path the planner chose, as evidence the fix took."""
    plan = await conn.fetch(
        f"EXPLAIN {_SEARCH_SQL}", qvec, BENCHMARK_MODEL, TARGET_VERSION, k
    )
    text = " ".join(r["QUERY PLAN"] for r in plan)
    if "Index Scan" in text:
        for token in text.split():
            if token.startswith("idx_trend_emb"):
                return f"index ({token})"
        return "index"
    return "seq scan"


async def _measure(
    conn: asyncpg.Connection, queries: list[str], k: int, ef_search: int = 40
) -> tuple[float, float, str]:
    """Return (recall@k, mean rows returned, scan type) for the current indexes."""
    hits = 0
    returned = 0
    for qvec in queries:
        exact = await _topk(conn, qvec, k, exact=True, ef_search=ef_search)
        approx = await _topk(conn, qvec, k, exact=False, ef_search=ef_search)
        hits += len(set(exact) & set(approx))
        returned += len(approx)
    n = len(queries)
    return hits / (n * k), returned / n, await _scan_type(conn, queries[0], k)


async def main_async(corpus: int, n_queries: int, k: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    centroids = rng.normal(0, 1, size=(N_CLUSTERS, EMBEDDING_DIMS))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)

    # Sweep how much of the table the filter keeps. The penalty should grow as
    # the filter gets more restrictive — a single data point could not show that.
    selectivities = (0.05, 0.10, 0.20)
    rows: list[list[object]] = []

    conn = await asyncpg.connect(DB_URL)
    try:
        for selectivity in selectivities:
            n_target = int(corpus * selectivity)
            n_decoy = corpus - n_target

            await _clear_benchmark(conn)
            await _load(
                conn, _sample_clustered(rng, n_target, centroids), TARGET_VERSION, "bench:t:"
            )
            await _load(
                conn, _sample_clustered(rng, n_decoy, centroids), DECOY_VERSION, "bench:d:"
            )
            queries = [
                _vector_literal(v) for v in _sample_clustered(rng, n_queries, centroids)
            ]

            # Each sweep step deletes and reloads the whole corpus, which leaves
            # the table-wide HNSW graph fragmented and its recall drifting for
            # reasons that have nothing to do with the filter. Rebuilding makes
            # every step an independent measurement.
            await conn.execute("REINDEX INDEX idx_trend_embeddings_vec")
            await conn.execute("ANALYZE trend_embeddings")

            # Only the table-wide HNSW index from the base schema exists here.
            before = await _measure(conn, queries, k)
            await ensure_partial_index(conn, BENCHMARK_MODEL, TARGET_VERSION)
            await conn.execute("ANALYZE trend_embeddings")
            after = await _measure(conn, queries, k)

            log.info(
                "selectivity=%4.0f%%  shared: recall=%.3f rows=%.2f  |  partial: recall=%.3f rows=%.2f",
                100 * selectivity, before[0], before[1], after[0], after[1],
            )
            rows.append(
                [
                    f"{100 * selectivity:.0f}%",
                    f"{before[0]:.3f}",
                    f"{before[1]:.2f} / {k}",
                    f"{after[0]:.3f}",
                    f"{after[1]:.2f} / {k}",
                ]
            )

        body = f"""
A {corpus:,}-vector corpus in which the `(model_name, model_version)` filter keeps
only the stated share of rows. Recall@{k} is measured against a sequential scan,
exact by construction, over {n_queries} queries per row, with `hnsw.ef_search`
pinned equal across both configurations so the comparison is about the filter and
not about search effort.

{markdown_table(
    ["filter keeps", f"shared recall@{k}", "shared rows", f"partial recall@{k}", "partial rows"],
    rows,
)}

The table-wide index never returns a full result set: it walks the graph unaware
of the filter, and neighbours from the other model version are discarded *after*
consuming the candidate budget. At 5% selectivity a query asking for 5 rows gets
under 3. Nothing raises an error — `search_similar` simply returns a short,
degraded list, which is why this survived until it was measured.

A partial index returns the full {k} rows at every selectivity, because every row
it contains already satisfies the predicate and the entire walk is usable.

The table-wide index walks the graph without knowing about the filter, so
neighbours from the other model version are found first and dropped afterwards.
Nothing errors — the query just returns fewer and worse rows. A partial index
contains only rows that already satisfy the predicate, so the entire walk counts.

pgvector's `iterative_scan` addresses this generally but needs 0.8+; the bundled
build is 0.6.2. A partial index is sufficient here because the filter is
low-cardinality and known ahead of time. `graph.embeddings.ensure_partial_index`
creates one for the active model version, and the background indexer calls it on
startup.

Reproduce with `python -m ml.eval_retrieval`.
"""
        upsert_section("Filtered vector search — recall", body)
        log.info("wrote results to docs/results.md")
    finally:
        await _clear_benchmark(conn)
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure filtered vector-search recall")
    parser.add_argument("--corpus", type=int, default=DEFAULT_CORPUS)
    parser.add_argument("--queries", type=int, default=DEFAULT_QUERIES)
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    asyncio.run(main_async(args.corpus, args.queries, args.k, args.seed))


if __name__ == "__main__":
    main()
