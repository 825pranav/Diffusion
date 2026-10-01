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
  iterative scan  the same shared index with pgvector 0.8's hnsw.iterative_scan,
                  which keeps walking the graph until enough rows pass the filter
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
import time

import asyncpg
import numpy as np

from config import DB_URL, configure_logging
from graph.embeddings import EMBEDDING_DIMS, _index_name, ensure_partial_index
from graph.models import CREATE_INDEXES
from ml.report import markdown_table, upsert_section

SHARED_HNSW_DDL = next(ddl for ddl in CREATE_INDEXES if "idx_trend_embeddings_vec" in ddl)

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

# Interpretation written against the measured tables; kept beside the code that
# produces them so a re-run that changes the picture is noticed in review.
RETRIEVAL_FINDINGS = """
Three things follow.

**At this scale the planner avoids the problem on its own.** pgvector 0.8
prices an HNSW walk against filtering through the `(model_name, model_version)`
B-tree and sorting what survives, and for a few hundred to two thousand rows the
exact path wins: full recall in 2–5 ms. On the embedded pgvector 0.6.2 this
repository ran on before, the same query walked the shared HNSW index and lost
rows.

**When the index is walked, iterative scan fixes the count, not the answer.**
`hnsw.iterative_scan` keeps walking the shared graph until enough rows pass the
filter, so every query comes back with 5 rows — but recall stays where the
shared index left it (0.320 at 5%), and it is the slowest of the three. The rows
it adds are whatever the walk reaches next, not the true neighbours.

**The partial index is still the right fix.** Every row in it satisfies the
filter, so the whole walk counts: 0.978 recall at 5% selectivity, and the
fastest of the three. Recall falls as the filter keeps more rows because the
index then holds more candidates at the same `ef_search` — the ordinary HNSW
trade-off, not a filter problem. `graph.embeddings.ensure_partial_index` creates
one for the active model version, and the background indexer calls it on
startup.
"""

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


async def _settings(
    conn: asyncpg.Connection, exact: bool, ef_search: int, iterative: str | None, force_hnsw: bool
) -> None:
    # Pinned so every configuration gets the same candidate-list budget and the
    # comparison is about the filter, not about search effort.
    await conn.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")
    if iterative:
        await conn.execute(f"SET LOCAL hnsw.iterative_scan = {iterative}")
    if exact:
        # Forcing a sequential scan makes the result exact by definition —
        # the ground truth every approximate configuration is scored against.
        await conn.execute("SET LOCAL enable_indexscan = off")
        await conn.execute("SET LOCAL enable_bitmapscan = off")
    elif force_hnsw:
        # Only an HNSW index can return rows already ordered by distance, so with
        # sorting priced out the planner must walk it. This is the path a large
        # table takes; at benchmark scale pgvector 0.8 prefers filtering through
        # the B-tree and sorting the survivors exactly.
        await conn.execute("SET LOCAL enable_sort = off")


async def _topk(
    conn: asyncpg.Connection,
    qvec: str,
    k: int,
    exact: bool,
    ef_search: int = 40,
    iterative: str | None = None,
    force_hnsw: bool = False,
) -> list[str]:
    async with conn.transaction():
        await _settings(conn, exact, ef_search, iterative, force_hnsw)
        rows = await conn.fetch(_SEARCH_SQL, qvec, BENCHMARK_MODEL, TARGET_VERSION, k)
    return [r["node_id"] for r in rows]


async def _scan_type(
    conn: asyncpg.Connection, qvec: str, k: int, iterative: str | None, force_hnsw: bool
) -> str:
    """Report the access path the planner chose under the same settings as the arm."""
    async with conn.transaction():
        await _settings(conn, False, 40, iterative, force_hnsw)
        plan = await conn.fetch(f"EXPLAIN {_SEARCH_SQL}", qvec, BENCHMARK_MODEL, TARGET_VERSION, k)
    text = " ".join(r["QUERY PLAN"] for r in plan)
    for token in text.split():
        if token.startswith("idx_trend_emb"):
            kind = "btree + sort" if token == "idx_trend_embeddings_model" else "hnsw"
            return f"{kind} ({token})"
    return "seq scan + sort"


async def _measure(
    conn: asyncpg.Connection,
    queries: list[str],
    k: int,
    ef_search: int = 40,
    iterative: str | None = None,
    force_hnsw: bool = False,
) -> tuple[float, float, str, float]:
    """Return (recall@k, mean rows returned, access path, mean query ms)."""
    hits = 0
    returned = 0
    elapsed = 0.0
    for qvec in queries:
        exact = await _topk(conn, qvec, k, exact=True, ef_search=ef_search)
        t0 = time.perf_counter()
        approx = await _topk(
            conn, qvec, k, exact=False, ef_search=ef_search, iterative=iterative, force_hnsw=force_hnsw
        )
        elapsed += time.perf_counter() - t0
        hits += len(set(exact) & set(approx))
        returned += len(approx)
    n = len(queries)
    path = await _scan_type(conn, queries[0], k, iterative, force_hnsw)
    return hits / (n * k), returned / n, path, 1000 * elapsed / n


async def _pgvector_version(conn: asyncpg.Connection) -> tuple[int, ...]:
    raw = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    return tuple(int(x) for x in raw.split("."))


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
        version = await _pgvector_version(conn)
        # iterative_scan arrived in pgvector 0.8; older builds reject the setting.
        iterative = "strict_order" if version >= (0, 8) else None
        latency: list[list[object]] = []
        planner: list[list[object]] = []
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
            await conn.execute(SHARED_HNSW_DDL)
            await conn.execute("REINDEX INDEX idx_trend_embeddings_vec")
            await conn.execute("ANALYZE trend_embeddings")

            # Only the table-wide HNSW index from the base schema exists here, so
            # every shared-index arm runs before the partial index is created.
            chosen = await _measure(conn, queries, k)
            before = await _measure(conn, queries, k, force_hnsw=True)
            walked = (
                await _measure(conn, queries, k, iterative=iterative, force_hnsw=True)
                if iterative else None
            )
            await ensure_partial_index(conn, BENCHMARK_MODEL, TARGET_VERSION)
            await conn.execute("ANALYZE trend_embeddings")
            # With both HNSW indexes present the planner sometimes still walks the
            # shared one, so it is dropped for this arm and rebuilt afterwards.
            # (Not inside a rolled-back transaction: SET LOCAL in the per-query
            # savepoints would outlive them and leak into every later query.)
            await conn.execute("DROP INDEX idx_trend_embeddings_vec")
            try:
                after = await _measure(conn, queries, k, force_hnsw=True)
            finally:
                await conn.execute(SHARED_HNSW_DDL)

            log.info(
                "selectivity=%4.0f%%  planner: %s recall=%.3f | hnsw shared: recall=%.3f rows=%.2f | "
                "iterative: %s | partial: recall=%.3f rows=%.2f [%s]",
                100 * selectivity, chosen[2], chosen[0], before[0], before[1],
                f"recall={walked[0]:.3f} rows={walked[1]:.2f}" if walked else "n/a",
                after[0], after[1], after[2],
            )
            na = ["n/a", "n/a"]
            rows.append(
                [
                    f"{100 * selectivity:.0f}%",
                    f"{before[0]:.3f}",
                    f"{before[1]:.2f} / {k}",
                    *([f"{walked[0]:.3f}", f"{walked[1]:.2f} / {k}"] if walked else na),
                    f"{after[0]:.3f}",
                    f"{after[1]:.2f} / {k}",
                ]
            )
            planner.append([f"{100 * selectivity:.0f}%", chosen[2], f"{chosen[0]:.3f}", f"{chosen[3]:.2f}"])
            latency.append(
                [
                    f"{100 * selectivity:.0f}%",
                    f"{before[3]:.2f}",
                    f"{walked[3]:.2f}" if walked else "n/a",
                    f"{after[3]:.2f}",
                ]
            )

        body = f"""
A {corpus:,}-vector corpus in which the `(model_name, model_version)` filter keeps
only the stated share of rows, on pgvector {".".join(map(str, version))}. Recall@{k}
is measured against a sequential scan, exact by construction, over {n_queries}
queries per row, with `hnsw.ef_search` pinned equal across configurations so the
comparison is about the filter and not about search effort.

**What the planner does on its own.**

{markdown_table(["filter keeps", "access path", f"recall@{k}", "ms / query"], planner)}

**When the HNSW index is walked** — forced here by pricing out the sort; a
table large enough that sorting every filtered row costs more than an index
walk takes this path unforced.

{markdown_table(
    ["filter keeps", f"shared recall@{k}", "shared rows",
     f"iterative recall@{k}", "iterative rows", f"partial recall@{k}", "partial rows"],
    rows,
)}

Mean latency per query on the HNSW path, in milliseconds:

{markdown_table(["filter keeps", "shared", "iterative scan", "partial"], latency)}

{RETRIEVAL_FINDINGS}

Reproduce with `python -m ml.eval_retrieval`.
"""
        upsert_section("Filtered vector search — recall", body)
        log.info("wrote results to docs/results.md")
    finally:
        await _clear_benchmark(conn)
        await conn.execute(SHARED_HNSW_DDL)  # never leave production without it
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
