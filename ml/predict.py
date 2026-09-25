"""
Inference wrapper around the trained virality classifier.

Loaded lazily and cached: the agent runs one investigation at a time and the
model is a few hundred kilobytes, so re-reading it per call would be pure waste,
while importing it at module load would make `ml` unimportable on a machine that
has not trained yet.

Absence of a model is not an error.  The agent keeps its heuristic tools and the
pipeline still runs — `score_cascade` returns None and the caller says so.
"""

from __future__ import annotations

import json
import logging
import pathlib

import joblib
import numpy as np

from ml.features import AUTHOR_WINDOW_SECONDS, FEATURE_COLUMNS, build_feature_table

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "models" / "virality_lgbm.joblib"
CONTRACT_PATH = REPO_ROOT / "models" / "virality_model.json"

# Number of per-prediction feature attributions to surface to the agent.
TOP_FEATURES = 5

# Seeds of every cascade whose reshare activity falls in the author-reuse
# window ending when this cascade began. A seed is a post that is reshared but
# is not itself a reshare.
_WINDOW_ROOTS_SQL = """
WITH target AS (
    SELECT min(ts) AS t0 FROM graph_edges WHERE source_id = $1 AND edge_type = 'reshare'
)
SELECT DISTINCT e.source_id
FROM graph_edges e, target
WHERE e.edge_type = 'reshare'
  AND e.ts >= target.t0 - make_interval(secs => $2)
  AND e.ts <= target.t0
  AND NOT EXISTS (
      SELECT 1 FROM graph_edges p
      WHERE p.target_id = e.source_id AND p.edge_type = 'reshare'
  )
"""

log = logging.getLogger(__name__)

_model = None
_contract: dict | None = None
_load_attempted = False


def load_model():
    """Return the cached model, or None if it has not been trained yet."""
    global _model, _contract, _load_attempted
    if _load_attempted:
        return _model
    _load_attempted = True

    if not MODEL_PATH.exists():
        log.warning(
            "no trained model at %s — classify_virality_model will be unavailable "
            "(run: python -m ml.train)",
            MODEL_PATH,
        )
        return None

    _model = joblib.load(MODEL_PATH)
    if CONTRACT_PATH.exists():
        _contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        trained_on = _contract.get("features", [])
        if trained_on != FEATURE_COLUMNS:
            # Feature order is positional in LightGBM; a drifted contract would
            # silently score the wrong columns.
            log.error(
                "feature contract mismatch — model expects %d features, code "
                "produces %d; retrain before trusting predictions",
                len(trained_on), len(FEATURE_COLUMNS),
            )
            _model = None
    log.info("loaded virality classifier from %s", MODEL_PATH)
    return _model


def _attributions(model, row: np.ndarray) -> list[dict[str, float]]:
    """
    Per-prediction feature contributions in log-odds.

    LightGBM's pred_contrib returns one column per feature plus a trailing base
    value, so the base is dropped before ranking.
    """
    contrib = model.booster_.predict(row.reshape(1, -1), pred_contrib=True)[0][:-1]
    order = np.argsort(np.abs(contrib))[::-1][:TOP_FEATURES]
    return [
        {
            "feature": FEATURE_COLUMNS[i],
            "value": round(float(row[i]), 4),
            "contribution": round(float(contrib[i]), 4),
        }
        for i in order
    ]


async def score_cascade(conn, root_id: str) -> dict | None:
    """
    Score one cascade.

    Returns {p_coordinated, top_features, cascade_size} or None when the model is
    untrained or the node has no reshare tree to describe.
    """
    model = load_model()
    if model is None:
        return None

    # The author-reuse and co-author features are defined against the cascades
    # that ran in the trailing window before this one. Featurising the root on
    # its own would leave that history empty and every one of those features at
    # zero — values the model only ever saw on the first cascades of the
    # training data — so the recent neighbours are featurised alongside it.
    rows = await conn.fetch(_WINDOW_ROOTS_SQL, root_id, AUTHOR_WINDOW_SECONDS)
    roots = sorted({r["source_id"] for r in rows} | {root_id})
    features = await build_feature_table(conn, roots)
    features = features[features["root_node_id"] == root_id]
    if features.empty:
        return None

    row = features[FEATURE_COLUMNS].iloc[0].to_numpy(dtype=float)
    p_coordinated = float(model.predict_proba(row.reshape(1, -1))[0][1])

    return {
        "p_coordinated": round(p_coordinated, 4),
        "cascade_size": int(features["size"].iloc[0]),
        "top_features": _attributions(model, row),
    }
