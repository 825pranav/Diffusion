"""
Dataset assembly — joins simulator ground truth to graph-derived features.

Shared by training and evaluation so both see exactly the same rows, feature
definitions, and split boundary.

The train/test split is **temporal**, not random: cascades are ordered by start
time and the most recent fraction is held out.  That mirrors how the system would
actually be used — fit on history, score what happens next — and it is the only
split that keeps the causal author-reuse features honest.  A random split would
put a cascade's own future neighbours in the training set.
"""

from __future__ import annotations

import logging
import pathlib

import asyncpg
import pandas as pd

from config import DB_URL
from ml.features import FEATURE_COLUMNS, build_feature_table

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_LABELS = REPO_ROOT / "data" / "simulation" / "labels.csv"

# Fraction of cascades (most recent by start time) reserved for evaluation.
TEST_FRACTION = 0.2

POSITIVE_CLASS = "coordinated"

log = logging.getLogger(__name__)


async def load_dataset(labels_path: pathlib.Path = DEFAULT_LABELS) -> pd.DataFrame:
    """
    Return one row per cascade: ground truth joined to graph-derived features.

    Cascades whose features could not be computed (no reshare edges) are dropped
    with a warning rather than silently imputed.
    """
    if not labels_path.exists():
        raise SystemExit(
            f"no labels at {labels_path} — run: python -m simulation.generate --n 2000"
        )

    labels = pd.read_csv(labels_path, parse_dates=["t0"])
    log.info("loaded %d labeled cascades from %s", len(labels), labels_path)

    # Keep only identity and ground truth. The CSV also stores the simulator's
    # generating parameters (delay_median_s, bot_frac, root_attach, ...) and the
    # true cascade size, which *determine* the class — merging them in would leak
    # the label outright, and several share a name with a real feature. They stay
    # in the CSV for analysis and never enter the dataset.
    identity = ["root_node_id", "label", "subtype", "topic_id", "t0"]
    labels = labels[identity]

    conn = await asyncpg.connect(DB_URL)
    try:
        features = await build_feature_table(conn, labels["root_node_id"].tolist())
    finally:
        await conn.close()

    merged = labels.merge(features, on="root_node_id", how="inner")
    missing = len(labels) - len(merged)
    if missing:
        log.warning("%d cascades had no computable features and were dropped", missing)

    merged["y"] = (merged["label"] == POSITIVE_CLASS).astype(int)
    return merged.sort_values("t0").reset_index(drop=True)


def temporal_split(
    data: pd.DataFrame, test_fraction: float = TEST_FRACTION
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split chronologically: earliest (1 - test_fraction) to train, the rest to test."""
    data = data.sort_values("t0").reset_index(drop=True)
    cut = int(len(data) * (1.0 - test_fraction))
    train, test = data.iloc[:cut].copy(), data.iloc[cut:].copy()
    log.info(
        "temporal split — train=%d (%.1f%% coordinated), test=%d (%.1f%% coordinated)",
        len(train), 100 * train["y"].mean(), len(test), 100 * test["y"].mean(),
    )
    return train, test


def feature_matrix(data: pd.DataFrame) -> pd.DataFrame:
    """The model input — features only, in a fixed column order."""
    return data[FEATURE_COLUMNS]
