"""
Train the cascade virality classifier.

Usage:
    python -m ml.train
    python -m ml.train --labels data/simulation/labels.csv --seed 42

Fits a LightGBM binary classifier (coordinated = positive) on graph-derived
cascade features, reports stratified 5-fold cross-validation on the training
split, and writes the model plus its feature contract to models/.

The held-out split is never touched here — it belongs to ml/evaluate.py, which
compares this model against the LLM agent.  Keeping the two scripts apart is
what stops the reported evaluation from quietly becoming a training score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from config import configure_logging
from ml.dataset import DEFAULT_LABELS, feature_matrix, load_dataset, temporal_split
from ml.features import FEATURE_COLUMNS
from ml.report import markdown_table, upsert_section

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "virality_lgbm.joblib"
CONTRACT_PATH = MODEL_DIR / "virality_model.json"

N_FOLDS = 5

# Deliberately modest capacity. The training split is ~1600 rows, and the point
# of the exercise is a trustworthy estimate rather than the highest possible F1.
LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)

log = logging.getLogger(__name__)


def cross_validate(X: pd.DataFrame, y: np.ndarray, seed: int) -> pd.DataFrame:
    """Stratified k-fold CV. Returns one row of metrics per fold."""
    folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
        model = LGBMClassifier(random_state=seed, **LGBM_PARAMS)
        model.fit(X.iloc[train_idx], y[train_idx])

        proba = model.predict_proba(X.iloc[valid_idx])[:, 1]
        pred = (proba >= 0.5).astype(int)
        truth = y[valid_idx]
        rows.append(
            {
                "fold": fold,
                "precision": precision_score(truth, pred, zero_division=0),
                "recall": recall_score(truth, pred, zero_division=0),
                "f1": f1_score(truth, pred, zero_division=0),
                "roc_auc": roc_auc_score(truth, proba),
                "pr_auc": average_precision_score(truth, proba),
            }
        )
    return pd.DataFrame(rows)


def _importance_frame(model: LGBMClassifier) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }
    )
    frame["share"] = frame["gain"] / frame["gain"].sum()
    return frame.sort_values("gain", ascending=False).reset_index(drop=True)


def _write_report(cv: pd.DataFrame, importance: pd.DataFrame, n_train: int, n_test: int) -> None:
    metric_rows = [
        [
            metric,
            f"{cv[metric].mean():.3f}",
            f"{cv[metric].std():.3f}",
            f"{cv[metric].min():.3f}",
            f"{cv[metric].max():.3f}",
        ]
        for metric in ("precision", "recall", "f1", "roc_auc", "pr_auc")
    ]
    top = importance.head(12)
    importance_rows = [
        [r.feature, f"{r.gain:,.0f}", f"{100 * r.share:.1f}%"] for r in top.itertuples()
    ]

    body = f"""
LightGBM, positive class = `coordinated`. {N_FOLDS}-fold stratified
cross-validation on the training split only ({n_train} cascades); {n_test}
cascades are held out for `ml/evaluate.py` and never seen here.

{markdown_table(["metric", "mean", "std", "min", "max"], metric_rows)}

### Feature importance (gain)

{markdown_table(["feature", "gain", "share"], importance_rows)}

Reproduce with `python -m ml.train`.
"""
    upsert_section("Virality classifier — cross-validation", body)


async def main_async(labels_path: pathlib.Path, seed: int) -> None:
    data = await load_dataset(labels_path)
    train, test = temporal_split(data)

    X = feature_matrix(train)
    y = train["y"].to_numpy()

    cv = cross_validate(X, y, seed)
    log.info(
        "CV over %d folds — f1=%.3f±%.3f  roc_auc=%.3f  precision=%.3f  recall=%.3f",
        N_FOLDS, cv["f1"].mean(), cv["f1"].std(),
        cv["roc_auc"].mean(), cv["precision"].mean(), cv["recall"].mean(),
    )

    model = LGBMClassifier(random_state=seed, **LGBM_PARAMS)
    model.fit(X, y)

    importance = _importance_frame(model)
    log.info("top features:\n%s", importance.head(8).to_string(index=False))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    CONTRACT_PATH.write_text(
        json.dumps(
            {
                "features": FEATURE_COLUMNS,
                "positive_class": "coordinated",
                "n_train": int(len(train)),
                "n_test_holdout": int(len(test)),
                "cv_f1_mean": float(cv["f1"].mean()),
                "cv_roc_auc_mean": float(cv["roc_auc"].mean()),
                "lgbm_params": LGBM_PARAMS,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("saved model to %s", MODEL_PATH)

    _write_report(cv, importance, len(train), len(test))
    log.info("wrote results to docs/results.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the virality classifier")
    parser.add_argument("--labels", type=pathlib.Path, default=DEFAULT_LABELS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    asyncio.run(main_async(args.labels, args.seed))


if __name__ == "__main__":
    main()
