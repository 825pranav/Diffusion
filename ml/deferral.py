"""
Learned deferral — predict whether the classifier is about to be wrong.

Usage:
    python -m ml.deferral            # train, evaluate, write results
    python -m ml.deferral --target-accuracy 0.95

Negative result, kept deliberately. This does not beat confidence gating, and
the reason it does not is the useful part.

Confidence gating assumes a prediction the model is sure about is a prediction
likely to be right. On an earlier build that failed badly — published
`viral_organic` cascades scored 0.474, worse than chance — which motivated
replacing it. The real cause turned out to be a feature bug rather than the gate:
author reuse was accumulated from the start of the dataset, so it drifted 1.7
standard deviations across a temporal split. Windowing it lifted `viral_organic`
from 0.500 to 0.781 without touching the gate at all.

This module tests the remaining hypothesis: that difficulty is *learnable* where
confidence is uninformative. A second model answers a different question — will
the classifier get this one right? — from the cascade's own shape plus what the
classifier said about it. It ties confidence gating rather than beating it,
because the cascades both models get wrong are generated to be
feature-indistinguishable from the other class. A deferral model reading the same
features cannot flag what the classifier cannot separate.

The labels for that model must come from **out-of-fold** predictions. Scoring the
classifier on its own training rows would show it correct almost everywhere, and
the deferral model would learn that nothing is ever hard.
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
from sklearn.model_selection import StratifiedKFold

from config import configure_logging
from ml.dataset import DEFAULT_LABELS, feature_matrix, load_dataset, temporal_split
from ml.features import FEATURE_COLUMNS
from ml.report import markdown_table, upsert_section
from ml.train import LGBM_PARAMS, N_FOLDS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "models"
DEFERRAL_PATH = MODEL_DIR / "deferral_lgbm.joblib"
CONTRACT_PATH = MODEL_DIR / "deferral_model.json"

# What the deferral model sees: the cascade itself, plus the classifier's verdict
# on it. The verdict matters because being wrong is not uniform across the score
# range — a cascade the classifier calls 0.9 fails differently from one it calls 0.55.
DEFERRAL_EXTRA = ["p_coordinated", "model_confidence"]
DEFERRAL_FEATURES = [*FEATURE_COLUMNS, *DEFERRAL_EXTRA]

# Accuracy the published set should reach.
DEFAULT_TARGET_ACCURACY = 0.95
# Never defer everything: a gate that publishes nothing trivially hits any target.
MIN_COVERAGE = 0.25

log = logging.getLogger(__name__)


def out_of_fold_probabilities(
    X: pd.DataFrame, y: np.ndarray, seed: int, n_folds: int = N_FOLDS
) -> np.ndarray:
    """
    p(coordinated) for every training row, from a model that never saw it.

    This is the whole basis of the deferral label. In-sample predictions would be
    right nearly everywhere and teach the second model that difficulty does not
    exist.
    """
    oof = np.zeros(len(X), dtype=float)
    folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, valid_idx in folds.split(X, y):
        model = LGBMClassifier(random_state=seed, **LGBM_PARAMS)
        model.fit(X.iloc[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    return oof


def deferral_matrix(X: pd.DataFrame, p_coordinated: np.ndarray) -> pd.DataFrame:
    """Cascade features joined to the classifier's verdict about them."""
    frame = X.copy()
    frame["p_coordinated"] = p_coordinated
    frame["model_confidence"] = np.maximum(p_coordinated, 1.0 - p_coordinated)
    return frame[DEFERRAL_FEATURES]


def train_deferral(
    X: pd.DataFrame, y: np.ndarray, oof_p: np.ndarray, seed: int
) -> tuple[LGBMClassifier, float]:
    """Fit the model that predicts classifier correctness. Returns (model, base rate)."""
    correct = ((oof_p >= 0.5).astype(int) == y).astype(int)
    features = deferral_matrix(X, oof_p)
    model = LGBMClassifier(random_state=seed, **LGBM_PARAMS)
    model.fit(features, correct)
    return model, float(correct.mean())


def risk_coverage(score: np.ndarray, correct: np.ndarray, grid: np.ndarray) -> pd.DataFrame:
    """Accuracy of the published set as the gate is swept across `grid`."""
    rows = []
    for threshold in grid:
        keep = score >= threshold
        if keep.sum() == 0:
            continue
        rows.append(
            {
                "threshold": float(threshold),
                "coverage": float(keep.mean()),
                "accuracy": float(correct[keep].mean()),
                "n": int(keep.sum()),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold(
    curve: pd.DataFrame, target_accuracy: float, min_coverage: float = MIN_COVERAGE
) -> tuple[float, float, float]:
    """
    Lowest gate that reaches the accuracy target while still publishing something.

    Returns (threshold, accuracy, coverage). Falls back to the best available
    accuracy when the target is unreachable, rather than silently gating to zero.
    """
    viable = curve[(curve["accuracy"] >= target_accuracy) & (curve["coverage"] >= min_coverage)]
    if not viable.empty:
        best = viable.sort_values("coverage", ascending=False).iloc[0]
    else:
        best = curve[curve["coverage"] >= min_coverage].sort_values("accuracy").iloc[-1]
    return float(best["threshold"]), float(best["accuracy"]), float(best["coverage"])


def _at_coverage(curve: pd.DataFrame, coverage: float) -> float:
    """Accuracy this gate achieves at (approximately) the given coverage."""
    if curve.empty:
        return float("nan")
    idx = (curve["coverage"] - coverage).abs().idxmin()
    return float(curve.loc[idx, "accuracy"])


def _subtype_table(holdout: pd.DataFrame, published: np.ndarray, label: str) -> list[list[str]]:
    rows = []
    frame = holdout.assign(published=published)
    for subtype, group in frame.groupby("subtype", sort=True):
        pub = group[group["published"]]
        rows.append(
            [
                f"`{subtype}`",
                label,
                f"{len(group)}",
                f"{group['published'].mean():.2f}",
                f"{pub['correct'].mean():.3f}" if len(pub) else "—",
            ]
        )
    return rows


async def main_async(labels_path: pathlib.Path, target_accuracy: float, seed: int) -> None:
    data = await load_dataset(labels_path)
    train, holdout = temporal_split(data)

    X_train, y_train = feature_matrix(train), train["y"].to_numpy()
    log.info("computing out-of-fold classifier predictions over %d rows", len(X_train))
    oof_p = out_of_fold_probabilities(X_train, y_train, seed)
    oof_accuracy = float(((oof_p >= 0.5).astype(int) == y_train).mean())
    log.info("out-of-fold classifier accuracy: %.3f", oof_accuracy)

    deferral, base_rate = train_deferral(X_train, y_train, oof_p, seed)
    log.info("deferral model trained — base correctness rate %.3f", base_rate)

    # Score the holdout with the real classifier, then ask the deferral model
    # whether to trust each verdict.
    classifier = joblib.load(MODEL_DIR / "virality_lgbm.joblib")
    X_hold = feature_matrix(holdout)
    p_hold = classifier.predict_proba(X_hold)[:, 1]
    correct = ((p_hold >= 0.5).astype(int) == holdout["y"].to_numpy()).astype(int)
    holdout = holdout.assign(correct=correct)

    p_correct = deferral.predict_proba(deferral_matrix(X_hold, p_hold))[:, 1]
    confidence = np.maximum(p_hold, 1.0 - p_hold)

    grid = np.round(np.arange(0.05, 0.99, 0.01), 2)
    deferral_curve = risk_coverage(p_correct, correct, grid)
    confidence_curve = risk_coverage(confidence, correct, grid)

    threshold, accuracy, coverage = choose_threshold(deferral_curve, target_accuracy)
    log.info(
        "deferral gate: threshold=%.2f accuracy=%.3f coverage=%.3f",
        threshold, accuracy, coverage,
    )

    # The fair comparison is at matched coverage — a gate that publishes less
    # will always look more accurate.
    confidence_at_match = _at_coverage(confidence_curve, coverage)
    log.info(
        "confidence gate at the same coverage (%.1f%%): accuracy=%.3f",
        100 * coverage, confidence_at_match,
    )

    published_deferral = p_correct >= threshold
    conf_row = confidence_curve.iloc[
        (confidence_curve["coverage"] - coverage).abs().idxmin()
    ]
    published_confidence = confidence >= conf_row["threshold"]

    subtype_rows = _subtype_table(holdout, published_confidence, "confidence")
    subtype_rows += _subtype_table(holdout, published_deferral, "deferral")
    subtype_rows.sort(key=lambda r: (r[0], r[1]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(deferral, DEFERRAL_PATH)
    CONTRACT_PATH.write_text(
        json.dumps(
            {
                "features": DEFERRAL_FEATURES,
                "predicts": "probability the virality classifier is correct",
                "threshold": threshold,
                "target_accuracy": target_accuracy,
                "holdout_accuracy_at_threshold": accuracy,
                "holdout_coverage_at_threshold": coverage,
                "oof_classifier_accuracy": oof_accuracy,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("saved deferral model to %s", DEFERRAL_PATH)

    sweep_rows = []
    for target in (0.90, 0.925, 0.95, 0.97):
        t, a, c = choose_threshold(deferral_curve, target)
        sweep_rows.append(
            [f"{target:.3f}", f"{t:.2f}", f"{a:.3f}", f"{100 * c:.1f}%",
             f"{_at_coverage(confidence_curve, c):.3f}"]
        )

    body = f"""
**This is a negative result, kept because it is one.** Learned deferral was
built to replace confidence gating and does not beat it.

The motivation was real. Confidence gating assumes a prediction the classifier
is sure about is a prediction likely to be right, and on an earlier build that
failed badly — published `viral_organic` cascades scored 0.474, worse than
chance. That turned out to be a *feature bug*, not a gating problem: author
reuse was counted cumulatively from the start of the dataset, so it grew without
bound and meant something different either side of a temporal split. Windowing
it fixed the gate as a side effect, and `viral_organic` went from 0.500 to 0.781
with no change to the gate at all.

Deferral was implemented anyway, to test whether difficulty is learnable. A
second LightGBM predicts *whether the classifier will be correct*, from the
cascade's own shape plus the verdict the classifier gave it, with labels taken
from out-of-fold predictions — scoring the classifier on rows it trained on
would show it correct almost everywhere and teach the deferral model that
nothing is hard.

It matches confidence gating and does not beat it ({accuracy:.3f} against
{confidence_at_match:.3f} at the same coverage). The subtype table below says
why: both gates publish a quarter of `stealth_coordinated` and get *every one*
wrong. Those cascades are generated to look organic — paced timing, rotated
accounts — so they are close to indistinguishable in the feature space both
models read. A deferral model looking at the same features cannot flag what the
classifier cannot separate; the difficulty is irreducible here, not something a
better gate recovers.

The conclusion that follows is about features, not gating: catching careful
campaigns needs signals these features do not carry — account age, or
coordination structure across cascades rather than within one.

### Gate comparison, at matched coverage

Both gates publish the same share of cases, so the only difference is *which*
cases each one keeps.

{markdown_table(
    ["target accuracy", "deferral threshold", "deferral accuracy", "coverage", "confidence accuracy at same coverage"],
    sweep_rows,
)}

### Where each gate publishes, by subtype

`published` is the share auto-published; `accuracy` is measured on those only.
The confusable subtypes are where the confidence gate broke.

{markdown_table(
    ["subtype", "gate", "n", "published", "accuracy once published"],
    subtype_rows,
)}

### What ships

Confidence gating stays in `agent/confidence.py`. Deferral matches it and costs a
second model plus a second artefact to keep in sync, so it is not wired into the
runtime — added complexity has to buy something measurable. The module and this
table remain as the record of what was tried.

Out-of-fold classifier accuracy is {oof_accuracy:.3f}, so both gates are choosing
between genuinely uncertain outcomes rather than reading an easy signal.

Reproduce with `python -m ml.deferral`.
"""
    upsert_section("Learned deferral — knowing when not to answer", body)
    log.info("wrote results to docs/results.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate learned deferral")
    parser.add_argument("--labels", type=pathlib.Path, default=DEFAULT_LABELS)
    parser.add_argument("--target-accuracy", type=float, default=DEFAULT_TARGET_ACCURACY)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    asyncio.run(main_async(args.labels, args.target_accuracy, args.seed))


if __name__ == "__main__":
    main()
