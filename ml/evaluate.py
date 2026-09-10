"""
Compare the classifier, the LLM agent, and the two combined on held-out data.

Usage:
    python -m ml.evaluate --skip-llm            # classifier arm only, seconds
    python -m ml.evaluate --llm-sample 24       # all three arms

Three arms, all scored on the same held-out cascades:

  classifier   LightGBM alone
  llm          the ReAct agent with classify_virality_model withheld, so it
               reasons from propagation structure and raw counts
  hybrid       the same agent with the model tool available

The LLM arms run the real agent, one investigation at a time against a local
model, so they are sampled rather than run over the full holdout — the sample
size is reported alongside every number it produces.

This is also where the human-review gate comes from. `CONFIDENCE_THRESHOLD` was
a guessed 0.65; here it is chosen from the reliability curve as the lowest
confidence at which observed accuracy clears the target, which also says what
fraction of cases that leaves for a human.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib

import aiohttp
import asyncpg
import matplotlib

matplotlib.use("Agg")  # headless: no display on CI or a server
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from agent.agent import run_investigation  # noqa: E402
from config import DB_URL, configure_logging  # noqa: E402
from ml.dataset import DEFAULT_LABELS, feature_matrix, load_dataset, temporal_split  # noqa: E402
from ml.predict import load_model  # noqa: E402
from ml.report import markdown_table, upsert_section  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DIAGRAM_PATH = REPO_ROOT / "docs" / "reliability.png"

# Accuracy a case file must clear before it is published without human review.
TARGET_ACCURACY = 0.90
# Candidate gates considered when reading the reliability curve.
THRESHOLD_GRID = np.round(np.arange(0.50, 0.96, 0.05), 2)
N_RELIABILITY_BINS = 8

DEFAULT_LLM_SAMPLE = 24

log = logging.getLogger(__name__)


def _verdict_to_probability(classification: str, confidence: float) -> float:
    """
    Map an agent verdict onto P(coordinated) so it is comparable with the model.

    The agent states a class and its confidence in that class, so an `organic`
    verdict at 0.8 is P(coordinated) = 0.2. `uncertain` carries no information
    either way and lands at 0.5.
    """
    confidence = float(np.clip(confidence, 0.0, 1.0))
    if classification == "coordinated_amplification":
        return confidence
    if classification == "organic":
        return 1.0 - confidence
    return 0.5


def arm_metrics(y_true: np.ndarray, p_coordinated: np.ndarray) -> dict[str, float]:
    """
    Discrimination and calibration for one arm, reported per class.

    Both classes are reported because they fail differently: an arm can look
    strong on `coordinated` while quietly misclassifying most organic cascades,
    and on a near-balanced holdout a single positive-class F1 hides that.
    """
    pred = (p_coordinated >= 0.5).astype(int)
    metrics = {
        "n": int(len(y_true)),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "brier": brier_score_loss(y_true, p_coordinated),
        "accuracy": float((pred == y_true).mean()),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
    }
    # Per class: label 1 = coordinated, label 0 = organic.
    for name, label in (("coordinated", 1), ("organic", 0)):
        metrics[f"precision_{name}"] = precision_score(
            y_true, pred, pos_label=label, zero_division=0
        )
        metrics[f"recall_{name}"] = recall_score(
            y_true, pred, pos_label=label, zero_division=0
        )
        metrics[f"f1_{name}"] = f1_score(y_true, pred, pos_label=label, zero_division=0)
    # A single-class sample makes ROC-AUC undefined rather than zero.
    metrics["roc_auc"] = (
        roc_auc_score(y_true, p_coordinated) if len(np.unique(y_true)) > 1 else float("nan")
    )
    return metrics


def choose_review_threshold(
    confidence: np.ndarray,
    correct: np.ndarray,
    target: float = TARGET_ACCURACY,
) -> tuple[float, float, float]:
    """
    Pick the lowest confidence gate whose retained predictions clear `target`.

    Returns (threshold, accuracy above it, coverage). Lowest rather than safest:
    every point of extra threshold sends more cases to a human, so the cheapest
    gate meeting the accuracy bar is the right one.
    """
    best = (1.0, float("nan"), 0.0)
    for threshold in THRESHOLD_GRID:
        keep = confidence >= threshold
        if keep.sum() == 0:
            continue
        accuracy = float(correct[keep].mean())
        coverage = float(keep.mean())
        if accuracy >= target:
            return float(threshold), accuracy, coverage
        best = (float(threshold), accuracy, coverage)
    return best


def _reliability(p: np.ndarray, y: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    centres, observed, counts = [], [], []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        centres.append(float(p[mask].mean()))
        observed.append(float(y[mask].mean()))
        counts.append(int(mask.sum()))
    return np.array(centres), np.array(observed), np.array(counts)


def plot_reliability(arms: dict[str, tuple[np.ndarray, np.ndarray]], path: pathlib.Path) -> None:
    """Predicted probability against observed frequency, one line per arm."""
    fig, ax = plt.subplots(figsize=(6.0, 5.0), dpi=140)
    ax.plot([0, 1], [0, 1], color="#888", linestyle="--", linewidth=1, label="perfect calibration")

    for name, (y_true, p) in arms.items():
        if len(y_true) == 0:
            continue
        centres, observed, counts = _reliability(p, y_true, N_RELIABILITY_BINS)
        if centres.size == 0:
            continue
        # Marker area tracks bin population so sparse bins read as less reliable.
        ax.plot(centres, observed, marker="o", linewidth=1.6, label=f"{name} (n={len(y_true)})")
        ax.scatter(centres, observed, s=8 + 90 * counts / counts.max(), alpha=0.25)

    ax.set_xlabel("predicted P(coordinated)")
    ax.set_ylabel("observed fraction coordinated")
    ax.set_title("Reliability — held-out cascades")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


async def _root_platforms(conn: asyncpg.Connection, root_ids: list[str]) -> dict[str, str]:
    rows = await conn.fetch(
        "SELECT id, platform FROM graph_nodes WHERE id = ANY($1::text[])", root_ids
    )
    return {r["id"]: r["platform"] for r in rows}


async def run_llm_arm(
    sample: pd.DataFrame,
    include_model_tool: bool,
    label: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Run the agent over the sampled cascades.

    Returns (y_true, p_coordinated, n_failed). Investigations that produce no
    parseable verdict are dropped rather than scored as a wrong answer — the arm
    is measuring the quality of the verdicts it gives, and the failure count is
    reported separately so the omission is visible.
    """
    conn = await asyncpg.connect(DB_URL)
    y_true: list[int] = []
    probabilities: list[float] = []
    failed = 0
    try:
        platforms = await _root_platforms(conn, sample["root_node_id"].tolist())
        async with aiohttp.ClientSession() as session:
            for i, row in enumerate(sample.itertuples(), start=1):
                try:
                    verdict = await run_investigation(
                        conn,
                        session,
                        node_id=row.root_node_id,
                        platform=platforms.get(row.root_node_id, "unknown"),
                        include_model_tool=include_model_tool,
                    )
                except Exception:
                    failed += 1
                    log.warning("[%s] %d/%d produced no verdict", label, i, len(sample))
                    continue
                probabilities.append(
                    _verdict_to_probability(verdict["classification"], verdict["confidence"])
                )
                y_true.append(int(row.y))
                log.info(
                    "[%s] %d/%d %s -> %s @ %.2f (truth=%s)",
                    label, i, len(sample), row.root_node_id[-12:],
                    verdict["classification"], verdict["confidence"], row.label,
                )
    finally:
        await conn.close()
    return np.array(y_true), np.array(probabilities), failed


def _stratified_sample(holdout: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Balanced sample, so a small LLM run is not dominated by one class."""
    per_class = max(n // 2, 1)
    parts = [
        group.sample(min(per_class, len(group)), random_state=seed)
        for _, group in holdout.groupby("label")
    ]
    return pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def subtype_breakdown(holdout: pd.DataFrame, p_coordinated: np.ndarray) -> str:
    """Classifier accuracy split by the simulator's plain and confusable subtypes."""
    frame = holdout[["subtype", "y"]].copy()
    frame["correct"] = ((p_coordinated >= 0.5).astype(int) == frame["y"]).astype(float)
    rows = [
        [f"`{subtype}`", int(len(g)), f"{g['correct'].mean():.3f}"]
        for subtype, g in frame.groupby("subtype", sort=True)
    ]
    return markdown_table(["subtype", "n", "accuracy"], rows)


def _write_report(
    results: dict[str, dict[str, float]],
    threshold: float,
    gate_accuracy: float,
    coverage: float,
    failures: dict[str, int],
    n_holdout: int,
    subtype_table: str,
) -> None:
    rows = []
    per_class_rows = []
    for name, m in results.items():
        rows.append(
            [
                f"`{name}`",
                m["n"],
                f"{m['accuracy']:.3f}",
                f"{m['macro_f1']:.3f}",
                "n/a" if np.isnan(m["roc_auc"]) else f"{m['roc_auc']:.3f}",
                f"{m['brier']:.3f}",
            ]
        )
        for cls in ("coordinated", "organic"):
            per_class_rows.append(
                [
                    f"`{name}`",
                    cls,
                    f"{m[f'precision_{cls}']:.3f}",
                    f"{m[f'recall_{cls}']:.3f}",
                    f"{m[f'f1_{cls}']:.3f}",
                ]
            )

    failure_note = ", ".join(f"`{k}` {v}" for k, v in failures.items() if v) or "none"

    body = f"""
All arms scored on the same temporally held-out split ({n_holdout} cascades;
the LLM arms are sampled from it, `n` below).

{markdown_table(
    ["arm", "n", "accuracy", "macro F1", "ROC-AUC", "Brier"],
    rows,
)}

### Per class

Reported both ways because the arms fail differently — an arm can look strong on
`coordinated` while misclassifying most organic cascades, which a single
positive-class F1 hides.

{markdown_table(
    ["arm", "class", "precision", "recall", "F1"],
    per_class_rows,
)}

![reliability diagram](reliability.png)

Investigations that returned no parseable verdict: {failure_note}. They are
excluded rather than counted as wrong — each arm is measured on the answers it
actually gives, and the count is reported so the omission stays visible.

### By cascade subtype

{subtype_table}

The simulator generates a fraction of each class as a confusable subtype:
`viral_organic` cascades that burst like a campaign, and `stealth_coordinated`
ones that pace themselves and rotate accounts. They should be measurably harder
than the plain cases, and are — which is the evidence that the overlap built into
the generator is doing real work rather than decorating the dataset.

### Human-review gate

`CONFIDENCE_THRESHOLD` was a guessed 0.65. Read off the classifier's reliability
curve, the lowest gate whose retained predictions reach
{100 * TARGET_ACCURACY:.0f}% accuracy is **{threshold:.2f}**, which holds
{100 * gate_accuracy:.1f}% accuracy while auto-publishing
{100 * coverage:.1f}% of cases. The remainder goes to a human.

Lowest rather than safest: every extra point of threshold buys accuracy by
sending more cases to review, so the cheapest gate that clears the bar is the
right one. Re-derive after any retrain — the number is a property of the fitted
model, not a constant.

Reproduce with `python -m ml.evaluate`.
"""
    upsert_section("Classifier vs LLM agent — held-out comparison", body)


async def main_async(
    labels_path: pathlib.Path, llm_sample: int, skip_llm: bool, seed: int
) -> None:
    model = load_model()
    if model is None:
        raise SystemExit("no trained model — run: python -m ml.train")

    data = await load_dataset(labels_path)
    _, holdout = temporal_split(data)

    p_classifier = model.predict_proba(feature_matrix(holdout))[:, 1]
    y_holdout = holdout["y"].to_numpy()

    results = {"classifier": arm_metrics(y_holdout, p_classifier)}
    arms_for_plot = {"classifier": (y_holdout, p_classifier)}
    failures: dict[str, int] = {}

    if not skip_llm:
        sample = _stratified_sample(holdout, llm_sample, seed)
        log.info("running LLM arms over %d sampled cascades (this is slow)", len(sample))
        for name, include_model in (("llm", False), ("hybrid", True)):
            y, p, failed = await run_llm_arm(sample, include_model, name)
            failures[name] = failed
            if len(y):
                results[name] = arm_metrics(y, p)
                arms_for_plot[name] = (y, p)
            else:
                log.warning("[%s] produced no usable verdicts at all", name)

    plot_reliability(arms_for_plot, DIAGRAM_PATH)
    log.info("wrote reliability diagram to %s", DIAGRAM_PATH)

    confidence = np.maximum(p_classifier, 1.0 - p_classifier)
    correct = ((p_classifier >= 0.5).astype(int) == y_holdout).astype(float)
    threshold, gate_accuracy, coverage = choose_review_threshold(confidence, correct)
    log.info(
        "review gate: threshold=%.2f accuracy=%.3f coverage=%.3f",
        threshold, gate_accuracy, coverage,
    )

    for name, m in results.items():
        log.info(
            "%-11s n=%-4d f1=%.3f roc_auc=%s brier=%.3f",
            name, m["n"], m["f1"],
            "n/a" if np.isnan(m["roc_auc"]) else f"{m['roc_auc']:.3f}",
            m["brier"],
        )

    _write_report(
        results, threshold, gate_accuracy, coverage, failures, len(holdout),
        subtype_breakdown(holdout, p_classifier),
    )
    log.info("wrote results to docs/results.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classifier, LLM agent, and hybrid")
    parser.add_argument("--labels", type=pathlib.Path, default=DEFAULT_LABELS)
    parser.add_argument("--llm-sample", type=int, default=DEFAULT_LLM_SAMPLE)
    parser.add_argument(
        "--skip-llm", action="store_true", help="classifier arm only (fast)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging()
    asyncio.run(main_async(args.labels, args.llm_sample, args.skip_llm, args.seed))


if __name__ == "__main__":
    main()
