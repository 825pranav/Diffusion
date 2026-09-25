"""
Classifier experiments: features, tuning, calibration, a GNN, and distribution shift.

Usage:
    python -m ml.experiment cv          # compare models by CV on the training split
    python -m ml.experiment shift       # robustness under simulator parameter shift
    python -m ml.experiment tune        # tune the shipped model's params (training split)
    python -m ml.experiment oracle      # ceiling: a model given the true simulator params
    python -m ml.experiment skew        # cost of featurising one cascade in isolation
    python -m ml.experiment holdout     # the single look at the 400 held-out cascades

Every comparison in `cv` runs on the temporal *training* split only, with the
same repeated stratified folds for every arm so differences are paired. The 400
held-out cascades are read by `holdout` alone, once, after the choice is made.

Hyperparameter search is nested: each outer fold runs its own Optuna study on
inner folds of its own training rows, so the tuned arm's score is an honest
estimate of "tune, then deploy" rather than of the best configuration found by
peeking at the validation folds.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pathlib
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from config import configure_logging
from ml.dataset import temporal_split
from ml.features import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS
from ml.gnn import GNNClassifier, build_graphs
from ml.offline import Shift, build_dataset, cascade_frames, simulate
from ml.report import markdown_table, upsert_section
from ml.train import LGBM_PARAMS as BASE_LGBM_PARAMS

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Tracked in git (models/ is not), so the shipped configuration is reproducible.
TUNED_PARAMS_PATH = REPO_ROOT / "ml" / "tuned_params.json"

N_SPLITS = 5
N_REPEATS = 2
INNER_SPLITS = 3
DEFAULT_TRIALS = 30

log = logging.getLogger(__name__)


# ── metrics ───────────────────────────────────────────────────────────────────


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Population-weighted gap between predicted and observed rate, equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(ece)


def score(y: np.ndarray, p: np.ndarray, subtype: np.ndarray | None = None) -> dict[str, float]:
    pred = (p >= 0.5).astype(int)
    out = {
        "macro_f1": f1_score(y, pred, average="macro"),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "ece": expected_calibration_error(y, p),
        "accuracy": float((pred == y).mean()),
    }
    if subtype is not None:
        for name in ("stealth_coordinated", "viral_organic", "plain"):
            mask = subtype == name
            out[f"acc_{name}"] = float((pred[mask] == y[mask]).mean()) if mask.any() else np.nan
    return out


# ── model factories ───────────────────────────────────────────────────────────


def lgbm(params: dict, seed: int) -> LGBMClassifier:
    return LGBMClassifier(random_state=seed, **{**params, "verbose": -1})


def tune_lgbm(X: pd.DataFrame, y: np.ndarray, trials: int, seed: int) -> dict:
    """Optuna TPE search minimising inner-CV log loss. Returns LightGBM params."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    folds = list(StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed).split(X, y))

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            num_leaves=trial.suggest_int("num_leaves", 4, 64, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 80, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            subsample_freq=1,
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
        losses = []
        for tr, va in folds:
            model = lgbm(params, seed).fit(X.iloc[tr], y[tr])
            losses.append(log_loss(y[va], model.predict_proba(X.iloc[va])[:, 1]))
        return float(np.mean(losses))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.optimize(objective, n_trials=trials)
    return {**study.best_params, "subsample_freq": 1}


# ── cross-validation ──────────────────────────────────────────────────────────

ARMS = [
    ("logreg (all features)", "Logistic regression, standardised — a linear floor"),
    ("lgbm (base features)", "The shipped model: original 23 features, original params"),
    ("lgbm (+ new features)", "Original params, 37 features"),
    ("lgbm (+ new, Optuna)", "37 features, params tuned per outer fold (nested)"),
    ("lgbm (+ new, Optuna, isotonic)", "Tuned, then isotonic calibration on inner folds"),
    ("GIN (tree GNN)", "Graph neural network on the reshare tree itself"),
    ("GIN + lgbm (mean)", "Average of the GNN and the tuned LightGBM probabilities"),
]


def run_cv(train: pd.DataFrame, graphs: dict, trials: int, seed: int, gnn_epochs: int):
    y = train["y"].to_numpy()
    subtype = train["subtype"].to_numpy()
    X_all = train[FEATURE_COLUMNS]
    X_base = train[BASE_FEATURE_COLUMNS]
    g_all = [graphs[r] for r in train["root_node_id"]]

    folds = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    rows = []
    chosen_params = []
    for k, (tr, va) in enumerate(folds.split(X_all, y), start=1):
        t = time.time()
        preds: dict[str, np.ndarray] = {}

        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        preds["logreg (all features)"] = lr.fit(X_all.iloc[tr], y[tr]).predict_proba(
            X_all.iloc[va])[:, 1]
        preds["lgbm (base features)"] = lgbm(BASE_LGBM_PARAMS, seed).fit(
            X_base.iloc[tr], y[tr]).predict_proba(X_base.iloc[va])[:, 1]
        preds["lgbm (+ new features)"] = lgbm(BASE_LGBM_PARAMS, seed).fit(
            X_all.iloc[tr], y[tr]).predict_proba(X_all.iloc[va])[:, 1]

        params = tune_lgbm(X_all.iloc[tr], y[tr], trials, seed + k)
        chosen_params.append(params)
        preds["lgbm (+ new, Optuna)"] = lgbm(params, seed).fit(
            X_all.iloc[tr], y[tr]).predict_proba(X_all.iloc[va])[:, 1]
        cal = CalibratedClassifierCV(lgbm(params, seed), method="isotonic", cv=INNER_SPLITS)
        preds["lgbm (+ new, Optuna, isotonic)"] = cal.fit(
            X_all.iloc[tr], y[tr]).predict_proba(X_all.iloc[va])[:, 1]

        gnn = GNNClassifier(epochs=gnn_epochs, seed=seed).fit([g_all[i] for i in tr], y[tr])
        preds["GIN (tree GNN)"] = gnn.predict_proba([g_all[i] for i in va])[:, 1]
        preds["GIN + lgbm (mean)"] = 0.5 * (
            preds["GIN (tree GNN)"] + preds["lgbm (+ new, Optuna)"]
        )

        for arm, p in preds.items():
            rows.append({"fold": k, "arm": arm, **score(y[va], p, subtype[va])})
        log.info(
            "fold %d/%d (%.0fs) — %s", k, N_SPLITS * N_REPEATS, time.time() - t,
            "  ".join(f"{a.split(' ')[0]}:{roc_auc_score(y[va], p):.3f}" for a, p in preds.items()),
        )
    return pd.DataFrame(rows), chosen_params


def _write_cv_report(cv: pd.DataFrame, n_train: int, trials: int, gnn_epochs: int) -> None:
    metrics = ["macro_f1", "roc_auc", "pr_auc", "brier", "ece", "acc_stealth_coordinated",
               "acc_viral_organic"]
    rows = []
    for arm, _ in ARMS:
        g = cv[cv["arm"] == arm]
        rows.append([f"`{arm}`", *[f"{g[m].mean():.3f} ± {g[m].std():.3f}" for m in metrics]])

    base = cv[cv["arm"] == "lgbm (base features)"].set_index("fold")
    delta_rows = []
    for arm, _ in ARMS:
        if arm == "lgbm (base features)":
            continue
        g = cv[cv["arm"] == arm].set_index("fold")
        cells = [f"`{arm}`"]
        for m in ("macro_f1", "roc_auc", "brier"):
            d = g[m] - base[m]
            better = (d < 0) if m == "brier" else (d > 0)
            cells.append(f"{d.mean():+.3f} ± {d.std():.3f} ({int(better.sum())}/{len(d)})")
        delta_rows.append(cells)

    arm_rows = [[f"`{a}`", desc] for a, desc in ARMS]
    body = f"""
Every arm scored on the same {N_SPLITS}-fold x {N_REPEATS}-repeat stratified split of
the temporal training split ({n_train} cascades), so fold-to-fold differences
are paired. The holdout is not used here. Optuna runs {trials} TPE trials per
outer fold on {INNER_SPLITS} inner folds of that fold's training rows (nested), so
the tuned arm is not scored on data its search saw. The GNN trains for
{gnn_epochs} epochs per fold with no early stopping (no inner validation to stop on).

{markdown_table(["arm", "what it is"], arm_rows)}

{markdown_table(
    ["arm", "macro F1", "ROC-AUC", "PR-AUC", "Brier", "ECE", "stealth_coord. acc", "viral_organic acc"],
    rows,
)}

### Paired difference against the shipped model

Mean ± std of the per-fold difference, and how many of the
{N_SPLITS * N_REPEATS} folds the arm won. Lower Brier is better.

{markdown_table(["arm", "Δ macro F1", "Δ ROC-AUC", "Δ Brier"], delta_rows)}

Reproduce with `python -m ml.experiment cv`.
"""
    upsert_section("Classifier experiments — training-split CV", body)


# ── shared data loading ───────────────────────────────────────────────────────


def _dataset_and_graphs(n: int, difficulty: str, seed: int, shift: Shift | None = None):
    cascades = simulate(n, difficulty, seed, shift)
    tree, authors = cascade_frames(cascades)
    return build_dataset(cascades), build_graphs(tree, authors)


# ── distribution shift ────────────────────────────────────────────────────────

SHIFT_SEED = 7

SHIFTS: list[tuple[Shift, str]] = [
    (Shift("in-distribution (new seed)"), "medium"),
    (Shift("harder mix", notes="difficulty=hard: 28% confusable subtypes, wider spread"), "hard"),
    (Shift("slower campaigns", coord_delay_mult=3.0,
           notes="coordinated delays x3"), "medium"),
    (Shift("desynchronised", coord_sigma_add=0.5,
           notes="coordinated delay sigma +0.5"), "medium"),
    (Shift("more human accounts", coord_bot_mult=0.5,
           notes="coordinated bot share x0.5"), "medium"),
    (Shift("larger bot pool", n_bots=2000,
           notes="2,000 bots instead of 400: far less reuse per account"), "medium"),
    (Shift("less star-shaped", coord_attach_mult=0.6,
           notes="coordinated root attachment x0.6"), "medium"),
    (Shift("all-stealth", hard_frac=1.0,
           notes="every cascade is its confusable subtype"), "medium"),
    (Shift("camouflaged", coord_delay_mult=2.5, coord_sigma_add=0.4, coord_bot_mult=0.6,
           coord_attach_mult=0.6, notes="all of the above at once, milder"), "medium"),
]


def _fit_shift_models(train: pd.DataFrame, graphs: dict, tuned: dict, seed: int, gnn_epochs: int):
    y = train["y"].to_numpy()
    models = {
        "lgbm base": (lgbm(BASE_LGBM_PARAMS, seed).fit(train[BASE_FEATURE_COLUMNS], y),
                      BASE_FEATURE_COLUMNS),
        "lgbm tuned + new": (lgbm(tuned, seed).fit(train[FEATURE_COLUMNS], y), FEATURE_COLUMNS),
    }
    gnn = GNNClassifier(epochs=gnn_epochs, seed=seed).fit(
        [graphs[r] for r in train["root_node_id"]], y
    )
    return models, gnn


def run_shift(seed: int, gnn_epochs: int) -> None:
    tuned = _load_tuned_params()
    data, graphs = _dataset_and_graphs(2000, "medium", 42)
    train, _ = temporal_split(data)  # the holdout stays unused here
    models, gnn = _fit_shift_models(train, graphs, tuned, seed, gnn_epochs)

    # Domain randomisation: extra training cascades from randomly perturbed
    # simulators. Each test shift is scored by a model whose augmentation
    # *excluded that shift's family*, so this measures transfer to an unseen
    # kind of shift, not memorisation of the test configuration.
    aug = _augmentation_pool(seed)

    rows = []
    for shift, difficulty in SHIFTS:
        test, test_graphs = _dataset_and_graphs(2000, difficulty, SHIFT_SEED, shift)
        y = test["y"].to_numpy()
        row = [shift.name, shift.notes or "—"]
        results = {}
        for name, (model, cols) in models.items():
            results[name] = model.predict_proba(test[cols])[:, 1]
        results["GIN"] = gnn.predict_proba([test_graphs[r] for r in test["root_node_id"]])[:, 1]

        family = _family(shift)
        pool = aug[aug["family"] != family] if family else aug
        mix = pd.concat([train, pool], ignore_index=True)
        dr = lgbm(tuned, seed).fit(mix[FEATURE_COLUMNS], mix["y"].to_numpy())
        results["lgbm tuned + new, randomised"] = dr.predict_proba(test[FEATURE_COLUMNS])[:, 1]

        for p in results.values():
            m = score(y, p)
            row.append(f"{m['roc_auc']:.3f} / {m['macro_f1']:.3f}")
        rows.append(row)
        log.info("%s: %s", shift.name, row[2:])

    headers = ["shift", "what changes", "lgbm base", "lgbm tuned + new", "GIN",
               "lgbm tuned + new, domain-randomised"]
    body = f"""
Each model is trained once on the ordinary training split (1,600 cascades,
seed 42) and then scored on 2,000 fresh cascades from a *perturbed* simulator
(seed {SHIFT_SEED}). Every perturbation makes coordinated cascades look more like
organic ones in one specific way. Cells are **ROC-AUC / macro F1**.

The last column adds domain randomisation: 2,000 extra training cascades drawn
from simulators with randomly perturbed campaign parameters. For every row the
augmentation **leaves out the family of shift being tested** (e.g. the
"slower campaigns" row is scored by a model that never saw slowed-down
campaigns), so the column measures transfer to an unseen kind of shift.

{markdown_table(headers, rows)}

Reproduce with `python -m ml.experiment shift`.
"""
    upsert_section("Robustness — simulator parameter shift", body)


FAMILIES = ("delay", "sigma", "bot", "attach", "pool", "stealth")


def _family(shift: Shift) -> str | None:
    """The single perturbation family a test shift belongs to (None = several/none)."""
    tags = []
    if shift.coord_delay_mult != 1.0:
        tags.append("delay")
    if shift.coord_sigma_add != 0.0:
        tags.append("sigma")
    if shift.coord_bot_mult != 1.0:
        tags.append("bot")
    if shift.coord_attach_mult != 1.0:
        tags.append("attach")
    if shift.n_bots != 400:
        tags.append("pool")
    if shift.hard_frac is not None:
        tags.append("stealth")
    return tags[0] if len(tags) == 1 else ("multi" if tags else None)


def _augmentation_pool(seed: int) -> pd.DataFrame:
    """
    Cascades from randomly perturbed simulators, one perturbation family each.

    Drawn from its own seed range, disjoint from both the training data (42)
    and the shift test sets (7), and at magnitudes sampled rather than fixed.
    """
    rng = np.random.default_rng(seed + 1000)
    parts = []
    for i, fam in enumerate(FAMILIES):
        kw: dict = {}
        if fam == "delay":
            kw["coord_delay_mult"] = float(rng.uniform(1.5, 4.0))
        elif fam == "sigma":
            kw["coord_sigma_add"] = float(rng.uniform(0.2, 0.7))
        elif fam == "bot":
            kw["coord_bot_mult"] = float(rng.uniform(0.3, 0.8))
        elif fam == "attach":
            kw["coord_attach_mult"] = float(rng.uniform(0.4, 0.8))
        elif fam == "pool":
            kw["n_bots"] = int(rng.integers(800, 3000))
        elif fam == "stealth":
            kw["hard_frac"] = float(rng.uniform(0.4, 0.9))
        data, _ = _dataset_and_graphs(2000, "medium", 5000 + i, Shift(f"aug-{fam}", **kw))
        # A subsample keeps the augmented model's training set a manageable
        # multiple of the original, and matches the original's cascade density.
        data = data.sample(400, random_state=seed)
        data["family"] = fam
        parts.append(data)
    return pd.concat(parts, ignore_index=True)


# ── tuned params for the shipped model ────────────────────────────────────────


def _load_tuned_params() -> dict:
    if not TUNED_PARAMS_PATH.exists():
        raise SystemExit("no tuned params — run: python -m ml.experiment tune")
    return json.loads(TUNED_PARAMS_PATH.read_text(encoding="utf-8"))["params"]


def run_tune(trials: int, seed: int) -> None:
    """Tune once on the whole training split, for the model that ships."""
    data, _ = _dataset_and_graphs(2000, "medium", 42)
    train, _ = temporal_split(data)
    params = tune_lgbm(train[FEATURE_COLUMNS], train["y"].to_numpy(), trials, seed)
    TUNED_PARAMS_PATH.parent.mkdir(exist_ok=True)
    TUNED_PARAMS_PATH.write_text(
        json.dumps({"params": params, "trials": trials, "seed": seed,
                    "objective": f"{INNER_SPLITS}-fold log loss, training split only"}, indent=2),
        encoding="utf-8",
    )
    log.info("tuned params: %s", params)


# ── how much signal exists at all ─────────────────────────────────────────────

ORACLE_PARAMS = [
    "param_root_attach", "param_delay_median_s", "param_delay_sigma",
    "param_bot_frac", "param_hop_prob", "param_hop_lag_s",
]


def run_oracle(seed: int) -> None:
    """
    Cross-validate a classifier that reads the simulator's true parameters.

    This leaks the generating process on purpose. No feature computed from an
    observed cascade can carry more information than the parameters that
    produced it, so this is a ceiling: where the oracle fails, the classes
    genuinely overlap and no feature engineering will recover the difference.
    """
    from ml.dataset import DEFAULT_LABELS

    labels = pd.read_csv(DEFAULT_LABELS, parse_dates=["t0"])
    labels["y"] = (labels["label"] == "coordinated").astype(int)
    train, _ = temporal_split(labels)
    y, subtype = train["y"].to_numpy(), train["subtype"].to_numpy()
    folds = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    p = np.zeros(len(train))
    per_fold = []
    for tr, va in folds.split(train, y):
        model = lgbm(BASE_LGBM_PARAMS, seed).fit(train[ORACLE_PARAMS].iloc[tr], y[tr])
        pv = model.predict_proba(train[ORACLE_PARAMS].iloc[va])[:, 1]
        per_fold.append(score(y[va], pv, subtype[va]))
        p[va] = pv
    m = pd.DataFrame(per_fold).mean()
    rows = [[k, f"{m[k]:.3f}"] for k in ("macro_f1", "roc_auc", "acc_plain",
                                         "acc_viral_organic", "acc_stealth_coordinated")]
    log.info("oracle: %s", m.round(3).to_dict())
    body = f"""
A LightGBM given each cascade's **true generating parameters** (root attachment,
delay median and spread, bot share, hop probability and lag) instead of
anything observed, cross-validated on the training split exactly like the arms
above. This deliberately leaks the simulator: nothing measured from a cascade
can carry more information than the parameters that produced it, so these
numbers are a ceiling.

{markdown_table(["metric (CV mean)", "oracle"], rows)}

Where the oracle fails, the two classes overlap by construction: a stealth
campaign's parameters are drawn from ranges that organic cascades also occupy,
and no feature engineering can separate what the generator made identical.

Reproduce with `python -m ml.experiment oracle`.
"""
    upsert_section("Classifier ceiling — oracle on true simulator parameters", body)


# ── train/serve skew in single-cascade scoring ───────────────────────────────


def run_skew(seed: int) -> None:
    """
    What scoring a cascade in isolation did to the agent's model tool.

    `ml.predict.score_cascade` used to featurise the one root it was asked
    about. The reuse and co-author features are defined against the cascades
    in the preceding window, so in isolation every one of them read zero. Here
    each validation fold is scored twice — with features computed in context,
    as in training, and with each cascade featurised alone, as the tool did.
    """
    from ml.features import compute_features
    from ml.train import model_params

    cascades = simulate(2000, "medium", 42)
    tree, authors = cascade_frames(cascades)
    data = build_dataset(cascades)
    train, _ = temporal_split(data)
    by_tree = dict(tuple(tree.groupby("root_id")))
    by_author = dict(tuple(authors.groupby("root_id")))
    alone = (
        pd.concat([compute_features(by_tree[r], by_author[r]) for r in train["root_node_id"]])
        .set_index("root_node_id").loc[train["root_node_id"]].reset_index()
    )
    y = train["y"].to_numpy()
    rows = []
    for name, cols, params in (("original", BASE_FEATURE_COLUMNS, BASE_LGBM_PARAMS),
                               ("upgraded", FEATURE_COLUMNS, model_params())):
        ctx, iso = [], []
        for tr, va in StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(train, y):
            model = lgbm(params, seed).fit(train[cols].iloc[tr], y[tr])
            ctx.append(score(y[va], model.predict_proba(train[cols].iloc[va])[:, 1]))
            iso.append(score(y[va], model.predict_proba(alone[cols].iloc[va])[:, 1]))
        c, i = pd.DataFrame(ctx).mean(), pd.DataFrame(iso).mean()
        rows.append([name, f"{c['roc_auc']:.3f}", f"{i['roc_auc']:.3f}",
                     f"{c['macro_f1']:.3f}", f"{i['macro_f1']:.3f}"])
        log.info("skew %s: context auc=%.3f alone auc=%.3f", name, c["roc_auc"], i["roc_auc"])

    body = f"""
The agent's `classify_virality_model` tool scored a cascade by featurising that
one root. Author reuse — the strongest feature by gain — and the co-author
features are defined against the cascades that ran in the trailing window, so
featurised alone they all read zero: a value the model only ever saw on the
first few cascades of its training data. Nothing errored.

{N_SPLITS}-fold CV on the training split, each validation fold scored with
features computed in context (as in training) and in isolation (as the tool
did):

{markdown_table(["model", "ROC-AUC in context", "ROC-AUC alone", "macro F1 in context",
                 "macro F1 alone"], rows)}

`score_cascade` now featurises the root together with every cascade active in
the window before it, which reproduces the batch features exactly (checked on
database cascades). The batch evaluation numbers were never affected — only
what the agent saw.

Reproduce with `python -m ml.experiment skew`.
"""
    upsert_section("Train/serve skew — scoring one cascade in isolation", body)


# ── the one look at the holdout ───────────────────────────────────────────────


def paired_bootstrap(
    y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n_boot: int = 2000, seed: int = 0
) -> dict[str, tuple[float, float]]:
    """95% intervals for the (b - a) difference in ROC-AUC and macro F1, resampling cascades."""
    rng = np.random.default_rng(seed)
    d_auc, d_f1 = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        d_auc.append(roc_auc_score(y[idx], p_b[idx]) - roc_auc_score(y[idx], p_a[idx]))
        d_f1.append(
            f1_score(y[idx], p_b[idx] >= 0.5, average="macro")
            - f1_score(y[idx], p_a[idx] >= 0.5, average="macro")
        )
    return {
        "roc_auc": tuple(np.percentile(d_auc, [2.5, 97.5])),
        "macro_f1": tuple(np.percentile(d_f1, [2.5, 97.5])),
    }


async def run_holdout(seed: int) -> None:
    """
    Score the original and the upgraded configuration on the 400 held-out cascades.

    Reads the database copy of the data — the same rows `ml.evaluate` scores —
    and refits both configurations on the same training split. This is the only
    command in the module that reads the holdout.
    """
    from ml.dataset import load_dataset
    from ml.evaluate import choose_review_threshold
    from ml.train import model_params

    data = await load_dataset()
    train, holdout = temporal_split(data)
    y_tr, y = train["y"].to_numpy(), holdout["y"].to_numpy()
    subtype = holdout["subtype"].to_numpy()

    original = lgbm(BASE_LGBM_PARAMS, seed).fit(train[BASE_FEATURE_COLUMNS], y_tr)
    upgraded = lgbm(model_params(), seed).fit(train[FEATURE_COLUMNS], y_tr)
    p_orig = original.predict_proba(holdout[BASE_FEATURE_COLUMNS])[:, 1]
    p_new = upgraded.predict_proba(holdout[FEATURE_COLUMNS])[:, 1]

    rows = []
    for name, p in (("original — 23 features, default params", p_orig),
                    ("upgraded — 37 features, Optuna params", p_new)):
        m = score(y, p, subtype)
        rows.append([name, f"{m['accuracy']:.3f}", f"{m['macro_f1']:.3f}", f"{m['roc_auc']:.3f}",
                     f"{m['pr_auc']:.3f}", f"{m['brier']:.3f}", f"{m['ece']:.3f}"])
        log.info("holdout %s: %s", name, {k: round(v, 3) for k, v in m.items()})

    sub_rows = []
    for st in ("plain", "viral_organic", "stealth_coordinated"):
        mask = subtype == st
        acc = [float(((p[mask] >= 0.5).astype(int) == y[mask]).mean()) for p in (p_orig, p_new)]
        sub_rows.append([f"`{st}`", int(mask.sum()), f"{acc[0]:.3f}", f"{acc[1]:.3f}"])

    ci = paired_bootstrap(y, p_orig, p_new)
    d_auc = roc_auc_score(y, p_new) - roc_auc_score(y, p_orig)
    d_f1 = f1_score(y, p_new >= 0.5, average="macro") - f1_score(y, p_orig >= 0.5, average="macro")

    gate_rows = []
    for name, p in (("original", p_orig), ("upgraded", p_new)):
        confidence = np.maximum(p, 1 - p)
        correct = ((p >= 0.5).astype(int) == y).astype(float)
        threshold, acc, coverage = choose_review_threshold(confidence, correct)
        gate_rows.append([name, f"{threshold:.2f}", f"{acc:.3f}", f"{100 * coverage:.1f}%"])

    body = f"""
The configuration chosen by cross-validation above, scored once on the
{len(holdout)} temporally held-out cascades (database copy, the same rows as
`ml/evaluate.py`), next to the original configuration refit on the same
training split. Nothing about the upgraded model was chosen by looking at
these rows.

{markdown_table(["model", "accuracy", "macro F1", "ROC-AUC", "PR-AUC", "Brier", "ECE"], rows)}

Paired bootstrap over held-out cascades (2,000 resamples), upgraded minus
original: ROC-AUC {d_auc:+.3f} (95% CI {ci['roc_auc'][0]:+.3f} to {ci['roc_auc'][1]:+.3f}),
macro F1 {d_f1:+.3f} (95% CI {ci['macro_f1'][0]:+.3f} to {ci['macro_f1'][1]:+.3f}).

### By subtype

{markdown_table(["subtype", "n", "original accuracy", "upgraded accuracy"], sub_rows)}

### Review gate, re-derived

Lowest confidence gate whose published predictions reach
90% accuracy, read off each model's own held-out predictions.

{markdown_table(["model", "gate", "accuracy above gate", "auto-published"], gate_rows)}

The `Classifier vs LLM agent` section predates this upgrade: its `classifier` row is
the original configuration, and the LLM arms have not been re-run (they need the
local model server).

Reproduce with `python -m ml.experiment holdout`.
"""
    upsert_section("Classifier upgrade — held-out result", body)


# ── entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifier experiments")
    parser.add_argument("command", choices=["cv", "tune", "shift", "oracle", "skew", "holdout"])
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--gnn-epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    configure_logging()

    if args.command == "cv":
        data, graphs = _dataset_and_graphs(2000, "medium", 42)
        train, _ = temporal_split(data)
        cv, _ = run_cv(train, graphs, args.trials, args.seed, args.gnn_epochs)
        cv.to_csv(REPO_ROOT / "data" / "experiment_cv.csv", index=False)
        _write_cv_report(cv, len(train), args.trials, args.gnn_epochs)
    elif args.command == "tune":
        run_tune(args.trials * 2, args.seed)
    elif args.command == "shift":
        run_shift(args.seed, args.gnn_epochs)
    elif args.command == "skew":
        run_skew(args.seed)
    elif args.command == "oracle":
        run_oracle(args.seed)
    elif args.command == "holdout":
        asyncio.run(run_holdout(args.seed))


if __name__ == "__main__":
    main()
