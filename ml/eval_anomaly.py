"""
Compare anomaly-detection baselines on replayed simulator traffic.

Usage:
    python -m ml.eval_anomaly
    python -m ml.eval_anomaly --topics 50 --tick 300

The detector's job is to decide *when to wake the agent*.  Judging it means
answering two questions:

  1. Does it fire for real campaigns, and how long after they start?
  2. How much of what it fires is noise?

Every topic's mention stream is replayed through the real VelocityScorer and
AnomalyDetector — no reimplementation — at a fixed tick, once per baseline.

A detection is *explained* if it lands shortly after some cascade actually began
on that topic, whether organic or coordinated.  Anything else fired on baseline
noise and is counted as a false positive.  Organic bursts deliberately do not
count as false positives: genuine virality is real anomalous velocity, and
telling it apart from a campaign is the classifier's job, not the detector's.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import pathlib

import asyncpg
import numpy as np
import pandas as pd

from config import DB_URL, configure_logging
from ml.dataset import DEFAULT_LABELS
from ml.features import to_epoch_seconds
from ml.report import markdown_table, upsert_section
from processing.anomaly_detector import AnomalyDetector
from processing.velocity_scorer import VelocityScorer

# Seconds between successive velocity samples during replay.
DEFAULT_TICK_SECONDS = 300
# Velocity window; matches the production VelocityScorer default.
WINDOW_SECONDS = 300
# A detection this soon after a cascade onset is attributed to that cascade.
ATTRIBUTION_WINDOW_SECONDS = 3600
# Topics replayed by default. Each carries ~10 cascades, which is plenty.
DEFAULT_TOPICS = 50

BASELINES = ("zscore", "robust", "ewma", "poisson", "negbin", "cusum")

# Threshold grids for the sweep. CUSUM's threshold is on the accumulated sum,
# so it has its own scale.
SWEEP_GRIDS = {
    "zscore": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0],
    "robust": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0],
    "ewma": [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0],
    "poisson": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
    "negbin": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
    "cusum": [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0],
}
# Operating point of the previous default detector, the reference every other
# baseline is matched against.
REFERENCE = ("robust", 2.5)

log = logging.getLogger(__name__)


async def load_topic_streams(
    conn: asyncpg.Connection, topic_ids: list[str]
) -> dict[str, np.ndarray]:
    """Event timestamps (epoch seconds) per topic, from its mention edges."""
    rows = await conn.fetch(
        """
        SELECT target_id AS topic_id, ts
        FROM graph_edges
        WHERE edge_type = 'mentions'
          AND target_id = ANY($1::text[])
        ORDER BY target_id, ts
        """,
        topic_ids,
    )
    frame = pd.DataFrame(rows, columns=["topic_id", "ts"])
    if frame.empty:
        return {}
    frame["epoch"] = to_epoch_seconds(frame["ts"])
    return {
        topic: group["epoch"].to_numpy(dtype=float)
        for topic, group in frame.groupby("topic_id", sort=False)
    }


def replay_topic(
    entity_id: str,
    events: np.ndarray,
    baseline: str,
    tick_seconds: int,
    threshold: float,
) -> list[float]:
    """
    Replay one topic's stream and return the epoch time of every detection.

    The scorer and detector are the production classes; only the clock is
    simulated.
    """
    if events.size < 2:
        return []

    scorer = VelocityScorer(window_seconds=WINDOW_SECONDS)
    detector = AnomalyDetector(baseline=baseline, threshold=threshold)

    start, end = float(events[0]), float(events[-1])
    if tick_seconds > 0:
        ticks = np.arange(start, end + tick_seconds, tick_seconds, dtype=float)
    else:
        # Per-event scoring, as processing.consumer does: one sample on every
        # arriving event rather than on a clock.
        ticks = events

    detections: list[float] = []
    cursor = 0
    for tick in ticks:
        while cursor < events.size and events[cursor] <= tick:
            scorer.record(entity_id, float(events[cursor]))
            cursor += 1
        score = scorer.score(entity_id, ts=float(tick))
        if detector.detect(score, platform="sim") is not None:
            detections.append(float(tick))
    return detections


def score_baseline(
    streams: dict[str, np.ndarray],
    onsets: pd.DataFrame,
    baseline: str,
    tick_seconds: int,
    threshold: float,
) -> dict[str, float]:
    """Replay every topic and summarise how well this baseline behaved."""
    onsets_by_topic = {t: g for t, g in onsets.groupby("topic_id", sort=False)}

    total_detections = 0
    explained = 0
    replay_hours = 0.0
    delays: list[float] = []
    detected_coordinated = 0
    detected_organic = 0
    n_coordinated = 0
    n_organic = 0

    for topic, events in streams.items():
        detections = replay_topic(topic, events, baseline, tick_seconds, threshold)
        total_detections += len(detections)
        replay_hours += (float(events[-1]) - float(events[0])) / 3600.0

        topic_onsets = onsets_by_topic.get(topic)
        if topic_onsets is None:
            continue

        n_coordinated += int((topic_onsets["label"] == "coordinated").sum())
        n_organic += int((topic_onsets["label"] == "organic").sum())

        detection_times = np.array(detections, dtype=float)
        for row in topic_onsets.itertuples():
            if detection_times.size == 0:
                continue
            after = detection_times[
                (detection_times >= row.epoch)
                & (detection_times <= row.epoch + ATTRIBUTION_WINDOW_SECONDS)
            ]
            if after.size:
                if row.label == "coordinated":
                    detected_coordinated += 1
                    delays.append(float(after.min() - row.epoch))
                else:
                    detected_organic += 1

        # A detection is explained if any cascade began shortly before it.
        if detection_times.size:
            starts = topic_onsets["epoch"].to_numpy(dtype=float)
            for t in detection_times:
                if np.any((starts <= t) & (t <= starts + ATTRIBUTION_WINDOW_SECONDS)):
                    explained += 1

    false_positives = total_detections - explained
    return {
        "detections": total_detections,
        "coordinated_recall": detected_coordinated / n_coordinated if n_coordinated else 0.0,
        "organic_recall": detected_organic / n_organic if n_organic else 0.0,
        "false_positive_rate": false_positives / total_detections if total_detections else 0.0,
        "fp_per_topic_day": 24.0 * false_positives / replay_hours if replay_hours else 0.0,
        "median_delay_min": float(np.median(delays)) / 60.0 if delays else float("nan"),
    }


def pareto_front(points: list[tuple[float, float]]) -> list[bool]:
    """
    Which (false alarms, recall) points no other point beats on both axes.

    A point is dominated when another has no more false alarms and at least the
    recall, and is strictly better on one of the two.
    """
    flags = []
    for i, (fp_i, r_i) in enumerate(points):
        dominated = any(
            fp_j <= fp_i and r_j >= r_i and (fp_j < fp_i or r_j > r_i)
            for j, (fp_j, r_j) in enumerate(points)
            if j != i
        )
        flags.append(not dominated)
    return flags


def choose_threshold(sweep: list[dict], min_recall: float) -> dict | None:
    """The fewest-false-alarm threshold whose coordinated recall reaches `min_recall`."""
    ok = [r for r in sweep if r["coordinated_recall"] >= min_recall]
    return min(ok, key=lambda r: (r["fp_per_topic_day"], -r["coordinated_recall"])) if ok else None


def _regime(tick: int) -> str:
    """Section suffix naming how the stream was sampled."""
    return "per event, as deployed" if tick <= 0 else f"{tick}s clock ticks"


def _fmt_delay(r: dict) -> str:
    return "n/a" if np.isnan(r["median_delay_min"]) else f"{r['median_delay_min']:.1f}"


def run_sweep(
    streams: dict[str, np.ndarray],
    onsets: pd.DataFrame,
    tune_topics: list[str],
    test_topics: list[str],
    tick: int,
) -> None:
    """
    Sweep thresholds on one set of topics, then report on a disjoint set.

    Picking thresholds and reporting results on the same topics would flatter
    whichever detector has the most knobs, so the choice is made on the tuning
    topics and every number in the headline table comes from topics the choice
    never saw.
    """
    tune = {t: streams[t] for t in tune_topics if t in streams}
    test = {t: streams[t] for t in test_topics if t in streams}

    sweeps: dict[str, list[dict]] = {}
    for baseline, grid in SWEEP_GRIDS.items():
        sweeps[baseline] = []
        for threshold in grid:
            r = score_baseline(tune, onsets, baseline, tick, threshold)
            r.update(baseline=baseline, threshold=threshold)
            sweeps[baseline].append(r)

    ref = next(r for r in sweeps[REFERENCE[0]] if r["threshold"] == REFERENCE[1])
    target = ref["coordinated_recall"]

    chosen: dict[str, dict] = {}
    for baseline, sweep in sweeps.items():
        pick = choose_threshold(sweep, target)
        if pick is not None:
            chosen[baseline] = pick
        log.info("%-7s tuned threshold: %s", baseline, pick and pick["threshold"])

    all_points = [r for s in sweeps.values() for r in s]
    front = pareto_front([(r["fp_per_topic_day"], r["coordinated_recall"]) for r in all_points])
    sweep_rows = [
        [
            f"`{r['baseline']}`", f"{r['threshold']:g}",
            f"{r['coordinated_recall']:.3f}", f"{r['fp_per_topic_day']:.2f}",
            _fmt_delay(r), "yes" if f else "",
        ]
        for r, f in zip(all_points, front, strict=True)
    ]

    runs = [(f"`{REFERENCE[0]}` @ {REFERENCE[1]:g} (previous default)", REFERENCE)]
    runs += [
        (f"`{b}` @ {c['threshold']:g} (tuned)", (b, c["threshold"]))
        for b, c in chosen.items()
        if (b, c["threshold"]) != REFERENCE
    ]
    test_rows = []
    for label, (baseline, threshold) in runs:
        r = score_baseline(test, onsets, baseline, tick, threshold)
        log.info("test %-28s recall=%.3f fp/topic-day=%.2f", label, r["coordinated_recall"],
                 r["fp_per_topic_day"])
        test_rows.append(
            [
                label, f"{r['detections']:,}", f"{r['coordinated_recall']:.3f}",
                f"{r['organic_recall']:.3f}", f"{r['false_positive_rate']:.3f}",
                f"{r['fp_per_topic_day']:.2f}", _fmt_delay(r),
            ]
        )

    missing = [b for b in sweeps if b not in chosen]
    missing_note = (
        "\n\nNo threshold in the grid reaches that recall for "
        + ", ".join(f"`{b}`" for b in missing)
        + "; left out of the table below."
        if missing else ""
    )
    sampling = (
        "Each topic is scored on every arriving event, exactly as "
        "`processing.consumer` does in production — so the history a baseline "
        "is built from is a history of event-time samples, not of a clock."
        if tick <= 0 else
        f"Each topic is scored on a {tick}s clock, like the fixed-threshold table."
    )
    body = f"""
{sampling}

The fixed-threshold table fixes every detector at 2.5, which is not a fair
fight: each baseline's z-scale means something different. Here each one is
swept, its threshold is chosen on **{len(tune)} tuning topics** (the ones in the
fixed-threshold replay table), and the chosen operating points are then scored on
**{len(test)} different topics** that none of the choices saw.

Selection rule: the fewest false alarms per topic-day among thresholds whose
coordinated recall on the tuning topics reaches the previous default's
(`{REFERENCE[0]}` at {REFERENCE[1]:g}: {target:.3f}). That holds recall fixed and
asks which detector pays the least noise for it.{missing_note}

### Held-out topics, at the tuned thresholds

{markdown_table(
    ["detector", "detections", "coord. recall", "organic recall", "FP rate",
     "FP / topic-day", "median delay (min)"],
    test_rows,
)}

### Tuning-topic sweep

`pareto` marks points that no other detector/threshold beats on both false
alarms and coordinated recall.

{markdown_table(
    ["baseline", "threshold", "coord. recall", "FP / topic-day", "median delay (min)", "pareto"],
    sweep_rows,
)}

Reproduce with `python -m ml.eval_anomaly --sweep --tick {tick}`.
"""
    upsert_section(f"Anomaly baselines — tuned on held-out topics ({_regime(tick)})", body)
    return {b: c["threshold"] for b, c in chosen.items()}


def offline_streams(
    clump_mean: float, seed: int = 42, clump_seed: int = 1
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """
    Rebuild the replay streams in memory, optionally with clustered chatter.

    With `clump_mean=0` this regenerates exactly the mention streams the
    database holds (same rng sequence as `simulation.generate`). A positive
    value gives every background mention a Poisson number of echoes a minute or
    two later — ordinary conversation that clumps without being a cascade.

    The simulator's background is a pure Poisson process, which is precisely
    the assumption the `poisson` detector makes; a detector evaluated only in
    the world it assumes is being flattered. Clumped chatter is overdispersed,
    which is what real mention counts look like.
    """
    from simulation.background import generate_background
    from simulation.cascade import CascadeSimulator

    rng = np.random.default_rng(seed)
    sim = CascadeSimulator(rng=rng, run_id=f"m{seed}", difficulty="medium")
    labels = ["organic"] * 1000 + ["coordinated"] * 1000
    rng.shuffle(labels)
    cascades = [sim.generate(label) for label in labels]
    topic_ids = sorted({c.topic_id for c in cascades})
    _, bg_edges = generate_background(
        rng=rng, run_id=f"m{seed}", topic_ids=topic_ids, window_days=sim.window_days,
        end_time=max(c.t0 for c in cascades),
    )

    events: dict[str, list[float]] = {t: [] for t in topic_ids}
    for c in cascades:
        for e in c.edges:
            if e.edge_type == "mentions":
                events[e.target_id].append(e.ts.timestamp())
    clump_rng = np.random.default_rng(clump_seed)
    for e in bg_edges:
        t = e.ts.timestamp()
        events[e.target_id].append(t)
        if clump_mean > 0:
            for _ in range(int(clump_rng.poisson(clump_mean))):
                events[e.target_id].append(t + float(clump_rng.exponential(90.0)))

    streams = {t: np.sort(np.array(v, dtype=float)) for t, v in events.items() if v}
    onsets = pd.DataFrame(
        {"topic_id": [c.topic_id for c in cascades], "label": [c.label for c in cascades],
         "epoch": [c.t0.timestamp() for c in cascades]}
    )
    return streams, onsets


STRESS_CLUMPS = (0.0, 1.0, 3.0)


def run_stress(chosen: dict[str, float], test_topics: list[str], tick: int) -> None:
    """Score the tuned operating points, unchanged, on clumpier chatter."""
    runs = [(f"`{REFERENCE[0]}` @ {REFERENCE[1]:g} (previous default)", REFERENCE)]
    runs += [(f"`{b}` @ {t:g}", (b, t)) for b, t in chosen.items() if (b, t) != REFERENCE]
    rows = []
    for clump in STRESS_CLUMPS:
        streams, onsets = offline_streams(clump)
        test = {t: streams[t] for t in test_topics if t in streams}
        for label, (baseline, threshold) in runs:
            r = score_baseline(test, onsets, baseline, tick, threshold)
            log.info("stress clump=%.0f %-24s recall=%.3f fp/topic-day=%.2f",
                     clump, label, r["coordinated_recall"], r["fp_per_topic_day"])
            rows.append([
                f"{clump:g}", label, f"{r['coordinated_recall']:.3f}",
                f"{r['organic_recall']:.3f}", f"{r['fp_per_topic_day']:.2f}", _fmt_delay(r),
            ])
    body = f"""
The simulator's background chatter is a pure (diurnal) Poisson process — which
is exactly what the `poisson` detector assumes. A detector scored only in the
world it assumes is being flattered, so here the thresholds tuned above are
kept **unchanged** and re-scored on the same {len(test_topics)} held-out topics
with overdispersed chatter: every background mention gets a Poisson(`clumps`)
number of echoes about a minute and a half later. That is ordinary
conversation that clumps without being a cascade, and it is what real mention
counts look like. `clumps = 0` regenerates the database streams in memory and
reproduces the matching held-out table (up to float rounding).

{markdown_table(
    ["clumps", "detector", "coord. recall", "organic recall", "FP / topic-day",
     "median delay (min)"],
    rows,
)}

Reproduce with `python -m ml.eval_anomaly --sweep --tick {tick}`.
"""
    upsert_section(f"Anomaly baselines — clumped-chatter stress test ({_regime(tick)})", body)


def _write_report(results: dict[str, dict[str, float]], tick: int, n_topics: int) -> None:
    rows = [
        [
            f"`{name}`",
            f"{r['detections']:,}",
            f"{r['coordinated_recall']:.3f}",
            f"{r['organic_recall']:.3f}",
            f"{r['false_positive_rate']:.3f}",
            f"{r['fp_per_topic_day']:.2f}",
            "n/a" if np.isnan(r["median_delay_min"]) else f"{r['median_delay_min']:.1f}",
        ]
        for name, r in results.items()
    ]
    body = f"""
Every topic's mention stream replayed through the production `VelocityScorer`
and `AnomalyDetector` at a {tick}s tick ({n_topics} topics). A detection is
*explained* if a cascade began on that topic within
{ATTRIBUTION_WINDOW_SECONDS // 60} minutes before it; unexplained detections
fired on baseline noise and are the false positives.

Organic bursts are not counted as false positives. Genuine virality is real
anomalous velocity — separating it from a campaign is the classifier's job, and
a detector that stayed silent for it would starve the agent of the harder half
of its cases.

{markdown_table(
    ["baseline", "detections", "coord. recall", "organic recall",
     "FP rate", "FP / topic-day", "median delay (min)"],
    rows,
)}

Reproduce with `python -m ml.eval_anomaly`.
"""
    upsert_section("Anomaly baselines — replay comparison", body)


async def main_async(
    labels_path: pathlib.Path, n_topics: int, tick: int, threshold: float, sweep: bool = False
) -> None:
    labels = pd.read_csv(labels_path, parse_dates=["t0"])
    labels["epoch"] = to_epoch_seconds(labels["t0"])

    if sweep:
        # Tune on the topics the fixed-threshold table reports, test on the rest.
        all_topics = sorted(labels["topic_id"].unique())
        tune_topics, test_topics = all_topics[:n_topics], all_topics[n_topics:]
        conn = await asyncpg.connect(DB_URL)
        try:
            streams = await load_topic_streams(conn, all_topics)
        finally:
            await conn.close()
        chosen = run_sweep(
            streams, labels[["topic_id", "label", "epoch"]], tune_topics, test_topics, tick
        )
        run_stress(chosen, test_topics, tick)
        log.info("wrote results to docs/results.md")
        return

    topic_ids = sorted(labels["topic_id"].unique())[:n_topics]
    onsets = labels[labels["topic_id"].isin(topic_ids)][["topic_id", "label", "epoch"]]
    log.info("replaying %d topics carrying %d cascades", len(topic_ids), len(onsets))

    conn = await asyncpg.connect(DB_URL)
    try:
        streams = await load_topic_streams(conn, topic_ids)
    finally:
        await conn.close()
    log.info("loaded %d topic streams", len(streams))

    results: dict[str, dict[str, float]] = {}
    for baseline in BASELINES:
        results[baseline] = score_baseline(streams, onsets, baseline, tick, threshold)
        r = results[baseline]
        log.info(
            "%-7s detections=%-6d coord_recall=%.3f fp_rate=%.3f fp/topic-day=%.2f delay=%.1fmin",
            baseline, r["detections"], r["coordinated_recall"],
            r["false_positive_rate"], r["fp_per_topic_day"], r["median_delay_min"],
        )

    _write_report(results, tick, len(streams))
    log.info("wrote results to docs/results.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare anomaly-detection baselines")
    parser.add_argument("--labels", type=pathlib.Path, default=DEFAULT_LABELS)
    parser.add_argument("--topics", type=int, default=DEFAULT_TOPICS)
    parser.add_argument(
        "--tick", type=int, default=DEFAULT_TICK_SECONDS,
        help="seconds between samples; 0 scores on every event, as the consumer does",
    )
    parser.add_argument("--threshold", type=float, default=2.5)
    parser.add_argument(
        "--sweep", action="store_true",
        help="sweep thresholds on the first --topics topics, report on the remainder",
    )
    args = parser.parse_args()

    configure_logging()
    asyncio.run(main_async(args.labels, args.topics, args.tick, args.threshold, args.sweep))


if __name__ == "__main__":
    main()
