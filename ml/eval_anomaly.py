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

BASELINES = ("zscore", "robust", "ewma")

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
    ticks = np.arange(start, end + tick_seconds, tick_seconds, dtype=float)

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


async def main_async(labels_path: pathlib.Path, n_topics: int, tick: int, threshold: float) -> None:
    labels = pd.read_csv(labels_path, parse_dates=["t0"])
    labels["epoch"] = to_epoch_seconds(labels["t0"])

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
    parser.add_argument("--tick", type=int, default=DEFAULT_TICK_SECONDS)
    parser.add_argument("--threshold", type=float, default=2.5)
    args = parser.parse_args()

    configure_logging()
    asyncio.run(main_async(args.labels, args.topics, args.tick, args.threshold))


if __name__ == "__main__":
    main()
