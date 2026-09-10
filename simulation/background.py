"""
Background chatter for the simulated graph.

Cascades alone make an unrealistically quiet world: every event belongs to a
cascade, so any velocity spike is by definition explainable and the anomaly
detector's false-positive rate is pinned at zero by construction rather than by
merit.

This adds low-rate, topic-level mentions with a diurnal cycle — the ordinary
traffic a topic attracts when nothing notable is happening.  The daily rhythm is
the point: it makes the baseline *non-stationary*, which is precisely the
condition under which a mean/standard-deviation z-score misbehaves and a robust
or drift-tracking baseline earns its place.

Background posts carry a `mentions` edge and no `reshare` edge, so they raise a
topic's measured velocity without belonging to any cascade.  Cascade feature
extraction traverses reshare edges only and never sees them.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np

from simulation.cascade import PLATFORMS, SimEdge, SimNode

BACKGROUND_CONTENT_FMT = "sim:{run}:bg:{n}"

# Mentions per topic per day, before the diurnal factor. High enough that a
# five-minute velocity window is usually non-empty: at a sparse rate almost every
# window reads zero, the baseline's spread collapses, and the detector abstains
# instead of ever producing a false positive.
DEFAULT_RATE_PER_DAY = 240.0
# Peak-to-mean swing of the daily cycle; 0.6 means quiet nights, busy evenings.
DIURNAL_AMPLITUDE = 0.6


def generate_background(
    rng: np.random.Generator,
    run_id: str,
    topic_ids: list[str],
    window_days: int,
    end_time: datetime,
    rate_per_day: float = DEFAULT_RATE_PER_DAY,
    diurnal_amplitude: float = DIURNAL_AMPLITUDE,
) -> tuple[list[SimNode], list[SimEdge]]:
    """
    Emit background mentions for each topic across the simulation window.

    Events are drawn by thinning: a homogeneous Poisson process at the peak rate,
    then each candidate kept with probability rate(t)/peak_rate.  That yields an
    exact inhomogeneous Poisson process without inverting the intensity.
    """
    nodes: list[SimNode] = []
    edges: list[SimEdge] = []
    start_time = end_time - timedelta(days=window_days)
    window_seconds = window_days * 86400
    counter = 0

    for topic_id in topic_ids:
        # Each topic peaks at its own hour, so the whole corpus does not pulse in
        # lockstep and a per-entity baseline has something specific to learn.
        phase = float(rng.uniform(0, 2 * np.pi))
        peak_rate = rate_per_day * (1.0 + diurnal_amplitude)
        n_candidates = int(rng.poisson(peak_rate * window_days))
        if n_candidates <= 0:
            continue

        offsets = np.sort(rng.uniform(0, window_seconds, size=n_candidates))
        intensity = 1.0 + diurnal_amplitude * np.sin(
            2 * np.pi * offsets / 86400.0 + phase
        )
        keep = rng.random(n_candidates) < (intensity / (1.0 + diurnal_amplitude))

        for offset in offsets[keep]:
            counter += 1
            ts = start_time + timedelta(seconds=float(offset))
            content_id = BACKGROUND_CONTENT_FMT.format(run=run_id, n=counter)
            platform = str(rng.choice(PLATFORMS))
            nodes.append(
                SimNode(
                    id=content_id,
                    type="content",
                    platform=platform,
                    label=f"Background mention {counter}",
                    metadata={"synthetic": True, "background": True},
                )
            )
            edges.append(SimEdge(content_id, topic_id, "mentions", platform, ts))

    return nodes, edges
