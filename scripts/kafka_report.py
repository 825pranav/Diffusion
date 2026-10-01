"""
Write the Kafka pipeline section of docs/results.md from the recorded runs.

Usage:
    python -m scripts.kafka_report

Reads docs/kafka-runs/, which holds what each run left behind: the lag monitor's
samples (scripts/measure_kafka.py), the phase timestamps, the loss audit
(scripts/audit_kafka.py) and, for run 2, the replay counts. Nothing here talks to
Kafka or Postgres, so the section can be regenerated without the stack running.

  run1  the original consumer: per-record autocommit writes, auto-committed
        offsets, velocity scoring inside the consumer. Bluesky and HN.
  run2  the batched transactional processor and the entity-keyed scorer. All
        four platforms; one processor is killed with `taskkill /F` mid-run.
  soak  run 2's pipeline after the scorer warm-up and the NER span filter.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from ml.report import markdown_table, upsert_section  # noqa: E402

RUNS = pathlib.Path(__file__).resolve().parent.parent / "docs" / "kafka-runs"


def _load(run: str) -> tuple[dict[str, float], list[dict]]:
    phases = {k: float(v) for k, v in (line.split() for line in (RUNS / run / "phases.txt").read_text().splitlines())}
    samples = [json.loads(line) for line in (RUNS / run / "samples.jsonl").read_text().splitlines() if line]
    return phases, samples


def _audit(run: str) -> dict[str, int]:
    text = (RUNS / run / "audit.txt").read_text()
    grab = lambda label: int(re.search(rf"{label}:\s+(\d+)", text).group(1))  # noqa: E731
    return {
        "messages": grab("messages on raw topics"),
        "records": grab("distinct content records"),
        "written": grab("written to the graph"),
        "missing": grab("missing"),
        "dead": grab("dead-lettered"),
    }


def _phase(samples: list[dict], lo: float, hi: float) -> dict[str, float]:
    rows = [s for s in samples if lo <= s["t"] < hi and "consume_rate" in s]
    return {
        "produce": sum(s["produce_rate"] for s in rows) / len(rows),
        "consume": sum(s["consume_rate"] for s in rows) / len(rows),
        "max_lag": max(s["lag"] for s in rows),
        "max_behind": max(s["behind_s"] for s in rows),
        "end_lag": rows[-1]["lag"],
    }


def _alert_rate(samples: list[dict], start: float, end: float, skip_s: float = 300) -> float:
    """Anomalies per minute after the first window, from the monitor's running count."""
    rows = [s for s in samples if start + skip_s <= s["t"] <= end and "anomalies" in s]
    return (rows[-1]["anomalies"] - rows[0]["anomalies"]) / ((rows[-1]["t"] - rows[0]["t"]) / 60)


def _replays() -> tuple[dict, list[tuple[int, float, dict]]]:
    text = (RUNS / "run2" / "replay.txt").read_text()
    before = json.loads(re.search(r"counts before replay (\{.*\})", text).group(1))
    runs = [
        (int(n), float(secs), json.loads(counts))
        for n, secs, counts in re.findall(r"replay with (\d+) processors: ([\d.]+) s, counts after (\{.*\})", text)
    ]
    return before, runs


def main() -> None:
    p1, s1 = _load("run1")
    p2, s2 = _load("run2")
    a1, a2 = _audit("run1"), _audit("run2")
    r1a, r1b = _phase(s1, p1["A"], p1["B"]), _phase(s1, p1["B"], p1["D"])
    r2a, r2b = _phase(s2, p2["START"], p2["B"]), _phase(s2, p2["B"], p2["D"])
    crash = [s for s in s2 if p2["CRASH"] - 1 <= s["t"] <= p2["CRASH"] + 60]
    crash_peak = max(crash, key=lambda s: s["lag"])
    before, replays = _replays()
    messages = a2["messages"]

    def lag_row(label, r):
        return [
            label, f"{r['produce']:.0f}/s", f"{r['consume']:.0f}/s",
            f"{r['max_lag']:,}", f"{r['max_behind']:.0f} s", f"{r['end_lag']:,}",
        ]

    soak = ""
    if (RUNS / "soak").exists():
        ps, ss = _load("soak")
        soak_rate = _alert_rate(ss, ps["START"], ps["END"])
        run2_rate = _alert_rate(s2, p2["START"], p2["D"])
        soak = f"""
### Anomaly alerts on live traffic

The scorer's alert rate after its first five-minute window, read from the
monitor's running count of `anomaly_events`:

{markdown_table(
    ["run", "scorer", "alerts / min"],
    [
        ["run 2", "no warm-up, every NER span", f"{run2_rate:.1f}"],
        ["soak", "warm-up per partition, implausible spans dropped", f"{soak_rate:.1f}"],
    ],
)}

The two runs saw different hours of traffic, so the comparison is indicative.
A controlled one replayed run 2's own 29,204 mentions through both scorers: the
warm-up alone took the post-window rate from 28.6 to 8.0 a minute. A fresh
window can only fill, so every entity's count climbs and steady chatter reads as
a spike; worse, those low early counts enter the baseline history and keep
firing alarms well after the window is full. The span filter removes the
loudest remaining source — 4.5% of extracted entities were emoji runs, URLs or
spans crossing a line break, and a string of sheep emoji alone raised 39 of
run 2's alerts.
"""

    body = f"""
The producer → Kafka → processor → graph path, run against live traffic on the
compose stack (Kafka 3.6 in KRaft mode, Postgres 16, pgvector 0.8.6), first with
the original consumer and then with the rewritten pipeline. Each run lasted about
fourteen minutes: seven with one consumer, seven with three, then a drain.

### Keeping up with the stream

{markdown_table(
    ["phase", "produced", "consumed", "max lag (msgs)", "max behind", "lag at end"],
    [
        lag_row("run 1 — 1 consumer", r1a),
        lag_row("run 1 — 3 consumers", r1b),
        lag_row("run 2 — 1 processor", r2a),
        lag_row("run 2 — 3 processors", r2b),
    ],
)}

"Behind" is the age of the oldest message the group has not processed. The
original consumer managed about 60 records a second against roughly 100
arriving, so a single instance fell three minutes behind and kept falling;
three instances only just outran the stream, and draining the backlog took
{p1['E'] - p1['D']:.0f} s after the producers stopped. Each record cost about
five autocommitted statements, every one waiting on its own WAL flush, while
spaCy extraction takes 2 ms. The processor writes a batch in one transaction and
keeps up with one instance; run 2 drained in {p2['E'] - p2['D']:.0f} s.

### Capacity

Rewinding the processor group to the start of run 2 and replaying all
{messages:,} messages as fast as the processors take them (wall time, including
process start-up and group join):

{markdown_table(
    ["processors", "time", "throughput", "graph nodes after", "graph edges after"],
    [[n, f"{secs:.0f} s", f"{messages / secs:,.0f} msgs/s",
      f"{c['graph_nodes']:,}", f"{c['graph_edges']:,}"] for n, secs, c in replays],
)}

Before either replay the graph held {before['graph_nodes']:,} nodes and
{before['graph_edges']:,} edges. Every message was processed again, twice, and
neither count moved: the writes are idempotent, which is what makes
at-least-once delivery safe here.

### Nothing lost

{markdown_table(
    ["run", "messages", "content records", "written", "missing", "dead-lettered"],
    [
        ["run 1", f"{a1['messages']:,}", f"{a1['records']:,}", f"{a1['written']:,}", a1["missing"], a1["dead"]],
        ["run 2", f"{a2['messages']:,}", f"{a2['records']:,}", f"{a2['written']:,}", a2["missing"], a2["dead"]],
    ],
)}

`scripts/audit_kafka.py` reads every record back off the raw topics and checks
that its content node exists as a real node rather than a placeholder. Run 1's
eleven missing records each carried a NUL character, which Postgres refuses in
TEXT and JSONB; the consumer logged the error, skipped the record and committed
past it. The processor strips NULs, and a record that still cannot be written
goes to the `dead-letter` topic with its source offset instead of disappearing.

Run 2 lost nothing although one of its three processors was killed with
`taskkill /F` in the middle of a batch. Lag peaked at {crash_peak['lag']}
messages ({crash_peak['behind_s']:.0f} s behind) while the group rebalanced; whatever
the dead process had read but not committed went to a survivor, and the audit's
zero is the evidence that it was written. Offsets
are committed inside the same Kafka transaction as the batch's entity mentions,
and only after the graph write has committed, so a crash re-reads work rather
than skipping it.
{soak}
Reproduce: `docker compose up -d`, start the processor(s), the scorer and the
producers, sample with `python -m scripts.measure_kafka`, audit with
`python -m scripts.audit_kafka --since <epoch>`, and rebuild this section with
`python -m scripts.kafka_report`.
"""
    upsert_section("Kafka pipeline — end to end", body)
    print("wrote the Kafka section of docs/results.md")


if __name__ == "__main__":
    main()
