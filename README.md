# Diffusion
**Distributed Trend Propagation Engine**

*It's not about what's trending. It's about how and why it spread.*

---

[![CI](https://github.com/825pranav/Diffusion/actions/workflows/ci.yml/badge.svg)](https://github.com/825pranav/Diffusion/actions/workflows/ci.yml)

**What it found**

| | |
|---|---|
| Coordinated-vs-organic classifier | macro F1 **0.910**, ROC-AUC **0.960** on 400 held-out cascades (up from 0.889 / 0.938; ΔAUC 95% CI +0.007 to +0.039) |
| Against an LLM agent | with its tools actually working, a local 7B (`qwen2.5:7b`) reasoning over graph structure alone got 13 and 9 of 24 held-out cascades right — chance; given the classifier as a tool it got 17 of 21. Earlier LLM figures were measured with an agent that never called a tool and are withdrawn |
| Against a graph neural network | a GIN over the raw reshare trees trails LightGBM on the same information (CV ROC-AUC 0.920 vs 0.948), though it holds up best on the two hardest simulator shifts (0.61 and 0.74 vs 0.55 and 0.73) |
| Robustness | scored on 2,000 cascades from each of nine perturbed simulators: ROC-AUC holds at 0.92–0.95 under single-parameter shifts, drops to 0.73 under a combined camouflage shift, and to near chance (0.55) when every cascade is a confusable subtype |
| Anomaly detection | per-event CUSUM over a Poisson count baseline: **0.08 false alarms per topic-day vs 1.35** for median/MAD at higher recall (0.846 vs 0.832), on 150 topics its threshold was not chosen on |
| Kafka pipeline | measured end to end on live traffic: one processor keeps up with the firehose (~100 msgs/s, ≤3 s behind) where the original consumer fell 173 s behind; **zero of 83,664 records lost** with a processor killed mid-batch; replay capacity 577 msgs/s on one processor, 1,276 on three |
| Anomaly alerts on live traffic | **28.1 → 3.9 a minute** after a per-partition warm-up and dropping junk NER spans |
| Filtered vector search | when the HNSW index is walked, a partial index returns all 5 requested rows at 0.978 recall where a table-wide index returns under 3; pgvector 0.8's iterative scan fills the count but not the recall (0.320) |
| Learned deferral | **negative result** — ties confidence gating, does not beat it |

Full tables, and the scripts that regenerate every figure, in [`docs/results.md`](docs/results.md).

---

## Overview

Diffusion ingests live signals from Bluesky, Mastodon, Hacker News, and GitHub, models how information spreads as a directed propagation graph, and deploys an autonomous LLM agent that activates only when anomalous spread patterns are detected. The agent investigates the pattern, scores the cascade with a trained classifier, retrieves semantically similar historical cases via vector search, and produces a structured **case file** classifying whether a trend spread organically or was coordinated.

Two architectural decisions shape everything else:

**The agent does not poll.** It sleeps until a statistically significant spike in propagation velocity triggers it. Everything downstream is a reaction to events, not a scheduled job. A backlog sweep sits behind that so a spike raised while the agent is down is still investigated rather than lost — Postgres `NOTIFY` is fire-and-forget and drops notifications with no live listener.

**The verdict is measured, not asserted.** Real social data carries no ground-truth label for "organic vs coordinated", so a simulator generates labeled cascades and a LightGBM classifier is trained and evaluated against them. The agent calls that classifier as a tool. Every number in [`docs/results.md`](docs/results.md) is produced by a script in `ml/`.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Ingestion Layer                    │
│  Bluesky · Mastodon · HN Firebase · GitHub Events   │
└────────────────────────┬────────────────────────────┘
                         │ idempotent producers, keyed by record id
                         ▼
┌─────────────────────────────────────────────────────┐
│                     Kafka                           │
│   bluesky-raw · mastodon-raw · hn-raw · gh-raw      │
└────────────────────────┬────────────────────────────┘
                         │ consumer group, batches of up to 500
                         ▼
┌─────────────────────────────────────────────────────┐
│               Processor  (×1–3)                     │
│   Deduplication → Entity Extraction (spaCy NER)     │
│   one Postgres transaction per batch, then one      │
│   Kafka transaction: mentions + input offsets       │──► dead-letter
└──────────┬──────────────────────────┬───────────────┘
           │ graph writes             │ entity-mentions, keyed by entity
           │                          ▼
           │             ┌──────────────────────────────┐
           │             │  Scorer                      │
           │             │  event-time velocity → CUSUM │
           │             └──────────────┬───────────────┘
           ▼                            ▼ anomaly events
┌─────────────────────────────────────────────────────┐
│              PostgreSQL + pgvector                  │
│   Propagation graph edges                           │
│   Historical trend embeddings                       │
│   Case file store                                   │
└───────────┬─────────────────────────┬───────────────┘
            │ LISTEN/NOTIFY           │ backlog sweep
            │ (fast path)             │ (durability)
            ▼                         ▼
┌─────────────────────────────────────────────────────┐
│      Tool-calling agent (Groq or local Ollama)      │
│   get_entity_activity     (who drove a topic)       │
│   get_propagation_path                              │
│   search_similar_trends (pgvector)                  │
│   classify_virality        (graph heuristics)       │
│   classify_virality_model  (trained LightGBM)       │
│   submit_verdict → evidence check → confidence gate │
│   → case file                                       │
│   Ragas evaluation on every output                  │
└────────────────────────┬────────────────────────────┘
                         │ REST · WebSocket · SSE
                         ▼
┌─────────────────────────────────────────────────────┐
│                    FastAPI                          │
│   Trend + case file endpoints                       │
│   Agent thought stream (SSE)                        │
│   Live graph deltas (WebSocket)                     │
└─────────────────────────────────────────────────────┘
```

> **What has actually run.** All of it, on the compose stack: four live producers → Kafka → processor → Postgres, the entity-keyed scorer behind it, the agent, embeddings and the API. Running Kafka for the first time found the compose healthcheck could never pass, the consumer silently dropping records that carried a NUL byte, and velocity scoring that went wrong as soon as a second consumer started — see [Kafka pipeline — end to end](docs/results.md#kafka-pipeline--end-to-end).

Offline, feeding the classifier the agent calls:

```
simulation/  labeled cascades + diurnal background traffic
     │
     ▼
ml/features   one recursive CTE → per-cascade structure, timing, author reuse
     │
     ▼
ml/train      LightGBM, temporal split, 5-fold CV → models/
     │
     ▼
ml/evaluate   classifier vs LLM vs hybrid, calibration, review gate
```

---

## Tech Stack

| Technology | Role |
|---|---|
| **Kafka** | Event bus: raw topics per platform, an entity-keyed `entity-mentions` topic for the scorer, and a dead-letter topic; offsets commit transactionally with the mentions |
| **PostgreSQL** | Source of truth — propagation graph edges, case files, metadata |
| **pgvector** | Vector search — semantic retrieval of historically similar trends |
| **LightGBM** | Cascade classifier — the agent's strongest evidence |
| **LlamaIndex** | Tool definitions; the agent drives native tool calls through the OpenAI-compatible API that Groq and Ollama both serve |
| **Groq** | LLM inference (primary), when a key serves the configured model |
| **Ollama** | LLM inference (fallback) — local, zero cost, and the embedding backend (`nomic-embed-text`) |
| **Ragas** | Retrieval and grounding scores on every case file — wired in but not yet installed, so scores currently record zeros |
| **FastAPI** | Backend — REST, WebSocket, and SSE endpoints |
| **Docker** | Kafka and Postgres via `docker compose up` |

---

## Key Design Decisions

**Event-driven, with a durable floor.**
The agent wakes when the anomaly detector sees a spike, so inference cost is zero during quiet periods. `NOTIFY` alone would silently drop anomalies raised while the agent was down, restarting, or mid-failure, so a periodic sweep re-drives anything still marked uninvestigated. Both paths feed one queue, de-duplicated by anomaly id.

**A count-aware anomaly baseline, tuned where it runs.**
Mean and standard deviation are both dragged upward by the spikes they are meant to detect, so a sustained campaign progressively hides itself; median and scaled MAD fixed that, and at a shared threshold of 2.5 cut false alarms 3.4× against the plain z-score. But a quiet topic's velocity is a small integer count, where the MAD is usually exactly zero and the robust baseline abstains. The default is now a one-sided CUSUM over Poisson tail probabilities: the window count is scored against a trimmed-mean Poisson rate, and modest sustained excess accumulates instead of waiting for one large sample. Thresholds are chosen per baseline on 50 tuning topics and reported on 150 others, in the per-event sampling the consumer actually uses — the regime matters: Poisson at its clock-tuned threshold of 5 catches 79% of campaigns on a 5-minute clock and 59% when scored per event.

**Graph modeling in PostgreSQL.**
Propagation relationships are stored as directed edges. Degree, spread depth, and cascade size are computed in SQL — no dedicated graph DB. Feature extraction expands every cascade's reshare tree in a single recursive CTE rather than one query per cascade.

**pgvector over a dedicated vector DB.**
Historical trend embeddings live in the same Postgres instance as relational data, so similarity search can be joined directly with graph queries.

**Provider-agnostic LLM inference.**
Both backends are driven through the same OpenAI-compatible tool-calling API, so `LLM_BACKEND` picks one without code changes. On `auto`, Groq is preferred and probed once at startup — the key must work *and* serve the configured model — because a revoked key or a decommissioned model name would otherwise fail every investigation while a working local model sat idle. Groq's free tier allows about one investigation a minute, so for live volume the local model is the one that keeps up.

**Evaluation as a first-class concern.**
The classifier is scored on a temporally held-out split against the LLM agent alone and the two combined. The human-review threshold is read off the reliability curve rather than guessed. Ragas scores retrieval relevance and grounding on each case file — note that these measure the *retrieval*, not confidence calibration, which comes from `ml/evaluate.py`. `ragas` and `datasets` are not in either requirements file yet, so the evaluator currently falls back to zero scores.

---

## The Case File

Every anomaly the agent investigates produces a structured case file:

```json
{
  "trend": "XYZ GitHub Repository",
  "platform_origin": "github",
  "detected_at": "2026-04-11T14:32:00Z",
  "classification": "coordinated_amplification",
  "confidence": 0.84,
  "signals": [
    "Classifier p(coordinated) = 0.91, driven by root fan-out and author reuse",
    "87% author overlap with earlier cascades on this topic",
    "Cross-platform jump to HN within 6 minutes"
  ],
  "similar_past_cases": [
    { "trend": "OSS library X", "similarity": 0.91 }
  ],
  "ragas_scores": {
    "retrieval_relevance": 0.88,
    "reasoning_consistency": 0.82,
    "answer_relevance": 0.79
  },
  "agent_reasoning_steps": 6
}
```

The values above are illustrative; in particular `ragas_scores` are currently all `0.0` because Ragas is not installed.

Case files below the confidence gate are flagged for human review and withheld from the published feed. An investigation that produces no parseable verdict writes **nothing** and leaves the anomaly uninvestigated for the sweep to retry — publishing a fabricated "uncertain at 0.0" would assert a verdict nothing supports and hide the failure permanently.

---

## Results

Full tables in [`docs/results.md`](docs/results.md), all regenerated by scripts.

| What | Measured |
|---|---|
| Cascade classifier | held out on 400 unseen cascades: macro F1 0.910, ROC-AUC 0.960, Brier 0.069 (original: 0.889 / 0.938 / 0.086). 14 new features won all 10 paired CV folds (+0.017 ROC-AUC); nested Optuna tuning mostly bought calibration (ECE 0.064 → 0.045) |
| Classifier vs LLM agent | classifier AUC 0.960 on the held-out split. The LLM arms are being re-measured: the old 0.571 / 0.359 came from an agent that never called its tools. First runs of the fixed agent: LLM alone at chance (13 and 9 of 24), with the classifier tool 17 of 21 |
| Classifier vs GNN | a plain-PyTorch GIN on the reshare trees, fed the same per-post signals, reaches CV ROC-AUC 0.920 against 0.948 for tuned LightGBM, and does not help in an ensemble |
| Distribution shift | nine perturbed simulators; the upgraded model's ROC-AUC beats the original's on eight of nine, loses on desynchronised campaigns (0.916 vs 0.928), and every model is near chance when every cascade is a confusable subtype |
| Hardest subtype | `stealth_coordinated` is **not** improved (held-out 0.393 → 0.286, n=28); an oracle reading the simulator's true parameters only reaches 0.56 on it |
| Anomaly baselines | at matched recall on 150 held-out topics, per-event CUSUM fires 0.08 false alarms per topic-day against 1.35 for median/MAD; under overdispersed chatter every detector degrades, CUSUM least |
| Filtered vector search | on the HNSW path, table-wide returns 2.64 of 5 rows at 5% selectivity, iterative scan 5 of 5 at the same 0.320 recall, a partial index 5 of 5 at 0.978; at benchmark scale pgvector 0.8's planner sidesteps the index and sorts exactly |
| Review gate | 0.55, read off the reliability curve, not guessed (the upgraded model clears 90% accuracy at every confidence, so the gate is kept as a floor) |
| Learned deferral | **negative result** — ties confidence gating, does not beat it |
| Live traffic | Bluesky, HN, Mastodon and GitHub through the full Kafka path: 59,834 reshare edges and 172,290 nodes in 14 minutes; cascade shape compared against the simulator |
| Kafka | zero records lost across a mid-batch processor kill; replaying all 84,957 messages twice left the graph unchanged; full tables in `docs/results.md` |

---

## Engineering Challenges

**Durable delivery on top of a fire-and-forget channel.**
`LISTEN/NOTIFY` gives sub-second wakeups and no durability: Postgres discards notifications with no live listener, so any anomaly raised while the agent was down was lost for good. Adding a queue would have meant new infrastructure for a problem the database could already answer. Solved by sweeping `anomaly_events WHERE investigated = FALSE` on a timer, with NOTIFY kept as the fast path and an in-flight set so the sweep never re-enqueues work already running.

**Filtered vector search silently loses recall.**
`search_similar` filters on `(model_name, model_version)` and orders by distance. A single HNSW index spanning the table walks the graph unaware of that filter, so neighbours from other model versions are found first and discarded afterwards — consuming the candidate budget. At 5% selectivity a query asking for 5 rows gets under 3, with no error raised. pgvector's `iterative_scan` fixes this generally but needs 0.8+. Solved with one partial HNSW index per model version, so every row in the index already satisfies the predicate.

**A feature that drifted across its own train/test split.**
Author-reuse counts accumulated from the start of the dataset, so they grew
without bound and did not mean the same thing on either side of a temporal split
— `prior_author_mean` shifted 1.7 standard deviations and `prior_author_frac`
saturated at exactly 1.000 on the holdout, carrying no information at all. It was
the strongest feature by gain. Nothing errored; the model simply learned
thresholds on small early counts and was tested on large late ones. Counting over
a trailing six-hour window instead made it stationary and moved F1 from 0.845 to
0.882, Brier from 0.119 to 0.086, and accuracy on the hardest subtype from 0.500
to 0.781. It is also the only version production could compute, since history
cannot be accumulated forever.

**The model tool never saw the feature it relied on most.**
The agent scores a cascade by featurising that one root. Author reuse and the
co-author features are defined against the cascades in the trailing window, so
featurised alone they all read zero — values the model only met on the first
cascades of its training data. Batch evaluation never exercised that path, so
every reported number was fine while the tool itself ran at ROC-AUC 0.874
instead of 0.929 (measured by scoring CV folds both ways). `score_cascade` now
featurises the root together with every cascade active in the window before it,
which reproduces the batch features exactly.

**A detector threshold only means something in its sampling regime.**
The anomaly replay scored topics on a 5-minute clock; the consumer scores on
every arriving event. Re-running the comparison per event moved every
baseline: z-score at a threshold of 5, which looked excellent on the clock,
caught 4% of campaigns per event. Thresholds are now chosen on tuning topics
and reported on 150 held-out topics in the per-event regime, and a stress test
rebuilds the streams with overdispersed chatter, because the simulator's
background is pure Poisson — exactly what a Poisson baseline assumes.

**Knowing when not to answer — and finding out you cannot learn it.**
The confidence gate published cascades the model was sure about, which held for
easy cases and failed on hard ones. A second model was built to predict whether
the classifier would be right, trained on out-of-fold labels so difficulty was
learned rather than memorised. It ties confidence gating and does not beat it,
because the cascades both models get wrong are generated to be
feature-indistinguishable from the other class — a deferral model reading the
same features cannot flag what the classifier cannot separate. Kept as a negative
result, not wired into the runtime: the remaining gap needs new features, not a
better gate.

**A benchmark that measures the generator instead of the problem.**
The first simulator produced a classifier at F1 0.97 — a number that says the two populations were trivially separable, not that the task was solved. `hop_prob` and `root_attach` had been given non-overlapping ranges per class. Fixed by making every class-conditional parameter range straddle its counterpart, drawing target size from a shared distribution so raw size cannot leak the label, and generating a fraction of each class as a confusable subtype. A test now asserts the overlap so the easy dataset cannot come back.

**At-least-once delivery, proven rather than assumed.**
The original consumer auto-committed offsets on a timer, so a record could be marked consumed before its write landed, and a record that failed to write was logged and skipped for good — eleven were, all carrying a NUL byte Postgres refuses. The processor now writes a batch in one transaction and commits the input offsets in the same Kafka transaction as the batch's entity mentions, only after the graph write. A loss audit reads every record back off the topics: zero of 83,664 missing with a processor killed mid-batch. The writes are idempotent — `ON CONFLICT` on the node key and on `(source_id, target_id, platform, ts)` — and replaying all 84,957 messages twice left the graph's node and edge counts unchanged.

**Scoring that broke as soon as it scaled.**
Velocity was counted inside the consumer, so with three consumers each saw roughly a third of an entity's mentions and judged against a third of its history. Mentions are now re-published to `entity-mentions` keyed by entity id, so every mention of one entity reaches the same scorer, which counts on event time rather than its own clock. A test checks that splitting a stream by entity raises exactly the alerts one scorer would, and that splitting it round-robin does not. The live run then exposed a second problem: a freshly started scorer's windows can only fill, so every count climbs and steady chatter reads as a spike, and the low early counts poison the baseline long after. A per-partition warm-up, plus dropping NER spans that were emoji runs or URLs, took live alerts from 28.1 to 3.9 a minute.

**An agent that never used its tools.**
The ReAct prompt asked for a bare JSON answer, and every model tested — hosted and local — obeyed by answering immediately, so no tool ever ran: an empty test node came back "coordinated at 0.95". Underneath, every tool advertised its parameters as `args` and `kwargs`, so even a model that tried could not call one correctly. The agent now uses native tool calls through the OpenAI-compatible API, submits its verdict through a tool of its own, and refuses a verdict with no tool result behind it. Anomalies fire on entities, which have no outgoing edges, so the tools also had to learn to look backwards from a topic to the posts and cascades behind it.

---

## Project Structure

```
diffusion/
├── ingestion/          # one async Kafka producer per platform
├── processing/         # processor (dedup, spaCy NER, batched writes), scorer, anomaly baselines
├── graph/              # schema, propagation queries, pgvector embeddings
├── agent/              # ReAct agent, tools, confidence gate, Ragas evaluation
├── api/                # FastAPI app, REST routes, WebSocket, SSE
├── simulation/         # labeled cascade generator + diurnal background traffic
│   ├── cascade.py      #   one growth engine, two parameterisations
│   ├── background.py   #   non-cascade chatter, so FP rate is measurable
│   └── generate.py     #   CLI: writes graph rows + labels.csv
├── ml/                 # the measured layer
│   ├── features.py     #   per-cascade features (recursive CTE + pandas)
│   ├── offline.py      #   the same features from in-memory simulator runs, plus shifts
│   ├── gnn.py          #   GIN baseline on reshare trees (plain PyTorch, CPU)
│   ├── experiment.py   #   nested CV, Optuna, shift benchmark, oracle, holdout
│   ├── dataset.py      #   ground truth join, temporal split
│   ├── train.py        #   LightGBM + cross-validation
│   ├── predict.py      #   inference for the agent tool
│   ├── evaluate.py     #   classifier vs LLM vs hybrid, calibration
│   ├── eval_anomaly.py #   anomaly baseline replay comparison
│   ├── eval_retrieval.py # filtered vector search recall
│   ├── deferral.py     #   learned deferral (negative result, see results.md)
│   └── analyze_real.py #   live cascade characterisation vs the simulator
├── tests/              # pure unit tests, no external services
├── scripts/            # Kafka lag monitor and loss audit, seed, e2e validation, benchmark
├── docs/results.md     # every measured number
└── docker-compose.yml
```

---

## Running Locally

**Prerequisites:** Python 3.11+ and Docker (for Kafka and Postgres).

```bash
git clone https://github.com/825pranav/diffusion
cd diffusion

python -m venv .venv
# Two passes: the agent's llama-index pins declare numpy<2 while the rest of
# the stack needs numpy>=2, so they cannot be resolved together.
.venv/bin/pip install -r requirements-agent.txt   # .venv/Scripts/pip on Windows
.venv/bin/pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Kafka and Postgres

```bash
cp .env.example .env
docker compose up -d     # Kafka (KRaft, no ZooKeeper), Postgres 16 + pgvector 0.8,
                         # and a one-shot job that creates the six topics
```

There is no separate migration step: the API, the processor and the scorer each
create the schema on startup if it is missing, serialised by an advisory lock so
several starting at once do not race.

### LLM

Set `GROQ_API_KEY` in `.env` for hosted inference, or run everything locally:

```bash
ollama pull qwen2.5:7b        # agent reasoning
ollama pull nomic-embed-text  # embeddings (768-dim, matches the schema)
```

The agent probes Groq once and falls back to Ollama if the key is missing, rejected, or does not serve `GROQ_MODEL` (default `openai/gpt-oss-120b`). The Ollama chat model comes from `OLLAMA_CHAT_MODEL`, which defaults to `llama3.2` — set `OLLAMA_CHAT_MODEL=qwen2.5:7b` to match the model the evaluation arms used.

### Run it

```bash
uvicorn api.main:app              # API + agent listener + embedding indexer
python -m processing.consumer     # raw topics -> graph; run up to 3 (one per partition)
python -m processing.scorer       # entity-mentions -> anomaly_events
python -m ingestion.bluesky_producer
python -m ingestion.hn_producer
python -m ingestion.mastodon_producer   # MASTODON_INSTANCE / MASTODON_ACCESS_TOKEN
python -m ingestion.github_producer     # GITHUB_TOKEN recommended: 60 requests/hour without
```

```bash
python -m scripts.measure_kafka --minutes 10 --out run.jsonl   # lag, throughput, latency
python -m scripts.audit_kafka --since <epoch>                  # every record reached the graph?
```

### The ML pipeline

```bash
python -m simulation.generate --n 2000 --truncate   # labeled cascades + background
python -m ml.train                                  # LightGBM + 5-fold CV
python -m ml.evaluate --skip-llm                    # classifier arm + review gate
python -m ml.evaluate --llm-sample 24               # all three arms (slow)
python -m ml.eval_anomaly                           # baseline comparison, fixed threshold
python -m ml.eval_anomaly --sweep --tick 0          # tuned thresholds, per-event, held-out topics
python -m ml.experiment cv                          # model comparison (needs requirements-experiments.txt)
python -m ml.experiment shift                       # robustness under simulator shift
python -m ml.experiment holdout                     # the one look at the 400 held-out cascades
python -m ml.eval_retrieval                         # filtered search recall
python -m ml.deferral                               # learned deferral (negative result)
```

### Live traffic

With the pipeline above running for a while:

```bash
python -m ml.analyze_real    # observed cascades vs the simulator
```

Each writes its own section of `docs/results.md`.

### Validation

```bash
pytest                                    # unit tests, no services needed
python scripts/validate_e2e.py            # end-to-end against a running stack
python scripts/benchmark.py --requests 200 --concurrency 10
```

---

## License

MIT
