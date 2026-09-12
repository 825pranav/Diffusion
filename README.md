# Diffusion
**Distributed Trend Propagation Engine**

*It's not about what's trending. It's about how and why it spread.*

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
                         │ async producers
                         ▼
┌─────────────────────────────────────────────────────┐
│                     Kafka                           │
│   bluesky-raw · mastodon-raw · hn-raw · gh-raw      │
└────────────────────────┬────────────────────────────┘
                         │ async consumers
                         ▼
┌─────────────────────────────────────────────────────┐
│            Stream Processing Layer                  │
│   Deduplication → Entity Extraction (spaCy NER)     │
│   Velocity Scoring → Anomaly Detection              │
│   (median/MAD · EWMA · z-score baselines)           │
└────────────────────────┬────────────────────────────┘
                         │ graph mutations + anomaly events
                         ▼
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
│            LlamaIndex ReAct Agent                   │
│   get_propagation_path                              │
│   search_similar_trends (pgvector)                  │
│   classify_virality        (graph heuristics)       │
│   classify_virality_model  (trained LightGBM)       │
│   confidence gate → case file                       │
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
| **Kafka** | Event bus — streams raw signals from all four sources |
| **PostgreSQL** | Source of truth — propagation graph edges, case files, metadata |
| **pgvector** | Vector search — semantic retrieval of historically similar trends |
| **LightGBM** | Cascade classifier — the agent's strongest evidence |
| **LlamaIndex** | Agent orchestration — stateful ReAct loop with tool calling |
| **Groq** | LLM inference (primary), when a key serves the configured model |
| **Ollama** | LLM inference (fallback) — local, zero cost, and the embedding backend |
| **Ragas** | Retrieval and grounding scores on every case file |
| **FastAPI** | Backend — REST, WebSocket, and SSE endpoints |
| **Docker** | Kafka and Postgres via `docker compose up` (optional — see below) |

---

## Key Design Decisions

**Event-driven, with a durable floor.**
The agent wakes when the anomaly detector sees a spike, so inference cost is zero during quiet periods. `NOTIFY` alone would silently drop anomalies raised while the agent was down, restarting, or mid-failure, so a periodic sweep re-drives anything still marked uninvestigated. Both paths feed one queue, de-duplicated by anomaly id.

**A robust anomaly baseline.**
Mean and standard deviation are both dragged upward by the spikes they are meant to detect, so a sustained campaign progressively hides itself. The default baseline is median and scaled MAD, which does not move until half the history is contaminated. Measured on replayed traffic it cuts false alarms 3.5× against the plain z-score for seven points of recall — the right trade when every detection costs an LLM investigation and the sweep makes a missed spike recoverable.

**Graph modeling in PostgreSQL.**
Propagation relationships are stored as directed edges. Degree, spread depth, and cascade size are computed in SQL — no dedicated graph DB. Feature extraction expands every cascade's reshare tree in a single recursive CTE rather than one query per cascade.

**pgvector over a dedicated vector DB.**
Historical trend embeddings live in the same Postgres instance as relational data, so similarity search can be joined directly with graph queries.

**Provider-agnostic LLM inference.**
Groq is preferred, with a local Ollama model behind it. The backend is probed once at startup — the key must work *and* serve the configured model — because a revoked key or a decommissioned model name would otherwise fail every investigation while a working local model sat idle.

**Evaluation as a first-class concern.**
The classifier is scored on a temporally held-out split against the LLM agent alone and the two combined. The human-review threshold is read off the reliability curve rather than guessed. Ragas scores retrieval relevance and grounding on each case file — note that these measure the *retrieval*, not confidence calibration, which comes from `ml/evaluate.py`.

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

Case files below the confidence gate are flagged for human review and withheld from the published feed. An investigation that produces no parseable verdict writes **nothing** and leaves the anomaly uninvestigated for the sweep to retry — publishing a fabricated "uncertain at 0.0" would assert a verdict nothing supports and hide the failure permanently.

---

## Results

Full tables in [`docs/results.md`](docs/results.md), all regenerated by scripts.

| What | Measured |
|---|---|
| Cascade classifier | F1 0.88, ROC-AUC 0.93, Brier 0.086 (5-fold CV, 2 000 simulated cascades) |
| Classifier vs LLM agent | the agent alone is near chance (AUC 0.60); given the classifier as a tool it reaches 0.83, still below the classifier alone |
| Anomaly baselines | median/MAD cuts false alarms 3.5× vs z-score at comparable delay |
| Filtered vector search | table-wide HNSW returns under 3 of 5 requested rows at 5% selectivity; a partial index returns all 5 |
| Review gate | 0.55, read off the reliability curve, not guessed |
| Learned deferral | **negative result** — ties confidence gating, does not beat it |
| Live traffic | real Bluesky and HN ingestion, cascade shape compared against the simulator |

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

**Idempotent edge insertion under Kafka redelivery.**
Kafka guarantees at-least-once delivery, so a consumer crash mid-processing redelivers the same event and duplicates edges. Solved with a composite unique constraint on `(source_id, target_id, platform, ts)` and `ON CONFLICT DO NOTHING` on every edge write.

---

## Project Structure

```
diffusion/
├── ingestion/          # one async Kafka producer per platform
├── processing/         # dedup, spaCy NER extraction, velocity, anomaly baselines
├── graph/              # schema, propagation queries, pgvector embeddings
├── agent/              # ReAct agent, tools, confidence gate, Ragas evaluation
├── api/                # FastAPI app, REST routes, WebSocket, SSE
├── simulation/         # labeled cascade generator + diurnal background traffic
│   ├── cascade.py      #   one growth engine, two parameterisations
│   ├── background.py   #   non-cascade chatter, so FP rate is measurable
│   └── generate.py     #   CLI: writes graph rows + labels.csv
├── ml/                 # the measured layer
│   ├── features.py     #   per-cascade features (recursive CTE + pandas)
│   ├── dataset.py      #   ground truth join, temporal split
│   ├── train.py        #   LightGBM + cross-validation
│   ├── predict.py      #   inference for the agent tool
│   ├── evaluate.py     #   classifier vs LLM vs hybrid, calibration
│   ├── eval_anomaly.py #   anomaly baseline replay comparison
│   └── eval_retrieval.py #  filtered vector search recall
│   ├── deferral.py     #   learned deferral (negative result, see results.md)
│   └── analyze_real.py #   live cascade characterisation vs the simulator
├── tests/              # pure unit tests, no external services
├── scripts/            # devdb, seed, live ingestion, e2e validation, benchmark
├── docs/results.md     # every measured number
└── docker-compose.yml
```

---

## Running Locally

**Prerequisites:** Python 3.11+, and either Docker or nothing at all (see below).

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

### Database

Docker is optional. The `pgserver` wheel bundles a self-contained PostgreSQL 16 with pgvector, which is enough to run the real schema:

```bash
python -m scripts.devdb start    # starts Postgres, creates the schema,
                                 # writes DATABASE_URL into .env
```

`devdb` also takes `url`, `stop`, and `reset`. It binds a dynamic port and rewrites `DATABASE_URL`, so `config.DB_URL` works unchanged.

To use the compose stack instead, `docker compose up -d` and set `DATABASE_URL` yourself. Compose is the only way to get Kafka, which the live ingestion path needs.

### LLM

Set `GROQ_API_KEY` in `.env` for hosted inference, or run everything locally:

```bash
ollama pull qwen2.5:7b        # agent reasoning
ollama pull nomic-embed-text  # embeddings (768-dim, matches the schema)
```

The agent probes Groq once and falls back to Ollama if the key is missing, rejected, or does not serve `GROQ_MODEL`.

### Run it

```bash
uvicorn api.main:app --reload     # API + agent listener + embedding indexer
python -m processing.consumer     # stream processor (needs Kafka)
python -m ingestion.hn_producer   # one producer per platform (needs Kafka)
```

### The ML pipeline

```bash
python -m simulation.generate --n 2000 --truncate   # labeled cascades + background
python -m ml.train                                  # LightGBM + 5-fold CV
python -m ml.evaluate --skip-llm                    # classifier arm + review gate
python -m ml.evaluate --llm-sample 24               # all three arms (slow)
python -m ml.eval_anomaly                           # baseline comparison
python -m ml.eval_retrieval                         # filtered search recall
python -m ml.deferral                               # learned deferral (negative result)
```

### Live traffic

```bash
python -m scripts.ingest_live --minutes 20   # real Bluesky + HN into the graph
python -m ml.analyze_real                    # observed cascades vs the simulator
```

`ingest_live` runs the real producer serialisation and the real
`consumer.handle()` against Postgres with the Kafka hop omitted, so a live run
works without Docker. Kafka is the transport; everything else is the production
path. Throughput through the broker itself has not been measured here.

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
