# Diffusion — where things stand

Working context. Read this first when picking the project back up.
Last updated 2026-10-01.

**What changed:** the project went from an LLM guessing at verdicts to a measured
ML system — a difficulty-calibrated cascade simulator, a trained and calibrated
classifier, a derived review gate — and, as of 2026-10-01, everything runs on
the real stack: four live producers through Kafka into Postgres, measured for
lag, capacity and loss, with an agent that now actually calls its tools.

---

## Resume in three commands

```bash
docker compose up -d               # Kafka + Postgres 16/pgvector 0.8, topics created
pytest                             # 170 tests, no services needed
uvicorn api.main:app               # API + agent listener + embedding indexer
```

then `python -m processing.consumer`, `python -m processing.scorer` and the four
`ingestion.*_producer`s. Everything else is listed in the README under
**Running Locally**. The compose database holds the simulated dataset *and* a
live capture taken through Kafka.

---

## Status

| # | Task | Status |
|---|------|--------|
| 0 | Unblock the pipeline | ✅ `validate_e2e.py` passes end to end |
| 1 | Labeled cascade simulator | ✅ 2 000 cascades, overlap calibrated |
| 2 | Virality classifier | ✅ F1 0.882, exposed as an agent tool |
| 3 | Evaluation + calibration | ✅ three arms, reliability diagram, derived gate |
| 4 | Anomaly baseline | ✅ median/MAD default, 3.5× fewer false alarms |
| 5 | Retrieval correctness | ✅ partial HNSW index, recall measured |
| 6 | Tests + CI | ✅ 145 tests, ruff clean, GitHub Actions |
| — | Live ingestion | ✅ real Bluesky + HN, cascades characterised |
| — | Propagation capture | ✅ HN comment threads and Mastodon replies, parents backfilled |
| — | Learned deferral | ✅ built, **negative result**, not wired in |
| 7 | Classifier upgrade | ✅ 14 features, nested Optuna, GIN baseline, shift benchmark, oracle ceiling |
| 8 | Anomaly upgrade | ✅ Poisson / negative binomial / CUSUM, tuned per event on held-out topics |
| — | Train/serve skew | ✅ single-cascade scoring now sees its trailing window |
| 9 | Kafka end to end | ✅ compose stack runs; batched transactional processor, entity-keyed scorer, loss audit |
| 10 | Agent grounding | ✅ native tool calls, verdicts refused without evidence, topics resolvable |

---

## Results

Full tables in [`docs/results.md`](docs/results.md), all script-generated.

| Measurement | Result |
|---|---|
| Classifier, 5-fold CV | F1 **0.893 ± 0.030**, ROC-AUC 0.946 (was 0.860, 0.929) |
| Classifier, held out (400) | macro F1 **0.910**, ROC-AUC **0.960**, Brier **0.069** (was 0.889, 0.938, 0.086) |
| By subtype (held out) | `plain` 0.971 · `viral_organic` 0.812 · `stealth_coordinated` 0.286 (was 0.941 · 0.781 · 0.393) |
| GIN on reshare trees | CV ROC-AUC 0.920 vs 0.948 tuned LightGBM — does not win |
| Oracle ceiling | true simulator params: ROC-AUC 0.981, `stealth_coordinated` 0.561 |
| Anomaly, per event, 150 held-out topics | CUSUM 0.08 FP/topic-day at recall 0.846 vs median/MAD 1.35 at 0.832 |
| Review gate | **0.55**, derived — 90.7% accuracy at 96.8% coverage |
| Anomaly baseline (fixed 2.5, clock ticks) | median/MAD cuts false alarms 3.4× vs z-score (4.99 → 1.48 per topic-day on the 50 topics the DB now replays; the earlier 5.06 → 1.44 was a 40-topic replay) |
| Filtered vector search | on the HNSW path (pgvector 0.8.6): table-wide 2.64 of 5 rows, iterative scan 5 of 5 at the same 0.32 recall, partial index 5 of 5 at **0.978** |
| Learned deferral | 0.958 vs 0.957 for confidence — **no gain** |
| Kafka throughput | one processor keeps up live (~100 msgs/s, ≤3 s behind; old consumer: 63/s, 173 s behind); replay 577 msgs/s, 1,276 on three |
| Kafka loss | **0 of 83,664** records missing with a processor killed mid-batch (old consumer: 11 lost to NUL bytes) |
| Kafka idempotence | replaying all 84,957 messages twice left node and edge counts unchanged |
| Live anomaly alerts | 28.1 → **3.9 a minute** with per-partition warm-up and junk-span filtering |
| Live capture (through Kafka) | 59,834 reshare edges, 172,290 nodes from Bluesky, HN, Mastodon, GitHub in 14 min |
| Live cascades | 3,000 observed, 768 with 3+ posts, max size 165, max depth 40 |
| Backlog sweep | verified on real data — recovered 50 anomalies (batch limit) that `NOTIFY` had dropped |

---

## The findings worth remembering

**0. (2026-09-26) New features beat new models, and the hard subtype is a ceiling.**
Fourteen features — log-delay spread, structural virality, a root-attachment
MLE, and co-author overlap across recent cascades — won all ten paired CV folds
and moved held-out ROC-AUC 0.938 → 0.960 (bootstrap 95% CI on the gain +0.007
to +0.039). Optuna under nested CV bought calibration, not discrimination. A
GIN over the raw trees lost to LightGBM on identical information, and an
ensemble did not help. `stealth_coordinated` did not improve (0.393 → 0.286 on
28 cascades): a model given the simulator's *true parameters* only reaches 0.56
on it, so most of that gap is the generator's deliberate overlap, not missing
features.

**0b. The agent's model tool had been scoring with its best feature zeroed.**
`score_cascade` featurised one root alone, so windowed author reuse read zero
at inference. Measured: ROC-AUC 0.929 → 0.874. Fixed by featurising the
root's trailing window with it.

**0c. Detector thresholds do not transfer between sampling regimes.**
The replay used a 5-minute clock; the consumer samples per event. Per event,
z-score at its clock-tuned 5 catches 4% of campaigns. The default is now
per-event-tuned CUSUM over Poisson tails.

**1. A feature drifted across its own train/test split.**
Author reuse was counted cumulatively from the start of the dataset, so it grew
without bound: `prior_author_mean` shifted 1.7 sd between train and holdout, and
`prior_author_frac` saturated at exactly 1.000 — no information at all — while
being the strongest feature by gain. Nothing errored. Windowing it to a trailing
six hours took F1 from 0.845 → 0.882, Brier 0.119 → 0.086, and `viral_organic`
accuracy 0.500 → 0.781.

**2. Learned deferral does not work here, and the reason matters.**
A second model was trained to predict whether the classifier would be right
(out-of-fold labels, so difficulty is learned not memorised). It ties confidence
gating. Both gates publish a quarter of `stealth_coordinated` and get *every one*
wrong — those cascades are generated to look organic, so they are
feature-indistinguishable from the real thing, and a deferral model reading the
same features cannot flag what the classifier cannot separate. **The remaining
gap needs new features, not a better gate.**

**3. The LLM is the weakest link — and the old LLM numbers measured nothing.**
(2026-10-01) The agent never called a tool: the prompt's "respond with ONLY
JSON" beat the ReAct format, and every tool's schema read `args`/`kwargs`. The
table below was measured on that broken agent and is withdrawn. With the agent
fixed, local `qwen2.5:7b` alone got 13 and then 9 of 24 right — still chance —
and with the classifier tool 17 of 21, so its value remains the case file it
writes rather than the label it picks. The old runs, for the record: But do not read
the `llm` and `hybrid` rows as a ranking. Three single runs of an unchanged
configuration produced:

    run    llm ROC-AUC    hybrid ROC-AUC
     1        0.602           0.833
     2        0.424           0.535
     3        0.571           0.359

The hybrid arm spans 0.47 AUC and the llm arm 0.18, on samples of ~15 verdicts.
Noise dwarfs the effect. `ml/evaluate.py` now takes `--llm-repeats` and reports
mean with the observed range, so the table cannot imply precision it does not
have. Getting a real comparison needs a stronger model and a sample in the
hundreds — a rate-limit and runtime problem, not a design one.

**4. The durability fix was validated against real traffic, not just tests.**
278 anomalies fired during the live capture while the agent was deliberately
stopped. Postgres `NOTIFY` drops notifications with no listener, so under the
original code every one would have been lost. On restart:

    backlog sweep recovered 50 uninvestigated anomalies
    investigating anomaly 6 - node=entity:org:utc platform=bluesky z=3.37

50 is `AGENT_SWEEP_BATCH`; the rest drain on subsequent ticks.

**5. Running on real traffic found a bug simulation never could.**
The anomaly detector's loudest "trending topics" were a Hindi phrase and a lone
Japanese bracket, firing repeatedly. `en_core_web_sm` is English-only and the
firehose is multilingual; given other languages it does not decline, it invents
entities, which then accumulate mentions and trip the detector. The detector was
working correctly on garbage input. Records already carried `langs`, so the guard
was free.

---

## Environment

Docker Desktop runs (WSL 2), so everything uses the compose stack; the embedded
`pgserver` database and the Kafka-bypassing `ingest_live.py` are gone.

| Component | How it runs | Notes |
|---|---|---|
| Kafka 3.6 (Confluent 7.6.1) | compose, KRaft | six topics created by `kafka-init` |
| PostgreSQL 16 + pgvector 0.8.6 | compose | schema created by whichever service starts first |
| LLM (hosted) | Groq `openai/gpt-oss-120b` | free tier: 8,000 tokens/min, about one investigation a minute |
| LLM (local) | Ollama `qwen2.5:7b` | default local model; `qwen3:8b` slower, `llama3.2` wrong too often |
| Embeddings | Ollama `nomic-embed-text` | 768-dim, matches the schema |

### Dependency notes

- `llama-index-core` pinned at **0.10.52** and now used only for `FunctionTool`;
  the tool-calling loop uses the `openai` client against Groq's and Ollama's
  OpenAI-compatible endpoints. It declares `numpy<2` but runs fine on the numpy 2
  that scipy and lightgbm need; install it *before* restoring numpy. The pip
  resolver warning is expected.
- `spacy` on 3.8.x — 3.7.5 has no Python 3.12 wheels.

---

## Open items

- [ ] **Groq bulk evaluation is rate-limited.** The key is valid and
      `openai/gpt-oss-120b` answers in ~1.6 s, but one investigation is 4–8 rapid
      calls and the free tier 429s throughout a 40-cascade run. `--llm-delay`
      exists; a paid tier or a smaller Groq model would fix it properly.
- [~] **Live classifier scores are out of domain.** Observed cascades were far
      shallower than simulated ones (`max_depth` p50 1 vs 4), because a firehose
      shows replies-to-posts far more often than replies-to-replies.
      `ingestion/bluesky_backfill.py` now resolves the missing parents through
      the public API — no auth needed — and climbs the reply chain. Measured on
      25 live replies: median depth 2, mean 2.8, max 7, with 32% reaching depth
      3 or more, against a flat 1 before. Not yet at the simulator's p50 of 4,
      so the domain gap is narrowed rather than closed.
- [ ] **Re-measure the LLM arms on the fixed agent.** Run
      `python -m ml.evaluate --backends ollama,groq --llm-repeats 3`. The local
      half takes ~20 minutes; Groq's free tier (8,000 tokens/min) makes its half
      about an hour, and unload other Ollama models first or they share VRAM.
- [~] **Anomalies re-fire on the same entity.** The warm-up and span filter
      cut live alerts from 28.1 to 3.9 a minute, but a sustained topic can still
      re-fire with no per-entity cooldown, and the local agent takes ~15 s an
      investigation.

---

## What I would do next, ranked

1. **Re-measure the live cascade distribution now that parents backfill.**
   `bluesky_backfill.py` reconstructs threads (median depth 2, max 7 on a
   25-reply sample), so `ml/analyze_real.py` should be re-run against a fresh
   capture to see how far the live distribution has moved toward the
   simulator's. That comparison is what decides whether live scores can be
   trusted, and it now has real thread structure to work with.
2. **`stealth_coordinated` is now mostly a simulator question.** Cross-cascade
   co-author features were added and did not move it; the oracle on true
   parameters reaches only 0.56. Further gains need either account-level
   signals the simulator does not yet model (account age, schedule regularity)
   or a decision that the stealth parameter ranges overlap organic by too much.
3. **Re-run the LLM arms properly** — and against the upgraded classifier.
   The `Classifier vs LLM agent` section still shows the original model's row.
   Needs Groq (rate limits permitting) or the local model server, and a sample
   in the hundreds.
4. **Validate CUSUM on live traffic.** It was chosen on replayed simulator
   streams, including an overdispersed stress test, but not on a live capture.
   Its `z_score` column now carries the CUSUM statistic, not a z-score.
5. **Desynchronised campaigns** are the one shift where the upgraded model
   loses to the original (0.916 vs 0.928 ROC-AUC): the log-delay features are
   what that shift attacks. Domain randomisation did not recover it.

---

## Decision log

- **Embedded Postgres over skipping the DB** (until 2026-10-01, when the compose
  stack replaced it). Every measured number needs real SQL; mocking it would have
  made them meaningless.
- **Transactional offsets over at-most-once speed.** Input offsets commit in the
  same Kafka transaction as the batch's entity mentions, after the graph write,
  so a crash costs a re-read rather than a lost record.
- **Re-key mentions by entity rather than share scorer state.** Kafka's
  partitioning puts every mention of one entity on one scorer; the alternative
  was a shared store on the hot path of every mention.
- **Temporal, not random, train/test split.** Fit on history, score what comes
  next — and it is the only split that keeps the causal author features honest.
- **Cascade size is deliberately not a class signal.** Target size is drawn from
  one distribution for both classes, so the model must learn timing, structure
  and author reuse. A useless `size` feature is an honest result.
- **Organic bursts are not anomaly false positives.** Genuine virality *is*
  anomalous velocity; separating it from a campaign is the classifier's job.
- **Failed investigations write nothing.** The old code wrote "uncertain at
  confidence 0.0" and marked the anomaly investigated — asserting a verdict
  nothing supported and hiding the failure permanently.
- **LightGBM over the GIN.** The GNN saw the same per-post information and lost
  on CV (0.920 vs 0.948); it is kept as a measured baseline, not shipped.
- **Calibration by tuning, not by wrapper.** Isotonic calibration on top of the
  tuned model changed nothing measurable, and a wrapper would break the
  per-prediction attributions the agent reads.
- **Deferral kept but not wired in.** It ties confidence gating, and added
  complexity has to buy something measurable.

---

## Bugs found and fixed — none of which raised an error where they lived

1. `consumer.py` called `detector.evaluate(vscore, platform=)` against a
   `(conn, score, platform)` signature. **Anomaly detection had never run.**
2. `embed_unindexed_nodes` had no callers, so `trend_embeddings` stayed empty and
   `search_similar_trends` returned `[]` on every investigation.
3. `store_embedding` used `ON CONFLICT` on columns with no unique index — every
   embedding write failed once the indexer actually ran.
4. `NOTIFY` had no durable backing; anomalies raised while the agent was down
   were lost permanently. The index for the sweep already existed, unused.
5. `validate_e2e.py` / `seed.py` couldn't `import config` when run as documented,
   and seeded nodes with a type nothing reads.
6. `validate_e2e.py` deleted `anomaly_events` before the `case_files` referencing
   them — a FK error that only appeared once the agent started succeeding.
7. The Groq→Ollama fallback the README promised did not exist.
8. The Groq probe used `urllib`'s default User-Agent, which Cloudflare 403s
   (error 1010) — it read a **valid** key as unusable, the exact failure it was
   added to prevent.
9. asyncpg returns `timestamptz` as `datetime64[us]`, so `.astype("int64")/1e9`
   was 1000× off. `peak_rate_per_min` was counting a 17-hour window.
10. Cascade delays were measured from the earliest *child* rather than the seed
    post, forcing the first root-attached child to zero.
11. Re-running the simulator with the same seed grafted two datasets together
    through shared node ids, inflating cascades past their target size.
12. **The extractor emitted no `reshare` edges at all**, so live data produced no
    cascades and the classifier could not be applied to it. The producer was
    discarding `reply.parent` and never subscribed to reposts.
13. Author reuse drifted 1.7 sd across the temporal split (see finding 1).
14. A name collision between the windowed history counter and per-cascade author
    frequency made every first-ever cascade look like it had a full history.
15. English-only NER ran on multilingual firehose text, inventing entities that
    then became the top "trending topics" the anomaly detector fired on.
16. The compose healthcheck probed `localhost:29092`, a listener bound to the
    `kafka` hostname, so Kafka never became healthy and the topics were never
    created. **The compose stack could not have started.**
17. Nothing on the compose path created the schema; only `devdb` did.
18. The consumer auto-committed offsets on a timer and skipped records that
    failed to write: 11 of 86,328 were lost to NUL bytes on the first Kafka run.
19. Velocity was counted per consumer process, so it was wrong as soon as a
    second consumer started.
20. A fresh scorer's filling windows read as spikes in everything, and poisoned
    the baseline for long after: 28 alerts a minute on live traffic.
21. Mastodon's default instance (and the one before it) now refuses anonymous
    public timelines, so the producer published nothing.
22. `.env.example` subscribed Jetstream to posts only, dropping every repost.
23. **The agent never called a tool.** The prompt's "respond with ONLY JSON"
    beat the ReAct format, and every tool's schema read `args`/`kwargs`, so no
    model could have called one correctly anyway. Verdicts were invented.
24. `classify_virality` ran two queries at once on one connection and failed on
    every call.
25. Anomalies fire on entities, which have no outgoing edges, so every tool
    returned nothing for a real anomaly.
