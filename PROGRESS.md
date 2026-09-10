# Diffusion — where things stand

Working context. Read this first when picking the project back up.
Last updated 2026-09-11.

**What changed:** the project went from an LLM guessing at verdicts to a measured
ML system — a difficulty-calibrated cascade simulator, a trained and calibrated
classifier, a derived review gate, and a live ingestion run against real Bluesky
and Hacker News traffic.

---

## Resume in three commands

```bash
python -m scripts.devdb start      # embedded Postgres + schema, writes DATABASE_URL
pytest                             # 95 tests, no services needed
uvicorn api.main:app --reload      # API + agent listener + embedding indexer
```

Everything else (`ml.train`, `ml.evaluate`, `scripts.ingest_live`, …) is listed in
the README under **Running Locally**. The `.venv` is already built and the
database still holds the simulated dataset *and* the live capture.

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
| 6 | Tests + CI | ✅ 95 tests, ruff clean, GitHub Actions |
| — | Live ingestion | ✅ real Bluesky + HN, cascades characterised |
| — | Learned deferral | ✅ built, **negative result**, not wired in |

---

## Results

Full tables in [`docs/results.md`](docs/results.md), all script-generated.

| Measurement | Result |
|---|---|
| Classifier, 5-fold CV | F1 **0.882 ± 0.039**, ROC-AUC 0.929 |
| Classifier, held out (400) | F1 **0.882**, ROC-AUC 0.938, Brier **0.086** |
| By subtype | `plain` 0.941 · `viral_organic` 0.781 · `stealth_coordinated` 0.393 |
| Review gate | **0.55**, derived — 90.7% accuracy at 96.8% coverage |
| Anomaly baseline | median/MAD cuts false alarms **3.5×** vs z-score (5.06 → 1.44 per topic-day) |
| Filtered vector search | table-wide HNSW returns **2.46 of 5** rows at 5% selectivity; partial index returns 5 |
| Learned deferral | 0.958 vs 0.957 for confidence — **no gain** |
| Live capture | **103,544 reshare edges**, 262,349 real nodes from Bluesky + HN |
| Live cascades | 3,000 observed, 980 with 3+ posts, max size 126, max depth 13 |

---

## The findings worth remembering

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

**3. The LLM is the weakest link, and the LLM measurement is itself weak.**
A local 7B reasoning over graph structure lands around chance, far below the
classifier's 0.938 — so routing a verdict through it costs accuracy, and its
value is the case file it writes rather than the label it picks. But do not read
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

**4. Running on real traffic found a bug simulation never could.**
The anomaly detector's loudest "trending topics" were a Hindi phrase and a lone
Japanese bracket, firing repeatedly. `en_core_web_sm` is English-only and the
firehose is multilingual; given other languages it does not decline, it invents
entities, which then accumulate mentions and trip the detector. The detector was
working correctly on garbage input. Records already carried `langs`, so the guard
was free.

---

## Environment

No Docker, WSL, or system PostgreSQL on this machine, so local development runs
on an embedded database rather than the compose stack.

| Component | How it runs | Notes |
|---|---|---|
| PostgreSQL 16.2 | `pgserver` wheel, embedded | `python -m scripts.devdb start` |
| pgvector 0.6.2 | bundled | **< 0.8**, so `iterative_scan` is unavailable — hence partial HNSW indexes |
| LLM (hosted) | Groq `openai/gpt-oss-120b` | key works; free tier 429s under bulk evaluation |
| LLM (local) | Ollama `qwen2.5:7b` | used for the evaluation arms; weak at ReAct |
| Embeddings | Ollama `nomic-embed-text` | 768-dim, matches the schema |
| Kafka | **not running** | Docker Desktop installed but won't start without a reboot |

### Dependency notes

- `llama-index-core` pinned at **0.10.52** — 0.14 removes `ReActAgent.from_tools`,
  which `agent/agent.py` is built on. It declares `numpy<2` but runs fine on the
  numpy 2 that scipy and lightgbm need; install it *before* restoring numpy. The
  pip resolver warning is expected.
- `spacy` on 3.8.x — 3.7.5 has no Python 3.12 wheels.

---

## Open items

- [ ] **Reboot to finish Docker Desktop.** It installs but the engine won't
      start until Windows restarts. Then `docker compose up -d` brings up Kafka
      (KRaft, no ZooKeeper) and the compose Postgres. The compose file is
      committed but **has never been run** — that is the one untested piece.
- [ ] **Groq bulk evaluation is rate-limited.** The key is valid and
      `openai/gpt-oss-120b` answers in ~1.6 s, but one investigation is 4–8 rapid
      calls and the free tier 429s throughout a 40-cascade run. `--llm-delay`
      exists; a paid tier or a smaller Groq model would fix it properly.
- [ ] **Live classifier scores are out of domain.** Observed cascades are far
      shallower than simulated ones (`max_depth` p50 1 vs 4), because a firehose
      shows replies-to-posts far more often than replies-to-replies. The model
      still returns confident numbers on inputs it was never trained on.
- [ ] **Anomalies re-fire on the same entity.** A sustained elevated topic
      triggers repeatedly with no per-entity cooldown, so one trend can occupy
      the agent many times over. 278 anomalies fired during a 25-minute capture,
      most of them repeats of a handful of entities.

---

## What I would do next, ranked

1. **Backfill cascade parents through the Bluesky API.** The single highest-value
   change. Right now a reply whose parent never floated past becomes a two-node
   tree, which is why real cascades look flat and why classifier scores on live
   data are out of domain. Fetching parents on demand reconstructs whole threads
   and makes the live numbers trustworthy.
2. **New features for `stealth_coordinated`.** Account age, posting-schedule
   regularity, or coordination structure *across* cascades rather than within
   one. This is the documented ceiling — no gating or model change touches it.
3. **Re-run the LLM arms properly.** Current numbers are a local 7B over ~17
   verdicts, where run-to-run variance swamps the difference between arms. Needs
   Groq (rate limits permitting) and a sample in the hundreds before the
   `llm` vs `hybrid` comparison means anything.
4. **Calibrate explicitly** (isotonic/Platt) and re-derive the gate. Brier is
   already 0.086 after the drift fix, so gains will be smaller than they would
   have been, but it makes the probabilities defensible rather than incidental.

---

## Decision log

- **Embedded Postgres over skipping the DB.** Every measured number needs real
  SQL; mocking it would have made them meaningless.
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
