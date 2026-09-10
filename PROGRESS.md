# Diffusion — improvement brief progress

Living document. Updated as each task lands. Started 2026-09-11.

**Goal:** turn the LLM-guessing agent into a real, evaluated ML system, keeping the
streaming architecture as-is.

---

## Status at a glance

| # | Task | Status |
|---|------|--------|
| — | Local environment (no Docker) | ✅ done |
| 0 | Unblock the pipeline | ✅ done — `validate_e2e.py` passes end to end |
| 1 | Labeled cascade simulator | ✅ done — 2 000 cascades, calibrated overlap |
| 2 | Real virality classifier | ✅ done — F1 0.854 ± 0.023, exposed as an agent tool |
| 3 | Evaluation harness + calibration | 🟡 classifier arm + gate done; LLM/hybrid arms running |
| 4 | Better anomaly baseline | ✅ done — median/MAD now the default |
| 5 | Retrieval correctness + eval | 🟡 fix landed; benchmark re-running after Task 3 |
| 6 | Tests + CI | ✅ done — 82 tests, ruff clean, GitHub Actions |
| — | Optional cleanup | ✅ done — KRaft, repinned deps, README rewritten |

---

## Next up

1. **Task 3** — the LLM and hybrid arms are running against local `qwen2.5:7b`
   (24 sampled holdout cascades per arm, roughly a minute each). When they finish,
   `docs/results.md` gets the three-way comparison and the reliability diagram.
2. **Task 5** — re-run `python -m ml.eval_retrieval` once the LLM arms are done.
   It was deferred deliberately: it loads 10 000 benchmark vectors into
   `trend_embeddings` and reindexes, which would perturb the similarity search the
   agent is using mid-evaluation.

---

## Results so far

Full tables in [`docs/results.md`](docs/results.md).

| Measurement | Result |
|---|---|
| Classifier, 5-fold CV | F1 **0.854 ± 0.023**, ROC-AUC 0.921 |
| Classifier, held-out (400) | F1 **0.854**, ROC-AUC 0.938, Brier 0.122 |
| Anomaly baseline | median/MAD cuts false alarms **3.5×** vs z-score (5.06 → 1.44 per topic-day) for 7 points of recall |
| Filtered vector search | table-wide HNSW returns **< 3 of 5** requested rows at 5% selectivity; partial index returns 5 |
| Review gate | **0.85** (91% accuracy, 75% auto-published), read off the reliability curve |

Top features by gain: `time_to_half_s`, `prior_author_mean`,
`cross_platform_edge_frac`, `delay_median_s`, `mean_children`. No single feature
dominates, which is the point — an early simulator gave F1 0.97 because two
parameters had non-overlapping ranges per class.

---

## Environment

The machine has no Docker, no WSL, and no system PostgreSQL, so the compose stack
could not be used. Rather than skip the database, local development runs on an
embedded PostgreSQL:

| Component | How it runs locally | Notes |
|---|---|---|
| PostgreSQL 16.2 | `pgserver` wheel, embedded | No Docker required |
| pgvector 0.6.2 | Bundled with `pgserver` | **< 0.8, so `iterative_scan` is unavailable** — Task 5 uses partial HNSW indexes |
| LLM | Ollama `qwen2.5:7b` | `llama3.2` (3B) could not finish a ReAct loop — it exhausted `max_iterations` every time |
| Embeddings | Ollama `nomic-embed-text` | 768-dim, matches the existing schema |
| Kafka | **not running** | Needs Docker. Only the live ingestion path depends on it — no Task 0–6 deliverable does. |

### Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m spacy download en_core_web_sm
python -m scripts.devdb start      # embedded Postgres + schema, writes DATABASE_URL
```

### Dependency notes

- `llama-index-core` is held at **0.10.52**. 0.14 removes `ReActAgent.from_tools`,
  which `agent/agent.py` is built on. It declares `numpy<2` but runs correctly on
  the numpy 2 that scipy and lightgbm require; install it *before* restoring numpy.
  The pip resolver warning is expected.
- `spacy` moved to 3.8.x — 3.7.5 has no Python 3.12 wheels.

---

## Open items requiring the user

- [ ] **Groq API key is rejected.** The key in `.env` returns HTTP 403 from
      `/v1/models`, and `GROQ_MODEL` still names `llama-3.3-70b-versatile`, which
      Groq has decommissioned. The agent now probes and falls back to Ollama, so
      nothing is blocked — but a working key would make Task 3's LLM and hybrid
      arms far stronger than a local 7B model can manage. Replace the key and set
      `GROQ_MODEL` to a model that key serves.
- [ ] **Docker Desktop** — `winget install --id Docker.DockerDesktop` (needs
      admin). Unblocks Kafka and the live ingestion path only. The KRaft compose
      file is committed but **untested**, since this machine cannot run it.

---

## Decision log

Choices worth remembering, with the reasoning:

- **Embedded Postgres over skipping the DB.** Tasks 0–5 all need real SQL; mocking
  it would have made every measured number meaningless.
- **pgvector 0.6.2 constrains Task 5.** `iterative_scan` (0.8+) is unavailable, so
  the filtered-search fix is partial HNSW indexes per `(model_name, model_version)`.
- **Ollama over waiting for a Groq key** — and then the key turned out to be
  rejected anyway, which is what exposed the missing fallback.
- **Size is deliberately not a class signal.** Target cascade size is drawn from
  one distribution for both classes, so the model cannot lean on raw size and has
  to learn timing, structure, and author reuse. A useless `size` feature is an
  honest result.
- **Temporal, not random, train/test split.** Fit on history, score what comes
  next. It is also the only split that keeps the causal author-reuse features
  honest — a random split would put a cascade's own future neighbours in training.
- **Organic bursts are not anomaly false positives.** Genuine virality is real
  anomalous velocity; separating it from a campaign is the classifier's job. A
  detector silent on it would starve the agent of its harder cases.
- **Failed investigations write nothing.** The old code wrote "uncertain at
  confidence 0.0" and marked the anomaly investigated, asserting an unsupported
  verdict and hiding the failure permanently.

---

## Bugs found and fixed along the way

Each of these was latent — none raised an error where it happened:

1. `consumer.py` called `detector.evaluate(vscore, platform=)` against a
   `(conn, score, platform)` signature. Anomaly detection had never run.
2. `embed_unindexed_nodes` had no callers, so `trend_embeddings` stayed empty and
   `search_similar_trends` returned `[]` on every investigation.
3. `store_embedding` used `ON CONFLICT` on columns with no matching unique index,
   so every embedding write failed once the indexer actually ran.
4. `NOTIFY` had no durable backing — anomalies raised while the agent was down were
   lost permanently. The index for the sweep already existed, unused.
5. `validate_e2e.py` and `seed.py` could not `import config` when run the way the
   README documented, and seeded nodes with a type nothing reads.
6. `validate_e2e.py` deleted `anomaly_events` before the `case_files` referencing
   them — a foreign-key error that only appeared once the agent started succeeding.
7. The Groq→Ollama fallback the README promised did not exist.
8. asyncpg returns `timestamptz` as `datetime64[us]`, so `.astype("int64")/1e9` was
   1 000× off. `peak_rate_per_min` was counting a 17-hour window. Tree splits are
   scale-invariant, which is exactly why it went unnoticed.
9. Cascade delays were measured from the earliest *child* rather than the seed
   post, forcing the first root-attached child to zero — and root attachment is
   what separates the two classes.
10. Re-running the simulator with the same seed grafted two datasets together
    through shared node ids, inflating cascades past their target size.
