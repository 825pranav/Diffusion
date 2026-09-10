# Diffusion — improvement brief progress

Living document. Updated as each task lands. Started 2026-09-11.

**Goal:** turn the LLM-guessing agent into a real, evaluated ML system, keeping the
streaming architecture as-is.

---

## Status at a glance

| # | Task | Status |
|---|------|--------|
| — | Local environment (no Docker) | ✅ done |
| 0 | Unblock the pipeline | ⬜ not started |
| 1 | Labeled cascade simulator | ⬜ not started |
| 2 | Real virality classifier | ⬜ not started |
| 3 | Evaluation harness + calibration | ⬜ not started |
| 4 | Better anomaly baseline | ⬜ not started |
| 5 | Retrieval correctness + eval | ⬜ not started |
| 6 | Tests + CI | ⬜ not started |
| — | Optional cleanup | ⬜ not started |

---

## Next up

**Task 0 — Unblock the pipeline.** Four defects, then `scripts/validate_e2e.py`
must pass end to end.

---

## Environment

The machine has no Docker, no WSL, and no system PostgreSQL, so the compose stack
could not be used. Rather than skip the database, local development now runs on an
embedded PostgreSQL:

| Component | How it runs locally | Notes |
|---|---|---|
| PostgreSQL 16.2 | `pgserver` wheel, embedded | No Docker required |
| pgvector 0.6.2 | Bundled with `pgserver` | **< 0.8, so `iterative_scan` is unavailable** — Task 5 must use partial HNSW indexes |
| LLM | Ollama `llama3.2` (local) | No Groq key configured; Groq stays the primary path when `GROQ_API_KEY` is set |
| Embeddings | Ollama `nomic-embed-text` | 768-dim, matches the existing schema |
| Kafka | **not running** | Needs Docker Desktop (admin install, pending). Only the live ingestion path depends on it — no Task 0–6 deliverable does. |

### Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
python -m scripts.devdb start      # embedded Postgres + schema, writes DATABASE_URL to .env
```

`scripts/devdb.py` (`start` / `url` / `stop` / `reset`) manages the embedded server.
It binds a dynamic port and rewrites `DATABASE_URL` in `.env`, so `config.DB_URL`
keeps working unchanged. The compose Postgres remains the deployment target — set
`DATABASE_URL` yourself and `devdb` is unnecessary.

### Dependency notes

- `llama-index-core` is held at **0.10.52**. 0.14.x removes `ReActAgent.from_tools`,
  which `agent/agent.py` is built on; upgrading means rewriting the agent onto the
  workflow API. 0.10.52 declares `numpy<2.0.0` but runs correctly on numpy 2.5.3,
  which `scipy`/`lightgbm` require. pip prints a resolver warning; it is benign and
  deliberate.
- `spacy` moved to 3.8.x — 3.7.5 has no Python 3.12 wheels.

---

## Open items requiring the user

- [ ] **Docker Desktop** — needs an admin install:
      `winget install --id Docker.DockerDesktop`. Unblocks Kafka and the live
      ingestion path only.
- [ ] **Groq API key** *(optional)* — `GROQ_API_KEY` in `.env` upgrades the agent
      from local `llama3.2` to a 70B model, which materially improves Task 3's
      LLM and hybrid arms.

---

## Decision log

Choices worth remembering, with the reasoning:

- **Embedded Postgres over skipping the DB.** Tasks 0–5 all need real SQL; mocking
  it would have made every measured number meaningless.
- **pgvector 0.6.2 constrains Task 5.** `iterative_scan` (0.8+) is not available, so
  the filtered-search fix must be partial HNSW indexes per `(model_name, model_version)`.
- **Ollama over waiting for a Groq key.** Keeps the whole pipeline runnable offline.
  A weak local model produces honest, modest LLM-arm numbers rather than no numbers.
