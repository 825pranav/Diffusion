# Diffusion
**Distributed Trend Propagation Engine**

*It's not about what's trending. It's about how and why it spread.*

> New dashboard in `new-front` branch — bento layout, 4 palettes, full backend wiring.

---

## Overview

Diffusion ingests live signals from Reddit, Hacker News, and GitHub, models how information spreads as a directed propagation graph, and deploys an autonomous LLM agent that activates only when anomalous spread patterns are detected. The agent investigates the pattern, retrieves semantically similar historical cases via vector search, and produces a structured **case file** classifying whether a trend spread organically or was coordinated.

The core architectural decision: the agent does not poll. It sleeps until a statistically significant spike in propagation velocity triggers it. Everything downstream is a reaction to events, not a scheduled job.

---

## Architecture

```
┌─────────────────────────────────────────────┐
│           Ingestion Layer                   │
│   Reddit (PRAW) · HN Firebase · GitHub API  │
└──────────────────┬──────────────────────────┘
                   │ async producers
                   ▼
┌─────────────────────────────────────────────┐
│              Kafka                          │
│   reddit-raw · hn-raw · gh-raw              │
└──────────────────┬──────────────────────────┘
                   │ async consumers
                   ▼
┌─────────────────────────────────────────────┐
│         Stream Processing Layer             │
│   Deduplication → Entity Extraction         │
│   Velocity Scoring → Z-score Anomaly Det.   │
└──────────────────┬──────────────────────────┘
                   │ graph mutations + anomaly events
                   ▼
┌─────────────────────────────────────────────┐
│         PostgreSQL + pgvector               │
│   Propagation graph edges                   │
│   Historical trend embeddings               │
│   Case file store                           │
└──────────────────┬──────────────────────────┘
                   │ LISTEN/NOTIFY on anomaly
                   ▼
┌─────────────────────────────────────────────┐
│         LlamaIndex ReAct Agent              │
│   get_propagation_path                      │
│   search_similar_trends (pgvector)          │
│   classify_virality                         │
│   confidence gate → case file               │
│   Ragas evaluation on every output          │
└──────────────────┬──────────────────────────┘
                   │ REST · WebSocket · SSE
                   ▼
┌─────────────────────────────────────────────┐
│         FastAPI + Next.js Dashboard         │
│   Live trend timeline                       │
│   Agent thought stream (SSE)                │
│   Case file feed with confidence scores     │
└─────────────────────────────────────────────┘
```

---

## Tech Stack

| Technology | Role |
|---|---|
| **Kafka** | Event bus — streams raw signals from HN and GitHub |
| **PostgreSQL** | Source of truth — propagation graph edges, case files, metadata |
| **pgvector** | Vector search — semantic retrieval of historically similar trends |
| **LlamaIndex** | Agent orchestration — stateful ReAct loop with tool calling |
| **Groq** | LLM inference (primary) — LLaMA 3.3 70B |
| **Ollama** | LLM inference (fallback) — local, zero cost, graceful degradation |
| **Ragas** | Evaluation — retrieval relevance, reasoning consistency, confidence calibration |
| **FastAPI** | Backend — REST, WebSocket, and SSE endpoints |
| **Next.js** | Frontend dashboard |
| **Docker** | Full stack containerization via `docker compose up` |

---

## Key Design Decisions

**Event-driven, not poll-based.**
The agent wakes only when the anomaly detector identifies a z-score deviation greater than 2.5 over a 5-minute rolling window. During quiet periods, inference cost is zero.

**Graph modeling in PostgreSQL.**
Propagation relationships are stored as directed edges `(source_id, target_id, platform, timestamp)`. Degree, spread depth, and cascade size are computed via SQL — no dedicated graph DB required. Keeps the stack lean without sacrificing graph-theoretic reasoning.

**pgvector over a dedicated vector DB.**
Historical trend embeddings live in the same Postgres instance as relational data. Relational queries and vector search run in the same transaction. Similarity search can be joined directly with graph queries in a single statement.

**Provider-agnostic LLM inference.**
Groq is the primary provider (~200ms inference). If Groq rate-limits, the system automatically falls back to a local Ollama instance with no agent interruption. Fault tolerance is explicit, not assumed.

**Evaluation as a first-class concern.**
Every case file produced by the agent is scored by Ragas across three dimensions: retrieval relevance, reasoning consistency, and confidence calibration. Agent quality is measurable, not just observable.

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
    "Repost rate increased 340% within a 4-minute window",
    "87% user overlap with a known amplification cluster",
    "Cross-platform jump to Reddit r/programming within 6 minutes"
  ],
  "similar_past_cases": [
    {
      "trend": "OSS library X",
      "similarity": 0.91,
      "outcome": "confirmed_coordinated"
    }
  ],
  "ragas_scores": {
    "retrieval_relevance": 0.88,
    "reasoning_consistency": 0.82,
    "confidence_calibration": 0.79
  },
  "agent_reasoning_steps": 6
}
```

Case files with confidence below 0.65 are flagged for human review and not auto-published.

---

## Engineering Challenges

**Idempotent edge insertion under Kafka redelivery.**
Kafka guarantees at-least-once delivery. A consumer crash mid-processing redelivers the same event, creating duplicate edges in the propagation graph and corrupting cascade size calculations. Solved with a composite unique constraint on `(source_id, target_id, platform, timestamp)` and `INSERT ... ON CONFLICT DO NOTHING` on every edge write.

**Embedding consistency across model versions.**
Trends embedded months apart may use different sentence-transformer checkpoints, making cosine similarity scores unreliable across time. Solved by storing `model_name` and `model_version` alongside every embedding and scoping all pgvector similarity searches to matching versions only.

**Agent activation without a scheduler.**
Triggering the agent on anomaly without polling required a clean handoff between the stream processing layer and the agent. Solved with a Postgres `LISTEN/NOTIFY` channel — the anomaly detector writes to a notify channel, and the agent process wakes on receive. No external queue, no polling loop.

---

## Project Structure

```
diffusion/
├── ingestion/
│   ├── reddit_producer.py       # PRAW async Kafka producer
│   ├── hn_producer.py           # HN Firebase API producer
│   └── github_producer.py       # GitHub Events API producer
├── processing/
│   ├── consumer.py              # Async Kafka consumer
│   ├── dedup.py                 # Deduplication logic
│   ├── entity_extractor.py      # Node and edge extraction
│   ├── velocity_scorer.py       # Rolling 5-minute rate-of-change
│   └── anomaly_detector.py      # Z-score spike detection
├── graph/
│   ├── models.py                # PostgreSQL schema
│   ├── queries.py               # Propagation path, degree, cascade
│   └── embeddings.py            # pgvector indexing and search
├── agent/
│   ├── agent.py                 # LlamaIndex ReAct agent
│   ├── tools.py                 # Tool implementations
│   ├── confidence.py            # Confidence gate
│   └── evaluator.py             # Ragas evaluation pipeline
├── api/
│   ├── main.py                  # FastAPI application
│   ├── routes.py                # REST endpoints
│   ├── websocket.py             # Live graph delta stream
│   └── sse.py                   # Agent thought stream
├── frontend/                    # Next.js dashboard
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## Running Locally

**Prerequisites:** Docker Desktop, Python 3.11+, Node.js 18+

```bash
# Clone the repository
git clone https://github.com/yourusername/diffusion
cd diffusion

# Start infrastructure
docker compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Start the backend
uvicorn api.main:app --reload

# Start the frontend
cd frontend && npm run dev
```

Open `http://localhost:3000` to view the dashboard.

**Services started by Docker Compose:**
- Kafka + Zookeeper → `localhost:9092`
- PostgreSQL + pgvector → `localhost:5432`

**Validation and benchmarking:**

```bash
# Seed synthetic graph data and fire anomaly events (wakes the agent if running)
python scripts/seed.py --nodes 30 --edges 60 --anomalies 2

# End-to-end smoke test — injects an anomaly and waits for the agent to produce a case file
python scripts/validate_e2e.py --timeout 120

# Latency benchmark — p50/p95/p99 across all read endpoints
python scripts/benchmark.py --requests 200 --concurrency 10
```

---

## Status

| Stage | Description | Status |
|---|---|---|
| 1 | Ingestion pipeline — Kafka producers for Reddit, HN, GitHub | ✅ Done |
| 2 | Stream processing — dedup, entity extraction, anomaly detection | ✅ Done |
| 3 | Graph + vector layer — PostgreSQL schema, pgvector embeddings | ✅ Done |
| 4 | Agent layer — LlamaIndex ReAct, tools, Ragas evaluation | ✅ Done |
| 5a | API — FastAPI REST, WebSocket graph delta stream, SSE agent thought stream | ✅ Done |
| 5b | Frontend — Next.js dashboard (live graph, case feed, SSE viewer) | ✅ Done |
| 6 | Benchmarks, end-to-end validation, performance profiling | ✅ Done |

---

## License

MIT
