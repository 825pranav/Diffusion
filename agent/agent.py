"""
LlamaIndex ReAct agent for anomaly investigation.

Wakes via PostgreSQL LISTEN/NOTIFY on 'anomaly_detected', with a periodic
backlog sweep behind it so anomalies raised while the agent was unavailable are
still investigated rather than silently dropped.

For each anomaly event the agent:
  1. Runs a bounded ReAct loop with the investigation tools
  2. Applies the confidence gate (< threshold → needs_review, withheld from feed)
  3. Scores the output with Ragas
  4. Writes a structured case file to PostgreSQL
  5. Marks the anomaly event as investigated

An investigation that produces no valid output writes nothing and leaves the
anomaly uninvestigated, so the sweep retries it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime

import aiohttp
import asyncpg
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM

from agent.confidence import apply_gate
from agent.evaluator import evaluate_case
from agent.tools import build_tools
from agent.types import Emitter
from config import DB_URL, configure_logging
from graph import embeddings as emb
from graph.models import ANOMALY_NOTIFY_CHANNEL
from graph.queries import get_uninvestigated_anomalies, mark_anomaly_investigated

MAX_AGENT_STEPS = int(os.getenv("AGENT_MAX_STEPS", "12"))
OLLAMA_REQUEST_TIMEOUT = 120.0   # seconds before Ollama inference is abandoned
GROQ_RATE_LIMIT_SLEEP = 5        # seconds to wait between investigations to avoid 429s

# Backlog sweep — recovers anomalies that NOTIFY never delivered.
SWEEP_INTERVAL_SECONDS = int(os.getenv("AGENT_SWEEP_INTERVAL", "60"))
SWEEP_BATCH_SIZE = int(os.getenv("AGENT_SWEEP_BATCH", "50"))

GROQ_API_BASE = "https://api.groq.com/openai/v1"
GROQ_PROBE_TIMEOUT = 10  # seconds to decide whether Groq is usable

# Inference backend, resolved once on first use.
_llm: LLM | None = None

log = logging.getLogger(__name__)

_INVESTIGATION_PROMPT = (
    "You are an intelligence analyst investigating how a trending topic spread online.\n"
    "Gather evidence with the available tools. classify_virality_model is the "
    "strongest single piece of evidence when it is available — weigh it above raw "
    "counts, but do not ignore what the other tools show.\n"
    "Call each tool at most once, then answer. Do not repeat a tool call you have "
    "already made.\n"
    "After your investigation respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "classification": "organic" | "coordinated_amplification" | "uncertain",\n'
    '  "confidence": <float 0.0-1.0>,\n'
    '  "signals": ["signal 1", "signal 2"],\n'
    '  "reasoning_steps": ["step 1", "step 2"]\n'
    "}"
)


def _groq_model_available(api_key: str, model: str) -> bool:
    """
    Check that the key works and actually serves the configured model.

    Presence of a key proves nothing: it can be revoked, scoped, or name a model
    that has since been decommissioned. Without this probe the agent selects Groq,
    fails on every investigation, and never falls back — the local model sitting
    ready the whole time.
    """
    request = urllib.request.Request(
        f"{GROQ_API_BASE}/models", headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=GROQ_PROBE_TIMEOUT) as response:
            payload = json.load(response)
    except Exception as exc:
        # Never log the exception body — it can echo the Authorization header.
        log.warning("Groq unreachable (%s); using local model", type(exc).__name__)
        return False

    available = {entry.get("id") for entry in payload.get("data", [])}
    if model not in available:
        log.warning(
            "Groq model %r not available to this key; using local model", model
        )
        return False
    return True


def _build_llm() -> LLM:
    """
    Select the inference backend once per process and cache it.

    Groq is preferred when usable; otherwise the local Ollama model. Rebuilding
    per investigation would repeat the probe and construct a fresh client every
    time the agent wakes.
    """
    global _llm
    if _llm is not None:
        return _llm

    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if groq_key and _groq_model_available(groq_key, groq_model):
        from llama_index.llms.groq import Groq

        log.info("LLM: Groq (%s)", groq_model)
        _llm = Groq(model=groq_model, api_key=groq_key)
        return _llm

    from llama_index.llms.ollama import Ollama

    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    log.info("LLM: Ollama (%s)", chat_model)
    _llm = Ollama(
        model=chat_model,
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
    )
    return _llm


_REQUIRED_KEYS = {"classification", "confidence", "signals", "reasoning_steps"}
_VALID_CLASSIFICATIONS = {"organic", "coordinated_amplification", "uncertain"}


def _parse_agent_response(raw: str) -> dict:
    """
    Extract and validate the JSON object from the agent's final response.

    Strips markdown code fences, then falls back to the first {...} block.
    Raises ValueError if parsing fails or any required key is absent.
    Returns a dict with all four keys coerced to their expected types so
    the caller never needs defensive .get() with defaults.
    """
    text = raw.strip()

    # Prefer a fenced block (```json ... ``` or ``` ... ```)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    elif "{" in text:
        text = text[text.find("{") : text.rfind("}") + 1]

    parsed = json.loads(text)

    missing = _REQUIRED_KEYS - parsed.keys()
    if missing:
        raise ValueError(f"agent response missing keys: {missing}")

    classification = str(parsed["classification"])
    if classification not in _VALID_CLASSIFICATIONS:
        classification = "uncertain"

    return {
        "classification": classification,
        "confidence": float(parsed["confidence"]),
        "signals": list(parsed["signals"]),
        "reasoning_steps": list(parsed["reasoning_steps"]),
    }


async def run_investigation(
    conn: asyncpg.Connection,
    session: aiohttp.ClientSession,
    node_id: str,
    platform: str,
    z_score: float = 0.0,
    velocity: float = 0.0,
    emit: Emitter | None = None,
    include_model_tool: bool = True,
) -> dict:
    """
    Run one bounded ReAct investigation and return the parsed verdict.

    Reasoning only — no database writes and no case file. Separated from
    investigate() so ml/evaluate.py can score the agent over a held-out set
    without persisting hundreds of evaluation runs as real case files, and so
    the model tool can be withheld to measure the LLM on its own.

    Raises if the agent produces no parseable verdict; callers decide what that
    means.
    """
    tools = build_tools(conn, session, emit=emit, include_model_tool=include_model_tool)
    agent = ReActAgent.from_tools(
        tools, llm=_build_llm(), max_iterations=MAX_AGENT_STEPS, verbose=False
    )
    query = (
        f"{_INVESTIGATION_PROMPT}\n\n"
        f"Investigate node '{node_id}' on platform '{platform}'. "
        f"Z-score: {z_score:.2f}, velocity: {velocity:.2f}."
    )
    response = await agent.aquery(query)
    return _parse_agent_response(str(response))


async def investigate(
    conn: asyncpg.Connection,
    session: aiohttp.ClientSession,
    event: dict,
    emit: Emitter | None = None,
) -> None:
    node_id: str = event.get("node_id", "")
    platform: str = event.get("platform", "")
    anomaly_id: int | None = event.get("id")

    log.info(
        "investigating anomaly %s — node=%s platform=%s z=%.2f",
        anomaly_id, node_id, platform, event.get("z_score", 0.0),
    )

    if emit:
        await emit({
            "type": "started",
            "node_id": node_id,
            "platform": platform,
            "z_score": event.get("z_score", 0.0),
            "velocity": event.get("velocity", 0.0),
        })

    try:
        result = await run_investigation(
            conn,
            session,
            node_id=node_id,
            platform=platform,
            z_score=event.get("z_score", 0.0),
            velocity=event.get("velocity", 0.0),
            emit=emit,
        )
    except Exception:
        # Do not fabricate a case file.  Writing "uncertain / confidence 0.0" here
        # would publish an unsupported verdict and — because the anomaly would be
        # marked investigated — permanently hide the failure.  Leaving the row
        # uninvestigated lets the backlog sweep retry it.
        log.exception(
            "agent produced no valid output for anomaly %s — "
            "leaving uninvestigated for retry",
            anomaly_id,
        )
        if emit:
            await emit({"type": "failed", "reason": "agent produced no valid output"})
            await emit(None)
        return

    classification: str = result["classification"]
    confidence: float = result["confidence"]
    signals: list = result["signals"]
    reasoning_steps: list = result["reasoning_steps"]

    needs_review = apply_gate(confidence)

    try:
        similar = await emb.search_similar(conn, node_id, session, limit=3)
    except Exception:
        log.warning("embedding search unavailable (Ollama not running) — skipping similar cases")
        similar = []

    ragas_scores = await evaluate_case(
        trend=node_id,
        classification=classification,
        signals=signals,
        similar_cases=similar,
        reasoning_steps=reasoning_steps,
    )

    await conn.execute(
        """
        INSERT INTO case_files (
            anomaly_event_id, trend, platform_origin, detected_at,
            classification, confidence, signals, similar_past_cases,
            ragas_scores, agent_reasoning_steps, reasoning_steps_detail, needs_review
        ) VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, $8::jsonb, $9::jsonb, $10, $11::jsonb, $12)
        """,
        anomaly_id,
        node_id,
        platform,
        datetime.now(UTC),
        classification,
        confidence,
        json.dumps(signals),
        json.dumps(similar),
        json.dumps(ragas_scores),
        len(reasoning_steps),
        json.dumps(reasoning_steps),
        needs_review,
    )

    if anomaly_id is not None:
        await mark_anomaly_investigated(conn, anomaly_id)

    log.info(
        "case file written — node=%s classification=%s confidence=%.2f needs_review=%s",
        node_id, classification, confidence, needs_review,
    )

    if emit:
        await emit({
            "type": "completed",
            "classification": classification,
            "confidence": confidence,
            "needs_review": needs_review,
        })
        await emit(None)


async def agent_listener(emit_factory: Callable[[int], Emitter] | None = None) -> None:
    """
    Background coroutine — drives investigations from two sources.

    NOTIFY is the fast path: the anomaly detector's INSERT fires a trigger and the
    agent wakes immediately.  It is not durable, though — Postgres drops
    notifications with no live listener, so any anomaly raised while this process
    is down, restarting, or busy would be lost for good.

    The backlog sweep makes delivery durable by polling for rows still marked
    uninvestigated.  Together they give low latency when healthy and no permanent
    loss when not.  Both feed one queue, de-duplicated by anomaly id so a sweep
    cannot re-enqueue work that NOTIFY is already running.

    Can be embedded in the API lifespan or run standalone via main().
    emit_factory(anomaly_id) returns an emit callable for SSE streaming; pass None to disable.
    """
    pool = await asyncpg.create_pool(DB_URL, min_size=2, max_size=5)
    notify_conn = await asyncpg.connect(DB_URL)
    queue: asyncio.Queue[dict] = asyncio.Queue()

    # Anomaly ids queued or in flight.  Guards against the sweep re-enqueueing an
    # anomaly that NOTIFY already delivered but which is not yet marked investigated.
    inflight: set[int] = set()

    def _enqueue(event: dict) -> bool:
        anomaly_id = event.get("id")
        if anomaly_id is not None:
            if anomaly_id in inflight:
                return False
            inflight.add(anomaly_id)
        queue.put_nowait(event)
        return True

    def _on_notify(_conn, _pid, _channel, payload: str) -> None:
        try:
            _enqueue(json.loads(payload))
        except json.JSONDecodeError:
            log.warning("non-JSON notify payload: %.200s", payload)

    async def _sweep_loop() -> None:
        """Re-drive anomalies that NOTIFY never delivered, forever."""
        while True:
            try:
                async with pool.acquire() as conn:
                    backlog = await get_uninvestigated_anomalies(conn, limit=SWEEP_BATCH_SIZE)
                recovered = sum(1 for event in backlog if _enqueue(event))
                if recovered:
                    log.info("backlog sweep recovered %d uninvestigated anomalies", recovered)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("backlog sweep failed — retrying next tick")
            await asyncio.sleep(SWEEP_INTERVAL_SECONDS)

    await notify_conn.add_listener(ANOMALY_NOTIFY_CHANNEL, _on_notify)
    log.info(
        "agent listening on channel '%s' (backlog sweep every %ds)",
        ANOMALY_NOTIFY_CHANNEL, SWEEP_INTERVAL_SECONDS,
    )

    sweep_task = asyncio.create_task(_sweep_loop())
    try:
        async with aiohttp.ClientSession() as session:
            while True:
                event = await queue.get()
                anomaly_id = event.get("id")
                emit = (
                    emit_factory(anomaly_id)
                    if emit_factory and anomaly_id is not None
                    else None
                )
                try:
                    async with pool.acquire() as conn:
                        await investigate(conn, session, event, emit=emit)
                    await asyncio.sleep(GROQ_RATE_LIMIT_SLEEP)
                except Exception:
                    log.exception("investigation failed for event %s", anomaly_id)
                finally:
                    # Always release the id: a failed investigation leaves the row
                    # uninvestigated, and the next sweep should be free to retry it.
                    if anomaly_id is not None:
                        inflight.discard(anomaly_id)
    finally:
        sweep_task.cancel()
        await asyncio.gather(sweep_task, return_exceptions=True)
        await notify_conn.remove_listener(ANOMALY_NOTIFY_CHANNEL, _on_notify)
        await notify_conn.close()
        await pool.close()


async def main() -> None:
    await agent_listener()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
