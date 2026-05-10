"""
LlamaIndex ReAct agent for anomaly investigation.

Wakes via PostgreSQL LISTEN/NOTIFY on 'anomaly_detected'.
For each anomaly event the agent:
  1. Runs a bounded ReAct loop with the three investigation tools
  2. Applies the confidence gate (< 0.65 → needs_review, withheld from feed)
  3. Scores the output with Ragas
  4. Writes a structured case file to PostgreSQL
  5. Marks the anomaly event as investigated
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Callable

import aiohttp
import asyncpg
from llama_index.core.agent import ReActAgent
from llama_index.core.llms import LLM

from config import DB_URL, configure_logging
from agent.confidence import apply_gate
from agent.evaluator import evaluate_case
from agent.tools import build_tools
from agent.types import Emitter
from graph import embeddings as emb
from graph.models import ANOMALY_NOTIFY_CHANNEL
from graph.queries import mark_anomaly_investigated

MAX_AGENT_STEPS = int(os.getenv("AGENT_MAX_STEPS", "12"))
OLLAMA_REQUEST_TIMEOUT = 120.0   # seconds before Ollama inference is abandoned
GROQ_RATE_LIMIT_SLEEP = 5        # seconds to wait between investigations to avoid 429s

log = logging.getLogger(__name__)

_INVESTIGATION_PROMPT = (
    "You are an intelligence analyst investigating how a trending topic spread online.\n"
    "Use get_propagation_path, search_similar_trends, and classify_virality to gather evidence.\n"
    "After your investigation respond with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "classification": "organic" | "coordinated_amplification" | "uncertain",\n'
    '  "confidence": <float 0.0-1.0>,\n'
    '  "signals": ["signal 1", "signal 2"],\n'
    '  "reasoning_steps": ["step 1", "step 2"]\n'
    "}"
)


def _build_llm() -> LLM:
    groq_key = os.getenv("GROQ_API_KEY", "")
    if groq_key:
        from llama_index.llms.groq import Groq
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        log.info("LLM: Groq (%s)", model)
        return Groq(model=model, api_key=groq_key)
    from llama_index.llms.ollama import Ollama
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    log.info("LLM: Ollama fallback (%s)", chat_model)
    return Ollama(
        model=chat_model,
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
        request_timeout=OLLAMA_REQUEST_TIMEOUT,
    )


def _parse_agent_response(raw: str) -> dict:
    """Strip markdown fences and parse JSON from the agent's final response."""
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]
    return json.loads(text)


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

    tools = build_tools(conn, session, emit=emit)
    agent = ReActAgent.from_tools(tools, llm=_build_llm(), max_iterations=MAX_AGENT_STEPS, verbose=False)

    query = (
        f"{_INVESTIGATION_PROMPT}\n\n"
        f"Investigate node '{node_id}' on platform '{platform}'. "
        f"Z-score: {event.get('z_score', 0.0):.2f}, velocity: {event.get('velocity', 0.0):.2f}."
    )

    try:
        response = await agent.aquery(query)
        result = _parse_agent_response(str(response))
    except Exception:
        log.exception("agent failed to produce valid output for anomaly %s", anomaly_id)
        result = {"classification": "uncertain", "confidence": 0.0, "signals": [], "reasoning_steps": []}

    classification: str = result.get("classification", "uncertain")
    confidence: float = float(result.get("confidence", 0.0))
    signals: list = result.get("signals", [])
    reasoning_steps: list = result.get("reasoning_steps", [])

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
        datetime.now(timezone.utc),
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
    Background coroutine — LISTENs on 'anomaly_detected' and runs investigations.
    Can be embedded in the API lifespan or run standalone via main().
    emit_factory(anomaly_id) returns an emit callable for SSE streaming; pass None to disable.
    """
    pool = await asyncpg.create_pool(DB_URL, min_size=2, max_size=5)
    notify_conn = await asyncpg.connect(DB_URL)
    queue: asyncio.Queue[dict] = asyncio.Queue()

    def _on_notify(_conn, _pid, _channel, payload: str) -> None:
        try:
            event = json.loads(payload)
            queue.put_nowait(event)
        except json.JSONDecodeError:
            log.warning("non-JSON notify payload: %.200s", payload)

    await notify_conn.add_listener(ANOMALY_NOTIFY_CHANNEL, _on_notify)
    log.info("agent listening on channel '%s'", ANOMALY_NOTIFY_CHANNEL)

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
                async with pool.acquire() as conn:
                    try:
                        await investigate(conn, session, event, emit=emit)
                        await asyncio.sleep(GROQ_RATE_LIMIT_SLEEP)
                    except Exception:
                        log.exception("investigation failed for event %s", anomaly_id)
    finally:
        await notify_conn.remove_listener(ANOMALY_NOTIFY_CHANNEL, _on_notify)
        await notify_conn.close()
        await pool.close()


async def main() -> None:
    await agent_listener()


if __name__ == "__main__":
    configure_logging()
    asyncio.run(main())
