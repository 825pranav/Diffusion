"""
LlamaIndex tool-calling agent for anomaly investigation.

Wakes via PostgreSQL LISTEN/NOTIFY on 'anomaly_detected', with a periodic
backlog sweep behind it so anomalies raised while the agent was unavailable are
still investigated rather than silently dropped.

For each anomaly event the agent:
  1. Runs a bounded tool-calling loop with the investigation tools
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
from openai import AsyncOpenAI, BadRequestError

from agent.confidence import apply_gate
from agent.evaluator import evaluate_case
from agent.tools import build_tools
from agent.types import Emitter
from config import DB_URL, configure_logging
from graph import embeddings as emb
from graph.models import ANOMALY_NOTIFY_CHANNEL
from graph.queries import get_uninvestigated_anomalies, mark_anomaly_investigated

MAX_AGENT_STEPS = int(os.getenv("AGENT_MAX_STEPS", "12"))
# Ceiling on confidence when no tool found anything about the node.
NO_EVIDENCE_CONFIDENCE = 0.3
OLLAMA_REQUEST_TIMEOUT = 120.0   # seconds before Ollama inference is abandoned
GROQ_RATE_LIMIT_SLEEP = 5        # seconds to wait between investigations to avoid 429s

# Backlog sweep — recovers anomalies that NOTIFY never delivered.
SWEEP_INTERVAL_SECONDS = int(os.getenv("AGENT_SWEEP_INTERVAL", "60"))
SWEEP_BATCH_SIZE = int(os.getenv("AGENT_SWEEP_BATCH", "50"))

GROQ_API_BASE = "https://api.groq.com/openai/v1"
# llama-3.3-70b-versatile, the previous default, has been decommissioned.
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_PROBE_TIMEOUT = 10  # seconds to decide whether Groq is usable
# Cloudflare fronts the Groq API and 403s urllib's default agent (error 1010).
GROQ_USER_AGENT = "diffusion-agent/0.1 (+https://github.com/825pranav/diffusion)"

# Which backend to use: "groq", "ollama", or "auto" (Groq when its key works).
LLM_BACKEND = os.getenv("LLM_BACKEND", "auto").lower()
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Inference backend, resolved once on first use: an OpenAI-compatible client and
# the model name to send. Groq and Ollama both serve /v1/chat/completions with
# native tool calls, so one loop drives either.
_llm: tuple[AsyncOpenAI, str] | None = None

log = logging.getLogger(__name__)

_INVESTIGATION_PROMPT = (
    "You are an intelligence analyst investigating how a trending topic spread online.\n"
    "Gather evidence with the available tools. Anomalies usually fire on a topic "
    "node (an id starting 'entity:' or 'github:repo:'); for those, start with "
    "get_entity_activity to see who produced the mentions. classify_virality_model "
    "is the strongest single piece of evidence when it is available — weigh it "
    "above raw counts, but do not ignore what the other tools show.\n"
    "Call each tool at most once. Do not repeat a tool call you have already made.\n"
    "When you have the evidence, call submit_verdict with your verdict. If you "
    "cannot call it, reply with ONLY a JSON object in this exact format:\n"
    "{\n"
    '  "classification": "organic" | "coordinated_amplification" | "uncertain",\n'
    '  "confidence": <float 0.0-1.0>,\n'
    '  "signals": ["signal 1", "signal 2"],\n'
    '  "reasoning_steps": ["step 1", "step 2"]\n'
    "}"
)


_UNKNOWN_TOOL = (
    "That tool does not exist. Use only the tools listed, and call submit_verdict "
    "to answer."
)
_GATHER_FIRST = (
    "You have not gathered any evidence yet. Call the tools first, then answer."
)
_ANSWER_NOW = (
    "You have all the evidence the tools can give. Call submit_verdict now. If "
    "the tools found little or nothing, say "
    '"uncertain" with a low confidence rather than guessing.'
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
        f"{GROQ_API_BASE}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            # Groq sits behind Cloudflare, which rejects urllib's default
            # "Python-urllib/x.y" agent with a 403 and Cloudflare error 1010.
            # Without this the probe reads a perfectly valid key as unusable and
            # silently downgrades to the local model — the exact failure it
            # exists to prevent.
            "User-Agent": GROQ_USER_AGENT,
        },
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


def _build_llm() -> tuple[AsyncOpenAI, str]:
    """
    Select the inference backend once per process and cache it.

    LLM_BACKEND=auto prefers Groq when the key works and serves the model, and
    otherwise uses the local Ollama model. Rebuilding per investigation would
    repeat the probe and construct a fresh client every time the agent wakes.
    """
    global _llm
    if _llm is not None:
        return _llm

    groq_key = os.getenv("GROQ_API_KEY", "")
    groq_model = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    if LLM_BACKEND == "groq" or (
        LLM_BACKEND == "auto" and groq_key and _groq_model_available(groq_key, groq_model)
    ):
        log.info("LLM: Groq (%s)", groq_model)
        # The free tier allows 8,000 tokens a minute, about one investigation;
        # the client honours Groq's retry-after on a 429, so let it wait rather
        # than fail the investigation.
        _llm = (AsyncOpenAI(base_url=GROQ_API_BASE, api_key=groq_key, max_retries=8), groq_model)
        return _llm

    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen2.5:7b")
    log.info("LLM: Ollama (%s)", chat_model)
    _llm = (
        AsyncOpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama", timeout=OLLAMA_REQUEST_TIMEOUT),
        chat_model,
    )
    return _llm


def use_backend(name: str) -> str:
    """Pin the backend for this process ("groq" or "ollama"); returns the model name."""
    global _llm, LLM_BACKEND
    _llm, LLM_BACKEND = None, name
    return _build_llm()[1]


async def _tool_loop(tools: list, query: str) -> str:
    """
    Drive one investigation through native tool calls; return the final answer.

    The model asks for a tool through the API, the tool runs against the graph,
    and its output goes back as a `tool` message — so every number the model
    can cite came from a tool, not from text it wrote in a tool's place.
    """
    client, model = _build_llm()
    specs = [t.metadata.to_openai_tool() for t in tools] + [_VERDICT_TOOL]
    by_name = {t.metadata.name: t for t in tools}
    messages: list[dict] = [
        {"role": "system", "content": _INVESTIGATION_PROMPT},
        {"role": "user", "content": query},
    ]
    called: set[str] = set()
    told_to_answer = False
    for step in range(MAX_AGENT_STEPS):
        # Once every tool has run there is nothing left to learn, and near the
        # step limit there is no room to: tell the model to answer. Without
        # this, a node whose tools all come back empty kept the model calling
        # them until the limit. The tools stay on offer — gpt-oss on Groq calls
        # one anyway when none are listed, and the API rejects that with a 400.
        if not told_to_answer and (called >= by_name.keys() or step >= MAX_AGENT_STEPS - 2):
            messages.append({"role": "user", "content": _ANSWER_NOW})
            told_to_answer = True
        try:
            response = await client.chat.completions.create(
                model=model, messages=messages, tools=specs, tool_choice="auto", temperature=0
            )
        except BadRequestError as exc:
            # Groq validates tool calls server-side and rejects the whole
            # request when the model names a tool that does not exist (gpt-oss
            # reaches for built-ins it was trained with). Point it back at the
            # real tools instead of losing the investigation.
            if "tool_use_failed" not in str(exc):
                raise
            messages.append({"role": "user", "content": _UNKNOWN_TOOL})
            continue
        message = response.choices[0].message
        if not message.tool_calls:
            if called:
                return message.content or ""
            # An answer before any evidence is a guess; send it back once.
            messages.append({"role": "assistant", "content": message.content or ""})
            messages.append({"role": "user", "content": _GATHER_FIRST})
            continue
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [call.model_dump() for call in message.tool_calls],
        })
        for call in message.tool_calls:
            name = call.function.name
            tool = by_name.get(name)
            if name == "submit_verdict":
                if called:
                    return call.function.arguments or ""
                output = "rejected: no evidence yet. Call the evidence tools first."
            elif name in called:
                output = "already called; use the result above"
            else:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    output = (await tool.acall(**args)).content if tool else f"no tool named {name}"
                    if tool:
                        called.add(name)
                except Exception as exc:
                    # Not marked as called: the model gets the error and can
                    # retry with corrected arguments.
                    output = f"tool error: {type(exc).__name__}: {exc}"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
    raise ValueError(f"no answer within {MAX_AGENT_STEPS} steps")


_REQUIRED_KEYS = {"classification", "confidence", "signals", "reasoning_steps"}
_VALID_CLASSIFICATIONS = {"organic", "coordinated_amplification", "uncertain"}

# The verdict is itself a tool call. gpt-oss on Groq, told to answer in JSON
# while tools are on offer, "calls" a tool named json that does not exist, and
# Groq rejects the request outright; a real tool with the verdict's schema gives
# it a legitimate way to answer, and gives every model structured output instead
# of free text to parse.
_VERDICT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_verdict",
        "description": "Submit the final verdict once the evidence is in.",
        "parameters": {
            "type": "object",
            "properties": {
                "classification": {"type": "string", "enum": sorted(_VALID_CLASSIFICATIONS)},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "signals": {"type": "array", "items": {"type": "string"}},
                "reasoning_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": sorted(_REQUIRED_KEYS),
        },
    },
}


def _parse_agent_response(raw: str) -> dict:
    """
    Extract and validate the JSON object from the agent's final response.

    Strips markdown code fences, then falls back to the first {...} block.
    Raises ValueError if parsing fails or any required key is absent.
    Returns a dict with all four keys coerced to their expected types so
    the caller never needs defensive .get() with defaults.
    """
    # Reasoning models (qwen3 through Ollama) can lead with a <think> block,
    # whose braces would otherwise be mistaken for the start of the answer.
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

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
    Run one bounded tool-calling investigation and return the parsed verdict.

    Reasoning only — no database writes and no case file. Separated from
    investigate() so ml/evaluate.py can score the agent over a held-out set
    without persisting hundreds of evaluation runs as real case files, and so
    the model tool can be withheld to measure the LLM on its own.

    Raises if the agent produces no parseable verdict, or reaches one without a
    single tool result behind it; callers decide what that means.

    Tools are called natively, through the API's function-calling interface,
    not through the text ReAct format this used to rely on. That format needs
    the model to write "Thought:/Action:" lines while the prompt demands a bare
    JSON answer; every backend tested resolved the conflict by answering at
    once, so no tool ever ran and each verdict — "coordinated at 0.95" on a
    node with no edges at all — was invented. gpt-oss on Groq went further and
    wrote its own "Observation:" lines.
    """
    executed: list[str] = []
    found: list[bool] = []

    async def record(event: dict | None) -> None:
        if event and event.get("type") == "tool_result":
            executed.append(f"{event['tool']}: {event['summary']}")
            # Similar past trends describe other nodes, not this one.
            if event["tool"] != "search_similar_trends":
                found.append(not event.get("empty", False))
        if emit:
            await emit(event)

    tools = build_tools(conn, session, emit=record, include_model_tool=include_model_tool)
    query = (
        f"Investigate node '{node_id}' on platform '{platform}'. "
        f"Z-score: {z_score:.2f}, velocity: {velocity:.2f}."
    )
    response = await _tool_loop(tools, query)
    if not executed:
        raise UngroundedVerdict("the model answered without calling any tool")
    result = _parse_agent_response(response)
    if not any(found):
        # Every evidence tool came back empty. The models still answer —
        # qwen2.5 said "organic at 0.85" about a node with no edges — so the
        # verdict is held at uncertain, below the review gate.
        result["classification"] = "uncertain"
        result["confidence"] = min(result["confidence"], NO_EVIDENCE_CONFIDENCE)
        result["signals"] = ["the tools found no graph evidence for this node", *result["signals"]]
    result["evidence"] = executed
    return result


class UngroundedVerdict(ValueError):
    """The model returned a verdict with no tool result behind it."""


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
