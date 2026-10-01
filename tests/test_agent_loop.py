"""Agent tool loop — native tool calls against a scripted model, no services needed."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import agent.agent as A
from agent.tools import build_tools


def _call(name: str, args: dict, call_id: str = "c1"):
    fn = SimpleNamespace(name=name, arguments=json.dumps(args))
    return SimpleNamespace(id=call_id, type="function", function=fn,
                           model_dump=lambda: {"id": call_id, "type": "function",
                                               "function": {"name": name, "arguments": json.dumps(args)}})


def _reply(content: str = "", calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=calls))])


class _ScriptedModel:
    """Returns the scripted replies in order and records every request."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append(kwargs)
        return self.replies.pop(0)


VERDICT = json.dumps({"classification": "organic", "confidence": 0.7,
                      "signals": ["p_coordinated=0.2"], "reasoning_steps": ["scored it"]})


def _tools(results: list):
    """Real FunctionTools whose bodies record what ran."""
    from llama_index.core.tools import FunctionTool
    from llama_index.core.tools.utils import create_schema_from_function

    async def classify_virality_model(node_id: str) -> str:
        results.append(node_id)
        return json.dumps({"p_coordinated": 0.2})

    return [FunctionTool.from_defaults(
        fn=lambda **_: None, async_fn=classify_virality_model, name="classify_virality_model",
        description="score", fn_schema=create_schema_from_function("classify_virality_model", classify_virality_model),
    )]


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    A._llm = None


def test_tools_advertise_their_real_parameters():
    """The schema used to come from a *args/**kwargs stub, hiding `node_id` from every model."""
    for tool in build_tools(None, None):
        params = tool.metadata.to_openai_tool()["function"]["parameters"]
        assert "args" not in params["properties"] and "kwargs" not in params["properties"]
    model_tool = {t.metadata.name: t for t in build_tools(None, None)}["classify_virality_model"]
    assert model_tool.metadata.to_openai_tool()["function"]["parameters"]["required"] == ["node_id"]


def test_an_answer_before_any_evidence_is_sent_back():
    ran: list = []
    model = _ScriptedModel([
        _reply(VERDICT),                                                   # guesses at once
        _reply(calls=[_call("classify_virality_model", {"node_id": "n"})]),  # told to gather first
        _reply(VERDICT),
    ])
    A._llm = (model, "m")
    assert asyncio.run(A._tool_loop(_tools(ran), "investigate n")) == VERDICT
    assert ran == ["n"]
    assert any(m["content"] == A._GATHER_FIRST for m in model.requests[1]["messages"] if m["role"] == "user")


def test_a_failed_call_can_be_retried():
    ran: list = []
    model = _ScriptedModel([
        _reply(calls=[_call("classify_virality_model", {"node": "n"})]),     # wrong argument name
        _reply(calls=[_call("classify_virality_model", {"node_id": "n"})]),  # corrected
        _reply(VERDICT),
    ])
    A._llm = (model, "m")
    asyncio.run(A._tool_loop(_tools(ran), "investigate n"))
    assert ran == ["n"]
    tool_msgs = [m["content"] for m in model.requests[1]["messages"] if m["role"] == "tool"]
    assert tool_msgs[0].startswith("tool error")


def test_once_every_tool_has_run_the_model_is_told_to_answer():
    ran: list = []
    model = _ScriptedModel([
        _reply(calls=[_call("classify_virality_model", {"node_id": "n"})]),
        _reply(VERDICT),
    ])
    A._llm = (model, "m")
    asyncio.run(A._tool_loop(_tools(ran), "investigate n"))
    last_user = [m for m in model.requests[1]["messages"] if m["role"] == "user"][-1]
    assert last_user["content"] == A._ANSWER_NOW
    assert model.requests[1]["tools"]  # still offered: gpt-oss calls one anyway otherwise


def test_a_verdict_with_no_tool_result_behind_it_is_rejected(monkeypatch):
    async def fake_loop(tools, query):
        return VERDICT  # an answer, but no tool ever emitted a result

    monkeypatch.setattr(A, "_tool_loop", fake_loop)
    with pytest.raises(A.UngroundedVerdict):
        asyncio.run(A.run_investigation(None, None, "n", "bluesky"))


def test_think_blocks_are_ignored_when_parsing():
    raw = "<think>maybe {not this}</think>\n" + VERDICT
    assert A._parse_agent_response(raw)["classification"] == "organic"


def test_the_verdict_arrives_through_its_own_tool():
    ran: list = []
    model = _ScriptedModel([
        _reply(calls=[_call("classify_virality_model", {"node_id": "n"})]),
        _reply(calls=[_call("submit_verdict", json.loads(VERDICT), "c2")]),
    ])
    A._llm = (model, "m")
    out = asyncio.run(A._tool_loop(_tools(ran), "investigate n"))
    assert A._parse_agent_response(out)["classification"] == "organic"
    assert any(spec["function"]["name"] == "submit_verdict" for spec in model.requests[0]["tools"])


def test_a_verdict_submitted_before_any_evidence_is_refused():
    ran: list = []
    model = _ScriptedModel([
        _reply(calls=[_call("submit_verdict", json.loads(VERDICT))]),        # straight to a verdict
        _reply(calls=[_call("classify_virality_model", {"node_id": "n"}, "c2")]),
        _reply(calls=[_call("submit_verdict", json.loads(VERDICT), "c3")]),
    ])
    A._llm = (model, "m")
    asyncio.run(A._tool_loop(_tools(ran), "investigate n"))
    first_reply = [m for m in model.requests[1]["messages"] if m["role"] == "tool"][0]
    assert first_reply["content"].startswith("rejected")
    assert ran == ["n"]


def test_a_rejected_tool_name_is_recovered_from():
    """Groq answers a made-up tool name with HTTP 400 for the whole request."""
    import httpx
    from openai import BadRequestError

    rejected = BadRequestError(
        "Tool call validation failed: attempted to call tool 'json' (tool_use_failed)",
        response=httpx.Response(400, request=httpx.Request("POST", "http://groq")),
        body=None,
    )
    ran: list = []
    replies = [
        _reply(calls=[_call("classify_virality_model", {"node_id": "n"})]),
        rejected,
        _reply(calls=[_call("submit_verdict", json.loads(VERDICT), "c2")]),
    ]

    class _Flaky(_ScriptedModel):
        async def _create(self, **kwargs):
            self.requests.append(kwargs)
            reply = self.replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

    model = _Flaky(replies)
    A._llm = (model, "m")
    out = asyncio.run(A._tool_loop(_tools(ran), "investigate n"))
    assert A._parse_agent_response(out)["classification"] == "organic"
    assert any(m.get("content") == A._UNKNOWN_TOOL for m in model.requests[2]["messages"])


def _emitting_tools(emit, empty: bool):
    from llama_index.core.tools import FunctionTool

    async def classify_virality_model(node_id: str) -> str:
        await emit({"type": "tool_result", "tool": "classify_virality_model", "summary": "x", "empty": empty})
        return "{}"

    return [FunctionTool.from_defaults(fn=lambda **_: None, async_fn=classify_virality_model,
                                       name="classify_virality_model", description="score")]


@pytest.mark.parametrize("empty", [True, False])
def test_a_verdict_on_no_evidence_is_held_at_uncertain(monkeypatch, empty):
    confident = json.dumps({"classification": "organic", "confidence": 0.85,
                            "signals": ["looks organic"], "reasoning_steps": ["guessed"]})
    monkeypatch.setattr(A, "build_tools", lambda conn, session, emit, include_model_tool: _emitting_tools(emit, empty))

    async def run_tools_then_answer(tools, query):
        for tool in tools:
            await tool.acall(node_id="n")
        return confident

    monkeypatch.setattr(A, "_tool_loop", run_tools_then_answer)
    result = asyncio.run(A.run_investigation(None, None, "n", "bluesky"))
    if empty:
        assert result["classification"] == "uncertain"
        assert result["confidence"] <= A.NO_EVIDENCE_CONFIDENCE
        assert result["signals"][0].startswith("the tools found no graph evidence")
    else:
        assert (result["classification"], result["confidence"]) == ("organic", 0.85)


def test_a_malformed_verdict_is_sent_back_for_another_try():
    ran: list = []
    broken = _call("submit_verdict", {}, "c2")
    broken.function.arguments = '{"classification": "organic", confidence: 0.7}'  # not JSON
    model = _ScriptedModel([
        _reply(calls=[_call("classify_virality_model", {"node_id": "n"})]),
        _reply(calls=[broken]),
        _reply(calls=[_call("submit_verdict", json.loads(VERDICT), "c3")]),
    ])
    A._llm = (model, "m")
    out = asyncio.run(A._tool_loop(_tools(ran), "investigate n"))
    assert A._parse_agent_response(out)["classification"] == "organic"
    retry_note = [m for m in model.requests[2]["messages"] if m["role"] == "tool"][-1]["content"]
    assert retry_note.startswith("rejected")
