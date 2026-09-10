"""Parsing the agent's final answer.

A local model rarely returns bare JSON: it wraps it in prose, fences it, or
invents a classification. Every one of those has to either parse or raise —
never half-populate a case file.
"""

from __future__ import annotations

import pytest

from agent.agent import _parse_agent_response

VALID = """{
  "classification": "organic",
  "confidence": 0.8,
  "signals": ["steady growth"],
  "reasoning_steps": ["checked the propagation path"]
}"""


def test_plain_json():
    result = _parse_agent_response(VALID)
    assert result["classification"] == "organic"
    assert result["confidence"] == 0.8
    assert result["signals"] == ["steady growth"]


def test_fenced_json_block():
    assert _parse_agent_response(f"```json\n{VALID}\n```")["classification"] == "organic"


def test_unlabelled_fence():
    assert _parse_agent_response(f"```\n{VALID}\n```")["classification"] == "organic"


def test_json_surrounded_by_prose():
    raw = f"Here is my analysis.\n{VALID}\nHope that helps!"
    assert _parse_agent_response(raw)["confidence"] == 0.8


def test_unknown_classification_becomes_uncertain():
    """An invented label must degrade to uncertain, not reach the database."""
    raw = VALID.replace('"organic"', '"definitely_bots"')
    assert _parse_agent_response(raw)["classification"] == "uncertain"


@pytest.mark.parametrize("missing", ["classification", "confidence", "signals", "reasoning_steps"])
def test_missing_key_raises(missing):
    import json

    payload = json.loads(VALID)
    payload.pop(missing)
    with pytest.raises(ValueError, match="missing keys"):
        _parse_agent_response(json.dumps(payload))


def test_prose_with_no_json_raises():
    # json.JSONDecodeError subclasses ValueError.
    with pytest.raises(ValueError):
        _parse_agent_response("I could not determine an answer.")


def test_types_are_coerced():
    raw = """{
      "classification": "coordinated_amplification",
      "confidence": "0.42",
      "signals": ["a"],
      "reasoning_steps": ["b"]
    }"""
    result = _parse_agent_response(raw)
    assert isinstance(result["confidence"], float)
    assert result["confidence"] == pytest.approx(0.42)
