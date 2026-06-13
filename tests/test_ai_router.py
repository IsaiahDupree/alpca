"""Unit tests for the AI router — all offline (no key, no network, no spend). Verifies routing,
credential gating, request shape, and that keys are never leaked into the built payload's logs."""

import json

import pytest

from alpca.ai.router import AIRouter, MissingCredential, DEFAULT_HAIKU, DEFAULT_OPENAI


def test_available_reflects_keys_without_revealing_them():
    r = AIRouter(anthropic_key="sk-ant-secret", openai_key=None)
    assert r.available() == {"haiku": True, "openai": False}
    # available() exposes booleans only, never the key string
    assert "secret" not in json.dumps(r.available())


def test_missing_credential_raises_clearly():
    r = AIRouter(anthropic_key=None, openai_key=None)
    with pytest.raises(MissingCredential):
        r.build_anthropic("hi")
    with pytest.raises(MissingCredential):
        r.build_openai("hi")


def test_anthropic_request_shape():
    r = AIRouter(anthropic_key="sk-ant-x", haiku_model="claude-haiku-4-5")
    url, headers, body = r.build_anthropic("propose an edge", system="be terse", max_tokens=256)
    assert url.endswith("/v1/messages")
    assert headers["x-api-key"] == "sk-ant-x" and headers["anthropic-version"]
    d = json.loads(body)
    assert d["model"] == "claude-haiku-4-5" and d["max_tokens"] == 256
    assert d["system"] == "be terse" and d["messages"][0]["content"] == "propose an edge"


def test_openai_request_shape():
    r = AIRouter(openai_key="sk-oai-y", openai_model="gpt-4o")
    url, headers, body = r.build_openai("critique this result", system="quant reviewer")
    assert url.endswith("/chat/completions")
    assert headers["Authorization"] == "Bearer sk-oai-y"
    d = json.loads(body)
    assert d["model"] == "gpt-4o"
    assert d["messages"][0] == {"role": "system", "content": "quant reviewer"}
    assert d["messages"][1]["content"] == "critique this result"


def test_route_picks_tier_by_weight(monkeypatch):
    r = AIRouter(anthropic_key="a", openai_key="o")
    calls = {}

    def fake_small(p, **k):
        calls["small"] = p
        return "S"

    def fake_think(p, **k):
        calls["think"] = p
        return "T"

    monkeypatch.setattr(r, "small", fake_small)
    monkeypatch.setattr(r, "think", fake_think)
    assert r.route("label this", heavy=False) == "S"
    assert r.route("design a falsification test", heavy=True) == "T"
    assert calls == {"small": "label this", "think": "design a falsification test"}


def test_defaults_present():
    assert DEFAULT_HAIKU.startswith("claude-haiku") and DEFAULT_OPENAI
