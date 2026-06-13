"""Unit tests for the AI router — all offline (no key, no network, no spend). Verifies routing,
credential gating, request shape, and that keys are never leaked into the built payload's logs."""

import json

import pytest

from alpca.ai.router import AIRouter, MissingCredential, DEFAULT_HAIKU, DEFAULT_OPENAI


def test_available_reflects_keys_without_revealing_them(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)   # hermetic: ignore ambient .env
    r = AIRouter(anthropic_key="sk-ant-secret", openai_key=None)
    av = r.available()
    assert av["haiku"] is True and av["openai"] is False and av["anthropic_mode"] == "api_key"
    # available() exposes booleans/mode only, never the key string
    assert "secret" not in json.dumps(av)


def test_missing_credential_raises_clearly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # force api_key mode with no key so it doesn't fall back to the (real) keychain OAuth token
    r = AIRouter(anthropic_key=None, openai_key=None, anthropic_mode="api_key")
    with pytest.raises(MissingCredential):
        r.build_anthropic("hi")
    with pytest.raises(MissingCredential):
        r.build_openai("hi")


def test_oauth_mode_uses_bearer_beta_and_claude_code_prefix(monkeypatch):
    import alpca.ai.router as R
    monkeypatch.setattr(R, "get_oauth_token", lambda *a, **k: "sk-ant-oat-FAKE")
    r = AIRouter(anthropic_key=None, anthropic_mode="oauth")
    assert r.available()["anthropic_mode"] == "oauth" and r.available()["haiku"] is True
    url, headers, body = r.build_anthropic("classify this", system="be terse")
    assert headers["Authorization"] == "Bearer sk-ant-oat-FAKE"
    assert headers["anthropic-beta"] == R.OAUTH_BETA and "x-api-key" not in headers
    d = json.loads(body)
    assert d["system"].startswith(R.CLAUDE_CODE_SYSTEM) and "be terse" in d["system"]


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
