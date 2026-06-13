"""
Multi-model router for the AI strategy-research loop (docs/SYSTEM_MAP.md §6).

Two tiers, cost-aware:
  - small(...)  -> Haiku (Anthropic, cheap/fast): parse/label/summarize/classify, one-line verdicts.
  - think(...)  -> an OpenAI medium model (larger reasoning): propose hypotheses, design falsification
                   tests, critique a result for overfit, pick the next vein.

Keys come ONLY from the environment (config.py loads .env; `.env` is gitignored). Nothing here logs
or returns a key. If a key is absent, the call raises a clear MissingCredential — the loop degrades
gracefully (and can run its deterministic parts without the model). Request-building is separated from
sending so it is fully unit-testable WITHOUT a key or a network call (and without spending money).

No new dependencies — uses urllib. The HONEST-HARNESS still adjudicates every proposal: the model can
suggest anything, but nothing counts until it clears the fresh-symbol + out-of-regime holdout.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Optional, Tuple

from alpca.ai.oauth import get_oauth_token

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_HAIKU = "claude-haiku-4-5"          # cheap/fast tier (override via env ALPCA_HAIKU_MODEL)
DEFAULT_OPENAI = "gpt-4o"                    # medium reasoning tier (override via env ALPCA_OPENAI_MODEL)
# OAuth (Claude Code subscription) tokens require these on the Messages API:
OAUTH_BETA = "oauth-2025-04-20"
CLAUDE_CODE_SYSTEM = "You are Claude Code, Anthropic's official CLI for Claude."


class MissingCredential(RuntimeError):
    pass


@dataclass
class AIRouter:
    anthropic_key: Optional[str] = None
    openai_key: Optional[str] = None
    haiku_model: str = ""
    openai_model: str = ""

    # auth mode for Anthropic: "api_key" (x-api-key) or "oauth" (Claude Code subscription Bearer)
    anthropic_mode: str = ""

    def __post_init__(self):
        self.openai_key = self.openai_key or os.environ.get("OPENAI_API_KEY")
        self.haiku_model = self.haiku_model or os.environ.get("ALPCA_HAIKU_MODEL", DEFAULT_HAIKU)
        self.openai_model = self.openai_model or os.environ.get("ALPCA_OPENAI_MODEL", DEFAULT_OPENAI)
        # Anthropic auth: an explicit/ env API key wins; else fall back to the Claude Code OAuth token.
        if self.anthropic_key or os.environ.get("ANTHROPIC_API_KEY"):
            self.anthropic_key = self.anthropic_key or os.environ.get("ANTHROPIC_API_KEY")
            self.anthropic_mode = self.anthropic_mode or "api_key"
        elif not self.anthropic_mode:
            self.anthropic_mode = "oauth"     # token fetched lazily from the keychain at call time

    def _anthropic_token(self) -> Optional[str]:
        return self.anthropic_key if self.anthropic_mode == "api_key" else get_oauth_token()

    # ---- introspection (never reveals key values) ----
    def available(self) -> dict:
        return {"haiku": bool(self._anthropic_token()), "openai": bool(self.openai_key),
                "anthropic_mode": self.anthropic_mode}

    # ---- request builders (pure; testable without a key or network) ----
    def build_anthropic(self, prompt: str, *, system: str = "", max_tokens: int = 1024) -> Tuple[str, dict, bytes]:
        tok = self._anthropic_token()
        if not tok:
            raise MissingCredential("No Anthropic credential — set ANTHROPIC_API_KEY, or log in to "
                                    "Claude Code (the keychain OAuth token is used automatically).")
        body = {"model": self.haiku_model, "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": prompt}]}
        if self.anthropic_mode == "oauth":
            # subscription tokens require the Claude Code system prefix + the oauth beta header
            body["system"] = (CLAUDE_CODE_SYSTEM + ("\n\n" + system if system else ""))
            headers = {"Authorization": f"Bearer {tok}", "anthropic-version": "2023-06-01",
                       "anthropic-beta": OAUTH_BETA, "content-type": "application/json"}
        else:
            if system:
                body["system"] = system
            headers = {"x-api-key": tok, "anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
        return ANTHROPIC_URL, headers, json.dumps(body).encode()

    def build_openai(self, prompt: str, *, system: str = "", max_tokens: int = 2048) -> Tuple[str, dict, bytes]:
        if not self.openai_key:
            raise MissingCredential("OPENAI_API_KEY not set — add it to .env (gitignored) for the medium model.")
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.openai_model, "messages": msgs, "max_tokens": max_tokens}
        headers = {"Authorization": f"Bearer {self.openai_key}", "content-type": "application/json"}
        return OPENAI_URL, headers, json.dumps(body).encode()

    # ---- senders ----
    @staticmethod
    def _post(url: str, headers: dict, body: bytes, timeout: float) -> dict:
        req = urllib.request.Request(url, data=body, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def small(self, prompt: str, *, system: str = "", max_tokens: int = 1024, timeout: float = 60.0) -> str:
        """Cheap/fast Haiku call for high-volume small tasks."""
        url, headers, body = self.build_anthropic(prompt, system=system, max_tokens=max_tokens)
        d = self._post(url, headers, body, timeout)
        return "".join(b.get("text", "") for b in d.get("content", []))

    def think(self, prompt: str, *, system: str = "", max_tokens: int = 2048, timeout: float = 120.0) -> str:
        """Medium OpenAI model for heavier reasoning (hypotheses, overfit critique, next-vein choice)."""
        url, headers, body = self.build_openai(prompt, system=system, max_tokens=max_tokens)
        d = self._post(url, headers, body, timeout)
        return d.get("choices", [{}])[0].get("message", {}).get("content", "")

    def route(self, prompt: str, *, heavy: bool = False, **kw) -> str:
        """Pick the tier by task weight: heavy reasoning -> OpenAI medium; else Haiku."""
        return self.think(prompt, **kw) if heavy else self.small(prompt, **kw)
