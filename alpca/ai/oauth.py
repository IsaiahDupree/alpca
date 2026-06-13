"""
Claude Code OAuth credential helper — reuse the token the local Claude Code login already holds
(same mechanism the ACD project relies on) so the AI router can call Haiku without a separate API key.

Precedence: env CLAUDE_CODE_OAUTH_TOKEN > macOS Keychain ("Claude Code-credentials") with auto-refresh.
The token rotates ~every 8h; when expired we refresh via the Claude Code public OAuth client and write
the rotated credentials back to the Keychain (exactly what Claude Code does), so its own login stays
healthy. NOTHING here logs or returns a token to anywhere it could be persisted in the repo.

This is read-mostly: if the token is still valid we touch nothing; we only write back the Keychain
after a SUCCESSFUL refresh (a failed/blocked refresh leaves the Keychain untouched).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.request
from typing import Optional

KEYCHAIN_SERVICE = "Claude Code-credentials"
KEYCHAIN_ACCOUNT = "Claude Code"
OAUTH_TOKEN_URL = "https://api.anthropic.com/v1/oauth/token"
OAUTH_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"   # Claude Code's public OAuth client id
_REFRESH_HEADERS = {"content-type": "application/json", "accept": "application/json",
                    "user-agent": "claude-cli/1.0 (external, cli)"}   # bare UA hits Cloudflare 1010


def _read_keychain() -> Optional[dict]:
    r = subprocess.run(["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    try:
        return json.loads(r.stdout.strip()).get("claudeAiOauth")
    except (ValueError, KeyError):
        return None


def _write_keychain(oa: dict) -> None:
    subprocess.run(["security", "add-generic-password", "-U", "-s", KEYCHAIN_SERVICE,
                    "-a", KEYCHAIN_ACCOUNT, "-w", json.dumps({"claudeAiOauth": oa})], check=True)


def _refresh(oa: dict, timeout: float = 30.0) -> dict:
    body = json.dumps({"grant_type": "refresh_token", "refresh_token": oa["refreshToken"],
                       "client_id": OAUTH_CLIENT_ID}).encode()
    req = urllib.request.Request(OAUTH_TOKEN_URL, data=body, headers=_REFRESH_HEADERS)
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    new = {"accessToken": r["access_token"],
           "refreshToken": r.get("refresh_token", oa["refreshToken"]),
           "expiresAt": int((time.time() + r.get("expires_in", 3600)) * 1000)}
    _write_keychain(new)            # only reached on a SUCCESSFUL refresh
    return new


def get_oauth_token(skew_s: int = 120) -> Optional[str]:
    """A valid Claude Code OAuth access token, or None. Env override first, else Keychain (auto-refresh)."""
    env = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN")
    if env:
        return env
    oa = _read_keychain()
    if not oa:
        return None
    if oa.get("expiresAt", 0) / 1000 <= time.time() + skew_s:
        try:
            oa = _refresh(oa)
        except Exception:
            return None             # Keychain untouched; caller degrades gracefully
    return oa.get("accessToken")
