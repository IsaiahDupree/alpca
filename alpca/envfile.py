"""
Minimal .env loader that injects KEY=VALUE pairs into os.environ.

Used so credentials can stay in an existing protected file (e.g. another
project's .env) and be loaded into THIS process only — never copied to a new
file, never printed. Existing environment values win unless override=True.
"""

from __future__ import annotations

import os
from typing import List


def load_env_file(path: str, *, override: bool = False) -> List[str]:
    """Load KEY=VALUE lines from `path` into os.environ. Returns the KEY names set."""
    loaded: List[str] = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if not key:
                continue
            if override or key not in os.environ:
                os.environ[key] = val
                loaded.append(key)
    return loaded
