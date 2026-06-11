"""
Shared panel alignment for multi-symbol backtests.

Several market-neutral / cross-asset strategies (PCA stat-arb, TSMOM, cross-sectional)
need the same first step: take a dict of {symbol: bars}, intersect timestamps across the
chosen symbols, and return a clean (T x N) simple-returns matrix on the common date index.
This was duplicated per module; consolidating it here keeps the alignment semantics
(intersection, >0 prices, simple returns, no look-ahead) identical everywhere.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def aligned_returns(bars_by_sym: Dict[str, List[dict]], symbols: Optional[List[str]] = None,
                    *, min_len: int = 0) -> Tuple[List[str], np.ndarray, List[int]]:
    """Return (syms, R, ts): R is a (T x N) simple-returns matrix on the timestamps common
    to all included symbols; ts is that common date index (length T, aligned to R rows).

    symbols: restrict to these (default: every symbol with >= min_len bars).
    min_len:  drop symbols with fewer bars before intersecting.
    Empty/degenerate inputs return ([], 0x0 array, [])."""
    if symbols is None:
        usable = {s: b for s, b in bars_by_sym.items() if len(b) >= min_len}
    else:
        usable = {s: bars_by_sym[s] for s in symbols if s in bars_by_sym and len(bars_by_sym[s]) >= min_len}
    if len(usable) < 2:
        return [], np.zeros((0, 0)), []

    maps: Dict[str, Dict[int, float]] = {}
    common = None
    for s, bars in usable.items():
        m = {int(b["timestamp"]): float(b["close"]) for b in bars if float(b.get("close", 0)) > 0}
        maps[s] = m
        common = set(m) if common is None else (common & set(m))
    ts = sorted(common or [])
    if len(ts) < 2:
        return [], np.zeros((0, 0)), []

    syms = sorted(usable)
    P = np.array([[maps[s][t] for s in syms] for t in ts])   # (T+1 x N) prices
    R = (P[1:] - P[:-1]) / P[:-1]                              # (T x N) simple returns
    return syms, R, ts[1:]
