"""
Rolling realized volatility from OHLC bars — the σ input every Almgren-style
market-impact fit needs (impact ~ η·σ·|x|^β). Pure-python, dependency-free.

Reimplemented clean from the standard realized-vol / Almgren methodology (the
shubhamcodez/Market-Impact-Model repo has no LICENSE; the math is public). We use
close-to-close log returns (captures the real price path incl. overnight gaps),
take the population stdev over a trailing window, and annualize by
sqrt(bars_per_year). bars_per_day is inferred from the bar timestamps (median bars
per ET session date) so the same code annualizes 1-min and daily bars correctly.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Dict, List, Optional

_TRADING_DAYS = 252
_DEFAULT_PRIOR = 0.15  # annualized σ fallback when there is too little history


def _infer_bars_per_day(bars: List[dict]) -> Optional[float]:
    from alpca.data.calendar import session_date
    counts: Dict[str, int] = {}
    for b in bars:
        ts = float(b.get("timestamp", 0) or 0)
        if ts <= 0:
            continue
        d = session_date(ts)
        counts[d] = counts.get(d, 0) + 1
    vals = sorted(counts.values())
    if len(vals) >= 2:
        return float(statistics.median(vals))
    return None


def _window_bars(bars: List[dict], lookback_days: float, bars_per_day: Optional[float]) -> int:
    bpd = bars_per_day or _infer_bars_per_day(bars) or float(len(bars))
    return max(2, int(round(lookback_days * bpd))), bpd


def compute_rolling_volatility(
    bars: List[dict],
    lookback_days: float = 10.0,
    *,
    annualize: bool = True,
    bars_per_day: Optional[float] = None,
    prior: float = _DEFAULT_PRIOR,
) -> float:
    """
    Annualized realized σ over the most recent `lookback_days` of bars. Returns
    `prior` when there are fewer than 2 usable returns. `bars_per_day` overrides
    the inferred cadence (use it for synthetic/integer-timestamp bars).
    """
    if not bars:
        return prior
    window, bpd = _window_bars(bars, lookback_days, bars_per_day)
    closes = [float(b["close"]) for b in bars[-(window + 1):]]
    rets = [math.log(closes[i] / closes[i - 1])
            for i in range(1, len(closes)) if closes[i - 1] > 0 and closes[i] > 0]
    if len(rets) < 2:
        return prior
    sigma = statistics.pstdev(rets)
    return sigma * math.sqrt(bpd * _TRADING_DAYS) if annualize else sigma


def build_vol_series(
    bars: List[dict],
    lookback_days: float = 10.0,
    *,
    annualize: bool = True,
    bars_per_day: Optional[float] = None,
    prior: float = _DEFAULT_PRIOR,
) -> Dict[float, float]:
    """
    {bar_timestamp -> trailing annualized σ at that bar}. O(n) via a sliding window
    of running sum/sum-of-squares, so it is safe on large (multi-year 1-min) series.
    Bars with <2 returns in their window get `prior`.
    """
    window, bpd = _window_bars(bars, lookback_days, bars_per_day)
    ann = math.sqrt(bpd * _TRADING_DAYS) if annualize else 1.0
    out: Dict[float, float] = {}
    w: deque = deque(maxlen=window)
    s = 0.0
    s2 = 0.0
    prev_close: Optional[float] = None
    for b in bars:
        ts = float(b.get("timestamp", 0) or 0)
        c = float(b["close"])
        if prev_close is not None and prev_close > 0 and c > 0:
            r = math.log(c / prev_close)
            if len(w) == w.maxlen:           # evict oldest before it drops off
                old = w[0]
                s -= old
                s2 -= old * old
            w.append(r)
            s += r
            s2 += r * r
        prev_close = c
        n = len(w)
        if n < 2:
            out[ts] = prior
        else:
            var = max(0.0, s2 / n - (s / n) ** 2)   # population variance
            out[ts] = math.sqrt(var) * ann
    return out
