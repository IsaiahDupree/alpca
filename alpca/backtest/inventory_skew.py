"""
Avellaneda-Stoikov INVENTORY-SKEW sizing, ported honestly to a price-TAKER on Alpaca.

We cannot quote (no L2, no rebates, ~1.2s fills — market making is infeasible here; see
docs/Alpca_Discovery_and_HFT_Assessment). But the A-S *reservation price* carries one
idea that survives the venue change: the optimal inventory is INVERSELY proportional to
risk-adjusted mispricing. From r = s − q·γ·σ²·(T−t), setting the reservation price to a
fair value μ and solving for the indifference inventory gives

    q* = (μ − s) / (γ · σ² · (T−t))        [clipped to ±max_pos]

i.e. hold a CONTINUOUS, VOL-SCALED position proportional to how far price sits below a
rolling fair value, shrunk by risk-aversion γ and by variance σ². This is a sizing layer,
NOT an alpha source — so we test it honestly two ways:
  (1) does A-S continuous vol-scaled sizing beat a naive BINARY z-score entry on the same
      signal (does the continuous/inventory-aware part add anything)?
  (2) does either beat BUY-AND-HOLD, out-of-sample + significant + stable?

The binary runner (backtest_resting) sizes every position to a fixed notional, so it
cannot represent a continuous target. This module is a small continuous-position
backtester whose equity curve is judged by the SAME harness primitives (sharpe_of,
sharpe_tstat, segment_sharpes, buy_and_hold) used everywhere else.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


def _closes(bars: List[dict]) -> List[float]:
    return [float(b["close"]) for b in bars]


def rolling_mean_var(closes: List[float], window: int) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """Trailing mean of PRICE (fair-value proxy μ) and trailing variance of simple
    RETURNS (σ², the A-S risk term), both as of bar t using the prior `window` bars.
    None during warmup. O(n) one-pass with running sums."""
    n = len(closes)
    mu: List[Optional[float]] = [None] * n
    var: List[Optional[float]] = [None] * n
    rets = [0.0] + [(closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] else 0.0
                    for i in range(1, n)]
    for t in range(n):
        if t < window:
            continue
        price_win = closes[t - window:t]
        mu[t] = sum(price_win) / window
        rw = rets[t - window + 1:t + 1]
        m = sum(rw) / len(rw)
        var[t] = sum((r - m) ** 2 for r in rw) / len(rw)
    return mu, var


def as_target(closes: List[float], *, window: int = 20, gamma: float = 1.0,
              horizon: float = 1.0, max_pos: float = 1.0,
              var_floor: float = 1e-6) -> List[float]:
    """A-S indifference inventory q* = (μ−s)/(γ·σ²·horizon), clipped to ±max_pos.
    Signed fraction-of-equity target per bar (0 during warmup). Long when price is
    below the rolling fair value, scaled down by risk-aversion and variance."""
    mu, var = rolling_mean_var(closes, window)
    out = [0.0] * len(closes)
    for t in range(len(closes)):
        if mu[t] is None or var[t] is None:
            continue
        v = max(var[t], var_floor)
        s = closes[t]
        q = (mu[t] - s) / (s * gamma * v * horizon)   # normalize mispricing by price -> return units
        out[t] = max(-max_pos, min(max_pos, q))
    return out


def binary_target(closes: List[float], *, window: int = 20, entry_z: float = 1.0,
                  exit_z: float = 0.25, max_pos: float = 1.0) -> List[float]:
    """Naive baseline: discrete z-score mean reversion. Full ±max_pos when |z|>=entry_z,
    flat when |z|<=exit_z, hold otherwise. Same signal, no inventory/vol scaling."""
    n = len(closes)
    out = [0.0] * n
    pos = 0.0
    for t in range(n):
        if t < window:
            continue
        win = closes[t - window:t]
        m = sum(win) / window
        sd = (sum((x - m) ** 2 for x in win) / window) ** 0.5
        if sd <= 0:
            out[t] = pos
            continue
        z = (closes[t] - m) / sd
        if z <= -entry_z:
            pos = max_pos          # cheap -> long
        elif z >= entry_z:
            pos = -max_pos         # rich -> short
        elif abs(z) <= exit_z:
            pos = 0.0
        out[t] = pos
    return out


def backtest_targets(closes: List[float], targets: List[float], *, cost_bps: float = 2.0,
                     starting_equity: float = 100_000.0) -> List[float]:
    """Equity curve for a continuous signed target series. Bar-t return =
    position_held_into_t × asset_return_t − turnover_cost. position_held_into_t is the
    PREVIOUS bar's target (no look-ahead: you size on bar t-1's close, earn t's move)."""
    eq = [starting_equity]
    prev_pos = 0.0
    for t in range(1, len(closes)):
        asset_ret = (closes[t] - closes[t - 1]) / closes[t - 1] if closes[t - 1] else 0.0
        pos = targets[t - 1]
        turnover = abs(pos - prev_pos)
        ret = pos * asset_ret - turnover * (cost_bps / 10_000.0)
        eq.append(eq[-1] * (1 + ret))
        prev_pos = pos
    return eq


def spread_series(a_bars: List[dict], b_bars: List[dict], hedge: float) -> Tuple[List[float], List[float]]:
    """Inner-join two symbols on timestamp; return (spread, ts) where spread =
    log(a) − hedge·log(b). Used to A-S-size a cointegrated pair's spread (market-neutral)."""
    bm = {int(b["timestamp"]): float(b["close"]) for b in b_bars}
    sp, ts = [], []
    for r in a_bars:
        t = int(r["timestamp"])
        if t in bm and float(r["close"]) > 0 and bm[t] > 0:
            sp.append(math.log(float(r["close"])) - hedge * math.log(bm[t]))
            ts.append(t)
    return sp, ts


def backtest_spread_targets(spread: List[float], targets: List[float], *, cost_bps: float = 4.0,
                            starting_equity: float = 100_000.0) -> List[float]:
    """Equity curve for a market-neutral spread position. The spread is already in
    log-return space, so bar-t pnl = position_into_t × Δspread_t (additive), minus
    turnover cost. Market-neutral: no buy-and-hold benchmark — the return IS the alpha."""
    eq = [starting_equity]
    prev = 0.0
    for t in range(1, len(spread)):
        d = spread[t] - spread[t - 1]
        pos = targets[t - 1]
        turnover = abs(pos - prev)
        ret = pos * d - turnover * (cost_bps / 10_000.0)
        eq.append(eq[-1] * (1 + ret))
        prev = pos
    return eq


def as_target_spread(spread: List[float], *, window: int = 20, gamma: float = 1.0,
                     max_pos: float = 1.0, var_floor: float = 1e-8) -> List[float]:
    """A-S inventory sizing on a spread: q* = (mean−spread)/(γ·var·) clipped. The fair
    value is the rolling spread mean; long the spread when it's below its mean."""
    n = len(spread)
    out = [0.0] * n
    for t in range(n):
        if t < window:
            continue
        win = spread[t - window:t]
        m = sum(win) / window
        v = max(sum((x - m) ** 2 for x in win) / window, var_floor)
        q = (m - spread[t]) / (gamma * v)
        out[t] = max(-max_pos, min(max_pos, q))
    return out


def binary_target_spread(spread: List[float], *, window: int = 20, entry_z: float = 2.0,
                         exit_z: float = 0.5, max_pos: float = 1.0) -> List[float]:
    """Naive binary z-score entry on a spread (the classic pairs rule), for comparison."""
    n = len(spread)
    out = [0.0] * n
    pos = 0.0
    for t in range(n):
        if t < window:
            continue
        win = spread[t - window:t]
        m = sum(win) / window
        sd = (sum((x - m) ** 2 for x in win) / window) ** 0.5
        if sd <= 0:
            out[t] = pos
            continue
        z = (spread[t] - m) / sd
        if z <= -entry_z:
            pos = max_pos
        elif z >= entry_z:
            pos = -max_pos
        elif abs(z) <= exit_z:
            pos = 0.0
        out[t] = pos
    return out
