"""
Betting-Against-Beta / low-volatility anomaly (Frazzini-Pedersen; Ang et al.). Low-beta / low-vol
stocks earn higher RISK-ADJUSTED returns than high-beta/high-vol names. Cross-sectional, dollar-
neutral: LONG the low-`signal` quantile / SHORT the high, rebalanced monthly-ish (low turnover).

  signal="beta"  → each stock's trailing beta to the benchmark (the classic BAB)
  signal="vol"   → each stock's trailing realized volatility (the low-vol / "lottery" anomaly)

HONEST CAVEAT baked into the test: the textbook BAB factor *levers* the low-beta leg to be
beta-neutral; this dollar-neutral, unlevered version is dominated by the raw beta differential, so in
a bull run (high-beta outruns low-beta) the unlevered long-low/short-high book can simply lose. The
harness will show whether the risk-adjusted effect survives without leverage. No look-ahead: the
signal uses trailing returns through the rebalance day; positions earn forward returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class LowBetaResult:
    symbols: List[str]
    n_days: int
    total_return: float
    sharpe: float
    max_drawdown: float
    n_rebalances: int
    top_frac: float
    signal: str
    avg_active: float
    avg_turnover: float
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    dates: List[int] = field(default_factory=list)


def backtest_low_beta(
    bars_by_sym: Dict[str, List[dict]],
    bench_bars: List[dict], *,
    signal: str = "beta",             # "beta" | "vol"
    lookback: int = 120,
    top_frac: float = 0.2,
    rebalance_days: int = 21,
    cost_bps: float = 2.0,
    reverse: bool = False,            # False = anomaly (long LOW signal); True = control (long HIGH)
    periods_per_year: float = 252.0,
    starting_equity: float = 100_000.0,
) -> LowBetaResult:
    syms = sorted(bars_by_sym)
    if len(syms) < max(5, int(round(2 / max(top_frac, 1e-9)))):
        return LowBetaResult(syms, 0, 0.0, 0.0, 0.0, 0, top_frac, signal, 0.0, 0.0, [starting_equity], [])

    sets = [set(int(b.get("timestamp", 0) or 0) for b in bars_by_sym[s]) for s in syms]
    bench_ts = set(int(b["timestamp"]) for b in bench_bars)
    common = sorted(set.intersection(*sets) & bench_ts) if sets else []
    if len(common) < lookback + rebalance_days + 2:
        return LowBetaResult(syms, len(common), 0.0, 0.0, 0.0, 0, top_frac, signal, 0.0, 0.0,
                             [starting_equity], [])
    cl = {s: {int(b["timestamp"]): float(b["close"]) for b in bars_by_sym[s]} for s in syms}
    price = np.array([[cl[s][t] for t in common] for s in syms], dtype=float).T   # (T, N)
    bmap = {int(b["timestamp"]): float(b["close"]) for b in bench_bars}
    bpx = np.array([bmap[t] for t in common])
    T, N = price.shape
    ret = np.zeros((T, N))
    ret[1:] = np.where(price[:-1] > 0, price[1:] / np.where(price[:-1] > 0, price[:-1], 1.0) - 1.0, 0.0)
    bret = np.zeros(T)
    bret[1:] = np.where(bpx[:-1] > 0, bpx[1:] / np.where(bpx[:-1] > 0, bpx[:-1], 1.0) - 1.0, 0.0)

    k = max(1, int(round(N * top_frac)))
    eq = [starting_equity]
    daily: List[float] = []
    actives: List[int] = []
    turnovers: List[float] = []
    rebals = 0
    w = np.zeros(N)
    prev_w = np.zeros(N)
    for t in range(1, T):
        if t > lookback and (t - 1) % rebalance_days == 0:
            win = ret[t - lookback:t]                  # trailing window, known entering day t
            if signal == "beta":
                bwin = bret[t - lookback:t]
                var = float(np.var(bwin))
                sig = np.array([np.cov(win[:, j], bwin)[0, 1] / var if var > 1e-12 else 0.0
                                for j in range(N)])
            else:  # vol
                sig = win.std(axis=0)
            ok = np.isfinite(sig)
            if ok.sum() >= 2 * k:
                order = np.argsort(np.where(ok, sig, np.inf))     # ascending: low signal first
                order = order[np.isin(order, np.where(ok)[0])]
                low, high = order[:k], order[-k:]
                lng, sht = (low, high) if not reverse else (high, low)
                w = np.zeros(N)
                w[lng] = 0.5 / k
                w[sht] = -0.5 / k
                rebals += 1
        turnover = float(np.abs(w - prev_w).sum())
        port_ret = float(w @ ret[t]) - turnover * (cost_bps / 1e4)
        eq.append(eq[-1] * (1 + port_ret))
        daily.append(port_ret)
        actives.append(int((w != 0).sum()))
        turnovers.append(turnover)
        prev_w = w

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return LowBetaResult(
        symbols=syms, n_days=len(daily), total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq), n_rebalances=rebals,
        top_frac=top_frac, signal=signal, avg_active=float(np.mean(actives)) if actives else 0.0,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        equity_curve=eq, daily_returns=daily, dates=common[1:])
