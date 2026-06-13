"""
52-week-high momentum (George & Hwang 2004) — a momentum variant that, in the literature, both
SUBSUMES traditional momentum and is more robust/longer-lived. The signal is each stock's
*proximity to its trailing 52-week high*: ratio = close / max(close over the last ~252 days), in
(0, 1]. Stocks near their high (ratio ~1) keep winning, those far below (ratio low) keep lagging —
attributed to anchoring + underreaction to news. Long the near-high names, short the far-below,
dollar-neutral.

We test it the honest way alongside the controls that matter: vs a long-only B&H proxy (is it alpha
or beta?), vs plain cross-sectional momentum (the literature's claim is it beats that), with a hold
sweep (overlapping tranches → ~1/hold turnover, the cost-wall guard), OOS split, and per-year regime
stability. No look-ahead: the position for day t uses the 52-wk window through t-1 and earns ret[t].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class High52Result:
    symbols: List[str]
    n_days: int
    total_return: float
    sharpe: float
    max_drawdown: float
    hold: int
    top_frac: float
    window: int
    avg_active: float
    avg_turnover: float
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    dates: List[int] = field(default_factory=list)


def backtest_high_52w(
    bars_by_sym: Dict[str, List[dict]], *,
    window: int = 252,
    hold: int = 20,
    top_frac: float = 0.2,
    cost_bps: float = 2.0,
    periods_per_year: float = 252.0,
    reverse: bool = False,                # False = momentum (long near-high); True = the control
    starting_equity: float = 100_000.0,
) -> High52Result:
    syms = sorted(bars_by_sym)
    if len(syms) < max(5, int(round(2 / max(top_frac, 1e-9)))):
        return High52Result(syms, 0, 0.0, 0.0, 0.0, hold, top_frac, window, 0.0, 0.0, [starting_equity], [])

    sets = [set(float(b.get("timestamp", 0) or 0) for b in bars_by_sym[s]) for s in syms]
    common = sorted(set.intersection(*sets)) if sets else []
    if len(common) < window + hold + 2:
        return High52Result(syms, len(common), 0.0, 0.0, 0.0, hold, top_frac, window, 0.0, 0.0,
                            [starting_equity], [])
    cl = {s: {float(b["timestamp"]): float(b["close"]) for b in bars_by_sym[s]} for s in syms}
    price = np.array([[cl[s][t] for t in common] for s in syms], dtype=float).T   # (T, N)
    T, N = price.shape
    k = max(1, int(round(N * top_frac)))

    ret = np.zeros((T, N))
    ret[1:] = np.where(price[:-1] > 0, price[1:] / np.where(price[:-1] > 0, price[:-1], 1.0) - 1.0, 0.0)

    # 52-week-high ratio per (t, j): close[t] / rolling max over the trailing `window`
    ratio = np.full((T, N), np.nan)
    for t in range(window - 1, T):
        hi = price[t - window + 1:t + 1].max(axis=0)
        ratio[t] = np.where(hi > 0, price[t] / hi, np.nan)

    # daily target weights from the ratio (long near-high / short far-below)
    targets = np.zeros((T, N))
    for t in range(window - 1, T):
        r = ratio[t]
        ok = np.isfinite(r)
        if ok.sum() < 2 * k:
            continue
        sig = r if not reverse else -r
        order = np.argsort(np.where(ok, sig, -np.inf))
        order = order[np.isin(order, np.where(ok)[0])]
        w = np.zeros(N)
        w[order[-k:]] = 0.5 / k         # long the highest ratio (nearest 52wk high)
        w[order[:k]] = -0.5 / k         # short the lowest ratio
        targets[t] = w

    eq = [starting_equity]
    daily: List[float] = []
    actives: List[int] = []
    turnovers: List[float] = []
    prev_held = np.zeros(N)
    for t in range(1, T):
        lo = max(0, t - hold)
        held = targets[lo:t].mean(axis=0)        # book entered by t-1 earns ret[t] (no look-ahead)
        turnover = float(np.abs(held - prev_held).sum())
        port_ret = float(held @ ret[t]) - turnover * (cost_bps / 1e4)
        eq.append(eq[-1] * (1 + port_ret))
        daily.append(port_ret)
        actives.append(int((held != 0).sum()))
        turnovers.append(turnover)
        prev_held = held

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return High52Result(
        symbols=syms, n_days=len(daily), total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq),
        hold=hold, top_frac=top_frac, window=window,
        avg_active=float(np.mean(actives)) if actives else 0.0,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        equity_curve=eq, daily_returns=daily, dates=[int(t) for t in common[1:]])
