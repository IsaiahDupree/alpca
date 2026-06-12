"""
Gap reversion — cross-sectional mean-reversion on the OVERNIGHT GAP, held for several days.

Distinct from Case 17 (overnight→intraday reversal), which was intraday-only and flat every
night → ~2×/day turnover that ate a real gross edge. Here the signal is the same family (fade
the gap) but the position is HELD for `hold` days via overlapping tranches, so only ~1/hold of
the book rotates daily. The whole question: does the gap-reversion edge survive once you stop
churning? Lower turnover is the only structural way past the cost wall that killed Case 17.

Signal: gap_j(t) = (open_j(t) − close_j(t−1)) / close_j(t−1), known at the open of day t. Reversal
goes LONG the biggest gap-DOWNs / SHORT the biggest gap-UPs, dollar-neutral. No look-ahead: the
target weight from day t's gap is entered at day t's CLOSE (avoids the slippery open print) and
earns close-to-close returns from t+1 onward; the held book is the mean of the last `hold` tranches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class GapResult:
    symbols: List[str]
    n_days: int
    total_return: float
    sharpe: float
    max_drawdown: float
    hold: int
    top_frac: float
    avg_active: float
    avg_turnover: float
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


def backtest_gap_reversion(
    bars_by_sym: Dict[str, List[dict]], *,
    hold: int = 5,
    top_frac: float = 0.2,
    cost_bps: float = 2.0,
    periods_per_year: float = 252.0,
    reverse: bool = True,
    starting_equity: float = 100_000.0,
) -> GapResult:
    """reverse=True fades the gap (long gap-downs / short gap-ups — the anomaly); reverse=False is
    the gap-momentum control (chase the gap), which should fail if the reversion is real."""
    syms = sorted(bars_by_sym)
    if len(syms) < max(5, int(round(2 / max(top_frac, 1e-9)))):
        return GapResult(syms, 0, 0.0, 0.0, 0.0, hold, top_frac, 0.0, 0.0, [starting_equity], [])

    sets = [set(float(b.get("timestamp", 0) or 0) for b in bars_by_sym[s]) for s in syms]
    common = sorted(set.intersection(*sets)) if sets else []
    if len(common) < hold + 5:
        return GapResult(syms, len(common), 0.0, 0.0, 0.0, hold, top_frac, 0.0, 0.0, [starting_equity], [])

    o = {s: {float(b["timestamp"]): float(b["open"]) for b in bars_by_sym[s]} for s in syms}
    c = {s: {float(b["timestamp"]): float(b["close"]) for b in bars_by_sym[s]} for s in syms}
    opens = np.array([[o[s][t] for t in common] for s in syms]).T    # (T, N)
    closes = np.array([[c[s][t] for t in common] for s in syms]).T
    T, N = opens.shape
    k = max(1, int(round(N * top_frac)))

    gap = np.full((T, N), np.nan)
    gap[1:] = np.where(closes[:-1] > 0, opens[1:] / np.where(closes[:-1] > 0, closes[:-1], 1.0) - 1.0, np.nan)
    ret = np.zeros((T, N))
    ret[1:] = np.where(closes[:-1] > 0, closes[1:] / np.where(closes[:-1] > 0, closes[:-1], 1.0) - 1.0, 0.0)

    # daily target weights from the gap (long gap-downs if reverse)
    targets = np.zeros((T, N))
    for t in range(1, T):
        g = gap[t]
        ok = np.isfinite(g)
        if ok.sum() < 2 * k:
            continue
        sig = -g if reverse else g                    # reversion: most-negative gap -> highest signal
        order = np.argsort(np.where(ok, sig, -np.inf))
        order = order[np.isin(order, np.where(ok)[0])]
        w = np.zeros(N)
        w[order[-k:]] = 0.5 / k                         # long the strongest reversion signal
        w[order[:k]] = -0.5 / k
        targets[t] = w

    eq = [starting_equity]
    daily: List[float] = []
    actives: List[int] = []
    turnovers: List[float] = []
    prev_held = np.zeros(N)
    for t in range(1, T):
        # book earning ret[t] (close[t-1]->close[t]) is the mean of tranches ENTERED at closes
        # t-1, t-2, ..., t-hold -> targets[t-hold .. t-1]. It excludes targets[t] (whose gap isn't
        # known until open[t]) so the gap move itself is never captured. ~1/hold rotates per day.
        lo = max(0, t - hold)
        held = targets[lo:t].mean(axis=0)
        turnover = np.abs(held - prev_held).sum()
        port_ret = float(held @ ret[t]) - turnover * (cost_bps / 1e4)
        eq.append(eq[-1] * (1 + port_ret))
        daily.append(port_ret)
        actives.append(int((held != 0).sum()))
        turnovers.append(turnover)
        prev_held = held

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return GapResult(
        symbols=syms, n_days=len(daily), total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq),
        hold=hold, top_frac=top_frac, avg_active=float(np.mean(actives)) if actives else 0.0,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        equity_curve=eq, daily_returns=daily)
