"""
Overnight→intraday cross-sectional REVERSAL — a market-neutral edge on a clock we have not
tested. Decompose each day's return into an OVERNIGHT leg (prev_close→open) and an INTRADAY
leg (open→close). The documented "tug of war" (Lou-Polk-Skouras 2019): a stock's overnight
and intraday returns are *negatively* related cross-sectionally — overnight winners tend to
give it back intraday, and vice-versa.

The tradeable, NO-LOOKAHEAD form: the overnight return (prev_close→open) is fully known *at
the open* of day t. Rank the universe on it, go LONG the overnight LOSERS / SHORT the overnight
WINNERS (reversal), enter at the open, and capture that day's INTRADAY return (open→close).
Flat every night — so it carries no overnight beta and a bull market can't flatter it.

Honest caveats baked into the harness, not hidden:
- This rebalances the ENTIRE book every day (on at the open, off at the close) → ~2.0 turnover
  per day. It is the most cost-sensitive edge class there is; the cost stress is the real test.
- It needs an open-print and close-print fill. We proved opens are slippage-heavy at ~1.2s
  fills, so a per-leg cost_bps materially understates reality — treat the cost sweep as the
  binding constraint, exactly as for PEAD's borrow.
- Dividend/split artifacts in the overnight gap are removed by using adjustment="all" bars.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class OvernightResult:
    symbols: List[str]
    n_days: int
    total_return: float
    sharpe: float
    max_drawdown: float
    n_rebalances: int
    signal_lookback: int
    top_frac: float
    avg_active: float
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


def backtest_overnight_reversal(
    bars_by_sym: Dict[str, List[dict]], *,
    signal_lookback: int = 1,
    top_frac: float = 0.2,
    cost_bps: float = 2.0,
    periods_per_year: float = 252.0,
    reverse: bool = True,
    starting_equity: float = 100_000.0,
) -> OvernightResult:
    """Cross-sectional overnight→intraday reversal, dollar-neutral, flat overnight.

      signal_lookback: 1 = signal is today's overnight return (prev_close→open); >1 = trailing
                       mean overnight return over the last N days (still known at today's open).
      top_frac:        fraction of the universe in each leg (0.2 = long bottom 20%, short top 20%).
      reverse:         True = REVERSAL (long overnight losers / short winners — the anomaly);
                       False = momentum (long winners / short losers — the control that should fail).
      cost_bps:        charged per unit of turnover; the book fully turns over daily (~2.0/day).
    """
    syms = sorted(bars_by_sym)
    need = max(2, int(round(1 / max(top_frac, 1e-9))))
    if len(syms) < need:
        return OvernightResult(syms, 0, 0.0, 0.0, 0.0, 0, signal_lookback, top_frac, 0.0, [starting_equity], [])

    # align on common timestamps; carry open+close per symbol
    sets = [set(float(b.get("timestamp", 0) or 0) for b in bars_by_sym[s]) for s in syms]
    common = sorted(set.intersection(*sets)) if sets else []
    if len(common) < signal_lookback + 3:
        return OvernightResult(syms, len(common), 0.0, 0.0, 0.0, 0, signal_lookback, top_frac, 0.0,
                               [starting_equity], [])

    o = {s: {float(b["timestamp"]): float(b["open"]) for b in bars_by_sym[s]} for s in syms}
    c = {s: {float(b["timestamp"]): float(b["close"]) for b in bars_by_sym[s]} for s in syms}
    opens = np.array([[o[s][t] for t in common] for s in syms], dtype=float)   # (N, T)
    closes = np.array([[c[s][t] for t in common] for s in syms], dtype=float)
    N, T = opens.shape

    # overnight[s,t] = open[t]/close[t-1] - 1 (t>=1); intraday[s,t] = close[t]/open[t] - 1
    overnight = np.full((N, T), np.nan)
    valid_prevc = closes[:, :-1] > 0
    overnight[:, 1:] = np.where(valid_prevc, opens[:, 1:] / np.where(valid_prevc, closes[:, :-1], 1.0) - 1.0, np.nan)
    intraday = np.where(opens > 0, closes / np.where(opens > 0, opens, 1.0) - 1.0, np.nan)

    k = max(1, int(round(N * top_frac)))
    eq = [starting_equity]
    daily: List[float] = []
    actives: List[int] = []
    rebals = 0
    for t in range(signal_lookback, T):
        # signal known at the open of day t: trailing mean overnight return over the window
        win = overnight[:, t - signal_lookback + 1:t + 1]
        sig = np.nanmean(win, axis=1)
        ok = np.isfinite(sig) & np.isfinite(intraday[:, t])
        if ok.sum() < 2 * k:
            eq.append(eq[-1]); daily.append(0.0); actives.append(0); continue
        order = np.argsort(np.where(ok, sig, np.inf))     # ascending: losers first, NaNs last
        order = order[np.isin(order, np.where(ok)[0])]
        losers, winners = order[:k], order[-k:]
        w = np.zeros(N)
        long_idx, short_idx = (losers, winners) if reverse else (winners, losers)
        w[long_idx] = 0.5 / k
        w[short_idx] = -0.5 / k
        gross_intraday = float(np.nansum(w * intraday[:, t]))
        # the book is flat overnight, so every day we pay to put it ON at the open AND take it
        # OFF at the close — full turnover BOTH ways (≈2.0/day on a 0.5/0.5 dollar-neutral book).
        turnover = 2.0 * float(np.abs(w).sum())
        port_ret = gross_intraday - turnover * (cost_bps / 1e4)
        eq.append(eq[-1] * (1.0 + port_ret))
        daily.append(port_ret)
        actives.append(int((w != 0).sum()))
        rebals += 1

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return OvernightResult(
        symbols=syms, n_days=len(daily), total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq),
        n_rebalances=rebals, signal_lookback=signal_lookback, top_frac=top_frac,
        avg_active=float(np.mean(actives)) if actives else 0.0,
        equity_curve=eq, daily_returns=daily)
