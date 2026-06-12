"""
Post-Earnings-Announcement-Drift (PEAD) backtest — event-driven cross-sectional.

For each earnings event: if the standardized surprise beats +entry_thr, go LONG the stock
for `hold` trading days starting `skip_days` after the report (PEAD is the DRIFT *after*
the announcement gap, not the gap itself); if it misses below -entry_thr, go SHORT. The
portfolio holds all currently-active events, equal-weighted.

The legs are judged SEPARATELY (the scout's instruction): the long leg is often just beta,
so the market-neutral edge — if any — lives in the dollar-neutral combination and the short
leg. `leg` selects: 'long' (equal-weight longs, sums to +1), 'short' (equal-weight shorts,
sums to -1), or 'both' (dollar-neutral: longs +0.5, shorts -0.5).

No look-ahead: a position for day t is set from events strictly before t and earns day t's
return. Events use only the reported surprise (known at the report). Honest null: PEAD has
decayed; market-neutral so the return itself is the alpha (no buy-and-hold to beat).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class PEADResult:
    leg: str
    equity_curve: List[float]
    total_return: float
    sharpe: float
    max_drawdown: float
    n_days: int
    n_events_used: int
    avg_active: float
    periods_per_year: float
    daily_returns: List[float] = field(default_factory=list)


def backtest_pead(
    bars_by_sym: Dict[str, List[dict]],
    events_by_sym: Dict[str, List[dict]], *,
    hold: int = 30,
    skip_days: int = 1,
    entry_thr: float = 2.0,
    leg: str = "both",
    cost_bps: float = 2.0,
    borrow_apr=0.0,
    no_borrow=None,
    starting_equity: float = 100_000.0,
    periods_per_year: float = 252.0,
) -> PEADResult:
    """Walk-forward PEAD over a universe. `events_by_sym[sym]` = [{date(epoch), surprise_pct}].

    SHORTING REALISM (the short leg is the fragile one):
      borrow_apr: annualized stock-borrow fee charged daily on the short notional
                  (apr/periods_per_year per day). Float = flat rate for all names, or a
                  {symbol: apr} dict for per-name rates. Large-cap GC ~0.3-1%; HTB much higher.
      no_borrow:  set of symbols with NO locate available -> their short events are dropped
                  entirely (you simply cannot put the trade on)."""
    syms = sorted(s for s in bars_by_sym if events_by_sym.get(s))
    if len(syms) < 3:
        return PEADResult(leg, [starting_equity], 0.0, 0.0, 0.0, 0, 0, 0.0, periods_per_year)
    no_borrow = set(no_borrow or ())

    # master trading calendar = sorted union of all symbols' bar timestamps
    master = sorted({int(b["timestamp"]) for s in syms for b in bars_by_sym[s]})
    idx = {t: i for i, t in enumerate(master)}
    T, N = len(master), len(syms)
    if T < hold + 20:
        return PEADResult(leg, [starting_equity], 0.0, 0.0, 0.0, 0, 0, 0.0, periods_per_year)

    ret = np.zeros((T, N))
    pos = np.zeros((T, N))
    n_used = 0
    # per-symbol annualized borrow rate (float -> flat; dict -> per name)
    apr = np.array([(borrow_apr.get(s, 0.0) if isinstance(borrow_apr, dict) else float(borrow_apr))
                    for s in syms])

    for j, s in enumerate(syms):
        bars = sorted(bars_by_sym[s], key=lambda b: int(b["timestamp"]))
        ts = [int(b["timestamp"]) for b in bars]
        cl = [float(b["close"]) for b in bars]
        # per-symbol simple returns mapped onto the master calendar
        for k in range(1, len(bars)):
            if cl[k - 1] > 0:
                ret[idx[ts[k]], j] = (cl[k] - cl[k - 1]) / cl[k - 1]
        # place each event's drift position
        for ev in events_by_sym[s]:
            surp = ev.get("surprise_pct")
            if surp is None or abs(surp) < entry_thr:
                continue
            sign = 1.0 if surp > 0 else -1.0
            if sign < 0 and s in no_borrow:
                continue                       # no locate available -> can't short this name
            # require the report to fall INSIDE this symbol's price window — else there's no
            # clean post-report entry (events before ts[0] would all pile in at bar 0).
            if ev["date"] < ts[0] or ev["date"] > ts[-1]:
                continue
            # first symbol-bar strictly after the report date, then skip_days more
            entry_k = next((k for k in range(len(ts)) if ts[k] > ev["date"]), None)
            if entry_k is None:
                continue
            entry_k += max(0, skip_days - 1)
            if entry_k >= len(ts):
                continue
            n_used += 1
            for k in range(entry_k, min(entry_k + hold, len(ts))):
                pos[idx[ts[k]], j] = sign

    eq = [starting_equity]
    daily: List[float] = []
    actives: List[int] = []
    prev_w = np.zeros(N)
    for t in range(1, T):
        p = pos[t - 1]
        longs, shorts = p > 0, p < 0
        w = np.zeros(N)
        if leg in ("long", "both") and longs.any():
            w[longs] = (1.0 if leg == "long" else 0.5) / longs.sum()
        if leg in ("short", "both") and shorts.any():
            w[shorts] = -(1.0 if leg == "short" else 0.5) / shorts.sum()
        turnover = np.abs(w - prev_w).sum()
        # daily borrow fee on the short notional (apr/periods_per_year per name held short)
        short_w = np.where(w < 0, -w, 0.0)
        borrow_drag = float((short_w * apr).sum()) / periods_per_year
        port_ret = float(w @ ret[t]) - turnover * (cost_bps / 10_000.0) - borrow_drag
        eq.append(eq[-1] * (1 + port_ret))
        daily.append(port_ret)
        actives.append(int(longs.sum() + shorts.sum()))
        prev_w = w

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return PEADResult(
        leg=leg, equity_curve=eq, total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq),
        n_days=len(daily), n_events_used=n_used,
        avg_active=float(np.mean(actives)) if actives else 0.0,
        periods_per_year=periods_per_year, daily_returns=daily)
