"""
EAR-PEAD — post-earnings drift driven by the EARNINGS ANNOUNCEMENT RETURN, not by the analyst
surprise. The literature (Quantpedia / Brandt-Kishore-Santa-Clara-Venkatachalam) finds the
3-day price reaction around the report (EAR) predicts a LONGER, cleaner drift than SUE, and the
drift is concentrated on the LONG side. This is the direct fix for our downgraded surprise-PEAD,
whose short leg (a) had no edge and (b) died to adverse-selection borrow.

Two things make this distinct from `backtest_pead`:
  1. SIGNAL = a price-only 3-day announcement return (EAR), so it needs NO analyst estimates —
     just the report date + bars. EAR_i = close[end of reaction window] / close[last pre-report] − 1.
  2. The short side is replaced by a CHEAP INDEX HEDGE. Shorting the crushed-earnings single
     names is exactly what adverse-selection borrow makes impossible/expensive; shorting SPY to
     neutralize market beta is general-collateral and trivial. So `mode="beta_hedged"` keeps the
     long alpha while removing market beta with a borrow we can actually get.

No look-ahead: EAR is measured over the reaction window [first post-report bar .. +ear_window−1];
the drift position is entered `skip_after_ear` bars LATER and earns subsequent returns only. The
signal is fully known before the position starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EARResult:
    mode: str
    equity_curve: List[float]
    total_return: float
    sharpe: float
    max_drawdown: float
    n_days: int
    n_events_used: int
    avg_active: float
    periods_per_year: float
    beta: float = 0.0                       # market beta of the long leg (0 for neutral modes)
    daily_returns: List[float] = field(default_factory=list)


def _ear_signal_events(bars_by_sym, events_by_sym, ear_window, skip_after_ear):
    """Per symbol, attach EAR (3-day announcement return) + the drift entry index to each event."""
    out = {}
    for s, bars in bars_by_sym.items():
        evs = events_by_sym.get(s)
        if not evs:
            continue
        b = sorted(bars, key=lambda x: int(x["timestamp"]))
        ts = [int(x["timestamp"]) for x in b]
        cl = [float(x["close"]) for x in b]
        tagged = []
        for ev in evs:
            r = ev.get("date")
            if r is None or r < ts[0] or r > ts[-1]:
                continue
            b0 = next((k for k in range(len(ts)) if ts[k] > r), None)   # first post-report bar
            if b0 is None or b0 == 0:
                continue
            end = b0 + ear_window - 1                                   # last bar of reaction window
            if end >= len(ts) or cl[b0 - 1] <= 0:
                continue
            ear = cl[end] / cl[b0 - 1] - 1.0                            # the 3-day announcement return
            entry = end + skip_after_ear                               # enter AFTER the window (no overlap)
            if entry >= len(ts):
                continue
            tagged.append({"ear": ear * 100.0, "entry_ts": ts[entry], "sym": s})
        if tagged:
            out[s] = tagged
    return out


def backtest_ear_pead(
    bars_by_sym: Dict[str, List[dict]],
    events_by_sym: Dict[str, List[dict]], *,
    hold: int = 40,
    ear_window: int = 3,
    skip_after_ear: int = 1,
    entry_thr: float = 2.0,
    mode: str = "long",                     # "long" | "neutral" | "beta_hedged"
    bench_bars: Optional[List[dict]] = None,
    cost_bps: float = 2.0,
    periods_per_year: float = 252.0,
    starting_equity: float = 100_000.0,
) -> EARResult:
    """EAR-driven PEAD. entry_thr is in EAR PERCENT units (2.0 = a +2% 3-day reaction).
    long: long the high-EAR names only. neutral: long high-EAR / short low-EAR (single-name short).
    beta_hedged: long high-EAR, short the index by the long leg's market beta (cheap GC short)."""
    syms = sorted(s for s in bars_by_sym if events_by_sym.get(s))
    if len(syms) < 3:
        return EARResult(mode, [starting_equity], 0.0, 0.0, 0.0, 0, 0, 0.0, periods_per_year)

    tagged = _ear_signal_events(bars_by_sym, events_by_sym, ear_window, skip_after_ear)
    master = sorted({int(b["timestamp"]) for s in syms for b in bars_by_sym[s]})
    idx = {t: i for i, t in enumerate(master)}
    T, N = len(master), len(syms)
    if T < hold + 20:
        return EARResult(mode, [starting_equity], 0.0, 0.0, 0.0, 0, 0, 0.0, periods_per_year)

    ret = np.zeros((T, N))
    pos = np.zeros((T, N))
    n_used = 0
    for j, s in enumerate(syms):
        bars = sorted(bars_by_sym[s], key=lambda x: int(x["timestamp"]))
        ts = [int(x["timestamp"]) for x in bars]
        cl = [float(x["close"]) for x in bars]
        for k in range(1, len(bars)):
            if cl[k - 1] > 0:
                ret[idx[ts[k]], j] = (cl[k] - cl[k - 1]) / cl[k - 1]
        for ev in tagged.get(s, ()):
            ear = ev["ear"]
            if abs(ear) < entry_thr:
                continue
            sign = 1.0 if ear > 0 else -1.0
            if mode in ("long", "beta_hedged") and sign < 0:
                continue                                  # long-side strategies skip low-EAR names
            entry_k = next((k for k in range(len(ts)) if ts[k] >= ev["entry_ts"]), None)
            if entry_k is None:
                continue
            n_used += 1
            for k in range(entry_k, min(entry_k + hold, len(ts))):
                pos[idx[ts[k]], j] = sign

    # benchmark daily returns on the master calendar (for the beta hedge)
    bench = np.zeros(T)
    if bench_bars:
        bmap = {int(b["timestamp"]): float(b["close"]) for b in bench_bars}
        bc = [bmap.get(t) for t in master]
        for t in range(1, T):
            if bc[t] is not None and bc[t - 1]:
                bench[t] = bc[t] / bc[t - 1] - 1.0

    eq = [starting_equity]
    daily: List[float] = []
    actives: List[int] = []
    long_daily: List[float] = []            # un-hedged long-leg return, for beta estimation
    prev_w = np.zeros(N)
    for t in range(1, T):
        p = pos[t - 1]
        longs, shorts = p > 0, p < 0
        w = np.zeros(N)
        if longs.any():
            # long-only / beta_hedged put full +1 on longs; neutral splits 0.5/0.5
            w[longs] = (0.5 if mode == "neutral" else 1.0) / longs.sum()
        if mode == "neutral" and shorts.any():
            w[shorts] = -0.5 / shorts.sum()
        turnover = np.abs(w - prev_w).sum()
        long_leg = float(w @ ret[t])
        port_ret = long_leg - turnover * (cost_bps / 1e4)
        eq.append(eq[-1] * (1 + port_ret))
        daily.append(port_ret)
        long_daily.append(long_leg)
        actives.append(int(longs.sum() + shorts.sum()))
        prev_w = w

    beta = 0.0
    if mode == "beta_hedged" and bench_bars:
        bl = bench[1:]                       # align with daily (t=1..T-1)
        ld = np.array(long_daily)
        var = float(np.var(bl))
        beta = float(np.cov(ld, bl)[0, 1] / var) if var > 1e-12 else 0.0
        # re-derive the hedged equity: long return minus beta*index return each day (GC index short)
        hedged = ld - beta * bl
        eq = [starting_equity]
        daily = []
        for x in hedged:
            eq.append(eq[-1] * (1 + float(x)))
            daily.append(float(x))

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return EARResult(
        mode=mode, equity_curve=eq, total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq),
        n_days=len(daily), n_events_used=n_used,
        avg_active=float(np.mean(actives)) if actives else 0.0,
        periods_per_year=periods_per_year, beta=beta, daily_returns=daily)
