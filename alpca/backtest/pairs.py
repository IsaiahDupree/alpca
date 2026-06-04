"""
Pairs / spread mean-reversion — the simplest MARKET-NEUTRAL strategy, and the first
real shot at ALPHA (not beta): hold equal dollars LONG one asset and SHORT a correlated
one, so net market exposure is ~0. The bet is on the SPREAD reverting, not on direction —
so a bull-market backtest can't flatter it the way it flatters long-biased strategies.

This needs a two-symbol backtester (the single-symbol runner can't express it), so it
lives here as a function rather than a registry Strategy.

Spread = log(price_a) - hedge*log(price_b) (hedge=1 => log-ratio). Rolling z-score:
  z >  entry  -> spread rich   -> SHORT A / LONG B   (bet spread falls)
  z < -entry  -> spread cheap  -> LONG A / SHORT B   (bet spread rises)
  |z| < exit  -> flat
Dollar-neutral: each leg sized to `leg_notional_pct` of equity; cost_bps charged per leg
on entry and exit.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class PairsResult:
    sym_a: str
    sym_b: str
    n_bars: int
    total_return: float
    sharpe: float
    max_drawdown: float
    n_trades: int
    hedge: float
    equity_curve: List[float] = field(default_factory=list)


def align(a_bars: List[dict], b_bars: List[dict]) -> List[Tuple[float, float, float]]:
    """Inner-join two bar lists on timestamp -> [(ts, close_a, close_b)] sorted by ts."""
    bmap: Dict[float, float] = {float(b.get("timestamp", 0) or 0): float(b["close"]) for b in b_bars}
    out = []
    for a in a_bars:
        ts = float(a.get("timestamp", 0) or 0)
        if ts in bmap:
            out.append((ts, float(a["close"]), bmap[ts]))
    out.sort(key=lambda r: r[0])
    return out


def _hedge_ratio(la: List[float], lb: List[float]) -> float:
    """OLS slope of log(a) on log(b) (the hedge that makes the spread most stationary)."""
    n = len(la)
    if n < 2:
        return 1.0
    mb = statistics.fmean(lb)
    ma = statistics.fmean(la)
    cov = sum((lb[i] - mb) * (la[i] - ma) for i in range(n))
    var = sum((lb[i] - mb) ** 2 for i in range(n))
    return cov / var if var > 0 else 1.0


def mean_reversion_stats(spread: List[float]) -> Tuple[float, float]:
    """
    Ornstein-Uhlenbeck mean-reversion speed via the AR(1) regression
        d(spread)_t = a + lam * spread_{t-1}
    lam < 0 => mean-reverting; half-life = -ln(2)/lam (in bars). Returns (lam, half_life);
    half_life = inf when not mean-reverting. This is the cointegration screen: a pair is
    only tradeable if its spread reverts on a sane horizon.
    """
    n = len(spread)
    if n < 5:
        return 0.0, float("inf")
    slag = spread[:-1]
    ds = [spread[i] - spread[i - 1] for i in range(1, n)]
    ml = statistics.fmean(slag)
    md = statistics.fmean(ds)
    var = sum((x - ml) ** 2 for x in slag)
    if var <= 0:
        return 0.0, float("inf")
    lam = sum((slag[i] - ml) * (ds[i] - md) for i in range(len(ds))) / var
    if lam >= 0:
        return lam, float("inf")
    return lam, -math.log(2) / lam


def screen_pairs(symbols: List[str], bars_by_sym: Dict[str, List[dict]], *,
                 min_overlap: int = 120, max_half_life: float = 60.0,
                 min_half_life: float = 2.0) -> List[dict]:
    """
    Rank every symbol pair by spread mean-reversion quality (cointegration screen).
    Returns pairs whose spread reverts with a half-life in [min_half_life, max_half_life],
    sorted by half-life ascending (faster reversion = more tradeable). Each entry:
    {a, b, hedge, half_life, lam, n}.
    """
    out = []
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            a, b = symbols[i], symbols[j]
            rows = align(bars_by_sym.get(a, []), bars_by_sym.get(b, []))
            if len(rows) < min_overlap:
                continue
            la = [math.log(c) for _, c, _ in rows]
            lb = [math.log(c) for _, _, c in rows]
            h = _hedge_ratio(la, lb)
            spread = [la[k] - h * lb[k] for k in range(len(rows))]
            lam, hl = mean_reversion_stats(spread)
            if min_half_life <= hl <= max_half_life:
                out.append({"a": a, "b": b, "hedge": round(h, 3), "half_life": round(hl, 1),
                            "lam": round(lam, 5), "n": len(rows)})
    out.sort(key=lambda r: r["half_life"])
    return out


def backtest_pairs(a_bars: List[dict], b_bars: List[dict], *, sym_a: str = "A", sym_b: str = "B",
                   lookback: int = 60, entry_z: float = 2.0, exit_z: float = 0.5,
                   starting_equity: float = 100_000.0, leg_notional_pct: float = 0.5,
                   cost_bps: float = 2.0, periods_per_year: float = 252.0,
                   hedge: Optional[float] = None) -> PairsResult:
    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of

    rows = align(a_bars, b_bars)
    if len(rows) < lookback + 5:
        return PairsResult(sym_a, sym_b, len(rows), 0.0, 0.0, 0.0, 0, hedge or 1.0, [])

    la = [math.log(c) for _, c, _ in rows]
    lb = [math.log(c) for _, _, c in rows]
    h = hedge if hedge is not None else _hedge_ratio(la, lb)
    spread = [la[i] - h * lb[i] for i in range(len(rows))]

    cash = starting_equity
    qa = qb = 0.0
    state = 0          # +1 long-spread (long A/short B), -1 short-spread, 0 flat
    trades = 0
    equity: List[float] = []
    win = deque(maxlen=lookback)

    def rebalance(target: int, pa: float, pb: float):
        nonlocal cash, qa, qb, state, trades
        if target == state:
            return
        # close existing legs
        cash += qa * pa + qb * pb
        cash -= cost_bps / 1e4 * (abs(qa) * pa + abs(qb) * pb)
        qa = qb = 0.0
        if target != 0:
            leg = leg_notional_pct * cash
            qa = (leg / pa) * target
            qb = (leg / pb) * (-target)
            cash -= qa * pa + qb * pb        # buy long leg (cash down), short leg (cash up)
            cash -= cost_bps / 1e4 * (abs(qa) * pa + abs(qb) * pb)
            trades += 1
        state = target

    for i in range(len(rows)):
        _, pa, pb = rows[i]
        win.append(spread[i])
        if len(win) >= lookback:
            mu = statistics.fmean(win)
            sd = statistics.pstdev(win)
            z = (spread[i] - mu) / sd if sd > 0 else 0.0
            if state == 0:
                if z > entry_z:
                    rebalance(-1, pa, pb)
                elif z < -entry_z:
                    rebalance(1, pa, pb)
            elif state == 1 and z >= -exit_z:
                rebalance(0, pa, pb)
            elif state == -1 and z <= exit_z:
                rebalance(0, pa, pb)
        equity.append(cash + qa * pa + qb * pb)

    total_return = (equity[-1] - starting_equity) / starting_equity if equity else 0.0
    return PairsResult(sym_a, sym_b, len(rows), total_return,
                       sharpe_of(equity, periods_per_year), max_drawdown_of(equity),
                       trades, h, equity)
