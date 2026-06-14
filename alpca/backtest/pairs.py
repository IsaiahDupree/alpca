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


def adf_stat(series: List[float]) -> float:
    """
    Dickey-Fuller test statistic for a unit root: regress d(y)_t = a + b*y_{t-1} + e and
    return t = b/SE(b). MORE NEGATIVE = more stationary (mean-reverting). Engle-Granger
    cointegration screen: run this on the pair's spread; require it below a critical value
    (~-2.86 at 5%, stricter ~-3.34 for an estimated-residual spread). Far stricter than a
    half-life cap (which passes most noise). Pure-python, no scipy.
    """
    n = len(series)
    if n < 12:
        return 0.0
    dy = [series[i] - series[i - 1] for i in range(1, n)]
    ylag = series[:-1]
    m = len(dy)
    mx = statistics.fmean(ylag)
    my = statistics.fmean(dy)
    sxx = sum((ylag[i] - mx) ** 2 for i in range(m))
    if sxx <= 0:
        return 0.0
    b = sum((ylag[i] - mx) * (dy[i] - my) for i in range(m)) / sxx
    a = my - b * mx
    rss = sum((dy[i] - (a + b * ylag[i])) ** 2 for i in range(m))
    if m <= 2 or rss <= 0:
        return 0.0
    se = math.sqrt(rss / (m - 2) / sxx)
    return b / se if se > 0 else 0.0


def screen_pairs(symbols: List[str], bars_by_sym: Dict[str, List[dict]], *,
                 min_overlap: int = 120, max_half_life: float = 60.0,
                 min_half_life: float = 2.0, max_adf: Optional[float] = None) -> List[dict]:
    """
    Rank every symbol pair by spread mean-reversion quality (cointegration screen).
    Returns pairs whose spread reverts with a half-life in [min_half_life, max_half_life]
    (and, when `max_adf` is set, whose spread ADF statistic is below it — a real
    stationarity test, much stricter than half-life alone). Sorted by ADF ascending if
    used, else half-life. Each entry: {a, b, hedge, half_life, lam, adf, n}.
    """
    # precompute {timestamp: close} per symbol ONCE (so a pair screen is a set-intersection
    # + lookup, not a per-pair dict rebuild — ~N^2 pairs, this is the hot loop).
    maps = {s: {float(b.get("timestamp", 0) or 0): float(b["close"]) for b in bars_by_sym.get(s, [])}
            for s in symbols}
    out = []
    for i in range(len(symbols)):
        a = symbols[i]
        ma = maps[a]
        ka = ma.keys()
        for j in range(i + 1, len(symbols)):
            b = symbols[j]
            mb = maps[b]
            ts = sorted(ka & mb.keys())
            if len(ts) < min_overlap:
                continue
            la = [math.log(ma[t]) for t in ts]
            lb = [math.log(mb[t]) for t in ts]
            h = _hedge_ratio(la, lb)
            spread = [la[k] - h * lb[k] for k in range(len(ts))]
            lam, hl = mean_reversion_stats(spread)
            if not (min_half_life <= hl <= max_half_life):
                continue
            adf = adf_stat(spread) if max_adf is not None else 0.0
            if max_adf is not None and adf >= max_adf:
                continue
            out.append({"a": a, "b": b, "hedge": round(h, 3), "half_life": round(hl, 1),
                        "lam": round(lam, 5), "adf": round(adf, 3), "n": len(ts)})
    out.sort(key=lambda r: r["adf"] if max_adf is not None else r["half_life"])
    return out


@dataclass
class WalkForwardResult:
    n_symbols: int
    n_windows: int
    n_oos_bars: int
    total_return: float
    sharpe: float
    max_drawdown: float
    train: int
    test: int
    top_n: int
    equity_curve: List[float] = field(default_factory=list)


def walkforward_pairs(bars_by_sym: Dict[str, List[dict]], *, train: int = 252, test: int = 63,
                      top_n: int = 15, max_half_life: float = 30.0, min_half_life: float = 3.0,
                      entry_z: float = 2.0, exit_z: float = 0.5, cost_bps: float = 2.0,
                      starting_equity: float = 100_000.0, periods_per_year: float = 252.0,
                      max_adf: Optional[float] = None, use_kalman: bool = False) -> "WalkForwardResult":
    """
    Rigorous walk-forward market-neutral pairs. Each step: SCREEN + hedge-fit on a trailing
    `train` window, then TRADE the selected top_n pairs on the next `test` window (genuinely
    out-of-sample — selection & hedge used only past data), roll forward by `test`. The OOS
    test-window basket returns are concatenated into one continuous equity curve, so the
    resulting Sharpe is the HONEST number (every trade is on unseen data). Pairs are
    re-selected each window (rolling re-screen) and the hedge is re-fit each window (rolling
    hedge) — the two upgrades over a single static screen.
    """
    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of

    syms = sorted(bars_by_sym)
    sets = [set(float(b.get("timestamp", 0) or 0) for b in bars_by_sym[s]) for s in syms]
    common = sorted(set.intersection(*sets)) if sets else []
    n = len(common)
    if n < train + 2 * test or len(syms) < 4:
        return WalkForwardResult(len(syms), 0, 0, 0.0, 0.0, 0.0, train, test, top_n, [])

    bymap = {s: {float(b["timestamp"]): b for b in bars_by_sym[s]} for s in syms}
    aligned = {s: [bymap[s][t] for t in common] for s in syms}

    oos_rets: List[float] = []
    windows = 0
    w = 0
    while w + train + test <= n:
        train_slice = {s: aligned[s][w:w + train] for s in syms}
        screened = screen_pairs(syms, train_slice, min_overlap=int(train * 0.8),
                                max_half_life=max_half_life, min_half_life=min_half_life, max_adf=max_adf)
        per_pair: List[List[float]] = []
        for r in screened[:top_n]:
            seg_a = aligned[r["a"]][w:w + train + test]
            seg_b = aligned[r["b"]][w:w + train + test]
            lb = int(max(20, min(120, r["half_life"] * 3)))
            res = backtest_pairs(seg_a, seg_b, lookback=lb, entry_z=entry_z, exit_z=exit_z,
                                 cost_bps=cost_bps, hedge=r["hedge"],  # hedge fit on TRAIN only
                                 use_kalman=use_kalman)
            eq = res.equity_curve
            seg = eq[-(test + 1):] if len(eq) > test else eq
            rr = [(seg[i] - seg[i - 1]) / seg[i - 1] for i in range(1, len(seg)) if seg[i - 1] > 0]
            if rr:
                per_pair.append(rr)
        if per_pair:
            m = min(len(x) for x in per_pair)
            for t in range(m):
                oos_rets.append(sum(x[t] for x in per_pair) / len(per_pair))
            windows += 1
        w += test

    eq = [starting_equity]
    for r in oos_rets:
        eq.append(eq[-1] * (1 + r))
    total = (eq[-1] - starting_equity) / starting_equity if len(eq) > 1 else 0.0
    return WalkForwardResult(len(syms), windows, len(oos_rets), total,
                             sharpe_of(eq, periods_per_year), max_drawdown_of(eq),
                             train, test, top_n, eq)


def kalman_spread(la: List[float], lb: List[float], *, delta: float = 1e-4, R: float = 1e-3):
    """
    Time-varying hedge via a 2-state Kalman filter (intercept + hedge ratio modeled as a
    random walk). Returns (betas, innovations, std): the innovation is the ADAPTIVE spread
    and std = sqrt(its variance S_t), so z_t = innovation/std. `delta` sets how fast the
    hedge tracks (bigger = faster); `R` is observation noise. Pure-python 2x2 (no numpy).
    """
    q = delta / (1.0 - delta)
    t0, t1 = 0.0, 1.0
    p00, p01, p11 = 1.0, 0.0, 1.0
    betas, innov, sd = [], [], []
    for i in range(len(la)):
        x, y = lb[i], la[i]
        p00 += q
        p11 += q                       # predict: add process noise to the diagonal
        hp0 = p00 + x * p01            # H·P, H=[1,x], P symmetric
        hp1 = p01 + x * p11
        S = hp0 + x * hp1 + R
        e = y - (t0 + t1 * x)
        k0, k1 = hp0 / S, hp1 / S
        t0 += k0 * e
        t1 += k1 * e
        p00 -= k0 * hp0
        p01 -= k0 * hp1
        p11 -= k1 * hp1
        betas.append(t1)
        innov.append(e)
        sd.append(math.sqrt(S) if S > 0 else 0.0)
    return betas, innov, sd


def backtest_pairs(a_bars: List[dict], b_bars: List[dict], *, sym_a: str = "A", sym_b: str = "B",
                   lookback: int = 60, entry_z: float = 2.0, exit_z: float = 0.5,
                   starting_equity: float = 100_000.0, leg_notional_pct: float = 0.5,
                   cost_bps: float = 2.0, periods_per_year: float = 252.0,
                   hedge: Optional[float] = None, use_kalman: bool = False) -> PairsResult:
    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of

    rows = align(a_bars, b_bars)
    if len(rows) < lookback + 5:
        return PairsResult(sym_a, sym_b, len(rows), 0.0, 0.0, 0.0, 0, hedge or 1.0, [])

    la = [math.log(c) for _, c, _ in rows]
    lb = [math.log(c) for _, _, c in rows]
    if use_kalman:
        # adaptive hedge: z from the Kalman innovation; warm up `lookback` bars to converge
        betas, innov, sdk = kalman_spread(la, lb)
        h = betas[-1]
        z_series = [innov[i] / sdk[i] if (i >= lookback and sdk[i] > 0) else None
                    for i in range(len(rows))]
    else:
        h = hedge if hedge is not None else _hedge_ratio(la, lb)
        spread = [la[i] - h * lb[i] for i in range(len(rows))]
        z_series = []
        win = deque(maxlen=lookback)
        for i in range(len(rows)):
            win.append(spread[i])
            if len(win) >= lookback:
                mu = statistics.fmean(win)
                sd = statistics.pstdev(win)
                z_series.append((spread[i] - mu) / sd if sd > 0 else 0.0)
            else:
                z_series.append(None)

    cash = starting_equity
    qa = qb = 0.0
    state = 0          # +1 long-spread (long A/short B), -1 short-spread, 0 flat
    trades = 0
    equity: List[float] = []

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
        z = z_series[i]
        if z is not None:
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


@dataclass
class DelistAwareResult:
    n_symbols: int
    n_delisted: int
    n_windows: int
    sharpe: float
    total_return: float
    max_drawdown: float
    delisted_leg_trades: int
    delisted_names_traded: List[str] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)


def delisting_aware_walkforward(
    bars_by_sym: Dict[str, List[dict]], *, delisted_syms: Optional[set] = None,
    train: int = 252, test: int = 63, top_n: int = 10, max_half_life: float = 30.0,
    min_half_life: float = 3.0, entry_z: float = 2.0, exit_z: float = 0.5, cost_bps: float = 2.0,
    max_adf: Optional[float] = -2.86, periods_per_year: float = 252.0,
) -> DelistAwareResult:
    """Walk-forward pairs that ALLOWS partial-history (delisted) names — the survivorship-honest
    version of `walkforward_pairs` (which uses a global timestamp INTERSECTION and so structurally
    excludes any name that delists mid-sample). Master calendar = UNION of all timestamps. Each window:
    screen among names with >=80% real bars in the TRAIN sub-window; backtest each top pair on the TEST
    sub-window via the pair's own bars (a leg that delists mid-window runs out of bars -> the position
    closes at its last real price, capturing an acquisition freeze or a crash). `delisted_syms` is only
    for accounting (which delisted names actually traded). On a survivor-only universe this reproduces
    `walkforward_pairs` exactly."""
    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of

    delisted_syms = delisted_syms or set()
    syms = sorted(bars_by_sym)
    master = sorted({int(b["timestamp"]) for s in syms for b in bars_by_sym[s]})
    bymap = {s: {int(b["timestamp"]): b for b in bars_by_sym[s]} for s in syms}
    n = len(master)
    if n < train + 2 * test or len(syms) < 4:
        return DelistAwareResult(len(syms), len(delisted_syms), 0, 0.0, 0.0, 0.0, 0, [], [1.0])
    oos_rets: List[float] = []
    windows = 0
    del_leg_trades = 0
    del_traded: set = set()
    w = 0
    while w + train + test <= n:
        train_ts = master[w:w + train]
        test_ts = master[w + train:w + train + test]
        cand = [s for s in syms if sum(1 for t in train_ts if t in bymap[s]) >= 0.8 * train]
        if len(cand) < 4:
            w += test
            continue
        train_slice = {s: [bymap[s][t] for t in train_ts if t in bymap[s]] for s in cand}
        screened = screen_pairs(cand, train_slice, min_overlap=int(train * 0.8),
                                max_half_life=max_half_life, min_half_life=min_half_life, max_adf=max_adf)
        win_span = train_ts + test_ts
        per_pair: List[List[float]] = []
        for r in screened[:top_n]:
            a, b = r["a"], r["b"]
            for leg in (a, b):
                if leg in delisted_syms:
                    del_leg_trades += 1
                    del_traded.add(leg)
            seg_a = [bymap[a][t] for t in win_span if t in bymap[a]]
            seg_b = [bymap[b][t] for t in win_span if t in bymap[b]]
            if len(seg_a) < train * 0.5 or len(seg_b) < train * 0.5:
                continue
            lb = int(max(20, min(120, r["half_life"] * 3)))
            res = backtest_pairs(seg_a, seg_b, lookback=lb, entry_z=entry_z, exit_z=exit_z,
                                 cost_bps=cost_bps, hedge=r["hedge"])
            eq = res.equity_curve
            seg = eq[-(test + 1):] if len(eq) > test else eq
            rr = [(seg[i] - seg[i - 1]) / seg[i - 1] for i in range(1, len(seg)) if seg[i - 1] > 0]
            if rr:
                per_pair.append(rr)
        if per_pair:
            m = min(len(x) for x in per_pair)
            for t in range(m):
                oos_rets.append(sum(x[t] for x in per_pair) / len(per_pair))
            windows += 1
        w += test
    eq = [1.0]
    for r in oos_rets:
        eq.append(eq[-1] * (1 + r))
    return DelistAwareResult(
        n_symbols=len(syms), n_delisted=len(delisted_syms), n_windows=windows,
        sharpe=sharpe_of(eq, periods_per_year), total_return=eq[-1] - 1.0,
        max_drawdown=max_drawdown_of(eq), delisted_leg_trades=del_leg_trades,
        delisted_names_traded=sorted(del_traded), equity_curve=eq)
