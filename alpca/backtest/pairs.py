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


def ou_sigma_eq(spread: List[float]) -> float:
    """Equilibrium (stationary) std of the OU/AR(1) fit on a spread: ou_std = sigma_eps / sqrt(1 - phi^2)
    where d(spread)_t = a + lam*spread_{t-1} + eps, phi = 1+lam (the AR(1) coefficient), and sigma_eps is
    the residual std of that regression. This is the natural z=1 width of the spread (equivalent to the OU
    sigma/sqrt(2*theta) in the continuous limit). Estimated TRAIN-only -> no look-ahead. Returns 0.0 if the
    spread is not mean-reverting (phi>=1) or too short."""
    n = len(spread)
    if n < 5:
        return 0.0
    slag = spread[:-1]
    ds = [spread[i] - spread[i - 1] for i in range(1, n)]
    ml = statistics.fmean(slag)
    md = statistics.fmean(ds)
    var = sum((x - ml) ** 2 for x in slag)
    if var <= 0:
        return 0.0
    lam = sum((slag[i] - ml) * (ds[i] - md) for i in range(len(ds))) / var
    a = md - lam * ml
    resid = [ds[i] - (a + lam * slag[i]) for i in range(len(ds))]
    if len(resid) < 3:
        return 0.0
    sig_eps = statistics.pstdev(resid)
    phi = 1.0 + lam
    if not (-1.0 < phi < 1.0) or sig_eps <= 0:
        return 0.0
    return sig_eps / math.sqrt(1.0 - phi * phi)


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
    daily_returns: List[float] = field(default_factory=list)
    dates: List[int] = field(default_factory=list)  # epoch per OOS daily return (test-window calendar)


def adf_pvalue(adf_t: float) -> float:
    """Crude Dickey-Fuller t-stat -> p-value via MacKinnon-style interpolation over the standard
    no-trend critical values (1% -3.43, 5% -2.86, 10% -2.57). Monotone, piecewise-linear, clipped to
    [0,1]. Used ONLY by the regime monitor (refinement c) to express rolling cointegration health on the
    article's p-value scale (warn 0.10 / halt 0.20). More-negative t -> smaller p (more stationary)."""
    pts = [(-3.43, 0.01), (-2.86, 0.05), (-2.57, 0.10), (-1.95, 0.30), (-1.62, 0.50),
           (-0.50, 0.80), (0.0, 0.95), (1.0, 0.99)]
    if adf_t <= pts[0][0]:
        return 0.005
    if adf_t >= pts[-1][0]:
        return 0.995
    for i in range(1, len(pts)):
        x0, p0 = pts[i - 1]
        x1, p1 = pts[i]
        if adf_t <= x1:
            return p0 + (p1 - p0) * (adf_t - x0) / (x1 - x0)
    return 0.995


def walkforward_pairs(bars_by_sym: Dict[str, List[dict]], *, train: int = 252, test: int = 63,
                      top_n: int = 15, max_half_life: float = 30.0, min_half_life: float = 3.0,
                      entry_z: float = 2.0, exit_z: float = 0.5, cost_bps: float = 2.0,
                      starting_equity: float = 100_000.0, periods_per_year: float = 252.0,
                      max_adf: Optional[float] = None, use_kalman: bool = False,
                      cost_cal_entry: bool = False, ou_sizing: bool = False,
                      regime_monitor: bool = False, regime_window: int = 60,
                      regime_warn_p: float = 0.10, regime_halt_p: float = 0.20) -> "WalkForwardResult":
    """
    Rigorous walk-forward market-neutral pairs. Each step: SCREEN + hedge-fit on a trailing
    `train` window, then TRADE the selected top_n pairs on the next `test` window (genuinely
    out-of-sample — selection & hedge used only past data), roll forward by `test`. The OOS
    test-window basket returns are concatenated into one continuous equity curve, so the
    resulting Sharpe is the HONEST number (every trade is on unseen data). Pairs are
    re-selected each window (rolling re-screen) and the hedge is re-fit each window (rolling
    hedge) — the two upgrades over a single static screen.

    REFINEMENTS (all default-OFF, additive, lookahead-free — params from TRAIN/trailing only):
      - `cost_cal_entry` (a): per pair, act_entry_z = max(entry_z, 4*cost_frac/ou_std) where ou_std is the
        equilibrium std of the OU/AR(1) fit on the TRAIN spread (z-units, since z is standardized by that
        same spread scale). Skips pairs whose threshold is unreachable (act_entry_z too large to fire).
      - `ou_sizing` (b): each leg sized leg_notional_pct * min(|z|/act_entry_z, 1.0) (max_fraction =
        leg_notional_pct -> pure reshape, no leverage bump).
      - `regime_monitor` (c): per pair, a rolling ADF p-value on the trailing `regime_window` of the hedged
        spread (TRAIN hedge) gates the TEST-window position: ACTIVE (p < warn) trade normally / WARNING
        (warn<=p<halt) hold, open nothing new / HALTED (p>=halt) flatten the pair. Risk overlay only.

    EMPIRICAL VERDICT (Case 56, 4-agent gauntlet — do NOT re-litigate):
      - `cost_cal_entry` + `ou_sizing` are INERT at the deployable ≤~20bps cost: at 2bps they reproduce the
        baseline daily returns bit-for-bit (4*cost_frac/ou_std << entry_z=2.0 for every selected pair, so
        act_entry_z=2.0 always and the OU reshape never binds). Lookahead-clean, no improvement, nothing to
        commit as an edge. They would only bind on far tighter spreads or much higher costs.
      - `regime_monitor` is a FOOTGUN — DO NOT ENABLE. It DESTROYS the edge (WF 0.83 -> -0.32, Sortino
        1.24 -> -0.41, ret +14.9% -> -1.1%): the rolling-ADF gate flattens pairs exactly when the spread is
        mean-reverting hardest (its most profitable state). Kept default-OFF for reproducibility only.
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
    cost_frac = cost_bps / 1e4

    oos_rets: List[float] = []
    oos_dates: List[int] = []
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
            h = r["hedge"]
            act_z: Optional[float] = None
            if cost_cal_entry or ou_sizing:
                # OU std on the TRAIN spread only (no look-ahead)
                tr_a = aligned[r["a"]][w:w + train]
                tr_b = aligned[r["b"]][w:w + train]
                tr_rows = align(tr_a, tr_b)
                la_t = [math.log(c) for _, c, _ in tr_rows]
                lb_t = [math.log(c) for _, _, c in tr_rows]
                tr_spread = [la_t[k] - h * lb_t[k] for k in range(len(tr_rows))]
                ou_std = ou_sigma_eq(tr_spread)
                if ou_std <= 0:
                    continue                                   # not mean-reverting on TRAIN -> skip
                act_z = max(entry_z, 4.0 * cost_frac / ou_std)
                if act_z > 8.0:        # unreachable threshold -> pair too tight to cover cost, skip
                    continue
            if regime_monitor:
                res = _backtest_pairs_regime(
                    seg_a, seg_b, lookback=lb, entry_z=entry_z, exit_z=exit_z, cost_bps=cost_bps,
                    hedge=h, act_entry_z=(act_z if cost_cal_entry else None),
                    ou_sizing=ou_sizing, regime_window=regime_window, train_len=len(align(
                        aligned[r["a"]][w:w + train], aligned[r["b"]][w:w + train])),
                    warn_p=regime_warn_p, halt_p=regime_halt_p)
            else:
                res = backtest_pairs(seg_a, seg_b, lookback=lb, entry_z=entry_z, exit_z=exit_z,
                                     cost_bps=cost_bps, hedge=h,  # hedge fit on TRAIN only
                                     use_kalman=use_kalman,
                                     act_entry_z=(act_z if cost_cal_entry else None),
                                     ou_sizing=ou_sizing)
            eq = res.equity_curve
            seg = eq[-(test + 1):] if len(eq) > test else eq
            rr = [(seg[i] - seg[i - 1]) / seg[i - 1] for i in range(1, len(seg)) if seg[i - 1] > 0]
            if rr:
                per_pair.append(rr)
        if per_pair:
            m = min(len(x) for x in per_pair)
            # the m aggregated returns are the LAST m bars of the test window (returns are
            # bar-to-bar, so they map to the final m of common[w+train : w+train+test]).
            test_ts = [int(t) for t in common[w + train:w + train + test]]
            date_tail = test_ts[-m:] if len(test_ts) >= m else test_ts
            for t in range(m):
                oos_rets.append(sum(x[t] for x in per_pair) / len(per_pair))
                oos_dates.append(date_tail[t] if t < len(date_tail) else 0)
            windows += 1
        w += test

    eq = [starting_equity]
    for r in oos_rets:
        eq.append(eq[-1] * (1 + r))
    total = (eq[-1] - starting_equity) / starting_equity if len(eq) > 1 else 0.0
    return WalkForwardResult(len(syms), windows, len(oos_rets), total,
                             sharpe_of(eq, periods_per_year), max_drawdown_of(eq),
                             train, test, top_n, eq, oos_rets, oos_dates)


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
                   hedge: Optional[float] = None, use_kalman: bool = False,
                   act_entry_z: Optional[float] = None,
                   ou_sizing: bool = False) -> PairsResult:
    """REFINEMENTS (default-OFF, additive — do not change validated behavior):
      - `act_entry_z` (refinement a, cost-calibrated min entry-z): if set, this REPLACES `entry_z` as the
        effective entry threshold (the caller computes act_entry_z = max(entry_z, 4*cost_frac/ou_std) on the
        TRAIN spread). A pair whose computed act_entry_z is unreachable simply never trades -> it is skipped.
      - `ou_sizing` (refinement b, OU-proportional sizing): when True, each leg is sized
        leg_notional_pct * min(|z|/act_entry_z, 1.0) at entry (max_fraction = leg_notional_pct -> pure reshape,
        no leverage bump). Biggest at the threshold, smaller deeper in (de-risks into convergence).
    """
    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of

    rows = align(a_bars, b_bars)
    if len(rows) < lookback + 5:
        return PairsResult(sym_a, sym_b, len(rows), 0.0, 0.0, 0.0, 0, hedge or 1.0, [])

    eff_entry = act_entry_z if act_entry_z is not None else entry_z
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

    def rebalance(target: int, pa: float, pb: float, frac: float = 1.0):
        nonlocal cash, qa, qb, state, trades
        if target == state:
            return
        # close existing legs
        cash += qa * pa + qb * pb
        cash -= cost_bps / 1e4 * (abs(qa) * pa + abs(qb) * pb)
        qa = qb = 0.0
        if target != 0:
            leg = leg_notional_pct * frac * cash      # frac=1.0 by default (validated); <1 only under ou_sizing
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
                if z > eff_entry:
                    f = min(abs(z) / eff_entry, 1.0) if ou_sizing else 1.0
                    rebalance(-1, pa, pb, f)
                elif z < -eff_entry:
                    f = min(abs(z) / eff_entry, 1.0) if ou_sizing else 1.0
                    rebalance(1, pa, pb, f)
            elif state == 1 and z >= -exit_z:
                rebalance(0, pa, pb)
            elif state == -1 and z <= exit_z:
                rebalance(0, pa, pb)
        equity.append(cash + qa * pa + qb * pb)

    total_return = (equity[-1] - starting_equity) / starting_equity if equity else 0.0
    return PairsResult(sym_a, sym_b, len(rows), total_return,
                       sharpe_of(equity, periods_per_year), max_drawdown_of(equity),
                       trades, h, equity)


def _backtest_pairs_regime(a_bars: List[dict], b_bars: List[dict], *, sym_a: str = "A", sym_b: str = "B",
                           lookback: int = 60, entry_z: float = 2.0, exit_z: float = 0.5,
                           starting_equity: float = 100_000.0, leg_notional_pct: float = 0.5,
                           cost_bps: float = 2.0, periods_per_year: float = 252.0,
                           hedge: Optional[float] = None, act_entry_z: Optional[float] = None,
                           ou_sizing: bool = False, regime_window: int = 60,
                           train_len: int = 0, warn_p: float = 0.10, halt_p: float = 0.20) -> PairsResult:
    """Refinement (c) — 3-state cointegration-health regime monitor as a RISK OVERLAY on the
    classic z-entry pairs backtest. Identical to `backtest_pairs` (fixed TRAIN hedge `h`, rolling
    z-score, optional cost-calibrated `act_entry_z` and `ou_sizing`) EXCEPT each bar first computes a
    rolling ADF p-value on the trailing `regime_window` of the hedged spread and gates positions:
      ACTIVE   (p <  warn_p)            -> trade normally (z-entry / z-exit as usual)
      WARNING  (warn_p <= p <  halt_p)  -> HOLD existing position, open NOTHING new
      HALTED   (p >= halt_p)            -> FLATTEN the pair immediately, open nothing
    Lookahead-free: the rolling ADF uses only the trailing `regime_window` spread values up to and
    INCLUDING the current bar (the spread itself is built with the TRAIN-only hedge `h`). `train_len`
    is informational (the regime monitor is active across the whole segment, gating the test window).
    No new leverage: sizing identical to `backtest_pairs`; this only ever REDUCES exposure."""
    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of

    rows = align(a_bars, b_bars)
    if len(rows) < lookback + 5:
        return PairsResult(sym_a, sym_b, len(rows), 0.0, 0.0, 0.0, 0, hedge or 1.0, [])

    eff_entry = act_entry_z if act_entry_z is not None else entry_z
    la = [math.log(c) for _, c, _ in rows]
    lb = [math.log(c) for _, _, c in rows]
    h = hedge if hedge is not None else _hedge_ratio(la, lb)
    spread = [la[i] - h * lb[i] for i in range(len(rows))]

    z_series: List[Optional[float]] = []
    win = deque(maxlen=lookback)
    for i in range(len(rows)):
        win.append(spread[i])
        if len(win) >= lookback:
            mu = statistics.fmean(win)
            sd = statistics.pstdev(win)
            z_series.append((spread[i] - mu) / sd if sd > 0 else 0.0)
        else:
            z_series.append(None)

    # rolling regime state per bar from a trailing-window ADF p-value (no look-ahead)
    rwin = max(20, regime_window)
    regime: List[int] = []  # 0 ACTIVE, 1 WARNING, 2 HALTED
    for i in range(len(rows)):
        if i + 1 < rwin:
            regime.append(0)  # warm-up: assume healthy (no gating until window fills)
            continue
        seg = spread[i + 1 - rwin:i + 1]
        p = adf_pvalue(adf_stat(seg))
        regime.append(2 if p >= halt_p else (1 if p >= warn_p else 0))

    cash = starting_equity
    qa = qb = 0.0
    state = 0
    trades = 0
    equity: List[float] = []

    def rebalance(target: int, pa: float, pb: float, frac: float = 1.0):
        nonlocal cash, qa, qb, state, trades
        if target == state:
            return
        cash += qa * pa + qb * pb
        cash -= cost_bps / 1e4 * (abs(qa) * pa + abs(qb) * pb)
        qa = qb = 0.0
        if target != 0:
            leg = leg_notional_pct * frac * cash
            qa = (leg / pa) * target
            qb = (leg / pb) * (-target)
            cash -= qa * pa + qb * pb
            cash -= cost_bps / 1e4 * (abs(qa) * pa + abs(qb) * pb)
            trades += 1
        state = target

    for i in range(len(rows)):
        _, pa, pb = rows[i]
        z = z_series[i]
        reg = regime[i]
        if reg == 2:                         # HALTED -> flatten, no trading
            if state != 0:
                rebalance(0, pa, pb)
        elif z is not None:
            if state == 0:
                if reg == 0:                 # only open NEW positions when ACTIVE
                    if z > eff_entry:
                        f = min(abs(z) / eff_entry, 1.0) if ou_sizing else 1.0
                        rebalance(-1, pa, pb, f)
                    elif z < -eff_entry:
                        f = min(abs(z) / eff_entry, 1.0) if ou_sizing else 1.0
                        rebalance(1, pa, pb, f)
            elif state == 1 and z >= -exit_z:  # exits always allowed (ACTIVE or WARNING)
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
    daily_returns: List[float] = field(default_factory=list)
    dates: List[int] = field(default_factory=list)         # epoch per daily return (test-window calendar)


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
    oos_dates: List[int] = []
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
        test_set = set(test_ts)
        # per pair: {test-window date -> that pair's return on that date}. Building it from the pair's
        # OWN joined calendar (not a tail slice of the equity curve) is what keeps a delisted leg honest:
        # a leg that delists mid-window simply has no joined bars after delisting, so its test-window
        # contribution naturally ends there — no train-window bars bleed in, no date mislabeling.
        per_pair: List[Dict[int, float]] = []
        for r in screened[:top_n]:
            a, b = r["a"], r["b"]
            for leg in (a, b):
                if leg in delisted_syms:
                    del_leg_trades += 1
                    del_traded.add(leg)
            joined_ts = [t for t in win_span if t in bymap[a] and t in bymap[b]]   # 1:1 w/ aligned eq rows
            if len(joined_ts) < train * 0.5:
                continue
            seg_a = [bymap[a][t] for t in joined_ts]
            seg_b = [bymap[b][t] for t in joined_ts]
            lb = int(max(20, min(120, r["half_life"] * 3)))
            res = backtest_pairs(seg_a, seg_b, lookback=lb, entry_z=entry_z, exit_z=exit_z,
                                 cost_bps=cost_bps, hedge=r["hedge"])
            eq = res.equity_curve                       # eq[i] = value after joined_ts[i]
            rr = {joined_ts[i]: eq[i] / eq[i - 1] - 1.0
                  for i in range(1, min(len(eq), len(joined_ts)))
                  if eq[i - 1] > 0 and joined_ts[i] in test_set}    # TEST-window days only, by real date
            if rr:
                per_pair.append(rr)
        if per_pair:
            # aggregate per calendar date: equal-weight the pairs that actually traded that day
            by_date: Dict[int, List[float]] = {}
            for pr in per_pair:
                for dt, ret in pr.items():
                    by_date.setdefault(dt, []).append(ret)
            for dt in sorted(by_date):
                vals = by_date[dt]
                oos_rets.append(sum(vals) / len(vals))
                oos_dates.append(dt)
            windows += 1
        w += test
    eq = [1.0]
    for r in oos_rets:
        eq.append(eq[-1] * (1 + r))
    return DelistAwareResult(
        n_symbols=len(syms), n_delisted=len(delisted_syms), n_windows=windows,
        sharpe=sharpe_of(eq, periods_per_year), total_return=eq[-1] - 1.0,
        max_drawdown=max_drawdown_of(eq), delisted_leg_trades=del_leg_trades,
        delisted_names_traded=sorted(del_traded), equity_curve=eq,
        daily_returns=oos_rets, dates=oos_dates)
