"""
Generic cross-sectional factor engine — one tested harness for the whole untested-factor backlog
(asset growth, net issuance, ROA, MAX, idio-vol, residual/vol-managed momentum, ...). A factor is
just a SIGNAL: a function `signal_fn(master, syms, price) -> (T,N)` array where `signal[t,j]` is the
factor value KNOWN AS OF day t (NaN where unavailable). The engine ranks it cross-sectionally each
rebalance, goes LONG the high (or low) quantile / SHORT the other, dollar-neutral, applies turnover
cost, and returns a gate-ready result (equity, per-year, dates).

No look-ahead: the book held through day t (earning ret[t]) is set from `signal[t-1]`. Signal builders
must therefore only use information available by their day index (trailing windows; fundamentals
lagged to the 10-K filing date).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np

from alpca.data.earnings import _epoch


@dataclass
class FactorResult:
    name: str
    symbols: List[str]
    n_days: int
    total_return: float
    sharpe: float
    max_drawdown: float
    n_rebalances: int
    top_frac: float
    avg_active: float
    avg_turnover: float
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    dates: List[int] = field(default_factory=list)


def _price_ret(bars_by_sym, syms, master):
    idx = {t: i for i, t in enumerate(master)}
    T, N = len(master), len(syms)
    price = np.full((T, N), np.nan)
    ret = np.zeros((T, N))
    for j, s in enumerate(syms):
        b = sorted(bars_by_sym[s], key=lambda x: int(x["timestamp"]))
        ts = [int(x["timestamp"]) for x in b]
        cl = [float(x["close"]) for x in b]
        for k in range(len(b)):
            price[idx[ts[k]], j] = cl[k]
            if k > 0 and cl[k - 1] > 0:
                ret[idx[ts[k]], j] = cl[k] / cl[k - 1] - 1.0
    return price, ret


def backtest_factor(
    bars_by_sym: Dict[str, List[dict]],
    signal_fn: Callable, *,
    name: str = "factor",
    top_frac: float = 0.2,
    rebalance_days: int = 21,
    cost_bps: float = 2.0,
    long_high: bool = True,           # True = long high-signal/short low; False = long low/short high
    periods_per_year: float = 252.0,
    starting_equity: float = 100_000.0,
) -> FactorResult:
    syms = sorted(bars_by_sym)
    if len(syms) < max(5, int(round(2 / max(top_frac, 1e-9)))):
        return FactorResult(name, syms, 0, 0.0, 0.0, 0.0, 0, top_frac, 0.0, 0.0, [starting_equity], [])
    master = sorted({int(b["timestamp"]) for s in syms for b in bars_by_sym[s]})
    T, N = len(master), len(syms)
    if T < rebalance_days + 5:
        return FactorResult(name, syms, 0, 0.0, 0.0, 0.0, 0, top_frac, 0.0, 0.0, [starting_equity], [])
    price, ret = _price_ret(bars_by_sym, syms, master)
    signal = signal_fn(master, syms, price)        # (T, N), value known AS OF day t

    k = max(1, int(round(N * top_frac)))
    eq = [starting_equity]
    daily: List[float] = []
    actives: List[int] = []
    turnovers: List[float] = []
    rebals = 0
    w = np.zeros(N)
    prev_w = np.zeros(N)
    for t in range(1, T):
        if (t - 1) % rebalance_days == 0:
            s = signal[t - 1]                       # known entering day t (no look-ahead)
            ok = np.isfinite(s)
            if ok.sum() >= 2 * k:
                order = np.argsort(np.where(ok, s, np.inf))   # ascending
                order = order[np.isin(order, np.where(ok)[0])]
                low, high = order[:k], order[-k:]
                lng, sht = (high, low) if long_high else (low, high)
                w = np.zeros(N)
                w[lng] = 0.5 / k
                w[sht] = -0.5 / k
                rebals += 1
        turnover = float(np.abs(w - prev_w).sum())
        port_ret = float(np.nansum(w * np.nan_to_num(ret[t]))) - turnover * (cost_bps / 1e4)
        eq.append(eq[-1] * (1 + port_ret))
        daily.append(port_ret)
        actives.append(int((w != 0).sum()))
        turnovers.append(turnover)
        prev_w = w

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return FactorResult(
        name=name, symbols=syms, n_days=len(daily), total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq), n_rebalances=rebals,
        top_frac=top_frac, avg_active=float(np.mean(actives)) if actives else 0.0,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        equity_curve=eq, daily_returns=daily, dates=master[1:])


# ---------- signal builders (each returns a signal_fn(master, syms, price) -> (T,N)) ----------
def _fundamental_step(fund_by_sym, value_of):
    """Build a step-function signal from annual fundamentals, lagged to the 10-K FILED date.
    `value_of(rows, i)` computes the factor value for fiscal-year row i (or None to skip)."""
    def fn(master, syms, price):
        T, N = len(master), len(syms)
        sig = np.full((T, N), np.nan)
        for j, s in enumerate(syms):
            rows = sorted(fund_by_sym.get(s, []), key=lambda r: r.get("fy_end", ""))
            for i in range(len(rows)):
                v = value_of(rows, i)
                ep = _epoch(rows[i].get("filed", ""), "%Y-%m-%d")
                if v is None or ep is None:
                    continue
                k0 = next((m for m, t in enumerate(master) if t >= ep), None)
                if k0 is not None:
                    sig[k0:, j] = v
        return sig
    return fn


def asset_growth_signal(fund_by_sym):
    def v(rows, i):
        if i == 0:
            return None
        a, ap = rows[i].get("total_assets"), rows[i - 1].get("total_assets")
        return (a - ap) / ap if (a and ap and ap > 0) else None
    return _fundamental_step(fund_by_sym, v)


def net_issuance_signal(fund_by_sym):
    def v(rows, i):
        if i == 0:
            return None
        s, sp = rows[i].get("shares"), rows[i - 1].get("shares")
        return (s - sp) / sp if (s and sp and sp > 0) else None
    return _fundamental_step(fund_by_sym, v)


def roa_signal(fund_by_sym):
    def v(rows, i):
        ni, a = rows[i].get("net_income"), rows[i].get("total_assets")
        return ni / a if (ni is not None and a and a > 0) else None
    return _fundamental_step(fund_by_sym, v)


def gross_profitability_signal(fund_by_sym):
    def v(rows, i):
        rev, cogs, a = rows[i].get("revenue"), rows[i].get("cogs"), rows[i].get("total_assets")
        return (rev - cogs) / a if (rev is not None and cogs is not None and a and a > 0) else None
    return _fundamental_step(fund_by_sym, v)


def si_change_signal(si_by_sym, pub_lag: int = 10):
    """Change in days-to-cover vs the prior FINRA settlement, lagged to the public dissemination date
    (settlement + pub_lag trading days). NOT the SI *level* (rejected, Case 21) — the *change*."""
    def fn(master, syms, price):
        T, N = len(master), len(syms)
        sig = np.full((T, N), np.nan)
        for j, s in enumerate(syms):
            obs = []
            for r in (si_by_sym.get(s) or []):
                ep = _epoch(r.get("settlement", ""), "%m/%d/%Y")
                if ep is not None and r.get("days_to_cover") is not None:
                    obs.append((int(ep), float(r["days_to_cover"])))
            obs.sort()
            for i in range(1, len(obs)):
                settle_ep, dtc = obs[i]
                change = dtc - obs[i - 1][1]
                k0 = next((m for m, t in enumerate(master) if t >= settle_ep), None)
                if k0 is None:
                    continue
                sig[min(k0 + pub_lag, T - 1):, j] = change
        return sig
    return fn


def short_horizon_return_signal(window: int = 5):
    """Trailing `window`-day cumulative return, known AS OF t-1. Rank LOW (long_high=False) to trade the
    short-horizon REVERSAL anomaly (long last-week's losers, short last-week's winners) — a different
    mechanism/horizon than the pairs basket's pairwise cointegration mean-reversion. Note: reversal is
    turnover-heavy, so the cost wall is the live test."""
    def fn(master, syms, price):
        T, N = price.shape
        sig = np.full((T, N), np.nan)
        for t in range(window, T):
            p0, p1 = price[t - window], price[t]
            sig[t] = np.where((p0 > 0) & np.isfinite(p1), p1 / p0 - 1.0, np.nan)
        return sig
    return fn


def max_return_signal(window: int = 21):
    def fn(master, syms, price):
        T, N = price.shape
        r = np.full((T, N), np.nan)
        r[1:] = np.where(price[:-1] > 0, price[1:] / np.where(price[:-1] > 0, price[:-1], 1.0) - 1.0, np.nan)
        sig = np.full((T, N), np.nan)
        for t in range(window, T):
            sig[t] = np.nanmax(r[t - window + 1:t + 1], axis=0)
        return sig
    return fn


def cross_sectional_seasonality_signal(min_prior: int = 15):
    """Heston-Sadka calendar seasonality: a stock's expected return THIS calendar month, estimated
    from its OWN returns in the SAME calendar month in PRIOR years only (strict no-lookahead — the
    current month's returns never enter its own signal). Long the historically-strong-this-month names,
    short the weak. Orthogonal by construction to trend (momentum) and to pairwise mean-reversion (the
    pairs basket): the P&L is on a calendar clock. `min_prior` = min same-month return observations from
    prior years before a name is rankable. Monthly rebalance is the natural cadence."""
    import time as _time

    def fn(master, syms, price):
        T, N = price.shape
        r = np.full((T, N), np.nan)
        r[1:] = np.where(price[:-1] > 0, price[1:] / np.where(price[:-1] > 0, price[:-1], 1.0) - 1.0, np.nan)
        cal_mon = np.array([_time.gmtime(int(t)).tm_mon for t in master])      # 1..12 per day
        # run id increments whenever (year,month) changes -> a "month-run"
        ym = [(_time.gmtime(int(t)).tm_year, _time.gmtime(int(t)).tm_mon) for t in master]
        run = np.zeros(T, dtype=int)
        for t in range(1, T):
            run[t] = run[t - 1] + (1 if ym[t] != ym[t - 1] else 0)
        # cumulative per-(column, calendar-month) sum/count of returns from COMPLETED prior runs
        csum = np.zeros((13, N)); ccnt = np.zeros((13, N))
        sig = np.full((T, N), np.nan)
        t = 0
        while t < T:
            t2 = t
            while t2 < T and run[t2] == run[t]:
                t2 += 1
            m = cal_mon[t]                                  # this run's calendar month
            with np.errstate(invalid="ignore"):
                mean = np.where(ccnt[m] >= min_prior, csum[m] / np.maximum(ccnt[m], 1), np.nan)
            sig[t:t2] = mean                                # signal for this run = prior-years' same-month mean
            seg = r[t:t2]                                   # now fold THIS run into the accumulator
            csum[m] += np.nansum(seg, axis=0)
            ccnt[m] += np.sum(np.isfinite(seg), axis=0)
            t = t2
        return sig
    return fn


def _bench_ret(bench_bars, master):
    bmap = {int(b["timestamp"]): float(b["close"]) for b in bench_bars}
    bc = [bmap.get(t) for t in master]
    br = np.zeros(len(master))
    for t in range(1, len(master)):
        if bc[t] and bc[t - 1]:
            br[t] = bc[t] / bc[t - 1] - 1.0
    return br


def idiosyncratic_vol_signal(bench_bars, window: int = 120):
    def fn(master, syms, price):
        T, N = price.shape
        r = np.zeros((T, N))
        r[1:] = np.where(price[:-1] > 0, price[1:] / np.where(price[:-1] > 0, price[:-1], 1.0) - 1.0, 0.0)
        br = _bench_ret(bench_bars, master)
        sig = np.full((T, N), np.nan)
        for t in range(window, T):
            bw = br[t - window + 1:t + 1]
            var = float(np.var(bw))
            for j in range(N):
                sw = r[t - window + 1:t + 1, j]
                beta = np.cov(sw, bw)[0, 1] / var if var > 1e-12 else 0.0
                sig[t, j] = np.std(sw - beta * bw)
        return sig
    return fn


def residual_momentum_signal(bench_bars, lookback: int = 120, skip: int = 21):
    """Trailing residual cumulative return (regress out the market), skipping the most recent month."""
    def fn(master, syms, price):
        T, N = price.shape
        r = np.zeros((T, N))
        r[1:] = np.where(price[:-1] > 0, price[1:] / np.where(price[:-1] > 0, price[:-1], 1.0) - 1.0, 0.0)
        br = _bench_ret(bench_bars, master)
        sig = np.full((T, N), np.nan)
        for t in range(lookback + skip, T):
            bw = br[t - lookback - skip:t - skip]
            var = float(np.var(bw))
            for j in range(N):
                sw = r[t - lookback - skip:t - skip, j]
                beta = np.cov(sw, bw)[0, 1] / var if var > 1e-12 else 0.0
                sig[t, j] = float(np.sum(sw - beta * bw))
        return sig
    return fn


def vol_managed_momentum_signal(lookback: int = 120, skip: int = 21, vol_window: int = 60):
    def fn(master, syms, price):
        T, N = price.shape
        r = np.zeros((T, N))
        r[1:] = np.where(price[:-1] > 0, price[1:] / np.where(price[:-1] > 0, price[:-1], 1.0) - 1.0, 0.0)
        sig = np.full((T, N), np.nan)
        for t in range(lookback + skip, T):
            mom = np.nansum(r[t - lookback - skip:t - skip], axis=0)
            vol = np.nanstd(r[t - vol_window:t], axis=0)
            sig[t] = np.where(vol > 1e-9, mom / vol, np.nan)
        return sig
    return fn
