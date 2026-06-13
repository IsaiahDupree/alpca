"""
Value composite (cross-sectional) — a second FUNDAMENTAL family, on SEC EDGAR data + price.

Rank the universe by a blend of yield metrics — E/P (earnings yield), FCF/P (free-cash-flow yield),
B/P (book-to-price) — and go LONG the cheap (high composite yield) / SHORT the expensive, dollar-
neutral. The value premium is the most-studied anomaly; it's orthogonal to momentum/positioning and
to the accruals signal, so a surviving version would genuinely diversify the combiner.

Market cap = shares_outstanding × price, so each yield re-prices daily as the stock moves; the
fundamental (NI / FCF / book equity / shares) is the most recent 10-K known at the rebalance day
(EDGAR `filed` date — no look-ahead). Rebalances every `rebalance_days` (value is slow → low turnover).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from alpca.data.earnings import _epoch


@dataclass
class ValueResult:
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


def _snapshots(rows: List[dict]) -> List[dict]:
    """Value fundamentals per fiscal year, public at `filed`. Needs shares (for market cap)."""
    out = []
    for r in sorted(rows, key=lambda x: x.get("fy_end", "")):
        sh = r.get("shares")
        ep = _epoch(r.get("filed", ""), "%Y-%m-%d")
        if sh and sh > 0 and ep is not None:
            out.append({"filed_epoch": int(ep), "ni": r.get("net_income"),
                        "fcf": r.get("fcf"), "book": r.get("book_equity"), "shares": float(sh)})
    return out


def _rank01(x: np.ndarray, ok: np.ndarray) -> np.ndarray:
    """Cross-sectional percentile rank in [0,1] among valid entries (higher value -> higher rank)."""
    out = np.full(x.shape, np.nan)
    idx = np.where(ok)[0]
    if len(idx) < 2:
        return out
    order = idx[np.argsort(x[idx])]
    for r, j in enumerate(order):
        out[j] = r / (len(order) - 1)
    return out


def backtest_value_composite(
    bars_by_sym: Dict[str, List[dict]],
    fund_by_sym: Dict[str, List[dict]], *,
    top_frac: float = 0.2,
    rebalance_days: int = 21,
    cost_bps: float = 2.0,
    reverse: bool = False,            # False = value (long cheap); True = the anti-value control
    periods_per_year: float = 252.0,
    starting_equity: float = 100_000.0,
) -> ValueResult:
    syms = sorted(s for s in bars_by_sym if _snapshots(fund_by_sym.get(s) or []))
    if len(syms) < max(5, int(round(2 / max(top_frac, 1e-9)))):
        return ValueResult(syms, 0, 0.0, 0.0, 0.0, 0, top_frac, 0.0, 0.0, [starting_equity], [])

    master = sorted({int(b["timestamp"]) for s in syms for b in bars_by_sym[s]})
    idx = {t: i for i, t in enumerate(master)}
    T, N = len(master), len(syms)
    if T < rebalance_days + 5:
        return ValueResult(syms, 0, 0.0, 0.0, 0.0, 0, top_frac, 0.0, 0.0, [starting_equity], [])

    price = np.full((T, N), np.nan)
    ret = np.zeros((T, N))
    ni = np.full((T, N), np.nan); fcf = np.full((T, N), np.nan)
    book = np.full((T, N), np.nan); shares = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        b = sorted(bars_by_sym[s], key=lambda x: int(x["timestamp"]))
        ts = [int(x["timestamp"]) for x in b]
        cl = [float(x["close"]) for x in b]
        for k in range(len(b)):
            price[idx[ts[k]], j] = cl[k]
            if k > 0 and cl[k - 1] > 0:
                ret[idx[ts[k]], j] = cl[k] / cl[k - 1] - 1.0
        for snap in _snapshots(fund_by_sym[s]):
            k0 = next((i for i, t in enumerate(master) if t >= snap["filed_epoch"]), None)
            if k0 is not None:
                ni[k0:, j] = snap["ni"] if snap["ni"] is not None else np.nan
                fcf[k0:, j] = snap["fcf"] if snap["fcf"] is not None else np.nan
                book[k0:, j] = snap["book"] if snap["book"] is not None else np.nan
                shares[k0:, j] = snap["shares"]

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
            p, sh = price[t - 1], shares[t - 1]      # known entering day t (no look-ahead)
            mc = np.where((p > 0) & (sh > 0), p * sh, np.nan)
            vmc = np.isfinite(mc)
            ep = np.where(vmc & np.isfinite(ni[t - 1]), ni[t - 1] / mc, np.nan)
            fp = np.where(vmc & np.isfinite(fcf[t - 1]), fcf[t - 1] / mc, np.nan)
            bp = np.where(vmc & np.isfinite(book[t - 1]), book[t - 1] / mc, np.nan)
            stack = np.vstack([_rank01(m, np.isfinite(m)) for m in (ep, fp, bp)])
            cnt = np.sum(np.isfinite(stack), axis=0)
            comp = np.where(cnt > 0, np.nansum(stack, axis=0) / np.maximum(cnt, 1), np.nan)
            ok = np.isfinite(comp)
            if ok.sum() >= 2 * k:
                order = np.argsort(np.where(ok, comp, -np.inf))
                order = order[np.isin(order, np.where(ok)[0])]
                cheap, expensive = order[-k:], order[:k]      # highest composite yield = cheapest
                lng, sht = (cheap, expensive) if not reverse else (expensive, cheap)
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
    return ValueResult(
        symbols=syms, n_days=len(daily), total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq), n_rebalances=rebals,
        top_frac=top_frac, avg_active=float(np.mean(actives)) if actives else 0.0,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        equity_curve=eq, daily_returns=daily, dates=master[1:])
