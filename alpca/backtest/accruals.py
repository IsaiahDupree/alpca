"""
Accruals anomaly (Sloan 1996) — the first FUNDAMENTAL edge we've tested, orthogonal to all our
price/positioning work. Earnings made of accruals (vs cash) are lower-quality and mean-revert, so
high-accrual firms underperform and low-accrual (cash-backed) firms outperform. Cash-flow accrual
ratio: ACC = (NetIncome − OperatingCashFlow) / avg(TotalAssets). LONG low-ACC / SHORT high-ACC,
dollar-neutral. Rebalances only when a new 10-K becomes public → annual, very low turnover.

NO LOOK-AHEAD: each fiscal year's accrual is acted on only from its 10-K FILING date (≈2 months
after fiscal year-end) — `fund_by_sym[s]` rows carry `filed`. The position entering day t uses the
most recent accrual whose filed date ≤ t-1, and earns day t's return.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from alpca.data.earnings import _epoch


@dataclass
class AccrualsResult:
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


def accrual_series(rows: List[dict]) -> List[dict]:
    """Per fiscal year: ACC = (NI − CFO)/avg_assets, public at `filed`. Needs rows sorted by fy_end."""
    rows = sorted(rows, key=lambda r: r["fy_end"])
    out = []
    for i, r in enumerate(rows):
        a_now = r["total_assets"]
        a_prev = rows[i - 1]["total_assets"] if i > 0 else a_now
        avg_a = (a_now + a_prev) / 2.0 if (a_now > 0 and a_prev > 0) else a_now
        if avg_a <= 0:
            continue
        acc = (r["net_income"] - r["cfo"]) / avg_a
        ep = _epoch(r["filed"], "%Y-%m-%d")
        if ep is not None:
            out.append({"filed_epoch": int(ep), "acc": acc})
    return out


def backtest_accruals(
    bars_by_sym: Dict[str, List[dict]],
    fund_by_sym: Dict[str, List[dict]], *,
    top_frac: float = 0.2,
    cost_bps: float = 2.0,
    reverse: bool = False,            # False = anomaly (long low-ACC / short high-ACC); True = control
    periods_per_year: float = 252.0,
    starting_equity: float = 100_000.0,
) -> AccrualsResult:
    syms = sorted(s for s in bars_by_sym if fund_by_sym.get(s))
    if len(syms) < max(5, int(round(2 / max(top_frac, 1e-9)))):
        return AccrualsResult(syms, 0, 0.0, 0.0, 0.0, 0, top_frac, 0.0, 0.0, [starting_equity], [])

    master = sorted({int(b["timestamp"]) for s in syms for b in bars_by_sym[s]})
    idx = {t: i for i, t in enumerate(master)}
    T, N = len(master), len(syms)
    if T < 30:
        return AccrualsResult(syms, 0, 0.0, 0.0, 0.0, 0, top_frac, 0.0, 0.0, [starting_equity], [])

    ret = np.zeros((T, N))
    acc = np.full((T, N), np.nan)     # step-function: accrual known entering each day
    for j, s in enumerate(syms):
        b = sorted(bars_by_sym[s], key=lambda x: int(x["timestamp"]))
        ts = [int(x["timestamp"]) for x in b]
        cl = [float(x["close"]) for x in b]
        for k in range(1, len(b)):
            if cl[k - 1] > 0:
                ret[idx[ts[k]], j] = cl[k] / cl[k - 1] - 1.0
        for ev in accrual_series(fund_by_sym[s]):
            k0 = next((i for i, t in enumerate(master) if t >= ev["filed_epoch"]), None)
            if k0 is not None:
                acc[k0:, j] = ev["acc"]      # known from the filing date forward (overwritten by newer)

    k = max(1, int(round(N * top_frac)))
    eq = [starting_equity]
    daily: List[float] = []
    actives: List[int] = []
    turnovers: List[float] = []
    rebals = 0
    prev_w = np.zeros(N)
    for t in range(1, T):
        a = acc[t - 1]
        ok = np.isfinite(a)
        w = np.zeros(N)
        if ok.sum() >= 2 * k:
            order = np.argsort(np.where(ok, a, np.inf))      # ascending ACC; NaNs last
            order = order[np.isin(order, np.where(ok)[0])]
            low, high = order[:k], order[-k:]                # low-ACC (long), high-ACC (short)
            long_idx, short_idx = (low, high) if not reverse else (high, low)
            w[long_idx] = 0.5 / k
            w[short_idx] = -0.5 / k
        turnover = float(np.abs(w - prev_w).sum())
        if turnover > 1e-9:
            rebals += 1
        port_ret = float(w @ ret[t]) - turnover * (cost_bps / 1e4)
        eq.append(eq[-1] * (1 + port_ret))
        daily.append(port_ret)
        actives.append(int((w != 0).sum()))
        turnovers.append(turnover)
        prev_w = w

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return AccrualsResult(
        symbols=syms, n_days=len(daily), total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq),
        n_rebalances=rebals, top_frac=top_frac,
        avg_active=float(np.mean(actives)) if actives else 0.0,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        equity_curve=eq, daily_returns=daily, dates=master[1:])
