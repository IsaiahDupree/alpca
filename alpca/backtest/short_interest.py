"""
Short-interest (borrow-fee) tilt — the scout's "hard-to-borrow" signal, on REAL Nasdaq short
interest. Days-to-cover (DTC = shares short / avg daily volume) is the fundamental driver of
borrow fees: crowded shorts go special. The documented short-interest anomaly says heavily-shorted
names UNDERPERFORM (short sellers are informed) → LONG low-DTC, SHORT high-DTC, dollar-neutral.

The honest crux (this is why it may not work net): the names the anomaly tells you to SHORT are
exactly the expensive-to-borrow ones, so the borrow fee eats the short leg — the same wall that
sank surprise-PEAD's short. `borrow` models that: flat apr, or DTC-scaled apr (apr grows with how
crowded the short is), charged daily on the short notional.

No look-ahead: short interest as of a settlement date is not disseminated until ~8 trading days
later, so each observation's signal is only acted on `pub_lag` trading days AFTER its settlement.
Rebalances bi-monthly (when a new batch goes public) → low turnover, unlike Cases 17/19/20.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from alpca.data.earnings import _epoch


@dataclass
class ShortInterestResult:
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


def backtest_short_interest_tilt(
    bars_by_sym: Dict[str, List[dict]],
    si_by_sym: Dict[str, List[dict]], *,
    top_frac: float = 0.2,
    pub_lag: int = 10,
    cost_bps: float = 2.0,
    borrow=None,                      # None | float (flat apr) | {"base":.., "per_dtc":.., "cap":..}
    reverse: bool = True,             # True = anomaly (short high-DTC); False = control (chase shorts)
    periods_per_year: float = 252.0,
    starting_equity: float = 100_000.0,
) -> ShortInterestResult:
    syms = sorted(s for s in bars_by_sym if si_by_sym.get(s))
    if len(syms) < max(5, int(round(2 / max(top_frac, 1e-9)))):
        return ShortInterestResult(syms, 0, 0.0, 0.0, 0.0, 0, top_frac, 0.0, 0.0, [starting_equity], [])

    master = sorted({int(b["timestamp"]) for s in syms for b in bars_by_sym[s]})
    idx = {t: i for i, t in enumerate(master)}
    T, N = len(master), len(syms)
    if T < 30:
        return ShortInterestResult(syms, 0, 0.0, 0.0, 0.0, 0, top_frac, 0.0, 0.0, [starting_equity], [])

    # returns matrix
    ret = np.zeros((T, N))
    for j, s in enumerate(syms):
        b = sorted(bars_by_sym[s], key=lambda x: int(x["timestamp"]))
        ts = [int(x["timestamp"]) for x in b]
        cl = [float(x["close"]) for x in b]
        for k in range(1, len(b)):
            if cl[k - 1] > 0:
                ret[idx[ts[k]], j] = cl[k] / cl[k - 1] - 1.0

    # DTC step-function matrix: DTC known at day t = most recent settlement whose public date
    # (settlement + pub_lag trading days) is <= t. NaN until the first public observation.
    dtc = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        obs = []
        for r in si_by_sym[s]:
            ep = _epoch(r.get("settlement", ""), "%m/%d/%Y")
            if ep is not None and r.get("days_to_cover") is not None:
                obs.append((int(ep), float(r["days_to_cover"])))
        obs.sort()
        for settle_ep, val in obs:
            # first master index >= settlement, then + pub_lag trading days = public date
            k0 = next((i for i, t in enumerate(master) if t >= settle_ep), None)
            if k0 is None:
                continue
            pub = min(k0 + pub_lag, T - 1)
            dtc[pub:, j] = val               # known from the public date forward (overwritten by newer)

    k = max(1, int(round(N * top_frac)))
    eq = [starting_equity]
    daily: List[float] = []
    actives: List[int] = []
    turnovers: List[float] = []
    rebals = 0
    prev_w = np.zeros(N)
    for t in range(1, T):
        d = dtc[t - 1]                       # signal known entering day t
        ok = np.isfinite(d)
        w = np.zeros(N)
        if ok.sum() >= 2 * k:
            order = np.argsort(np.where(ok, d, np.inf))      # ascending DTC; NaNs last
            order = order[np.isin(order, np.where(ok)[0])]
            low, high = order[:k], order[-k:]
            long_idx, short_idx = (low, high) if reverse else (high, low)
            w[long_idx] = 0.5 / k
            w[short_idx] = -0.5 / k
        turnover = np.abs(w - prev_w).sum()
        if turnover > 1e-9:
            rebals += 1
        # borrow drag on the short notional (DTC-scaled or flat)
        short_w = np.where(w < 0, -w, 0.0)
        apr = _borrow_apr(borrow, d, ok)
        borrow_drag = float((short_w * apr).sum()) / periods_per_year
        port_ret = float(w @ ret[t]) - turnover * (cost_bps / 1e4) - borrow_drag
        eq.append(eq[-1] * (1 + port_ret))
        daily.append(port_ret)
        actives.append(int((w != 0).sum()))
        turnovers.append(turnover)
        prev_w = w

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return ShortInterestResult(
        symbols=syms, n_days=len(daily), total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq),
        n_rebalances=rebals, top_frac=top_frac,
        avg_active=float(np.mean(actives)) if actives else 0.0,
        avg_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        equity_curve=eq, daily_returns=daily)


def _borrow_apr(borrow, d, ok) -> np.ndarray:
    """Per-name annualized borrow rate. None->0; float->flat; dict->DTC-scaled (base+per_dtc*DTC, capped)."""
    n = d.shape[0]
    if borrow is None:
        return np.zeros(n)
    if isinstance(borrow, (int, float)):
        return np.full(n, float(borrow))
    base = borrow.get("base", 0.01)
    per = borrow.get("per_dtc", 0.0)
    cap = borrow.get("cap", 1.0)
    dd = np.where(ok, d, 0.0)
    return np.minimum(base + per * dd, cap)
