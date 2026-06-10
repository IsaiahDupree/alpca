"""
Time-Series Momentum (TSMOM) on a diversified ETF panel — the classic CTA edge — with
the Kim (2016) critique baked in as the honest null.

Moskowitz-Ooi-Pedersen (2012): for each asset, take the SIGN of the trailing ~12-month
return and hold it sized to a constant target volatility; equal risk across a diversified
panel, rebalanced monthly. The diversification across uncorrelated trends is the edge;
monthly turnover survives the cost wall.

THE CATCH (Kim, Tse & Wald 2016): the famous TSMOM results are largely driven by the
VOLATILITY-SCALING, not the momentum timing. So this module backtests THREE things on the
same panel so the harness can separate them:
  - "tsmom"     : sign(trailing return) * (target_vol / asset_vol)        [momentum + vol-scale]
  - "long_vol"  : +1 * (target_vol / asset_vol)                          [vol-scale ONLY, the null]
  - "ew_bh"     : equal-weight buy-and-hold                              [plain panel beta]
If tsmom ~= long_vol, the "momentum" is illusory and we report that as a clean negative.

Walk-forward: signal + vol use only trailing data; weights set on the rebalance day are
held over the following days' returns (no look-ahead).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class TSMOMResult:
    mode: str
    equity_curve: List[float]
    total_return: float
    sharpe: float
    max_drawdown: float
    n_days: int
    avg_gross_leverage: float
    periods_per_year: float
    daily_returns: List[float] = field(default_factory=list)


def _aligned_returns(bars_by_sym: Dict[str, List[dict]], syms: List[str]):
    common = None
    maps = {}
    for s in syms:
        bars = bars_by_sym.get(s, [])
        m = {int(b["timestamp"]): float(b["close"]) for b in bars if float(b.get("close", 0)) > 0}
        maps[s] = m
        common = set(m) if common is None else (common & set(m))
    ts = sorted(common or [])
    if len(ts) < 300:
        return [], np.zeros((0, 0))
    use = [s for s in syms if s in maps and len(maps[s]) >= len(ts)]
    P = np.array([[maps[s][t] for s in use] for t in ts])  # (T+1 x N)
    R = (P[1:] - P[:-1]) / P[:-1]
    return use, R


def backtest_tsmom(
    bars_by_sym: Dict[str, List[dict]], syms: List[str], *,
    mode: str = "tsmom",
    lookback: int = 252,
    vol_window: int = 60,
    target_vol: float = 0.10,
    rebalance: int = 21,
    max_leverage: float = 3.0,
    cost_bps: float = 2.0,
    starting_equity: float = 100_000.0,
    periods_per_year: float = 252.0,
) -> TSMOMResult:
    """Backtest one of {tsmom, long_vol, ew_bh} on the panel. Vol-scaled modes size each
    asset to target_vol/asset_vol (capped at max_leverage), rebalanced every `rebalance`
    days; ew_bh holds equal weights throughout. Net of cost_bps*turnover."""
    use, R = _aligned_returns(bars_by_sym, syms)
    T, N = R.shape if R.ndim == 2 else (0, 0)
    if T < lookback + rebalance + 20 or N < 3:
        return TSMOMResult(mode, [starting_equity], 0.0, 0.0, 0.0, 0, 0.0, periods_per_year)

    eq = [starting_equity]
    daily: List[float] = []
    levs: List[float] = []
    w = np.zeros(N)
    prev_w = np.zeros(N)

    for t in range(lookback, T):
        if (t - lookback) % rebalance == 0:               # set weights on rebalance days
            if mode == "ew_bh":
                w = np.ones(N) / N
            else:
                vol = R[t - vol_window:t].std(axis=0, ddof=1) * np.sqrt(periods_per_year)
                vol = np.where(vol > 1e-6, vol, 1e-6)
                scale = np.clip(target_vol / vol, 0.0, max_leverage)
                if mode == "long_vol":
                    sign = np.ones(N)
                else:  # tsmom
                    trailing = R[t - lookback:t].sum(axis=0)   # trailing cumulative return proxy
                    sign = np.sign(trailing)
                    sign = np.where(sign == 0, 1.0, sign)
                w = sign * scale / N                        # equal risk budget per asset
        turnover = np.abs(w - prev_w).sum()
        port_ret = float(w @ R[t]) - turnover * (cost_bps / 10_000.0)
        eq.append(eq[-1] * (1 + port_ret))
        daily.append(port_ret)
        levs.append(float(np.abs(w).sum()))
        prev_w = w

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return TSMOMResult(
        mode=mode, equity_curve=eq, total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq),
        n_days=len(daily), avg_gross_leverage=float(np.mean(levs)) if levs else 0.0,
        periods_per_year=periods_per_year, daily_returns=daily)
