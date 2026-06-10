"""
Avellaneda-Lee PCA / eigenportfolio residual statistical arbitrage.

The principled generalization of hand-picked cointegration pairs: instead of finding two
co-moving names, strip the SYSTEMATIC risk from every stock by regressing its daily returns
on the top-m PCA "eigenportfolios" of the universe, then trade the mean-reversion of the
IDIOSYNCRATIC residual. Market-neutral by construction (factor exposure is regressed out).

Method (Avellaneda & Lee, "Statistical Arbitrage in the U.S. Equities Market", 2009):
  1. On a trailing window of M days, standardize each stock's returns (subtract mean / vol).
  2. Eigendecompose the NxN correlation matrix; the top m eigenvectors define eigenportfolio
     weights Q_j,i = v_j,i / sigma_i, giving factor returns F_j = sum_i Q_j,i R_i.
  3. Regress every stock's returns on the m factors (ONE lstsq for all stocks — same factors):
     R_i = b_i0 + sum_j b_ij F_j + residual_i.
  4. Cumulative residual X_i = cumsum(residual_i); fit OU via AR(1) X_t = a + b X_{t-1} + e.
     s-score s_i = (X_i,last - m_eq) / sigma_eq, with m_eq=a/(1-b), sigma_eq=std(e)/sqrt(1-b^2).
  5. Trade AGAINST the residual: s < -open -> LONG, s > +open -> SHORT; close near 0. Keep
     only fast-reverting names (OU half-life below max_half_life). Dollar-neutral, equal-weight.

Walk-forward by construction: every day's s-score uses only the trailing window; the target
weights are held over the NEXT day's return (no look-ahead). Judged market-neutral (the
return itself is the alpha — there is no buy-and-hold to beat); the honest NULL is our
existing cointegration-pairs basket (OOS Sharpe ~0.54).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class PCAStatArbResult:
    equity_curve: List[float]
    total_return: float
    sharpe: float
    max_drawdown: float
    n_days: int
    avg_gross_names: float
    avg_daily_turnover: float
    periods_per_year: float
    daily_returns: List[float] = field(default_factory=list)


def _aligned_returns(bars_by_sym: Dict[str, List[dict]], min_len: int):
    """Intersect timestamps across symbols with enough history; return (syms, R) where
    R is a (T x N) simple-returns matrix aligned on the common date index."""
    usable = {s: b for s, b in bars_by_sym.items() if len(b) >= min_len}
    if len(usable) < 5:
        return [], np.zeros((0, 0)), []
    # common timestamp set (intersection)
    common = None
    maps = {}
    for s, bars in usable.items():
        m = {int(b["timestamp"]): float(b["close"]) for b in bars if float(b.get("close", 0)) > 0}
        maps[s] = m
        common = set(m) if common is None else (common & set(m))
    ts = sorted(common or [])
    if len(ts) < min_len:
        return [], np.zeros((0, 0)), []
    syms = sorted(usable)
    P = np.array([[maps[s][t] for s in syms] for t in ts])   # (T+1 x N) prices
    R = (P[1:] - P[:-1]) / P[:-1]                              # (T x N) simple returns
    return syms, R, ts[1:]


def _sscores(Rwin: np.ndarray, n_factors: int, max_half_life: float):
    """Given a (M x N) window of returns, return (s, eligible) where s is the per-stock
    s-score and eligible is a bool mask of fast-mean-reverting names. Vectorized."""
    M, N = Rwin.shape
    mu = Rwin.mean(axis=0)
    sd = Rwin.std(axis=0, ddof=1)
    sd_safe = np.where(sd > 1e-12, sd, 1e-12)
    Y = (Rwin - mu) / sd_safe                       # standardized returns (M x N)
    C = (Y.T @ Y) / (M - 1)                          # correlation matrix (N x N)
    # top-m eigenvectors (symmetric -> eigh, ascending; take the last m)
    vals, vecs = np.linalg.eigh(C)
    m = min(n_factors, N - 1)
    V = vecs[:, -m:]                                 # (N x m)
    Q = V / sd_safe[:, None]                         # eigenportfolio weights (N x m)
    F = Rwin @ Q                                     # factor returns (M x m)
    # one lstsq for ALL stocks: regress Rwin (M x N) on [1, F] (M x (m+1))
    A = np.column_stack([np.ones(M), F])             # (M x (m+1))
    B, *_ = np.linalg.lstsq(A, Rwin, rcond=None)     # (m+1 x N)
    resid = Rwin - A @ B                             # (M x N)
    X = np.cumsum(resid, axis=0)                     # cumulative residual (M x N)
    # OU via AR(1) per column, vectorized: X_t = a + b X_{t-1} + e
    x0 = X[:-1]                                       # (M-1 x N)
    x1 = X[1:]
    mx0 = x0.mean(axis=0)
    mx1 = x1.mean(axis=0)
    cov = ((x0 - mx0) * (x1 - mx1)).mean(axis=0)
    var0 = ((x0 - mx0) ** 2).mean(axis=0)
    var0 = np.where(var0 > 1e-18, var0, 1e-18)
    b = cov / var0
    a = mx1 - b * mx0
    # mean-reversion only where 0 < b < 1; compute s-score there
    eligible = (b > 1e-4) & (b < 1.0 - 1e-6)
    kappa = np.where(eligible, -np.log(np.clip(b, 1e-6, 1 - 1e-9)), 0.0)  # per-day speed
    with np.errstate(divide="ignore", invalid="ignore"):
        half_life = np.where(kappa > 0, np.log(2.0) / kappa, np.inf)
    eligible &= half_life <= max_half_life
    m_eq = np.where(np.abs(1 - b) > 1e-9, a / (1 - b), 0.0)
    resid_var = ((x1 - (a + b * x0)) ** 2).mean(axis=0)
    sig_eq = np.sqrt(np.maximum(resid_var, 1e-18) / np.maximum(1 - b * b, 1e-9))
    s = np.where(eligible & (sig_eq > 1e-12), (X[-1] - m_eq) / sig_eq, np.nan)
    return s, eligible & np.isfinite(s)


def backtest_pca_statarb(
    bars_by_sym: Dict[str, List[dict]], *,
    lookback: int = 60,
    n_factors: int = 15,
    max_half_life: float = 30.0,
    s_open: float = 1.25,
    s_close: float = 0.50,
    cost_bps: float = 2.0,
    starting_equity: float = 100_000.0,
    periods_per_year: float = 252.0,
    min_len: int = 400,
) -> PCAStatArbResult:
    """Walk-forward PCA residual stat-arb over the whole universe. Returns a market-neutral
    equity curve (dollar-neutral, equal-weight long/short books) net of cost_bps*turnover."""
    syms, R, _ = _aligned_returns(bars_by_sym, min_len)
    T, N = R.shape if R.ndim == 2 else (0, 0)
    if T < lookback + 20 or N < 5:
        return PCAStatArbResult([starting_equity], 0.0, 0.0, 0.0, 0, 0.0, 0.0, periods_per_year)

    pos = np.zeros(N)        # signed unit position per name: +1 long, -1 short, 0 flat
    eq = [starting_equity]
    daily_rets: List[float] = []
    gross_counts, turnovers = [], []
    prev_w = np.zeros(N)

    for t in range(lookback, T):
        Rwin = R[t - lookback:t]                      # returns up to day t-1 (known at close t-1)
        s, elig = _sscores(Rwin, n_factors, max_half_life)
        # update band positions (Avellaneda-Lee open/close thresholds)
        for i in range(N):
            if not elig[i] or np.isnan(s[i]):
                pos[i] = 0.0                            # drop names that lost mean-reversion
                continue
            if pos[i] == 0.0:
                if s[i] < -s_open:
                    pos[i] = 1.0                       # residual low -> underpriced -> LONG
                elif s[i] > s_open:
                    pos[i] = -1.0                      # residual high -> SHORT
            elif pos[i] > 0 and s[i] > -s_close:
                pos[i] = 0.0                            # close long near equilibrium
            elif pos[i] < 0 and s[i] < s_close:
                pos[i] = 0.0                            # close short near equilibrium
        # dollar-neutral equal-weight target weights
        longs = pos > 0
        shorts = pos < 0
        w = np.zeros(N)
        if longs.any():
            w[longs] = 0.5 / longs.sum()
        if shorts.any():
            w[shorts] = -0.5 / shorts.sum()
        # realize NEXT day's return with these weights (no look-ahead)
        turnover = np.abs(w - prev_w).sum()
        port_ret = float(w @ R[t]) - turnover * (cost_bps / 10_000.0)
        eq.append(eq[-1] * (1 + port_ret))
        daily_rets.append(port_ret)
        gross_counts.append(int(longs.sum() + shorts.sum()))
        turnovers.append(float(turnover))
        prev_w = w

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return PCAStatArbResult(
        equity_curve=eq, total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq),
        n_days=len(daily_rets),
        avg_gross_names=float(np.mean(gross_counts)) if gross_counts else 0.0,
        avg_daily_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        periods_per_year=periods_per_year, daily_returns=daily_rets)
