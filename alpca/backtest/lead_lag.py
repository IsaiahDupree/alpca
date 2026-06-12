"""
Lead-lag cross-predictability — a market-neutral mechanism we have never tested. Some stocks'
returns lead others' (slow information diffusion / investor inattention): if leader i moves
today, follower j tends to move tomorrow. Estimate the lead-lag map on a TRAIN window, then
trade it FORWARD on a held-out TEST window — long the stocks your leaders predict will rise,
short those they predict will fall, dollar-neutral.

Why this design is honest (the scout warned the source repo's in-sample Sharpe ~1.95 is almost
certainly overfit — the leader→follower map is itself a fitted object, a selection-bias minefield):
  - WALK-FORWARD: the lead-lag correlation matrix C is estimated ONLY on past (train) data; the
    test window's positions use that frozen map and lagged returns. No lookahead.
  - SHUFFLE PLACEBO: `shuffle_leaders=True` randomly permutes each follower's leader assignment.
    If the real map does not beat the placebo, the "edge" is just fitted noise — the decisive control.
  - The signal changes daily (it keys off yesterday's leader returns) → high turnover, so cost is
    a first-class stress, same lesson as Case 17.

C[i,j] = corr(leader i's return at t-lag, follower j's return at t), over the train window. Each
follower keeps its top-`n_leaders` by |C|; predicted_j(t) = Σ_leaders C[i,j]·r_i(t-lag).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class LeadLagResult:
    symbols: List[str]
    n_days: int
    total_return: float
    sharpe: float
    max_drawdown: float
    n_windows: int
    train: int
    test: int
    lag: int
    n_leaders: int
    shuffled: bool
    avg_active: float
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


def _zscore_cols(m):
    mu = m.mean(axis=0)
    sd = m.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (m - mu) / sd


def backtest_lead_lag(
    bars_by_sym: Dict[str, List[dict]], *,
    train: int = 252,
    test: int = 63,
    lag: int = 1,
    n_leaders: int = 5,
    top_frac: float = 0.2,
    cost_bps: float = 2.0,
    periods_per_year: float = 252.0,
    shuffle_leaders: bool = False,
    seed: int = 0,
    starting_equity: float = 100_000.0,
) -> LeadLagResult:
    syms = sorted(bars_by_sym)
    if len(syms) < max(5, int(round(2 / max(top_frac, 1e-9)))):
        return LeadLagResult(syms, 0, 0.0, 0.0, 0.0, 0, train, test, lag, n_leaders,
                             shuffle_leaders, 0.0, [starting_equity], [])

    # align on common timestamps -> returns matrix R (T x N)
    sets = [set(float(b.get("timestamp", 0) or 0) for b in bars_by_sym[s]) for s in syms]
    common = sorted(set.intersection(*sets)) if sets else []
    if len(common) < train + test + lag + 2:
        return LeadLagResult(syms, len(common), 0.0, 0.0, 0.0, 0, train, test, lag, n_leaders,
                             shuffle_leaders, 0.0, [starting_equity], [])
    cl = {s: {float(b["timestamp"]): float(b["close"]) for b in bars_by_sym[s]} for s in syms}
    price = np.array([[cl[s][t] for t in common] for s in syms], dtype=float).T   # (T, N)
    R = np.zeros_like(price)
    R[1:] = np.where(price[:-1] > 0, price[1:] / np.where(price[:-1] > 0, price[:-1], 1.0) - 1.0, 0.0)
    T, N = R.shape
    k = max(1, int(round(N * top_frac)))
    rng = np.random.RandomState(seed)

    eq = [starting_equity]
    daily: List[float] = []
    actives: List[int] = []
    n_windows = 0
    start = train
    prev_w = np.zeros(N)
    while start + test <= T:
        tr = R[start - train:start]                       # train returns
        A = _zscore_cols(tr[:-lag])                        # leaders at t-lag
        B = _zscore_cols(tr[lag:])                         # followers at t
        C = (A.T @ B) / max(A.shape[0], 1)                 # (N, N): C[i,j]=corr(lead i, follow j)
        np.fill_diagonal(C, 0.0)                            # a stock can't be its own leader
        # keep each follower's top-n leaders by |C|, zero the rest -> sparse predictor Cmask
        Cmask = np.zeros_like(C)
        for j in range(N):
            idx = np.argsort(-np.abs(C[:, j]))[:n_leaders]
            if shuffle_leaders:
                idx = rng.permutation(N)[:n_leaders]        # placebo: random leaders, real strengths
            Cmask[idx, j] = C[idx, j]
        # trade the held-out test window with the frozen map
        for t in range(start, start + test):
            pred = R[t - lag] @ Cmask                       # predicted follower returns
            pred = pred - pred.mean()                       # cross-sectional demean (dollar-neutral)
            order = np.argsort(pred)
            w = np.zeros(N)
            w[order[-k:]] = 0.5 / k                          # long the top predicted
            w[order[:k]] = -0.5 / k                          # short the bottom predicted
            turnover = np.abs(w - prev_w).sum()
            port_ret = float(w @ R[t]) - turnover * (cost_bps / 1e4)
            eq.append(eq[-1] * (1 + port_ret))
            daily.append(port_ret)
            actives.append(int((w != 0).sum()))
            prev_w = w
        n_windows += 1
        start += test

    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    return LeadLagResult(
        symbols=syms, n_days=len(daily), total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq),
        n_windows=n_windows, train=train, test=test, lag=lag, n_leaders=n_leaders,
        shuffled=shuffle_leaders, avg_active=float(np.mean(actives)) if actives else 0.0,
        equity_curve=eq, daily_returns=daily)
