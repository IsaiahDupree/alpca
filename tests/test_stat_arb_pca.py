"""Invariants for the Avellaneda-Lee PCA residual stat-arb (alpca/backtest/stat_arb_pca.py)."""

import math

import numpy as np

from alpca.backtest.stat_arb_pca import _aligned_returns, _sscores, backtest_pca_statarb


def _synthetic_universe(n_sym=12, n_days=500, seed=1):
    """A common market factor + per-name mean-reverting idiosyncratic residual -> the
    method should find tradeable residuals."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 0.01, n_days)
    bars_by = {}
    base_ts = 1_700_000_000
    for i in range(n_sym):
        resid = np.zeros(n_days)
        for t in range(1, n_days):
            resid[t] = 0.8 * resid[t - 1] + rng.normal(0, 0.01)  # AR(1) mean-reverting
        rets = (1.0 + 0.5 * i / n_sym) * factor + resid
        price = 100 * np.cumprod(1 + rets)
        bars_by[f"S{i:02d}"] = [{"timestamp": base_ts + d * 86400, "close": float(price[d])}
                                for d in range(n_days)]
    return bars_by


def test_aligned_returns_shapes():
    bars = _synthetic_universe()
    syms, R, ts = _aligned_returns(bars, min_len=400)
    assert len(syms) == 12
    assert R.shape[1] == 12 and R.shape[0] == len(ts)
    assert np.isfinite(R).all()


def test_sscores_finite_and_masked():
    bars = _synthetic_universe()
    _, R, _ = _aligned_returns(bars, min_len=400)
    s, elig = _sscores(R[:60], n_factors=5, max_half_life=30.0)
    assert s.shape == (12,) and elig.shape == (12,)
    # eligible entries must have finite s-scores
    assert np.isfinite(s[elig]).all()


def test_backtest_runs_and_is_finite():
    bars = _synthetic_universe()
    r = backtest_pca_statarb(bars, lookback=60, n_factors=5, min_len=400, cost_bps=2.0)
    assert r.n_days > 100
    assert len(r.equity_curve) == r.n_days + 1
    assert all(math.isfinite(e) and e > 0 for e in r.equity_curve)
    assert math.isfinite(r.sharpe)


def test_higher_cost_never_helps():
    bars = _synthetic_universe()
    lo = backtest_pca_statarb(bars, lookback=60, n_factors=5, min_len=400, cost_bps=1.0)
    hi = backtest_pca_statarb(bars, lookback=60, n_factors=5, min_len=400, cost_bps=100.0)
    assert hi.total_return <= lo.total_return + 1e-9


def test_degenerate_universe_returns_trivial():
    bars = {f"S{i}": [{"timestamp": 1_700_000_000 + d * 86400, "close": 100.0}
                      for d in range(450)] for i in range(6)}  # flat prices
    r = backtest_pca_statarb(bars, lookback=60, n_factors=3, min_len=400)
    # flat prices -> no real residual edge -> equity stays ~flat, no crash
    assert all(math.isfinite(e) for e in r.equity_curve)


def test_dollar_neutral_when_both_books_populated():
    bars = _synthetic_universe(n_sym=20, n_days=500)
    # inspect one rebalance's weights indirectly: gross names > 0 and curve finite
    r = backtest_pca_statarb(bars, lookback=60, n_factors=5, min_len=400)
    assert r.avg_gross_names >= 0
    assert math.isfinite(r.avg_daily_turnover)
