"""Invariants for TSMOM + the vol-scaled null (alpca/backtest/tsmom.py)."""

import math

import numpy as np

from alpca.backtest.tsmom import _aligned_returns, backtest_tsmom


def _panel(n=6, n_days=900, seed=2):
    rng = np.random.default_rng(seed)
    bars_by = {}
    base = 1_700_000_000
    for i in range(n):
        drift = 0.0003 * (1 if i % 2 == 0 else -0.2)  # some up-trenders, some not
        rets = drift + rng.normal(0, 0.012, n_days)
        price = 100 * np.cumprod(1 + rets)
        bars_by[f"E{i}"] = [{"timestamp": base + d * 86400, "close": float(price[d])}
                            for d in range(n_days)]
    return bars_by, list(bars_by)


def test_aligned_shapes():
    bars, syms = _panel()
    use, R = _aligned_returns(bars, syms)
    assert len(use) == len(syms)
    assert R.shape[1] == len(syms) and np.isfinite(R).all()


def test_three_modes_run_finite():
    bars, syms = _panel()
    for mode in ("tsmom", "long_vol", "ew_bh"):
        r = backtest_tsmom(bars, syms, mode=mode, lookback=252, rebalance=21)
        assert r.mode == mode
        assert r.n_days > 100
        assert all(math.isfinite(e) and e > 0 for e in r.equity_curve)
        assert math.isfinite(r.sharpe)


def test_ew_bh_leverage_is_one():
    bars, syms = _panel()
    r = backtest_tsmom(bars, syms, mode="ew_bh", lookback=252, rebalance=21)
    assert abs(r.avg_gross_leverage - 1.0) < 1e-6  # equal weights sum to 1.0 gross


def test_vol_scaled_leverage_responds_to_target_vol():
    bars, syms = _panel()
    lo = backtest_tsmom(bars, syms, mode="long_vol", target_vol=0.05, rebalance=21)
    hi = backtest_tsmom(bars, syms, mode="long_vol", target_vol=0.20, rebalance=21)
    assert hi.avg_gross_leverage > lo.avg_gross_leverage  # higher target vol -> more leverage


def test_higher_cost_never_helps():
    bars, syms = _panel()
    lo = backtest_tsmom(bars, syms, mode="tsmom", cost_bps=1.0, rebalance=21)
    hi = backtest_tsmom(bars, syms, mode="tsmom", cost_bps=200.0, rebalance=21)
    assert hi.total_return <= lo.total_return + 1e-9


def test_max_leverage_cap_respected():
    bars, syms = _panel()
    r = backtest_tsmom(bars, syms, mode="long_vol", target_vol=5.0, max_leverage=2.0, rebalance=21)
    # each asset capped at 2.0/N gross; total gross <= 2.0 + small slack
    assert r.avg_gross_leverage <= 2.0 + 1e-6
