"""Invariants for the overnight→intraday cross-sectional reversal backtest."""

import math
import random

from alpca.backtest.overnight import backtest_overnight_reversal


def _book_with_reversal(n_sym=20, n_days=300, strength=0.6, seed=0):
    """Synthetic universe where overnight winners systematically REVERSE intraday — so the
    reversal leg should make money at zero cost and the momentum leg should lose."""
    rng = random.Random(seed)
    bars_by = {}
    base = 1_600_000_000
    for j in range(n_sym):
        prev_close = 100.0
        bars = []
        for i in range(n_days):
            overnight = rng.gauss(0.0, 0.02)
            op = prev_close * (1 + overnight)
            intraday = -strength * overnight + rng.gauss(0.0, 0.005)   # reversal built in
            cl = op * (1 + intraday)
            bars.append({"timestamp": base + i * 86400, "open": op, "high": max(op, cl) * 1.001,
                         "low": min(op, cl) * 0.999, "close": cl, "volume": 1e6})
            prev_close = cl
        bars_by[f"S{j}"] = bars
    return bars_by


def test_reversal_profits_when_momentum_loses_at_zero_cost():
    bars = _book_with_reversal()
    rev = backtest_overnight_reversal(bars, signal_lookback=1, cost_bps=0.0, reverse=True)
    mom = backtest_overnight_reversal(bars, signal_lookback=1, cost_bps=0.0, reverse=False)
    assert rev.total_return > 0          # the built-in reversal is harvestable gross
    assert mom.total_return < rev.total_return   # the momentum control is strictly worse
    assert mom.total_return < 0


def test_cost_is_monotonic_drag():
    bars = _book_with_reversal()
    free = backtest_overnight_reversal(bars, signal_lookback=1, cost_bps=0.0, reverse=True)
    cheap = backtest_overnight_reversal(bars, signal_lookback=1, cost_bps=1.0, reverse=True)
    dear = backtest_overnight_reversal(bars, signal_lookback=1, cost_bps=5.0, reverse=True)
    assert free.total_return > cheap.total_return > dear.total_return   # ~2x/day turnover bites


def test_equity_curve_finite_and_neutral_leg_counts():
    bars = _book_with_reversal(n_sym=20)
    r = backtest_overnight_reversal(bars, signal_lookback=1, top_frac=0.2, cost_bps=2.0)
    assert all(math.isfinite(e) and e > 0 for e in r.equity_curve)
    # long and short legs are equal-sized (dollar-neutral): 20 syms * 0.2 = 4 per leg = 8 active
    assert abs(r.avg_active - 8) < 1.0


def test_tiny_universe_returns_empty():
    bars = {"A": [{"timestamp": 1, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}]}
    r = backtest_overnight_reversal(bars, top_frac=0.2)
    assert r.n_days == 0 and r.equity_curve == [100_000.0]


def test_lookback_window_known_at_open_no_crash():
    # a multi-day signal window still runs and stays finite (no-lookahead by construction:
    # signal uses overnight returns through today's open, captures today's intraday)
    bars = _book_with_reversal(seed=3)
    for lb in (1, 2, 3, 5):
        r = backtest_overnight_reversal(bars, signal_lookback=lb, cost_bps=1.0)
        assert all(math.isfinite(e) for e in r.equity_curve)
        assert r.signal_lookback == lb
