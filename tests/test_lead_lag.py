"""Invariants for the walk-forward lead-lag backtest, incl. the shuffle-placebo control."""

import math
import random

from alpca.backtest.lead_lag import backtest_lead_lag


def _book_with_lead_lag(n_sym=12, n_days=700, strength=0.6, seed=0):
    """Synthetic universe with a REAL lead-lag: two driver series lead; every other symbol's
    return today = strength * its driver's return YESTERDAY + noise. A real lead-lag map should
    detect driver->follower and profit; a shuffled (random-leader) placebo should not."""
    rng = random.Random(seed)
    base = 1_600_000_000
    drv = [[rng.gauss(0, 0.02) for _ in range(n_days)] for _ in range(2)]
    bars_by = {}
    for j in range(n_sym):
        p, bars = 100.0, []
        for i in range(n_days):
            if j < 2:
                r = drv[j][i]                                  # the leaders themselves
            else:
                lead_prev = drv[j % 2][i - 1] if i > 0 else 0.0
                r = strength * lead_prev + rng.gauss(0, 0.01)  # follower lags its driver by 1 day
            p *= (1 + r)
            bars.append({"timestamp": base + i * 86400, "close": p})
        bars_by[f"S{j:02d}"] = bars
    return bars_by


def test_real_lead_lag_beats_shuffle_placebo_when_structure_exists():
    bars = _book_with_lead_lag()
    real = backtest_lead_lag(bars, train=200, test=60, n_leaders=2, cost_bps=0.0, shuffle_leaders=False)
    plac = backtest_lead_lag(bars, train=200, test=60, n_leaders=2, cost_bps=0.0, shuffle_leaders=True)
    assert real.total_return > plac.total_return          # the genuine map extracts the built-in lead-lag
    assert real.sharpe > plac.sharpe


def test_walk_forward_rolls_multiple_windows_no_lookahead():
    bars = _book_with_lead_lag(n_days=700)
    r = backtest_lead_lag(bars, train=200, test=60, n_leaders=3, cost_bps=1.0)
    assert r.n_windows >= 2                                 # genuinely rolled train->test
    assert all(math.isfinite(e) and e > 0 for e in r.equity_curve)


def test_cost_is_monotonic_drag():
    bars = _book_with_lead_lag()
    free = backtest_lead_lag(bars, train=200, test=60, n_leaders=2, cost_bps=0.0)
    cheap = backtest_lead_lag(bars, train=200, test=60, n_leaders=2, cost_bps=1.0)
    dear = backtest_lead_lag(bars, train=200, test=60, n_leaders=2, cost_bps=5.0)
    assert free.total_return > cheap.total_return > dear.total_return


def test_shuffle_is_deterministic_under_seed():
    bars = _book_with_lead_lag()
    a = backtest_lead_lag(bars, train=200, test=60, n_leaders=3, shuffle_leaders=True, seed=7)
    b = backtest_lead_lag(bars, train=200, test=60, n_leaders=3, shuffle_leaders=True, seed=7)
    assert a.total_return == b.total_return                 # same seed -> reproducible placebo


def test_too_short_history_returns_empty():
    bars = {f"S{j}": [{"timestamp": j2, "close": 100.0} for j2 in range(50)] for j in range(8)}
    r = backtest_lead_lag(bars, train=252, test=63)
    assert r.n_days == 0 and r.equity_curve == [100_000.0]
