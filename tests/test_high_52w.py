"""Invariants for the 52-week-high momentum backtest."""

import math
import random

from alpca.backtest.high_52w import backtest_high_52w


def _book_with_high_momentum(n_sym=16, n_days=420, strength=0.0008, seed=0):
    """Synthetic universe where names near their trailing high keep rising (momentum built in):
    each symbol has a persistent drift, and the high-drift names ride near their 52wk high. The
    near-high (momentum) leg should then profit and the reversal control should lose."""
    rng = random.Random(seed)
    base = 1_600_000_000
    bars = {}
    for j in range(n_sym):
        drift = strength * (j - n_sym / 2)            # spread of persistent drifts
        p, rows = 100.0, []
        for i in range(n_days):
            p *= (1 + drift + rng.gauss(0, 0.01))
            rows.append({"timestamp": base + i * 86400, "close": p})
        bars[f"S{j:02d}"] = rows
    return bars


def test_momentum_profits_when_reversal_loses_with_built_in_trend():
    bars = _book_with_high_momentum()
    mom = backtest_high_52w(bars, window=200, hold=20, cost_bps=0.0, reverse=False)
    rev = backtest_high_52w(bars, window=200, hold=20, cost_bps=0.0, reverse=True)
    assert mom.total_return > rev.total_return        # near-high rides the built-in trend
    assert rev.total_return < 0


def test_ratio_is_bounded_and_turnover_drops_with_hold():
    bars = _book_with_high_momentum()
    short = backtest_high_52w(bars, window=200, hold=10, cost_bps=0.0)
    longh = backtest_high_52w(bars, window=200, hold=60, cost_bps=0.0)
    assert longh.avg_turnover < short.avg_turnover    # overlapping tranches -> longer hold churns less
    assert all(math.isfinite(e) and e > 0 for e in short.equity_curve)


def test_cost_is_monotonic_drag():
    bars = _book_with_high_momentum()
    free = backtest_high_52w(bars, window=200, hold=20, cost_bps=0.0)
    dear = backtest_high_52w(bars, window=200, hold=20, cost_bps=5.0)
    assert free.total_return > dear.total_return


def test_dates_align_with_returns():
    bars = _book_with_high_momentum()
    r = backtest_high_52w(bars, window=200, hold=20)
    assert len(r.dates) == len(r.daily_returns)


def test_too_short_history_returns_empty():
    bars = {f"S{j}": [{"timestamp": j2, "close": 100.0} for j2 in range(50)] for j in range(8)}
    r = backtest_high_52w(bars, window=252, hold=20)
    assert r.n_days == 0 and r.equity_curve == [100_000.0]
