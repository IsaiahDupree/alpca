"""Invariants for cross-sectional gap-reversion (multi-day hold)."""

import math
import random

from alpca.backtest.gap_reversion import backtest_gap_reversion


def _book_with_gap_reversion(n_sym=12, n_days=500, strength=0.5, seed=0):
    """Synthetic universe where the overnight gap REVERTS: a stock that gaps down today drifts
    up over the next close-to-close. So the reversion leg (long gap-downs) should profit gross
    and the gap-momentum control should not."""
    rng = random.Random(seed)
    base = 1_600_000_000
    bars_by = {}
    for j in range(n_sym):
        prev_close = 100.0
        prev_gap = 0.0
        bars = []
        for i in range(n_days):
            gap = rng.gauss(0.0, 0.02)
            op = prev_close * (1 + gap)
            cc = -strength * prev_gap + rng.gauss(0.0, 0.004)   # close-to-close reverts YESTERDAY's gap
            cl = prev_close * (1 + cc)                          # today's close vs yesterday's close
            bars.append({"timestamp": base + i * 86400, "open": op, "high": max(op, cl) * 1.001,
                         "low": min(op, cl) * 0.999, "close": cl, "volume": 1e6})
            prev_close, prev_gap = cl, gap
        bars_by[f"S{j:02d}"] = bars
    return bars_by


def test_reversion_profits_when_momentum_loses_gross():
    bars = _book_with_gap_reversion()
    rev = backtest_gap_reversion(bars, hold=3, cost_bps=0.0, reverse=True)
    mom = backtest_gap_reversion(bars, hold=3, cost_bps=0.0, reverse=False)
    assert rev.total_return > 0
    assert mom.total_return < rev.total_return


def test_turnover_decreases_with_hold():
    bars = _book_with_gap_reversion()
    short = backtest_gap_reversion(bars, hold=1, cost_bps=0.0)
    longh = backtest_gap_reversion(bars, hold=10, cost_bps=0.0)
    assert longh.avg_turnover < short.avg_turnover     # longer hold -> less of the book rotates daily


def test_cost_is_monotonic_drag():
    bars = _book_with_gap_reversion()
    free = backtest_gap_reversion(bars, hold=5, cost_bps=0.0, reverse=True)
    cheap = backtest_gap_reversion(bars, hold=5, cost_bps=1.0, reverse=True)
    dear = backtest_gap_reversion(bars, hold=5, cost_bps=5.0, reverse=True)
    assert free.total_return > cheap.total_return > dear.total_return


def test_equity_finite_and_dollar_neutral_legs():
    bars = _book_with_gap_reversion(n_sym=20)
    r = backtest_gap_reversion(bars, hold=5, top_frac=0.2, cost_bps=2.0)
    assert all(math.isfinite(e) and e > 0 for e in r.equity_curve)
    # each daily tranche is 2*k=8 names; the smoothed book (mean of `hold` tranches) holds more
    # distinct names but never more than the universe.
    assert 8 <= r.avg_active <= 20


def test_too_few_symbols_returns_empty():
    bars = {"A": [{"timestamp": 1, "open": 1.0, "close": 1.0}]}
    r = backtest_gap_reversion(bars)
    assert r.n_days == 0 and r.equity_curve == [100_000.0]
