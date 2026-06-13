"""Invariants for the Betting-Against-Beta / low-vol backtest."""

import math
import random

from alpca.backtest.low_beta import backtest_low_beta


def _universe(n_sym=16, n_days=400, seed=0):
    """Synthetic where the low-vol anomaly holds: half the names are low-vol with a positive drift,
    half are high-vol with a negative drift, on top of a shared market factor. So signal='vol' long
    low-vol should profit and the control should lose. A benchmark series is provided for beta."""
    rng = random.Random(seed)
    base = 1_600_000_000
    mkt = [0.0]
    for _ in range(n_days):
        mkt.append(rng.gauss(0.0004, 0.008))
    bars = {}
    p = 100.0
    bench = []
    for i in range(n_days):
        p *= (1 + mkt[i + 1])
        bench.append({"timestamp": base + i * 86400, "close": p})
    for j in range(n_sym):
        low = (j % 2 == 0)
        vol = 0.006 if low else 0.022
        drift = 0.0008 if low else -0.0008
        q, rows = 100.0, []
        for i in range(n_days):
            q *= (1 + 0.3 * mkt[i + 1] + drift + rng.gauss(0, vol))
            rows.append({"timestamp": base + i * 86400, "close": q})
        bars[f"S{j:02d}"] = rows
    return bars, bench


def test_low_vol_anomaly_profits_when_control_loses():
    bars, bench = _universe()
    anom = backtest_low_beta(bars, bench, signal="vol", lookback=100, top_frac=0.25,
                             rebalance_days=21, cost_bps=0.0, reverse=False)
    ctrl = backtest_low_beta(bars, bench, signal="vol", lookback=100, top_frac=0.25,
                             rebalance_days=21, cost_bps=0.0, reverse=True)
    assert anom.total_return > ctrl.total_return       # long low-vol (which drifts up) wins
    assert ctrl.total_return < 0


def test_beta_signal_runs_finite():
    bars, bench = _universe()
    r = backtest_low_beta(bars, bench, signal="beta", lookback=100, top_frac=0.2, rebalance_days=21)
    assert r.signal == "beta" and all(math.isfinite(e) and e > 0 for e in r.equity_curve)
    assert len(r.dates) == len(r.daily_returns)


def test_low_turnover_on_slow_rebalance():
    bars, bench = _universe()
    r = backtest_low_beta(bars, bench, signal="vol", lookback=100, rebalance_days=42, cost_bps=2.0)
    assert r.avg_turnover < 0.1


def test_cost_is_monotonic_drag():
    bars, bench = _universe()
    free = backtest_low_beta(bars, bench, signal="vol", lookback=100, cost_bps=0.0)
    dear = backtest_low_beta(bars, bench, signal="vol", lookback=100, cost_bps=10.0)
    assert free.total_return >= dear.total_return


def test_too_few_symbols_returns_empty():
    r = backtest_low_beta({"A": [{"timestamp": 1, "close": 1.0}]}, [{"timestamp": 1, "close": 1.0}])
    assert r.n_days == 0 and r.equity_curve == [100_000.0]
