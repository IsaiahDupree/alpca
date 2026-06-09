"""
Invariants for the Avellaneda-Stoikov inventory-skew sizing (alpca/backtest/inventory_skew.py).
Pins the math/behavior; the harness-level economic verdict lives in scripts/test_inventory_skew.py.
"""

import math

import pytest

from alpca.backtest.inventory_skew import (
    as_target,
    as_target_spread,
    backtest_spread_targets,
    backtest_targets,
    binary_target,
    binary_target_spread,
    rolling_mean_var,
    spread_series,
)


def _wave(n=300, period=20, amp=0.05, base=100.0):
    """Mean-reverting price wave so reversion sizing has something to act on."""
    return [base * (1 + amp * math.sin(2 * math.pi * i / period)) for i in range(n)]


def test_target_clipped_to_max_pos():
    cl = _wave()
    for g in (0.01, 0.1, 1.0):  # tiny gamma -> huge raw target -> must clip
        t = as_target(cl, window=20, gamma=g, max_pos=1.0)
        assert all(-1.0 - 1e-9 <= x <= 1.0 + 1e-9 for x in t)


def test_warmup_is_flat():
    cl = _wave()
    t = as_target(cl, window=20)
    assert all(x == 0.0 for x in t[:20])


def test_sign_long_when_below_fair_value():
    # price strictly below its trailing mean -> A-S wants LONG (positive target)
    cl = [100.0] * 25 + [90.0]  # last bar well below the 100-mean
    t = as_target(cl, window=20, gamma=1.0)
    assert t[-1] > 0
    # and strictly above -> short
    cl2 = [100.0] * 25 + [110.0]
    t2 = as_target(cl2, window=20, gamma=1.0)
    assert t2[-1] < 0


def test_higher_gamma_shrinks_inventory():
    cl = _wave()
    lo = as_target(cl, window=20, gamma=0.5, max_pos=10.0)   # high cap so clipping doesn't mask it
    hi = as_target(cl, window=20, gamma=5.0, max_pos=10.0)
    # mean absolute target must be smaller for the more risk-averse (higher gamma)
    al = sum(abs(x) for x in lo) / len(lo)
    ah = sum(abs(x) for x in hi) / len(hi)
    assert ah < al


def test_flat_price_zero_target_flat_equity():
    cl = [100.0] * 200
    t = as_target(cl, window=20, gamma=1.0)
    assert all(x == 0.0 for x in t)             # no mispricing -> no inventory
    eq = backtest_targets(cl, t, cost_bps=2.0)
    assert all(abs(e - eq[0]) < 1e-6 for e in eq)  # flat target on flat price -> flat equity


def test_cost_monotonic_reduces_return():
    cl = _wave()
    t = as_target(cl, window=20, gamma=1.0)
    r_lo = backtest_targets(cl, t, cost_bps=1.0)[-1]
    r_hi = backtest_targets(cl, t, cost_bps=50.0)[-1]
    assert r_hi <= r_lo  # more cost on the same turnover never helps


def test_zero_target_series_is_flat():
    cl = _wave()
    eq = backtest_targets(cl, [0.0] * len(cl), cost_bps=10.0)
    assert all(abs(e - eq[0]) < 1e-9 for e in eq)


def test_rolling_mean_var_shapes():
    cl = _wave()
    mu, var = rolling_mean_var(cl, 20)
    assert mu[:20] == [None] * 20 and var[:20] == [None] * 20
    assert all(m is not None and v is not None and v >= 0 for m, v in zip(mu[20:], var[20:]))


def test_binary_target_discrete_levels():
    cl = _wave()
    t = binary_target(cl, window=20, entry_z=1.0, exit_z=0.25, max_pos=1.0)
    assert set(round(x, 6) for x in t) <= {-1.0, 0.0, 1.0}


def test_spread_series_inner_joins_on_timestamp():
    a = [{"timestamp": i, "close": 100 + i} for i in range(10)]
    b = [{"timestamp": i, "close": 50 + i} for i in range(3, 13)]  # overlap ts 3..9
    sp, ts = spread_series(a, b, hedge=1.0)
    assert ts == list(range(3, 10))
    assert len(sp) == len(ts)


def test_spread_targets_zero_is_flat_and_sign_correct():
    sp = [math.sin(i / 3.0) for i in range(200)]
    # below rolling mean -> long the spread (positive)
    t = as_target_spread(sp, window=20, gamma=1.0)
    assert all(-1.0 - 1e-9 <= x <= 1.0 + 1e-9 for x in t)
    eq = backtest_spread_targets(sp, [0.0] * len(sp), cost_bps=4.0)
    assert all(abs(e - eq[0]) < 1e-9 for e in eq)


def test_binary_spread_discrete():
    sp = [math.sin(i / 3.0) for i in range(200)]
    t = binary_target_spread(sp, window=20, entry_z=2.0, exit_z=0.5, max_pos=1.0)
    assert set(round(x, 6) for x in t) <= {-1.0, 0.0, 1.0}
