"""Invariants for the value-composite backtest."""

import math
import random
import datetime

from alpca.backtest.value import backtest_value_composite, _rank01
import numpy as np


def _value_universe(n_sym=16, n_days=420, seed=0):
    """Synthetic: each symbol has a fixed earnings yield; CHEAP (high-yield) names drift UP, so the
    value leg (long cheap) profits and the anti-value control loses. Fundamentals carry NI/FCF/book/
    shares so all three yields compute; one 10-K filed early so the signal is live for most of the window."""
    rng = random.Random(seed)
    base = 1_600_000_000
    filed = datetime.datetime.fromtimestamp(base + 40 * 86400, datetime.timezone.utc).strftime("%Y-%m-%d")
    fyend = datetime.datetime.fromtimestamp(base + 5 * 86400, datetime.timezone.utc).strftime("%Y-%m-%d")
    bars_by, fund_by = {}, {}
    for j in range(n_sym):
        yld = (j + 1) / n_sym                     # earnings yield rank in (0,1]; higher = cheaper
        drift = 0.0010 * (yld - 0.5)              # cheap names drift up, expensive down
        p, bars = 100.0, []
        for i in range(n_days):
            p *= (1 + drift + rng.gauss(0, 0.01))
            bars.append({"timestamp": base + i * 86400, "close": p})
        bars_by[f"S{j:02d}"] = bars
        ni = yld * 1e9                            # at price 100, shares 1e9 -> E/P proportional to yld
        fund_by[f"S{j:02d}"] = [{"fy_end": fyend, "filed": filed, "net_income": ni, "cfo": ni,
                                 "total_assets": 1e10, "capex": 0.0, "fcf": ni,
                                 "book_equity": ni * 5, "shares": 1e9}]
    return bars_by, fund_by


def test_rank01_orders_and_handles_nan():
    x = np.array([0.1, 0.9, 0.5, np.nan])
    ok = np.isfinite(x)
    r = _rank01(x, ok)
    assert r[1] == 1.0 and r[0] == 0.0 and 0 < r[2] < 1 and math.isnan(r[3])


def test_value_long_cheap_profits_when_control_loses():
    bars, fund = _value_universe()
    val = backtest_value_composite(bars, fund, top_frac=0.25, rebalance_days=21, cost_bps=0.0, reverse=False)
    anti = backtest_value_composite(bars, fund, top_frac=0.25, rebalance_days=21, cost_bps=0.0, reverse=True)
    assert val.total_return > anti.total_return       # long cheap (which drifts up) wins
    assert anti.total_return < 0


def test_low_turnover_on_slow_rebalance():
    bars, fund = _value_universe()
    r = backtest_value_composite(bars, fund, top_frac=0.25, rebalance_days=42, cost_bps=2.0)
    assert all(math.isfinite(e) and e > 0 for e in r.equity_curve)
    assert r.avg_turnover < 0.1                       # monthly+ rebalance -> low daily turnover


def test_handles_missing_book_and_fcf_via_available_metrics():
    bars, fund = _value_universe(n_sym=12)
    for rows in fund.values():                        # strip book + fcf -> composite uses E/P only
        rows[0]["book_equity"] = None
        rows[0]["fcf"] = None
    r = backtest_value_composite(bars, fund, top_frac=0.25, rebalance_days=21)
    assert r.n_days > 0 and all(math.isfinite(e) for e in r.equity_curve)


def test_requires_shares_and_enough_symbols():
    bars, fund = _value_universe(n_sym=12)
    for rows in fund.values():
        rows[0]["shares"] = None                      # no shares -> no market cap -> dropped
    r = backtest_value_composite(bars, fund, top_frac=0.25)
    assert r.n_days == 0 and r.equity_curve == [100_000.0]
