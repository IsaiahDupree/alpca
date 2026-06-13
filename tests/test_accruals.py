"""Invariants for the accruals-anomaly backtest."""

import math
import random

from alpca.backtest.accruals import accrual_series, backtest_accruals


def _fund_universe(n_sym=16, n_days=500, seed=0):
    """Synthetic: each symbol has a fixed accrual level; HIGH-accrual names drift DOWN (the anomaly).
    Fundamentals filed once mid-history so the no-lookahead step-function activates partway through."""
    rng = random.Random(seed)
    base = 1_600_000_000
    day = 86400
    bars_by, fund_by = {}, {}
    for j in range(n_sym):
        acc_level = (j / n_sym) - 0.5                    # spread of accrual levels in [-0.5, +0.5]
        drift = -0.002 * acc_level                       # high accrual -> negative drift
        p, rows = 100.0, []
        for i in range(n_days):
            p *= (1 + drift + rng.gauss(0, 0.01))
            rows.append({"timestamp": base + i * day, "close": p})
        bars_by[f"S{j:02d}"] = rows
        # one 10-K filed ~day 60 (so the signal is live for most of the window); NI-CFO encodes acc_level
        import datetime
        filed = datetime.datetime.fromtimestamp(base + 60 * day, datetime.timezone.utc).strftime("%Y-%m-%d")
        fyend = datetime.datetime.fromtimestamp(base + 10 * day, datetime.timezone.utc).strftime("%Y-%m-%d")
        # acc = (NI - CFO)/assets = acc_level  -> set assets=1e9, NI-CFO = acc_level*1e9
        fund_by[f"S{j:02d}"] = [{"fy_end": fyend, "filed": filed,
                                 "net_income": acc_level * 1e9, "cfo": 0.0, "total_assets": 1e9}]
    return bars_by, fund_by


def test_accrual_series_computes_ratio_and_public_date():
    rows = [{"fy_end": "2022-12-31", "filed": "2023-02-15", "net_income": 5e8, "cfo": 2e8, "total_assets": 1e10},
            {"fy_end": "2023-12-31", "filed": "2024-02-15", "net_income": 6e8, "cfo": 8e8, "total_assets": 1.2e10}]
    s = accrual_series(rows)
    assert len(s) == 2
    # 2nd year accrual = (6e8-8e8)/avg(1e10,1.2e10)= -2e8/1.1e10 ~ -0.0182
    assert abs(s[1]["acc"] - (-2e8 / 1.1e10)) < 1e-6
    assert s[0]["filed_epoch"] < s[1]["filed_epoch"]


def test_anomaly_profits_when_control_loses():
    bars, fund = _fund_universe()
    anom = backtest_accruals(bars, fund, top_frac=0.25, reverse=False, cost_bps=0.0)
    ctrl = backtest_accruals(bars, fund, top_frac=0.25, reverse=True, cost_bps=0.0)
    assert anom.total_return > ctrl.total_return      # long low-accrual (which drifts up) wins
    assert ctrl.total_return < 0


def test_low_turnover_annual_rebalance():
    bars, fund = _fund_universe()
    r = backtest_accruals(bars, fund, top_frac=0.25, cost_bps=2.0)
    assert r.avg_turnover < 0.05                        # annual signal -> tiny daily turnover
    assert all(math.isfinite(e) and e > 0 for e in r.equity_curve)


def test_cost_barely_matters_due_to_low_turnover():
    bars, fund = _fund_universe()
    free = backtest_accruals(bars, fund, top_frac=0.25, cost_bps=0.0)
    dear = backtest_accruals(bars, fund, top_frac=0.25, cost_bps=10.0)
    assert free.total_return >= dear.total_return
    assert abs(free.total_return - dear.total_return) < 0.05   # low turnover -> cost is nearly free


def test_too_few_symbols_returns_empty():
    r = backtest_accruals({"A": []}, {"A": []})
    assert r.n_days == 0 and r.equity_curve == [100_000.0]
