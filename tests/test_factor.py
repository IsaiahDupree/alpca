"""Invariants for the generic cross-sectional factor engine + signal builders."""

import math
import random
import datetime
import numpy as np

from alpca.backtest.factor import (
    backtest_factor, asset_growth_signal, net_issuance_signal, roa_signal,
    max_return_signal, idiosyncratic_vol_signal, residual_momentum_signal, vol_managed_momentum_signal)


def _universe(n_sym=16, n_days=400, seed=0, score_drives=True):
    """Each symbol has a fixed score in [0,1]; if score_drives, high-score names drift UP. A signal_fn
    returning that score should then profit at long_high=True and lose at long_high=False."""
    rng = random.Random(seed)
    base = 1_600_000_000
    bars, scores = {}, {}
    for j in range(n_sym):
        sc = (j + 1) / n_sym
        scores[f"S{j:02d}"] = sc
        drift = 0.0010 * (sc - 0.5) if score_drives else 0.0
        p, rows = 100.0, []
        for i in range(n_days):
            p *= (1 + drift + rng.gauss(0, 0.01))
            rows.append({"timestamp": base + i * 86400, "close": p})
        bars[f"S{j:02d}"] = rows
    return bars, scores


def test_engine_long_high_profits_long_low_loses_on_driving_signal():
    bars, scores = _universe()
    def sig(master, syms, price):
        T, N = price.shape
        m = np.tile(np.array([scores[s] for s in syms]), (T, 1))   # constant per-symbol score
        return m
    hi = backtest_factor(bars, sig, top_frac=0.25, rebalance_days=21, cost_bps=0.0, long_high=True)
    lo = backtest_factor(bars, sig, top_frac=0.25, rebalance_days=21, cost_bps=0.0, long_high=False)
    assert hi.total_return > 0 and lo.total_return < 0 and hi.total_return > lo.total_return


def test_cost_and_dates_and_empty():
    bars, scores = _universe()
    sig = lambda m, s, p: np.tile(np.array([scores[x] for x in s]), (len(m), 1))
    free = backtest_factor(bars, sig, cost_bps=0.0, long_high=True)
    dear = backtest_factor(bars, sig, cost_bps=10.0, long_high=True)
    assert free.total_return >= dear.total_return
    assert len(free.dates) == len(free.daily_returns)
    r = backtest_factor({"A": [{"timestamp": 1, "close": 1.0}]}, sig)
    assert r.n_days == 0 and r.equity_curve == [100_000.0]


def _funds(syms, growth_by_sym):
    base = 1_600_000_000
    filed0 = datetime.datetime.fromtimestamp(base + 5 * 86400, datetime.timezone.utc).strftime("%Y-%m-%d")
    filed1 = datetime.datetime.fromtimestamp(base + 200 * 86400, datetime.timezone.utc).strftime("%Y-%m-%d")
    out = {}
    for s in syms:
        g = growth_by_sym[s]
        out[s] = [{"fy_end": "2020-12-31", "filed": filed0, "total_assets": 1e9, "shares": 1e8,
                   "net_income": 1e8, "revenue": 5e8, "cogs": 3e8},
                  {"fy_end": "2021-12-31", "filed": filed1, "total_assets": 1e9 * (1 + g),
                   "shares": 1e8 * (1 + g), "net_income": 1e8, "revenue": 5e8, "cogs": 3e8}]
    return out


def test_fundamental_signal_builders_produce_values():
    bars, _ = _universe(n_sym=8)
    funds = _funds(list(bars), {s: (j - 4) * 0.05 for j, s in enumerate(bars)})
    master = sorted({int(b["timestamp"]) for v in bars.values() for b in v})
    price = np.full((len(master), len(bars)), 100.0)
    for fn in (asset_growth_signal(funds), net_issuance_signal(funds), roa_signal(funds)):
        sig = fn(master, sorted(bars), price)
        assert sig.shape == (len(master), len(bars))
        assert np.isfinite(sig[-1]).any()        # signal live by the end (after the 2nd 10-K filed)


def test_price_signal_builders_run():
    bars, _ = _universe()
    master = sorted({int(b["timestamp"]) for v in bars.values() for b in v})
    syms = sorted(bars)
    from alpca.backtest.factor import _price_ret
    price, _ = _price_ret(bars, syms, master)
    bench = bars[syms[0]]
    for fn in (max_return_signal(21), idiosyncratic_vol_signal(bench, 100),
               residual_momentum_signal(bench, 100, 21), vol_managed_momentum_signal(100, 21, 60)):
        sig = fn(master, syms, price)
        assert sig.shape == (len(master), len(syms)) and np.isfinite(sig[-1]).any()
