"""Invariants for the short-interest (borrow-fee) tilt backtest."""

import math
import random

from alpca.backtest.short_interest import backtest_short_interest_tilt, _borrow_apr
import numpy as np


def _si_book(n_sym=12, n_days=400, seed=0):
    """Synthetic: high days-to-cover names UNDERPERFORM (the short-interest anomaly built in), so
    the anomaly leg (short high-DTC) profits gross and the control loses. SI observations are spaced
    ~bi-monthly with MM/DD/YYYY settlement dates so the no-lookahead pub-lag logic exercises."""
    rng = random.Random(seed)
    base_ep = 1_600_000_000
    day = 86400
    bars_by, si_by = {}, {}
    # assign each symbol a fixed DTC level; higher DTC => more negative drift
    dtc_level = {f"S{j:02d}": 1.0 + 4.0 * (j / n_sym) for j in range(n_sym)}
    for j in range(n_sym):
        s = f"S{j:02d}"
        drift = -0.0006 * dtc_level[s]                       # high DTC -> negative drift
        p, bars = 100.0, []
        for i in range(n_days):
            p *= (1 + drift + rng.gauss(0, 0.01))
            bars.append({"timestamp": base_ep + i * day, "close": p})
        bars_by[s] = bars
        # bi-monthly SI rows with settlement dates spread across the window
        rows = []
        import datetime
        for k in range(0, n_days, 10):
            ep = base_ep + k * day
            dt = datetime.datetime.fromtimestamp(ep, datetime.timezone.utc)
            rows.append({"settlement": dt.strftime("%m/%d/%Y"),
                         "interest": dtc_level[s] * 1e6, "avg_vol": 1e6,
                         "days_to_cover": dtc_level[s] + rng.gauss(0, 0.05)})
        si_by[s] = rows
    return bars_by, si_by


def test_anomaly_profits_when_control_loses_gross():
    bars, si = _si_book()
    anom = backtest_short_interest_tilt(bars, si, top_frac=0.25, reverse=True, borrow=None, cost_bps=0.0)
    ctrl = backtest_short_interest_tilt(bars, si, top_frac=0.25, reverse=False, borrow=None, cost_bps=0.0)
    assert anom.total_return > ctrl.total_return        # shorting high-DTC (which fall) wins
    assert ctrl.total_return < 0


def test_borrow_is_a_drag_on_the_short_leg():
    bars, si = _si_book()
    free = backtest_short_interest_tilt(bars, si, reverse=True, borrow=None, cost_bps=0.0)
    flat = backtest_short_interest_tilt(bars, si, reverse=True, borrow=0.10, cost_bps=0.0)
    scaled = backtest_short_interest_tilt(bars, si, reverse=True,
                                          borrow={"base": 0.02, "per_dtc": 0.05, "cap": 0.6}, cost_bps=0.0)
    assert free.total_return > flat.total_return        # any borrow fee reduces the short-leg return
    assert free.total_return > scaled.total_return


def test_borrow_apr_helper_modes():
    d = np.array([1.0, 3.0, 10.0])
    ok = np.array([True, True, True])
    assert np.allclose(_borrow_apr(None, d, ok), 0.0)
    assert np.allclose(_borrow_apr(0.05, d, ok), 0.05)
    scaled = _borrow_apr({"base": 0.01, "per_dtc": 0.02, "cap": 0.15}, d, ok)
    assert np.allclose(scaled, [0.03, 0.07, 0.15])         # 0.01+0.02*10=0.21 capped at 0.15


def test_low_turnover_bimonthly():
    bars, si = _si_book()
    r = backtest_short_interest_tilt(bars, si, top_frac=0.2, cost_bps=2.0)
    assert all(math.isfinite(e) and e > 0 for e in r.equity_curve)
    assert r.avg_turnover < 0.2                          # bi-monthly rebalance -> low daily turnover


def test_dates_align_with_returns_for_regime_breakdown():
    bars, si = _si_book()
    r = backtest_short_interest_tilt(bars, si, top_frac=0.25, cost_bps=2.0)
    assert len(r.dates) == len(r.daily_returns)     # per-year regime breakdown needs this alignment
    assert all(isinstance(d, int) for d in r.dates)


def test_too_few_symbols_returns_empty():
    r = backtest_short_interest_tilt({"A": []}, {"A": []})
    assert r.n_days == 0 and r.equity_curve == [100_000.0]
