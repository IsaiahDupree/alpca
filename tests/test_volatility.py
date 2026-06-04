"""
Phase 0 keystone: rolling realized-volatility helper (calibration/volatility.py).
"""

import math

from alpca.calibration.volatility import build_vol_series, compute_rolling_volatility


def _daily_bars(closes, start_ts=1_700_000_000.0):
    # one bar per calendar day -> bars_per_day inference resolves to 1
    day = 86_400.0
    return [{"open": c, "high": c, "low": c, "close": c, "volume": 1e6,
             "timestamp": start_ts + i * day, "symbol": "T"} for i, c in enumerate(closes)]


def test_constant_prices_zero_vol():
    bars = _daily_bars([100.0] * 30)
    assert compute_rolling_volatility(bars, lookback_days=10, bars_per_day=1) == 0.0


def test_too_few_bars_returns_prior():
    assert compute_rolling_volatility([], prior=0.2) == 0.2
    one = _daily_bars([100.0])
    assert compute_rolling_volatility(one, prior=0.15, bars_per_day=1) == 0.15


def test_annualization_factor_is_sqrt_252_for_daily():
    bars = _daily_bars([100, 101, 100, 102, 99, 103, 98])
    raw = compute_rolling_volatility(bars, lookback_days=10, bars_per_day=1, annualize=False)
    ann = compute_rolling_volatility(bars, lookback_days=10, bars_per_day=1, annualize=True)
    assert raw > 0
    assert abs(ann - raw * math.sqrt(252)) < 1e-9


def test_monotonic_in_volatility():
    calm = _daily_bars([100, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0])
    wild = _daily_bars([100, 110, 95, 112, 90, 115, 88])
    assert (compute_rolling_volatility(wild, bars_per_day=1) >
            compute_rolling_volatility(calm, bars_per_day=1))


def test_build_series_last_matches_point_estimate():
    closes = [100, 101, 100, 102, 99, 103, 98, 104]
    bars = _daily_bars(closes)
    series = build_vol_series(bars, lookback_days=100, bars_per_day=1)  # window covers all
    point = compute_rolling_volatility(bars, lookback_days=100, bars_per_day=1)
    last_ts = bars[-1]["timestamp"]
    assert abs(series[last_ts] - point) < 1e-9
    assert len(series) == len(bars)


def test_build_series_is_finite_and_positive_on_trending_data():
    bars = _daily_bars([100 * (1.01 ** i) for i in range(40)])  # steady uptrend
    series = build_vol_series(bars, lookback_days=10, bars_per_day=1)
    vals = [v for v in series.values()]
    assert all(math.isfinite(v) and v >= 0 for v in vals)
