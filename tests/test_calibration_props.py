"""
Calibration property/invariant tests — records, rolling volatility, the pure
Nelder-Mead optimizer, and the calibrate() pipeline.

Complements test_calibration.py (known-param recovery) and test_calibration_records.py
(JSONL round-trip) with broader sweeps and invariants.
"""

import math

import pytest

from alpca.calibration.fit import _nelder_mead, calibrate, fit_nonlinear_impact
from alpca.calibration.records import CalibrationRecord, CalibrationStore
from alpca.calibration.volatility import (
    build_vol_series,
    compute_rolling_volatility,
)


def _rec(side, intended, fill, **kw):
    return CalibrationRecord(symbol="SPY", side=side, qty=kw.pop("qty", 10.0),
                             intended_price=intended, fill_price=fill, **kw)


# ----------------------------------------------------------------- record props
@pytest.mark.parametrize("intended,fill", [(100.0, 100.5), (100.0, 101.0), (50.0, 50.1)])
def test_buy_above_intended_is_positive_slippage(intended, fill):
    assert _rec("BUY", intended, fill).slippage_bps > 0


@pytest.mark.parametrize("intended,fill", [(100.0, 99.5), (100.0, 99.0), (50.0, 49.9)])
def test_buy_below_intended_is_negative_slippage(intended, fill):
    assert _rec("BUY", intended, fill).slippage_bps < 0


@pytest.mark.parametrize("intended,fill", [(100.0, 99.5), (100.0, 99.0)])
def test_sell_below_intended_is_positive_slippage(intended, fill):
    # selling LOWER than intended is adverse -> positive slippage
    assert _rec("SELL", intended, fill).slippage_bps > 0


@pytest.mark.parametrize("intended,fill", [(100.0, 100.5), (100.0, 101.0)])
def test_sell_above_intended_is_negative_slippage(intended, fill):
    assert _rec("SELL", intended, fill).slippage_bps < 0


def test_slippage_bps_magnitude():
    # buy 100 -> 100.5 = 50 bps
    assert _rec("BUY", 100.0, 100.5).slippage_bps == pytest.approx(50.0)


def test_slippage_none_without_intended():
    r = CalibrationRecord(symbol="X", side="BUY", qty=1, intended_price=0.0, fill_price=10.0)
    assert r.slippage_bps is None


@pytest.mark.parametrize("qty,vol,exp", [(10.0, 1000.0, 0.01), (50.0, 1000.0, 0.05), (5.0, 100.0, 0.05)])
def test_participation(qty, vol, exp):
    assert _rec("BUY", 100, 100, qty=qty, bar_volume=vol).participation == pytest.approx(exp)


def test_participation_none_without_volume():
    assert _rec("BUY", 100, 100).participation is None


@pytest.mark.parametrize("bid,ask,exp", [(99.0, 101.0, 100.0), (100.0, 100.2, 10.0)])
def test_quoted_half_spread_bps(bid, ask, exp):
    r = _rec("BUY", 100, 100, bid=bid, ask=ask)
    assert r.quoted_half_spread_bps == pytest.approx(exp, rel=1e-3)


@pytest.mark.parametrize("bid,ask", [(None, 101.0), (101.0, 100.0), (100.0, 100.0)])
def test_quoted_half_spread_none_on_bad_quote(bid, ask):
    assert _rec("BUY", 100, 100, bid=bid, ask=ask).quoted_half_spread_bps is None


def test_store_roundtrip_with_quote_fields(tmp_path):
    store = CalibrationStore(str(tmp_path / "c.jsonl"))
    store.append(_rec("BUY", 100.0, 100.4, bar_volume=2000.0, realized_vol=0.21,
                      bid=99.99, ask=100.01, bid_size=300.0, ask_size=120.0, quote_ts=5.0))
    back = store.read_all()
    assert len(back) == 1
    assert back[0].realized_vol == 0.21
    assert back[0].bid == 99.99 and back[0].ask == 100.01


# ----------------------------------------------------------- Nelder-Mead optimizer
@pytest.mark.parametrize("target", [(3.0, 5.0), (-2.0, 7.0), (0.0, 0.0), (10.0, -4.0)])
def test_nelder_mead_minimizes_quadratic(target):
    tx, ty = target

    def f(p):
        return (p[0] - tx) ** 2 + (p[1] - ty) ** 2

    best, fval = _nelder_mead(f, [0.0, 0.0], max_iter=800)
    assert best[0] == pytest.approx(tx, abs=1e-3)
    assert best[1] == pytest.approx(ty, abs=1e-3)
    assert fval == pytest.approx(0.0, abs=1e-5)


def test_nelder_mead_1d():
    best, _ = _nelder_mead(lambda p: (p[0] - 4.2) ** 2, [0.0], max_iter=500)
    assert best[0] == pytest.approx(4.2, abs=1e-3)


# --------------------------------------------------------- nonlinear impact fit
def test_nonlinear_impact_recovers_sqrt_law():
    # build records on slippage = 2 + 8*sqrt(participation) (sigma folded into eta)
    recs = []
    for i in range(40):
        part = 0.01 + 0.01 * (i % 20)         # 0.01 .. 0.20, varied
        slip = 2.0 + 8.0 * math.sqrt(part)
        recs.append(_rec("BUY", 100.0, 100.0 * (1 + slip / 1e4),
                         qty=part * 10_000.0, bar_volume=10_000.0, realized_vol=1.0))
    out = fit_nonlinear_impact(recs, half_spread_hint=2.0)
    assert out is not None
    assert out["beta"] == pytest.approx(0.5, abs=0.2)       # sqrt law
    assert out["eta"] > 0


def test_nonlinear_impact_none_when_participation_too_narrow():
    recs = [_rec("BUY", 100.0, 100.5, qty=100.0, bar_volume=10_000.0, realized_vol=1.0)
            for _ in range(10)]  # all identical participation
    assert fit_nonlinear_impact(recs) is None


# --------------------------------------------------------- rolling volatility
def _daily_bars(closes, start=1_700_000_000.0):
    return [{"close": c, "timestamp": start + i * 86400} for i, c in enumerate(closes)]


def test_vol_prior_on_empty():
    assert compute_rolling_volatility([], prior=0.33) == 0.33


def test_vol_prior_on_single_bar():
    assert compute_rolling_volatility(_daily_bars([100.0]), prior=0.33) == 0.33


@pytest.mark.parametrize("closes", [
    [100, 101, 100, 102, 99, 103, 98],
    [100, 100, 100, 100, 100, 100],     # zero-vol -> sigma 0
    [100, 110, 90, 120, 80, 130],
])
def test_vol_nonnegative(closes):
    v = compute_rolling_volatility(_daily_bars([float(c) for c in closes]), bars_per_day=1.0)
    assert v >= 0.0


def test_flat_series_has_zero_vol():
    v = compute_rolling_volatility(_daily_bars([100.0] * 10), bars_per_day=1.0)
    assert v == pytest.approx(0.0, abs=1e-12)


def test_more_volatile_series_has_higher_vol():
    calm = compute_rolling_volatility(_daily_bars([100, 100.5, 100, 100.5, 100, 100.5]), bars_per_day=1.0)
    wild = compute_rolling_volatility(_daily_bars([100, 110, 95, 115, 90, 120]), bars_per_day=1.0)
    assert wild > calm


def test_build_vol_series_keys_and_nonneg():
    bars = _daily_bars([100, 101, 99, 102, 98, 103, 97, 104])
    series = build_vol_series(bars, bars_per_day=1.0)
    assert set(series.keys()) == {b["timestamp"] for b in bars}
    assert all(v >= 0.0 for v in series.values())


def test_build_vol_series_first_bar_is_prior():
    bars = _daily_bars([100, 101, 102])
    series = build_vol_series(bars, bars_per_day=1.0, prior=0.15)
    assert series[bars[0]["timestamp"]] == 0.15  # <2 returns yet


# -------------------------------------------------------------- calibrate()
def test_calibrate_returns_result_with_latency():
    recs = []
    for i in range(12):
        recs.append(_rec("BUY", 100.0, 100.0 * (1 + (2.0 + i * 0.1) / 1e4),
                         qty=10.0 + i, bar_volume=10_000.0,
                         submit_to_ack_ms=200.0 + i, ack_to_fill_ms=50.0 + i))
    res = calibrate(recs)
    assert res.half_spread_bps is not None
    assert res.latency is not None
    assert res.n_records == 12
