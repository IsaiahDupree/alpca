"""
Phase 1: nonlinear (η,β) impact fit, vol-regime split, and spread decomposition
in calibration/fit.py. Legacy half_spread/impact_coef behavior must be unchanged.
"""

import math

from alpca.calibration.fit import (
    _regime_split_impact,
    calibrate,
    fit_nonlinear_impact,
)
from alpca.calibration.records import CalibrationRecord


def _rec(part, slip_bps, *, sigma=0.2, vol=1e6, bid=None, ask=None, side="BUY"):
    intended = 100.0
    fill = intended * (1 + slip_bps / 1e4) if side == "BUY" else intended * (1 - slip_bps / 1e4)
    return CalibrationRecord(symbol="SPY", side=side, qty=part * vol,
                             intended_price=intended, fill_price=fill, bar_volume=vol,
                             realized_vol=sigma, bid=bid, ask=ask)


def test_nonlinear_recovers_power_curve():
    # noiseless slippage = eta*sigma*part^beta (c=0); fit should reproduce it
    eta_t, sig, beta_t = 100.0, 0.25, 0.55
    parts = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
    recs = [_rec(p, eta_t * sig * (p ** beta_t), sigma=sig) for p in parts]
    fit = fit_nonlinear_impact(recs)
    assert fit is not None
    assert 0.4 <= fit["beta"] <= 0.7                       # recovers the exponent region
    # the fitted curve reproduces the points
    for p in parts:
        pred = fit["c"] + fit["eta"] * sig * (p ** fit["beta"])
        actual = eta_t * sig * (p ** beta_t)
        assert abs(pred - actual) < 0.05


def test_nonlinear_returns_none_on_narrow_participation():
    recs = [_rec(1e-5 * (1 + 0.01 * i), 3.0) for i in range(6)]  # ratio < 1.5
    assert fit_nonlinear_impact(recs) is None


def test_nonlinear_returns_none_on_too_few_points():
    recs = [_rec(1e-5, 2.0), _rec(1e-3, 5.0), _rec(1e-4, 3.0)]   # 3 < min_points
    assert fit_nonlinear_impact(recs) is None


def test_regime_split_recovers_distinct_coefficients():
    parts = [1e-5, 1e-4, 5e-4, 1e-3]
    low = [_rec(p, 2 + 5.0 * math.sqrt(p), sigma=0.10) for p in parts]
    high = [_rec(p, 2 + 30.0 * math.sqrt(p), sigma=0.40) for p in parts]
    out = _regime_split_impact(low + high)
    assert out is not None
    coef_low, coef_high, thr = out
    assert coef_low is not None and coef_high is not None
    assert coef_high > coef_low                            # high-vol regime = bigger impact
    assert 0.10 < thr < 0.40


def test_calibrate_measures_half_spread_from_quotes():
    # bid/ask spread 0.03 around mid 100 -> half-spread 1.5 bps
    recs = [_rec(1e-5 * (1 + i), 1.5, bid=99.985, ask=100.015) for i in range(6)]
    res = calibrate(recs)
    assert res.half_spread_measured is True
    assert abs(res.half_spread_bps - 1.5) < 0.1


def test_calibrate_backcompat_without_quotes():
    recs = [_rec(p, 2 + 8.0 * math.sqrt(p)) for p in (1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 3e-3)]
    res = calibrate(recs)
    assert res.half_spread_measured is False              # no quotes -> legacy path
    assert res.impact_fitted is True                      # legacy sqrt OLS still runs
    assert res.impact_coef_bps > 0
