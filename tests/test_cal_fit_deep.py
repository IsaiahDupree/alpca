"""
Deep, deterministic, network-free tests for alpca.calibration.fit.

Covers the pure/offline calibration math:
  - _percentile, _ols_two_param, _nelder_mead
  - fit_nonlinear_impact (beta bounds, eta>=0, None gates)
  - _regime_split_impact (vol-stratified sqrt-impact)
  - calibrate() branch coverage (priors / impact-fit / narrow / no-vol /
    measured-half-spread / latency) and the resulting flags
  - CalibrationResult fields + to_fill_model() coefficient mapping
  - LatencyPreset.to_dict
All inputs are constructed by hand and fully deterministic. No network, no
mocks, no live Alpaca paths are touched.
"""

from __future__ import annotations

import math

import pytest

from alpca.calibration import fit as F
from alpca.calibration.fit import (
    CalibrationResult,
    LatencyPreset,
    _nelder_mead,
    _ols_two_param,
    _percentile,
    _regime_split_impact,
    calibrate,
    fit_nonlinear_impact,
)
from alpca.calibration.records import CalibrationRecord


# --------------------------------------------------------------------------
# tiny self-contained builders (no import from other tests/ files)
# --------------------------------------------------------------------------
def make_rec(
    *,
    side: str = "BUY",
    qty: float = 10.0,
    intended_price: float = 100.0,
    slip_bps: float | None = 0.0,
    fill_price: float | None = None,
    bar_volume: float | None = None,
    realized_vol: float | None = None,
    bid: float | None = None,
    ask: float | None = None,
    submit_to_ack_ms: float | None = None,
    ack_to_fill_ms: float | None = None,
) -> CalibrationRecord:
    """Build a CalibrationRecord. If `slip_bps` is given (and fill_price isn't),
    compute the fill_price that yields exactly that signed slippage."""
    if fill_price is None:
        if slip_bps is None:
            # force slippage_bps property to None via bad intended_price
            intended_price = 0.0
            fill_price = 1.0
        else:
            diff = slip_bps / 10_000.0 * intended_price
            fill_price = intended_price + diff if side == "BUY" else intended_price - diff
    return CalibrationRecord(
        symbol="TEST",
        side=side,
        qty=qty,
        intended_price=intended_price,
        fill_price=fill_price,
        bar_volume=bar_volume,
        realized_vol=realized_vol,
        bid=bid,
        ask=ask,
        submit_to_ack_ms=submit_to_ack_ms,
        ack_to_fill_ms=ack_to_fill_ms,
    )


def impact_recs(coef: float, half: float = 1.0, vol: float | None = None,
                volumes=(10000, 5000, 2000, 1000, 500, 200), qty: float = 10.0):
    """Records whose slippage exactly follows half + coef*sqrt(participation)."""
    recs = []
    for bv in volumes:
        part = qty / bv
        slip = half + coef * math.sqrt(part)
        recs.append(make_rec(qty=qty, bar_volume=bv, slip_bps=slip, realized_vol=vol))
    return recs


APPROX = 1e-6


# ==========================================================================
# _percentile
# ==========================================================================
def test_percentile_empty_is_none():
    assert _percentile([], 0.5) is None


def test_percentile_single_returns_value_for_any_q():
    assert _percentile([7.0], 0.0) == 7.0
    assert _percentile([7.0], 0.99) == 7.0


@pytest.mark.parametrize(
    "vals,q,expected",
    [
        ([1.0, 2.0, 3.0, 4.0], 0.0, 1.0),
        ([1.0, 2.0, 3.0, 4.0], 1.0, 4.0),
        ([1.0, 2.0, 3.0, 4.0], 0.5, 2.5),
        ([1.0, 2.0, 3.0, 4.0], 0.25, 1.75),
        ([1.0, 2.0, 3.0, 4.0], 0.95, 1.0 + 2.85),  # idx=2.85 -> 3.85
        ([10.0, 20.0], 0.5, 15.0),
        ([0.0, 100.0], 0.5, 50.0),
    ],
)
def test_percentile_linear_interpolation(vals, q, expected):
    assert _percentile(vals, q) == pytest.approx(expected, abs=APPROX)


def test_percentile_q_landing_on_index_no_interp():
    # idx = 0.5*(3-1) = 1.0 exactly -> returns element 1
    assert _percentile([5.0, 9.0, 13.0], 0.5) == 9.0


def test_percentile_handles_negative_and_extreme_magnitudes():
    vals = [-1e9, 0.0, 1e9]
    assert _percentile(vals, 0.5) == 0.0
    assert _percentile(vals, 0.0) == -1e9
    assert _percentile(vals, 1.0) == 1e9


# ==========================================================================
# _ols_two_param
# ==========================================================================
def test_ols_recovers_exact_line():
    # y = 1 + 2x
    a, b = _ols_two_param([0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 5.0, 7.0])
    assert a == pytest.approx(1.0, abs=APPROX)
    assert b == pytest.approx(2.0, abs=APPROX)


def test_ols_recovers_negative_slope():
    a, b = _ols_two_param([0.0, 1.0, 2.0], [10.0, 7.0, 4.0])
    assert a == pytest.approx(10.0, abs=APPROX)
    assert b == pytest.approx(-3.0, abs=APPROX)


def test_ols_too_few_points_is_none():
    assert _ols_two_param([], []) is None
    assert _ols_two_param([1.0], [2.0]) is None


def test_ols_constant_x_is_none():
    # zero variance in x => degenerate => None
    assert _ols_two_param([2.0, 2.0, 2.0], [1.0, 2.0, 3.0]) is None


def test_ols_least_squares_on_noisy_data():
    # symmetric residuals around y = x; slope must be ~1, intercept ~0
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [0.1, 0.9, 2.1, 2.9, 4.0]
    a, b = _ols_two_param(xs, ys)
    assert b == pytest.approx(0.98, abs=0.05)
    assert abs(a) < 0.1


# ==========================================================================
# _nelder_mead
# ==========================================================================
def test_nelder_mead_minimizes_quadratic_bowl():
    f = lambda x: (x[0] - 3.0) ** 2 + (x[1] + 1.0) ** 2
    best, fbest = _nelder_mead(f, [0.0, 0.0])
    assert best[0] == pytest.approx(3.0, abs=1e-3)
    assert best[1] == pytest.approx(-1.0, abs=1e-3)
    assert fbest == pytest.approx(0.0, abs=1e-6)


def test_nelder_mead_one_dimensional():
    # A 1-element simplex (n=1) cannot shrink toward the optimum as tightly as
    # the 2-D bowl; assert it lands in the right neighborhood and the achieved
    # value is close to the floor (2.0), strictly above it.
    f = lambda x: (x[0] - 5.0) ** 2 + 2.0
    best, fbest = _nelder_mead(f, [0.0])
    assert best[0] == pytest.approx(5.0, abs=0.5)
    assert 2.0 <= fbest <= 2.1


def test_nelder_mead_starts_from_zero_coordinate():
    # x0 contains a zero coordinate -> the +step branch in simplex build
    f = lambda x: (x[0] - 1.0) ** 2 + (x[1] - 2.0) ** 2
    best, _ = _nelder_mead(f, [0.0, 0.0])
    assert best[0] == pytest.approx(1.0, abs=1e-3)
    assert best[1] == pytest.approx(2.0, abs=1e-3)


def test_nelder_mead_respects_objective_penalty_clamp():
    # objective penalizes x<0 heavily; min of (x-(-2))^2 unconstrained is -2,
    # but penalty pushes the optimum toward 0.
    def f(x):
        v = x[0]
        pen = (min(0.0, v)) ** 2 * 1e6
        return (v + 2.0) ** 2 + pen
    best, _ = _nelder_mead(f, [1.0])
    assert best[0] >= -1e-3  # pinned at/above the boundary


# ==========================================================================
# fit_nonlinear_impact
# ==========================================================================
def test_nonlinear_recovers_sqrt_law_params():
    # slippage = 1 + 5*sigma*part^0.5 with sigma=1.0
    recs = impact_recs(coef=5.0, half=1.0, vol=1.0)
    out = fit_nonlinear_impact(recs)
    assert out is not None
    assert out["beta"] == pytest.approx(0.5, abs=0.05)
    assert out["eta"] == pytest.approx(5.0, abs=0.2)
    assert out["c"] == pytest.approx(1.0, abs=0.3)


def test_nonlinear_beta_clamped_to_bounds():
    # true beta=0.7 -> within or clamped to [0.2,1.0]; whatever the fit, the
    # RETURNED beta must respect the documented bounds.
    recs = []
    for bv in (10000, 5000, 2000, 1000, 500, 200):
        part = 10.0 / bv
        recs.append(make_rec(qty=10, bar_volume=bv,
                             slip_bps=2.0 + 4.0 * part ** 0.7, realized_vol=1.0))
    out = fit_nonlinear_impact(recs)
    assert out is not None
    assert 0.2 <= out["beta"] <= 1.0


def test_nonlinear_eta_and_c_nonnegative():
    recs = impact_recs(coef=3.0, half=0.5, vol=1.0)
    out = fit_nonlinear_impact(recs)
    assert out is not None
    assert out["eta"] >= 0.0
    assert out["c"] >= 0.0


def test_nonlinear_none_when_too_few_points():
    recs = impact_recs(coef=5.0, volumes=(10000, 5000, 2000))  # 3 < default min_points 4
    assert fit_nonlinear_impact(recs) is None


def test_nonlinear_respects_custom_min_points():
    recs = impact_recs(coef=5.0, vol=1.0, volumes=(10000, 5000, 2000))  # 3 points
    assert fit_nonlinear_impact(recs, min_points=3) is not None
    assert fit_nonlinear_impact(recs, min_points=4) is None


def test_nonlinear_none_when_participation_range_too_narrow():
    # all identical bar_volume => identical participation => ratio 1.0 < 1.5
    recs = [make_rec(qty=10, bar_volume=10000, slip_bps=2.0, realized_vol=1.0)
            for _ in range(8)]
    assert fit_nonlinear_impact(recs) is None


def test_nonlinear_threshold_boundary_excludes_exactly_1_5x():
    # max/min == 1.5 exactly -> NOT > threshold via `< threshold` is False so...
    # gate is `(max/min) < threshold` => returns None. Build ratio just under 1.5.
    recs = []
    for bv in (1000, 900, 800, 700):  # parts 0.01..0.0143 ratio ~1.43 < 1.5
        recs.append(make_rec(qty=10, bar_volume=bv, slip_bps=2.0, realized_vol=1.0))
    ratio = max(10.0 / bv for bv in (1000, 900, 800, 700)) / min(10.0 / bv for bv in (1000, 900, 800, 700))
    assert ratio < 1.5
    assert fit_nonlinear_impact(recs) is None


def test_nonlinear_skips_records_without_participation():
    # records with no bar_volume have participation None and are filtered out;
    # only 3 valid remain -> None
    good = impact_recs(coef=5.0, vol=1.0, volumes=(10000, 4000, 1000))
    novol = [make_rec(qty=10, slip_bps=2.0) for _ in range(5)]
    assert fit_nonlinear_impact(good + novol) is None  # only 3 usable


def test_nonlinear_empty_is_none():
    assert fit_nonlinear_impact([]) is None


# ==========================================================================
# _regime_split_impact
# ==========================================================================
def test_regime_split_recovers_per_regime_coef():
    recs = []
    for vol, coef in ((0.10, 3.0), (0.40, 9.0)):
        recs += impact_recs(coef=coef, half=1.0, vol=vol,
                            volumes=(10000, 2000, 1000, 500))
    out = _regime_split_impact(recs)
    assert out is not None
    low, high, thr = out
    assert low == pytest.approx(3.0, abs=0.1)
    assert high == pytest.approx(9.0, abs=0.1)
    # median split of [0.10]*4 + [0.40]*4 interpolates to 0.25
    assert thr == pytest.approx(0.25, abs=APPROX)


def test_regime_split_none_when_too_few_records():
    # need >= 2*min_per_regime = 8; provide 6
    recs = impact_recs(coef=5.0, vol=0.2, volumes=(10000, 5000, 2000, 1000, 500, 200))
    assert _regime_split_impact(recs) is None


def test_regime_split_none_without_realized_vol():
    # realized_vol None on all => filtered out => fewer than 8 => None
    recs = impact_recs(coef=5.0, vol=None, volumes=(10000, 5000, 2000, 1000, 500, 200, 100, 50))
    assert _regime_split_impact(recs) is None


def test_regime_split_one_regime_narrow_returns_other():
    # low-vol regime has wide participation (fits); high-vol regime all identical
    # participation (narrow) -> high coef None, but low present => tuple returned.
    low = impact_recs(coef=4.0, half=1.0, vol=0.1, volumes=(10000, 2000, 1000, 500))
    high = [make_rec(qty=10, bar_volume=1000, slip_bps=3.0, realized_vol=0.9)
            for _ in range(4)]
    out = _regime_split_impact(low + high)
    assert out is not None
    coef_low, coef_high, thr = out
    assert coef_low == pytest.approx(4.0, abs=0.1)
    assert coef_high is None


def test_regime_split_none_when_both_regimes_narrow():
    # both regimes identical participation -> both coefs None -> overall None
    recs = []
    for vol in (0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9):
        recs.append(make_rec(qty=10, bar_volume=1000, slip_bps=2.0, realized_vol=vol))
    assert _regime_split_impact(recs) is None


def test_regime_split_empty_is_none():
    assert _regime_split_impact([]) is None


# ==========================================================================
# calibrate() — branch coverage + flags
# ==========================================================================
def test_calibrate_empty_returns_priors():
    r = calibrate([])
    assert r.n_records == 0
    assert r.n_buys == 0
    assert r.n_sells == 0
    assert r.half_spread_bps == 1.0
    assert r.impact_coef_bps == 8.0
    assert r.impact_fitted is False
    assert r.slippage_p50_bps is None
    assert r.slippage_p95_bps is None
    assert r.latency is None
    assert any("no valid records" in n for n in r.notes)


def test_calibrate_all_slippage_none_returns_priors():
    # intended_price <= 0 makes slippage_bps None -> treated as no valid records
    recs = [make_rec(slip_bps=None) for _ in range(5)]
    r = calibrate(recs)
    assert r.n_records == 0
    assert r.impact_fitted is False
    assert any("no valid records" in n for n in r.notes)


def test_calibrate_low_confidence_note_below_min_records():
    recs = [make_rec(slip_bps=2.0, qty=1) for _ in range(3)]
    r = calibrate(recs, min_records=6)
    assert r.n_records == 3
    assert any("low-confidence" in n for n in r.notes)


def test_calibrate_no_low_confidence_note_at_or_above_min():
    recs = [make_rec(slip_bps=2.0, qty=1) for _ in range(6)]
    r = calibrate(recs, min_records=6)
    assert not any("low-confidence" in n for n in r.notes)


def test_calibrate_buy_sell_counts():
    recs = [make_rec(side="BUY", slip_bps=1.0) for _ in range(4)] + \
           [make_rec(side="SELL", slip_bps=1.0) for _ in range(3)]
    r = calibrate(recs)
    assert r.n_records == 7
    assert r.n_buys == 4
    assert r.n_sells == 3


def test_calibrate_no_volume_context_keeps_prior_coef():
    recs = [make_rec(slip_bps=2.0, qty=1) for _ in range(6)]  # no bar_volume
    r = calibrate(recs)
    assert r.impact_fitted is False
    assert r.impact_coef_bps == 8.0
    assert r.half_spread_bps == pytest.approx(2.0, abs=APPROX)
    assert any("no volume context" in n for n in r.notes)


def test_calibrate_impact_fitted_recovers_coef():
    recs = impact_recs(coef=5.0, half=1.0)
    r = calibrate(recs)
    assert r.impact_fitted is True
    assert r.impact_coef_bps == pytest.approx(5.0, abs=0.05)
    assert any("impact fitted" in n for n in r.notes)


def test_calibrate_half_spread_from_bottom_quartile():
    # half-spread is the median slippage at the smallest-participation fills
    recs = impact_recs(coef=5.0, half=1.0)
    r = calibrate(recs)
    # cut = p25 of participations = 0.00275 -> bottom-quartile fills are the two
    # smallest (parts 0.001, 0.002); half-spread = median of their slippages.
    s1 = 1.0 + 5.0 * math.sqrt(0.001)
    s2 = 1.0 + 5.0 * math.sqrt(0.002)
    expected = round((s1 + s2) / 2.0, 3)
    assert r.half_spread_bps == pytest.approx(expected, abs=APPROX)


def test_calibrate_narrow_participation_keeps_prior():
    recs = [make_rec(qty=10, bar_volume=10000, slip_bps=2.0) for _ in range(6)]
    r = calibrate(recs)
    assert r.impact_fitted is False
    assert r.impact_coef_bps == 8.0
    assert any("too narrow" in n for n in r.notes)


def test_calibrate_fewer_than_four_points_not_fitted():
    # wide participation but only 3 records -> len(with_part) >= 4 fails
    recs = impact_recs(coef=5.0, volumes=(10000, 1000, 200))
    r = calibrate(recs)
    assert r.impact_fitted is False
    assert r.impact_coef_bps == 8.0


def test_calibrate_half_spread_measured_from_quotes():
    # bid=99.9 ask=100.1 mid=100 -> half-spread = (0.2/2)/100*1e4 = 10 bps
    recs = [make_rec(bid=99.9, ask=100.1, slip_bps=2.0) for _ in range(5)]
    r = calibrate(recs)
    assert r.half_spread_measured is True
    assert r.half_spread_bps == pytest.approx(10.0, abs=0.001)
    assert any("half-spread measured" in n for n in r.notes)


def test_calibrate_quotes_below_four_not_measured():
    recs = [make_rec(bid=99.9, ask=100.1, slip_bps=2.0) for _ in range(3)]
    r = calibrate(recs)
    assert r.half_spread_measured is False


def test_calibrate_half_spread_nonnegative():
    # negative slippage (favorable fills) must not produce a negative half-spread
    recs = [make_rec(slip_bps=-5.0, qty=1) for _ in range(6)]
    r = calibrate(recs)
    assert r.half_spread_bps >= 0.0


def test_calibrate_percentiles_computed_and_rounded():
    # slippages 1..6 bps; p50 interp of sorted [1..6] = 3.5
    recs = [make_rec(slip_bps=float(s), qty=1) for s in range(1, 7)]
    r = calibrate(recs)
    assert r.slippage_p50_bps == pytest.approx(3.5, abs=APPROX)
    assert r.slippage_p95_bps is not None
    assert r.slippage_p95_bps >= r.slippage_p50_bps


def test_calibrate_latency_preset_populated():
    recs = [make_rec(qty=1, submit_to_ack_ms=100.0, ack_to_fill_ms=50.0)
            for _ in range(6)]
    r = calibrate(recs)
    assert r.latency is not None
    # sub_ack p50 = 100 -> split 50/50; ack_fill p50 = 50
    assert r.latency.submit_latency_ms == pytest.approx(50.0, abs=APPROX)
    assert r.latency.ack_latency_ms == pytest.approx(50.0, abs=APPROX)
    assert r.latency.fill_latency_ms == pytest.approx(50.0, abs=APPROX)
    assert r.latency.jitter_ms == pytest.approx(0.0, abs=APPROX)


def test_calibrate_latency_jitter_from_p95_spread():
    # varied submit_to_ack -> nonzero jitter (p95 - p50)
    vals = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    recs = [make_rec(qty=1, submit_to_ack_ms=v, ack_to_fill_ms=10.0) for v in vals]
    r = calibrate(recs)
    assert r.latency is not None
    assert r.latency.jitter_ms > 0.0


def test_calibrate_no_latency_note_when_absent():
    recs = [make_rec(slip_bps=2.0, qty=1) for _ in range(6)]
    r = calibrate(recs)
    assert r.latency is None
    assert any("no latency captured" in n for n in r.notes)


def test_calibrate_nonlinear_and_regime_extras_populated():
    recs = []
    for vol, coef in ((0.10, 3.0), (0.40, 9.0)):
        recs += impact_recs(coef=coef, half=1.0, vol=vol,
                            volumes=(10000, 2000, 1000, 500))
    r = calibrate(recs)
    assert r.beta_fitted is True
    assert r.eta is not None and r.eta >= 0.0
    assert r.beta is not None and 0.2 <= r.beta <= 1.0
    assert r.impact_coef_low_vol == pytest.approx(3.0, abs=0.1)
    assert r.impact_coef_high_vol == pytest.approx(9.0, abs=0.1)
    assert r.vol_threshold == pytest.approx(0.25, abs=APPROX)


def test_calibrate_extras_absent_when_not_fittable():
    recs = [make_rec(slip_bps=2.0, qty=1) for _ in range(6)]  # no volume
    r = calibrate(recs)
    assert r.beta_fitted is False
    assert r.eta is None
    assert r.beta is None
    assert r.impact_coef_low_vol is None
    assert r.impact_coef_high_vol is None
    assert r.vol_threshold is None


def test_calibrate_handles_extreme_and_out_of_order_slippage():
    # huge magnitudes + reversed order must not crash; percentiles ordered
    vals = [1e4, 1.0, -1e4, 500.0, 2.0, 3.0]
    recs = [make_rec(slip_bps=v, qty=1) for v in vals]
    r = calibrate(recs)
    assert r.n_records == 6
    assert r.slippage_p50_bps <= r.slippage_p95_bps


# ==========================================================================
# CalibrationResult.to_fill_model + to_dict + LatencyPreset
# ==========================================================================
def test_to_fill_model_maps_coefficients():
    r = CalibrationResult(
        n_records=5, n_buys=3, n_sells=2,
        half_spread_bps=2.5, impact_coef_bps=6.0, impact_fitted=True,
        slippage_p50_bps=2.0, slippage_p95_bps=4.0, latency=None,
    )
    fm = r.to_fill_model()
    assert fm.half_spread_bps == pytest.approx(2.5, abs=APPROX)
    assert fm.impact_coef_bps == pytest.approx(6.0, abs=APPROX)
    assert fm.participation_cap == pytest.approx(0.10, abs=APPROX)
    assert fm.min_tick == pytest.approx(0.01, abs=APPROX)


def test_to_fill_model_custom_cap_and_tick():
    r = calibrate([])
    fm = r.to_fill_model(participation_cap=0.5, min_tick=0.05)
    assert fm.participation_cap == pytest.approx(0.5, abs=APPROX)
    assert fm.min_tick == pytest.approx(0.05, abs=APPROX)


def test_to_fill_model_clamps_negative_coefficients():
    r = CalibrationResult(
        n_records=1, n_buys=1, n_sells=0,
        half_spread_bps=-5.0, impact_coef_bps=-3.0, impact_fitted=False,
        slippage_p50_bps=None, slippage_p95_bps=None, latency=None,
    )
    fm = r.to_fill_model()
    assert fm.half_spread_bps == 0.0
    assert fm.impact_coef_bps == 0.0


def test_to_dict_serializes_latency_nested():
    lat = LatencyPreset(1.0, 2.0, 3.0, 4.0)
    r = CalibrationResult(
        n_records=2, n_buys=1, n_sells=1,
        half_spread_bps=1.0, impact_coef_bps=8.0, impact_fitted=False,
        slippage_p50_bps=1.0, slippage_p95_bps=2.0, latency=lat,
    )
    d = r.to_dict()
    assert d["latency"] == {
        "submit_latency_ms": 1.0,
        "ack_latency_ms": 2.0,
        "fill_latency_ms": 3.0,
        "jitter_ms": 4.0,
    }
    assert d["half_spread_bps"] == 1.0
    assert d["impact_fitted"] is False


def test_to_dict_latency_none_passthrough():
    r = calibrate([])
    d = r.to_dict()
    assert d["latency"] is None


def test_latency_preset_to_dict_keys():
    lp = LatencyPreset(10.0, 20.0, 30.0, 5.0)
    assert lp.to_dict() == {
        "submit_latency_ms": 10.0,
        "ack_latency_ms": 20.0,
        "fill_latency_ms": 30.0,
        "jitter_ms": 5.0,
    }


def test_calibration_result_default_flags():
    r = CalibrationResult(
        n_records=0, n_buys=0, n_sells=0,
        half_spread_bps=1.0, impact_coef_bps=8.0, impact_fitted=False,
        slippage_p50_bps=None, slippage_p95_bps=None, latency=None,
    )
    assert r.notes == []
    assert r.half_spread_measured is False
    assert r.eta is None
    assert r.beta is None
    assert r.beta_fitted is False
    assert r.impact_coef_low_vol is None
    assert r.impact_coef_high_vol is None
    assert r.vol_threshold is None


# ==========================================================================
# record-level properties used by the fitter (sanity invariants)
# ==========================================================================
@pytest.mark.parametrize(
    "side,intended,fill,expected",
    [
        ("BUY", 100.0, 101.0, 100.0),    # paid more -> worse -> +100 bps
        ("SELL", 100.0, 99.0, 100.0),    # received less -> worse -> +100 bps
        ("BUY", 100.0, 99.0, -100.0),    # paid less -> favorable -> -100 bps
        ("SELL", 100.0, 101.0, -100.0),  # received more -> favorable -> -100 bps
    ],
)
def test_record_slippage_sign_convention(side, intended, fill, expected):
    rec = CalibrationRecord("X", side, 1.0, intended, fill)
    assert rec.slippage_bps == pytest.approx(expected, abs=APPROX)


def test_record_slippage_none_on_nonpositive_intended():
    assert CalibrationRecord("X", "BUY", 1.0, 0.0, 1.0).slippage_bps is None


def test_record_participation_none_without_volume():
    assert CalibrationRecord("X", "BUY", 1.0, 100.0, 100.0).participation is None
    assert CalibrationRecord("X", "BUY", 1.0, 100.0, 100.0, bar_volume=0).participation is None


def test_record_quoted_half_spread_none_when_crossed_or_missing():
    # crossed book (ask <= bid) -> None
    assert CalibrationRecord("X", "BUY", 1.0, 100.0, 100.0,
                             bid=100.2, ask=100.1).quoted_half_spread_bps is None
    # missing bid -> None
    assert CalibrationRecord("X", "BUY", 1.0, 100.0, 100.0,
                             ask=100.1).quoted_half_spread_bps is None
