"""
Robustness tests for alpca.calibration.* (records, fit, volatility).

Focus: degenerate / edge inputs must NOT crash. calibrate() always returns a
CalibrationResult with safe defaults (priors / None) plus explanatory notes, and
fit_nonlinear_impact() returns None (rather than raising) when it cannot fit.

All inputs are deterministic and constructed in-process. No network, no mocks,
no other tests/ imports. Pure/offline calibration logic only.
"""

from __future__ import annotations

import math
import statistics

import pytest

from alpca.calibration.fit import (
    CalibrationResult,
    LatencyPreset,
    calibrate,
    fit_nonlinear_impact,
    _fit_latency,
    _ols_two_param,
    _percentile,
    _regime_split_impact,
)
from alpca.calibration.records import CalibrationRecord
from alpca.calibration.volatility import (
    build_vol_series,
    compute_rolling_volatility,
)

# Priors hardcoded in source — kept in sync with alpca.calibration.*
PRIOR_VOL = 0.15
PRIOR_HALF_SPREAD = 1.0
PRIOR_IMPACT_COEF = 8.0


# --------------------------------------------------------------------------- #
# tiny self-contained helpers
# --------------------------------------------------------------------------- #
def rec(side="BUY", qty=10.0, ip=100.0, fp=100.0, vol=None, **kw):
    return CalibrationRecord(
        symbol="TST", side=side, qty=qty, intended_price=ip, fill_price=fp,
        bar_volume=vol, **kw,
    )


def bar(close, ts):
    return {"close": float(close), "timestamp": float(ts)}


def make_sqrt_impact_records(n=10, half=2.0, coef=5.0, vol_shares=100_000.0):
    """Synthetic fills following slippage = half + coef*sqrt(participation),
    with a real participation range so impact is fittable and recoverable."""
    out = []
    for i in range(n):
        part = (i + 1) * 0.001               # 0.001 .. n*0.001 (>1.5x spread)
        qty = part * vol_shares
        slip = half + coef * math.sqrt(part)
        fp = 100.0 * (1.0 + slip / 10_000.0)
        out.append(rec(side="BUY", qty=qty, ip=100.0, fp=fp, vol=vol_shares,
                       submit_to_ack_ms=20.0 + i, ack_to_fill_ms=50.0 + i))
    return out


# =========================================================================== #
# CalibrationRecord.slippage_bps
# =========================================================================== #
@pytest.mark.parametrize(
    "side, ip, fp, expected",
    [
        ("BUY", 100.0, 100.10, 10.0),     # paid 0.10 more -> +10 bps (worse)
        ("SELL", 100.0, 99.90, 10.0),     # received 0.10 less -> +10 bps (worse)
        ("BUY", 100.0, 99.90, -10.0),     # price improvement on a buy -> negative
        ("SELL", 100.0, 100.10, -10.0),   # price improvement on a sell -> negative
        ("BUY", 100.0, 100.0, 0.0),       # exact fill -> zero slippage
    ],
)
def test_slippage_bps_signed_directions(side, ip, fp, expected):
    got = rec(side=side, ip=ip, fp=fp).slippage_bps
    assert got == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("ip", [0.0, -1.0, -100.0])
def test_slippage_bps_nonpositive_intended_returns_none(ip):
    # intended_price <= 0 is explicitly guarded -> None, never a div-by-zero.
    assert rec(ip=ip, fp=100.0).slippage_bps is None


def test_slippage_bps_none_prices_return_none():
    assert rec(ip=None, fp=100.0).slippage_bps is None          # type: ignore[arg-type]
    assert rec(ip=100.0, fp=None).slippage_bps is None          # type: ignore[arg-type]


def test_slippage_bps_nan_intended_is_not_guarded_propagates_nan():
    # NaN <= 0 is False, so the guard does not trip and NaN flows through.
    # Document the ACTUAL behavior: result is NaN, not None.
    out = rec(ip=float("nan"), fp=100.0).slippage_bps
    assert out is not None and math.isnan(out)


def test_slippage_bps_nan_fill_propagates_nan():
    out = rec(ip=100.0, fp=float("nan")).slippage_bps
    assert out is not None and math.isnan(out)


# =========================================================================== #
# CalibrationRecord.participation
# =========================================================================== #
@pytest.mark.parametrize(
    "qty, vol, expected",
    [
        (100.0, 1000.0, 0.1),
        (1.0, 4.0, 0.25),
        (50.0, 50.0, 1.0),
    ],
)
def test_participation_basic(qty, vol, expected):
    assert rec(qty=qty, vol=vol).participation == pytest.approx(expected)


@pytest.mark.parametrize("vol", [None, 0.0, -10.0])
def test_participation_missing_or_nonpositive_volume_returns_none(vol):
    assert rec(qty=10.0, vol=vol).participation is None


# =========================================================================== #
# CalibrationRecord.quoted_half_spread_bps
# =========================================================================== #
def test_quoted_half_spread_basic():
    # bid 99.5 / ask 100.5 -> spread 1.0, half 0.5, mid 100 -> 50 bps
    assert rec(bid=99.5, ask=100.5).quoted_half_spread_bps == pytest.approx(50.0)


@pytest.mark.parametrize(
    "bid, ask",
    [
        (None, 100.5),     # missing bid
        (99.5, None),      # missing ask
        (100.5, 99.5),     # inverted (ask <= bid)
        (100.0, 100.0),    # crossed/equal
        (0.0, 1.0),        # falsy bid
    ],
)
def test_quoted_half_spread_degenerate_returns_none(bid, ask):
    assert rec(bid=bid, ask=ask).quoted_half_spread_bps is None


def test_to_dict_includes_derived_fields():
    d = rec(side="BUY", ip=100.0, fp=100.1, vol=1000.0, bid=99.5, ask=100.5).to_dict()
    assert d["slippage_bps"] == pytest.approx(10.0)
    assert d["participation"] == pytest.approx(10.0 / 1000.0)
    assert d["quoted_half_spread_bps"] == pytest.approx(50.0)
    # original dataclass fields are present too
    assert d["symbol"] == "TST" and d["side"] == "BUY"


# =========================================================================== #
# calibrate() — degenerate inputs return safe defaults, never crash
# =========================================================================== #
def test_calibrate_empty_returns_priors():
    r = calibrate([])
    assert isinstance(r, CalibrationResult)
    assert (r.n_records, r.n_buys, r.n_sells) == (0, 0, 0)
    assert r.half_spread_bps == PRIOR_HALF_SPREAD
    assert r.impact_coef_bps == PRIOR_IMPACT_COEF
    assert r.impact_fitted is False
    assert r.slippage_p50_bps is None and r.slippage_p95_bps is None
    assert r.latency is None
    assert any("no valid records" in n for n in r.notes)


def test_calibrate_all_records_unscorable_treated_as_empty():
    # every record has intended_price <= 0 -> slippage None -> no valid records
    bad = [rec(ip=0.0, fp=100.0) for _ in range(5)]
    r = calibrate(bad)
    assert r.n_records == 0
    assert r.half_spread_bps == PRIOR_HALF_SPREAD
    assert any("no valid records" in n for n in r.notes)


def test_calibrate_single_record_low_confidence():
    r = calibrate([rec(side="BUY", ip=100.0, fp=100.1)])
    assert r.n_records == 1 and r.n_buys == 1 and r.n_sells == 0
    # no volume -> half-spread is the median slippage (10 bps), impact unfit
    assert r.half_spread_bps == pytest.approx(10.0)
    assert r.impact_fitted is False
    assert r.impact_coef_bps == PRIOR_IMPACT_COEF
    assert any("low-confidence" in n for n in r.notes)
    assert any("no volume context" in n for n in r.notes)


def test_calibrate_no_volume_keeps_prior_impact():
    recs = [rec(side="BUY", ip=100.0, fp=100.0 + i * 0.01) for i in range(8)]
    r = calibrate(recs)
    assert r.impact_fitted is False
    assert r.impact_coef_bps == PRIOR_IMPACT_COEF
    assert any("no volume context" in n for n in r.notes)
    # half-spread is the median slippage of the 8 fills
    slips = sorted(x.slippage_bps for x in recs)
    assert r.half_spread_bps == pytest.approx(round(statistics.median(slips), 3))


def test_calibrate_identical_participation_impact_unfittable():
    # same qty + same bar_volume -> participation identical -> ratio == 1 < 1.5
    recs = [rec(side="BUY", qty=10.0, ip=100.0, fp=100.0 + i * 0.01, vol=1000.0)
            for i in range(8)]
    r = calibrate(recs)
    assert r.impact_fitted is False
    assert r.impact_coef_bps == PRIOR_IMPACT_COEF
    assert any("range too narrow" in n for n in r.notes)


def test_calibrate_all_identical_fills_zero_slippage_variance():
    # identical fills (zero slippage) but varying participation -> impact slope 0
    recs = [rec(side="BUY", qty=10.0, ip=100.0, fp=100.0, vol=1000.0 + i * 500.0)
            for i in range(8)]
    r = calibrate(recs)
    assert r.half_spread_bps == 0.0
    assert r.slippage_p50_bps == 0.0 and r.slippage_p95_bps == 0.0
    # participation varies so an OLS fit happens, but the slope is ~0
    assert r.impact_fitted is True
    assert r.impact_coef_bps == pytest.approx(0.0, abs=1e-6)


def test_calibrate_recovers_sqrt_impact_coefficient():
    r = calibrate(make_sqrt_impact_records(n=10, half=2.0, coef=5.0))
    assert r.impact_fitted is True
    assert r.impact_coef_bps == pytest.approx(5.0, abs=0.2)
    assert any("impact fitted" in n for n in r.notes)


def test_calibrate_buy_sell_counts_partition_records():
    recs = ([rec(side="BUY", ip=100.0, fp=100.1) for _ in range(3)] +
            [rec(side="SELL", ip=100.0, fp=99.9) for _ in range(2)])
    r = calibrate(recs)
    assert r.n_records == 5
    assert r.n_buys == 3 and r.n_sells == 2
    assert r.n_buys + r.n_sells == r.n_records


def test_calibrate_quoted_half_spread_measured_when_nbbo_present():
    recs = [rec(side="BUY", ip=100.0, fp=100.1, vol=1000.0,
                bid=99.95, ask=100.05) for _ in range(5)]
    r = calibrate(recs)
    assert r.half_spread_measured is True
    # spread 0.10, half 0.05, mid 100 -> 5 bps
    assert r.half_spread_bps == pytest.approx(5.0, abs=1e-6)
    assert any("half-spread measured" in n for n in r.notes)


def test_calibrate_few_quotes_does_not_trigger_measured():
    # only 3 quoted records (< 4 threshold) -> not measured
    recs = ([rec(side="BUY", ip=100.0, fp=100.1, vol=1000.0, bid=99.95, ask=100.05)
             for _ in range(3)] +
            [rec(side="BUY", ip=100.0, fp=100.1, vol=1000.0) for _ in range(3)])
    r = calibrate(recs)
    assert r.half_spread_measured is False


def test_calibrate_latency_preset_from_measured_legs():
    r = calibrate(make_sqrt_impact_records(n=10))
    assert r.latency is not None
    assert isinstance(r.latency, LatencyPreset)
    # submit+ack legs split 50/50 from sub->ack median; non-negative jitter
    assert r.latency.submit_latency_ms == r.latency.ack_latency_ms
    assert r.latency.jitter_ms >= 0.0
    assert r.latency.fill_latency_ms > 0.0


def test_calibrate_no_latency_when_legs_absent():
    recs = [rec(side="BUY", ip=100.0, fp=100.1) for _ in range(6)]
    r = calibrate(recs)
    assert r.latency is None
    assert any("no latency captured" in n for n in r.notes)


def test_calibrate_result_round_trips_to_dict():
    r = calibrate(make_sqrt_impact_records(n=10))
    d = r.to_dict()
    assert d["n_records"] == 10
    assert d["impact_fitted"] is True
    assert isinstance(d["latency"], dict)
    assert "submit_latency_ms" in d["latency"]
    assert isinstance(d["notes"], list)


def test_calibrate_to_fill_model_clamps_negatives_nonnegative():
    # craft a result where half_spread would be negative-ish; ensure clamp at 0
    r = calibrate([rec(side="BUY", ip=100.0, fp=99.9) for _ in range(6)])  # negative slip
    fm = r.to_fill_model()
    assert fm.half_spread_bps >= 0.0
    assert fm.impact_coef_bps >= 0.0


def test_calibrate_idempotent_on_same_input():
    recs = make_sqrt_impact_records(n=10)
    a = calibrate(recs).to_dict()
    b = calibrate(recs).to_dict()
    # latency dicts compare structurally; whole result must be deterministic
    assert a == b


def test_calibrate_handles_extreme_magnitude_prices():
    # huge and tiny prices must not overflow / crash
    recs = [rec(side="BUY", ip=1e9, fp=1e9 * 1.0001, vol=1e12) for _ in range(6)]
    r = calibrate(recs)
    assert isinstance(r, CalibrationResult)
    assert r.half_spread_bps >= 0.0
    assert math.isfinite(r.half_spread_bps)


def test_calibrate_order_independent_for_summary_stats():
    recs = make_sqrt_impact_records(n=10)
    forward = calibrate(recs)
    reverse = calibrate(list(reversed(recs)))
    assert forward.n_records == reverse.n_records
    assert forward.half_spread_bps == pytest.approx(reverse.half_spread_bps, abs=1e-6)
    assert forward.impact_coef_bps == pytest.approx(reverse.impact_coef_bps, abs=1e-6)
    assert forward.slippage_p50_bps == pytest.approx(reverse.slippage_p50_bps, abs=1e-6)


# =========================================================================== #
# fit_nonlinear_impact() — returns None (never raises) on un-fittable input
# =========================================================================== #
def test_fit_nonlinear_empty_returns_none():
    assert fit_nonlinear_impact([]) is None


def test_fit_nonlinear_too_few_points_returns_none():
    # default min_points=4
    recs = make_sqrt_impact_records(n=3)
    assert fit_nonlinear_impact(recs) is None


def test_fit_nonlinear_no_volume_returns_none():
    recs = [rec(side="BUY", ip=100.0, fp=100.1) for _ in range(8)]  # no participation
    assert fit_nonlinear_impact(recs) is None


def test_fit_nonlinear_narrow_participation_returns_none():
    # identical participation -> ratio 1 < threshold 1.5
    recs = [rec(side="BUY", qty=10.0, ip=100.0, fp=100.0 + i * 0.01, vol=1000.0)
            for i in range(8)]
    assert fit_nonlinear_impact(recs) is None


def test_fit_nonlinear_recovers_beta_half_and_eta():
    # realized_vol = 1.0 so eta absorbs the coef; expect eta~coef, beta~0.5
    recs = []
    for i in range(12):
        part = (i + 1) * 0.001
        qty = part * 100_000.0
        slip = 2.0 + 5.0 * math.sqrt(part)
        fp = 100.0 * (1.0 + slip / 10_000.0)
        recs.append(rec(side="BUY", qty=qty, ip=100.0, fp=fp, vol=100_000.0,
                        realized_vol=1.0))
    out = fit_nonlinear_impact(recs, half_spread_hint=2.0)
    assert out is not None
    assert set(out) == {"c", "eta", "beta"}
    assert 0.2 <= out["beta"] <= 1.0
    assert out["c"] >= 0.0 and out["eta"] >= 0.0
    assert out["beta"] == pytest.approx(0.5, abs=0.2)
    assert out["eta"] == pytest.approx(5.0, rel=0.25)


# =========================================================================== #
# _regime_split_impact() — None when not splittable
# =========================================================================== #
def test_regime_split_none_without_realized_vol():
    recs = make_sqrt_impact_records(n=12)  # realized_vol all None
    assert _regime_split_impact(recs) is None


def test_regime_split_none_too_few_records():
    recs = [rec(side="BUY", qty=10.0 * (i + 1), ip=100.0, fp=100.1,
                vol=100_000.0, realized_vol=0.1 + 0.01 * i) for i in range(5)]
    # 2*min_per_regime defaults to 8; 5 < 8 -> None
    assert _regime_split_impact(recs) is None


def test_regime_split_returns_threshold_when_splittable():
    recs = []
    for i in range(16):
        part = (i + 1) * 0.001
        qty = part * 100_000.0
        slip = 2.0 + 5.0 * math.sqrt(part)
        fp = 100.0 * (1.0 + slip / 10_000.0)
        recs.append(rec(side="BUY", qty=qty, ip=100.0, fp=fp, vol=100_000.0,
                        realized_vol=0.10 + 0.01 * i))
    out = _regime_split_impact(recs)
    assert out is not None
    low, high, thr = out
    assert thr is not None and thr > 0.0
    # at least one regime produces a coefficient
    assert low is not None or high is not None


# =========================================================================== #
# _percentile / _ols_two_param helpers
# =========================================================================== #
def test_percentile_empty_returns_none():
    assert _percentile([], 0.5) is None


def test_percentile_single_value():
    assert _percentile([7.0], 0.5) == 7.0
    assert _percentile([7.0], 0.0) == 7.0
    assert _percentile([7.0], 1.0) == 7.0


@pytest.mark.parametrize(
    "q, expected",
    [
        (0.0, 0.0),
        (0.5, 2.0),    # midpoint of [0,1,2,3,4]
        (1.0, 4.0),
        (0.25, 1.0),
    ],
)
def test_percentile_interpolation(q, expected):
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert _percentile(vals, q) == pytest.approx(expected)


def test_ols_two_param_too_few_points_returns_none():
    assert _ols_two_param([1.0], [2.0]) is None
    assert _ols_two_param([], []) is None


def test_ols_two_param_degenerate_x_returns_none():
    # all x equal -> sxx ~ 0 -> None
    assert _ols_two_param([3.0, 3.0, 3.0], [1.0, 2.0, 3.0]) is None


def test_ols_two_param_recovers_known_line():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [1.0, 3.0, 5.0, 7.0]  # y = 1 + 2x
    a, b = _ols_two_param(xs, ys)
    assert a == pytest.approx(1.0)
    assert b == pytest.approx(2.0)


# =========================================================================== #
# _fit_latency()
# =========================================================================== #
def test_fit_latency_none_when_no_legs():
    recs = [rec(side="BUY", ip=100.0, fp=100.1) for _ in range(4)]
    assert _fit_latency(recs) is None


def test_fit_latency_uses_only_ack_fill_leg():
    recs = [rec(side="BUY", ip=100.0, fp=100.1, ack_to_fill_ms=40.0 + i)
            for i in range(5)]
    lp = _fit_latency(recs)
    assert lp is not None
    # no submit->ack data -> submit/ack legs are 0, fill leg is the median
    assert lp.submit_latency_ms == 0.0 and lp.ack_latency_ms == 0.0
    assert lp.fill_latency_ms == pytest.approx(42.0)
    assert lp.jitter_ms == 0.0


# =========================================================================== #
# compute_rolling_volatility() — degenerate series
# =========================================================================== #
def test_vol_empty_returns_prior():
    assert compute_rolling_volatility([]) == PRIOR_VOL


def test_vol_single_bar_returns_prior():
    assert compute_rolling_volatility([bar(100.0, 1)]) == PRIOR_VOL


def test_vol_flat_series_is_zero():
    bars = [bar(100.0, i) for i in range(20)]
    assert compute_rolling_volatility(bars, bars_per_day=1.0) == 0.0


@pytest.mark.parametrize("price", [0.0, -1.0, -100.0])
def test_vol_nonpositive_prices_return_prior(price):
    # log of <=0 prices is guarded -> no usable returns -> prior
    bars = [bar(price, i) for i in range(20)]
    assert compute_rolling_volatility(bars) == PRIOR_VOL


def test_vol_custom_prior_respected():
    assert compute_rolling_volatility([], prior=0.42) == 0.42
    assert compute_rolling_volatility([bar(100.0, 1)], prior=0.42) == 0.42


def test_vol_non_annualized_matches_population_stdev():
    closes = [100.0, 101.0, 102.0, 101.0, 103.0]
    bars = [bar(c, i) for i, c in enumerate(closes)]
    got = compute_rolling_volatility(bars, lookback_days=100.0,
                                     bars_per_day=1.0, annualize=False)
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    assert got == pytest.approx(statistics.pstdev(rets))


def test_vol_annualization_scales_by_sqrt_bars_per_year():
    closes = [100.0, 101.0, 102.0, 101.0, 103.0]
    bars = [bar(c, i) for i, c in enumerate(closes)]
    raw = compute_rolling_volatility(bars, lookback_days=100.0,
                                     bars_per_day=1.0, annualize=False)
    ann = compute_rolling_volatility(bars, lookback_days=100.0,
                                     bars_per_day=1.0, annualize=True)
    assert ann == pytest.approx(raw * math.sqrt(1.0 * 252))


def test_vol_missing_close_raises_keyerror():
    # close is a required key (no .get fallback); document the actual behavior.
    with pytest.raises(KeyError):
        compute_rolling_volatility([{"timestamp": 1}, {"timestamp": 2}],
                                   bars_per_day=1.0)


def test_vol_nonfinite_close_filtered_falls_back_to_prior():
    # NaN close: `nan > 0` is False, so the positivity guard drops every return
    # that touches the NaN bar. Here only 101->102 survives (1 return < 2) so the
    # function safely returns the prior rather than producing/propagating NaN.
    bars = [bar(100.0, 0), {"close": float("nan"), "timestamp": 1},
            bar(101.0, 2), bar(102.0, 3)]
    out = compute_rolling_volatility(bars, lookback_days=100.0, bars_per_day=1.0,
                                     annualize=False)
    assert out == PRIOR_VOL
    assert math.isfinite(out)


# =========================================================================== #
# build_vol_series() — degenerate series
# =========================================================================== #
def test_build_vol_series_empty_returns_empty_dict():
    assert build_vol_series([]) == {}


def test_build_vol_series_single_bar_gets_prior():
    out = build_vol_series([bar(100.0, 1)])
    assert out == {1.0: PRIOR_VOL}


def test_build_vol_series_flat_series_all_zero_after_warmup():
    bars = [bar(100.0, i) for i in range(10)]
    out = build_vol_series(bars, lookback_days=1.0, bars_per_day=5.0)
    vals = list(out.values())
    # first couple bars are prior (n<2), the rest are 0 (flat)
    assert vals[0] == PRIOR_VOL
    assert all(v in (PRIOR_VOL, 0.0) for v in vals)
    assert vals[-1] == 0.0


def test_build_vol_series_keys_are_all_timestamps():
    bars = [bar(100.0 + (i % 3), i) for i in range(12)]
    out = build_vol_series(bars, lookback_days=1.0, bars_per_day=4.0)
    assert set(out.keys()) == {float(b["timestamp"]) for b in bars}
    assert len(out) == len(bars)


def test_build_vol_series_sliding_window_matches_direct_compute():
    # for a window covering the whole series, the last sliding value should equal
    # the direct non-annualized compute over the same closes.
    closes = [100.0, 101.0, 99.0, 102.0, 98.0, 103.0]
    bars = [bar(c, i) for i, c in enumerate(closes)]
    series = build_vol_series(bars, lookback_days=100.0, bars_per_day=1.0,
                              annualize=False)
    direct = compute_rolling_volatility(bars, lookback_days=100.0,
                                        bars_per_day=1.0, annualize=False)
    assert series[float(len(closes) - 1)] == pytest.approx(direct, abs=1e-12)


def test_build_vol_series_handles_nonpositive_prices_without_crash():
    bars = [bar(0.0, 0), bar(100.0, 1), bar(-5.0, 2), bar(101.0, 3)]
    out = build_vol_series(bars, lookback_days=100.0, bars_per_day=1.0)
    assert len(out) == 4
    # every value is a finite float (prior or computed); no exception
    assert all(isinstance(v, float) for v in out.values())
