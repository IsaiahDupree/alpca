"""
Real, deterministic tests for alpca.calibration.fit — focused on the
CalibrationResult I/O surface:

  * to_dict() / save() JSON serialization (including a result built DIRECTLY,
    not via calibrate()), round-tripped through json.load from a tmp_path file;
  * LatencyPreset values flowing into to_fill_model() / SimAdapter latency;
  * slippage_p50_bps / slippage_p95_bps and the other scalar fields;
  * degenerate / edge inputs (None latency, empty notes, NaN/inf, negative,
    extreme magnitudes, idempotency).

Everything offline + pure: no network, no Alpaca, no mocks. calibrate() is used
only with hand-built CalibrationRecords (deterministic) to produce real results
that we then serialize.
"""

from __future__ import annotations

import json
import math
import os

import pytest

from alpca.calibration.fit import (
    CalibrationResult,
    LatencyPreset,
    calibrate,
    _percentile,
)
from alpca.calibration.records import CalibrationRecord
from alpca.execution.fills import FillModel
from alpca.execution.adapters.sim import SimAdapter


# --------------------------------------------------------------------------
# tiny self-contained helpers (do NOT import from other tests/ files)
# --------------------------------------------------------------------------

def make_preset(submit=10.0, ack=12.0, fill=30.0, jitter=4.0) -> LatencyPreset:
    return LatencyPreset(submit_latency_ms=submit, ack_latency_ms=ack,
                         fill_latency_ms=fill, jitter_ms=jitter)


def make_result(**over) -> CalibrationResult:
    """A fully-specified CalibrationResult built DIRECTLY (not via calibrate)."""
    base = dict(
        n_records=10, n_buys=6, n_sells=4,
        half_spread_bps=1.25, impact_coef_bps=7.5, impact_fitted=True,
        slippage_p50_bps=2.0, slippage_p95_bps=9.0,
        latency=make_preset(), notes=["built directly"],
    )
    base.update(over)
    return CalibrationResult(**base)


def make_records(n, *, side_cycle=("BUY", "SELL"), base_price=100.0,
                 slip_bps=3.0, bar_volume=10_000.0, qty=10.0,
                 with_latency=True, with_quote=False, vary_part=False):
    """Build n deterministic CalibrationRecords with a known per-record slippage.

    slippage_bps is determined by (fill_price - intended_price)/intended_price.
    For BUY: fill above intended -> positive slip. We solve fill_price so that
    each record's slippage_bps == slip_bps exactly.
    """
    out = []
    for i in range(n):
        side = side_cycle[i % len(side_cycle)]
        intended = base_price
        adj = slip_bps / 10_000.0
        if side == "BUY":
            fill = intended * (1 + adj)
        else:
            fill = intended * (1 - adj)
        q = qty
        bv = bar_volume
        if vary_part:
            # spread participation across a wide range to enable impact fit
            q = qty * (1 + i)
        rec = CalibrationRecord(
            symbol="AAPL", side=side, qty=q,
            intended_price=intended, fill_price=fill,
            bar_volume=bv,
            submit_to_ack_ms=200.0 if with_latency else None,
            ack_to_fill_ms=60.0 if with_latency else None,
            ts=float(1_700_000_000 + i),
        )
        if with_quote:
            rec.bid = base_price - 0.05
            rec.ask = base_price + 0.05
        out.append(rec)
    return out


# --------------------------------------------------------------------------
# _percentile invariants
# --------------------------------------------------------------------------

@pytest.mark.parametrize("vals,q,expected", [
    ([], 0.5, None),
    ([42.0], 0.5, 42.0),
    ([42.0], 0.0, 42.0),
    ([42.0], 1.0, 42.0),
    ([1.0, 2.0, 3.0], 0.0, 1.0),
    ([1.0, 2.0, 3.0], 1.0, 3.0),
    ([1.0, 2.0, 3.0], 0.5, 2.0),
    ([0.0, 10.0], 0.5, 5.0),
    ([0.0, 10.0], 0.25, 2.5),
    ([0.0, 4.0, 8.0, 12.0], 0.5, 6.0),
])
def test_percentile_exact(vals, q, expected):
    got = _percentile(sorted(vals), q)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)


def test_percentile_monotone_in_q():
    vals = sorted([3.0, 1.0, 9.0, 4.0, 7.0, 2.0])
    prev = _percentile(vals, 0.0)
    for q in (0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        cur = _percentile(vals, q)
        assert cur >= prev - 1e-12
        prev = cur


# --------------------------------------------------------------------------
# LatencyPreset.to_dict + into SimAdapter
# --------------------------------------------------------------------------

def test_latency_preset_to_dict_keys():
    p = make_preset(1.0, 2.0, 3.0, 4.0)
    d = p.to_dict()
    assert d == {
        "submit_latency_ms": 1.0,
        "ack_latency_ms": 2.0,
        "fill_latency_ms": 3.0,
        "jitter_ms": 4.0,
    }


@pytest.mark.parametrize("submit,ack,fill,jitter", [
    (0.0, 0.0, 0.0, 0.0),
    (5.0, 8.0, 20.0, 4.0),
    (120.0, 128.0, 60.0, 25.0),
    (1e6, 1e6, 1e6, 1e3),
])
def test_latency_preset_flows_into_sim_adapter(submit, ack, fill, jitter):
    p = make_preset(submit, ack, fill, jitter)
    d = p.to_dict()
    adapter = SimAdapter(
        submit_latency_ms=d["submit_latency_ms"],
        ack_latency_ms=d["ack_latency_ms"],
        fill_latency_ms=d["fill_latency_ms"],
        latency_jitter_ms=d["jitter_ms"],
        sleep=False, seed=7,
    )
    assert adapter.submit_latency_ms == submit
    assert adapter.ack_latency_ms == ack
    assert adapter.fill_latency_ms == fill
    assert adapter.jitter == jitter


def test_sim_adapter_draw_latency_deterministic_with_seed():
    p = make_preset(100.0, 100.0, 100.0, 10.0)
    a1 = SimAdapter(
        submit_latency_ms=p.submit_latency_ms,
        ack_latency_ms=p.ack_latency_ms,
        fill_latency_ms=p.fill_latency_ms,
        latency_jitter_ms=p.jitter_ms, sleep=False, seed=123)
    a2 = SimAdapter(
        submit_latency_ms=p.submit_latency_ms,
        ack_latency_ms=p.ack_latency_ms,
        fill_latency_ms=p.fill_latency_ms,
        latency_jitter_ms=p.jitter_ms, sleep=False, seed=123)
    draws1 = [a1._draw_latency(a1.submit_latency_ms) for _ in range(20)]
    draws2 = [a2._draw_latency(a2.submit_latency_ms) for _ in range(20)]
    assert draws1 == draws2
    # within base +/- jitter, and never negative
    for d in draws1:
        assert d >= 0.0
        assert 90.0 - 1e-9 <= d <= 110.0 + 1e-9


def test_sim_adapter_draw_latency_clamped_nonnegative():
    # base smaller than jitter -> some draws would go negative, must clamp to 0
    a = SimAdapter(submit_latency_ms=1.0, latency_jitter_ms=100.0,
                   sleep=False, seed=1)
    draws = [a._draw_latency(a.submit_latency_ms) for _ in range(200)]
    assert all(d >= 0.0 for d in draws)
    assert min(draws) == 0.0  # with this seed/range, clamping does engage


# --------------------------------------------------------------------------
# to_fill_model
# --------------------------------------------------------------------------

@pytest.mark.parametrize("hs,coef", [
    (1.0, 8.0),
    (0.0, 0.0),
    (2.5, 12.3),
    (1e4, 1e4),
])
def test_to_fill_model_passthrough(hs, coef):
    r = make_result(half_spread_bps=hs, impact_coef_bps=coef)
    fm = r.to_fill_model()
    assert isinstance(fm, FillModel)
    assert fm.half_spread_bps == pytest.approx(hs)
    assert fm.impact_coef_bps == pytest.approx(coef)
    assert fm.participation_cap == pytest.approx(0.10)
    assert fm.min_tick == pytest.approx(0.01)


@pytest.mark.parametrize("hs,coef", [
    (-1.0, -8.0),
    (-0.001, -100.0),
    (-1e9, -1e9),
])
def test_to_fill_model_clamps_negative_to_zero(hs, coef):
    r = make_result(half_spread_bps=hs, impact_coef_bps=coef)
    fm = r.to_fill_model()
    assert fm.half_spread_bps == 0.0
    assert fm.impact_coef_bps == 0.0


def test_to_fill_model_custom_kwargs():
    r = make_result(half_spread_bps=3.0, impact_coef_bps=4.0)
    fm = r.to_fill_model(participation_cap=0.5, min_tick=0.05)
    assert fm.participation_cap == pytest.approx(0.5)
    assert fm.min_tick == pytest.approx(0.05)
    assert fm.half_spread_bps == pytest.approx(3.0)


def test_to_fill_model_then_fill_uses_calibrated_spread():
    # a fill from the calibrated model must apply exactly the calibrated bps
    r = make_result(half_spread_bps=5.0, impact_coef_bps=0.0)
    fm = r.to_fill_model(min_tick=0.0)
    res = fm.fill(side_buy=True, ref_price=100.0, qty=1.0, bar_volume=None)
    assert res.slippage_bps == pytest.approx(5.0)
    assert res.price == pytest.approx(100.0 * (1 + 5.0 / 10_000.0))


# --------------------------------------------------------------------------
# to_dict on a DIRECTLY-built result
# --------------------------------------------------------------------------

def test_to_dict_direct_result_has_all_fields():
    r = make_result()
    d = r.to_dict()
    for k in ("n_records", "n_buys", "n_sells", "half_spread_bps",
              "impact_coef_bps", "impact_fitted", "slippage_p50_bps",
              "slippage_p95_bps", "latency", "notes", "half_spread_measured",
              "eta", "beta", "beta_fitted", "impact_coef_low_vol",
              "impact_coef_high_vol", "vol_threshold"):
        assert k in d, k


def test_to_dict_latency_is_nested_dict_not_dataclass():
    r = make_result(latency=make_preset(1.0, 2.0, 3.0, 4.0))
    d = r.to_dict()
    assert isinstance(d["latency"], dict)
    assert d["latency"]["submit_latency_ms"] == 1.0
    assert d["latency"]["jitter_ms"] == 4.0


def test_to_dict_none_latency():
    r = make_result(latency=None)
    d = r.to_dict()
    assert d["latency"] is None


def test_to_dict_preserves_optional_none_fields():
    r = make_result(slippage_p50_bps=None, slippage_p95_bps=None,
                    eta=None, beta=None, vol_threshold=None,
                    impact_coef_low_vol=None, impact_coef_high_vol=None)
    d = r.to_dict()
    assert d["slippage_p50_bps"] is None
    assert d["slippage_p95_bps"] is None
    assert d["eta"] is None
    assert d["beta"] is None
    assert d["vol_threshold"] is None


def test_to_dict_phase1_extras_roundtrip():
    r = make_result(eta=0.5, beta=0.42, beta_fitted=True,
                    impact_coef_low_vol=3.0, impact_coef_high_vol=9.0,
                    vol_threshold=0.25, half_spread_measured=True)
    d = r.to_dict()
    assert d["eta"] == 0.5
    assert d["beta"] == 0.42
    assert d["beta_fitted"] is True
    assert d["impact_coef_low_vol"] == 3.0
    assert d["impact_coef_high_vol"] == 9.0
    assert d["vol_threshold"] == 0.25
    assert d["half_spread_measured"] is True


# --------------------------------------------------------------------------
# save() -> JSON file -> json.load round-trip (to a tmp_path)
# --------------------------------------------------------------------------

def _roundtrip(result: CalibrationResult, tmp_path, name="cal.json"):
    path = os.path.join(str(tmp_path), name)
    returned = result.save(path)
    assert returned == path
    assert os.path.exists(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh), path


def test_save_returns_path_and_writes_file(tmp_path):
    r = make_result()
    loaded, path = _roundtrip(r, tmp_path)
    assert isinstance(loaded, dict)
    assert path.endswith("cal.json")


def test_save_roundtrip_scalar_fields(tmp_path):
    r = make_result(n_records=33, n_buys=20, n_sells=13,
                    half_spread_bps=1.111, impact_coef_bps=6.789,
                    slippage_p50_bps=2.5, slippage_p95_bps=11.0)
    loaded, _ = _roundtrip(r, tmp_path)
    assert loaded["n_records"] == 33
    assert loaded["n_buys"] == 20
    assert loaded["n_sells"] == 13
    assert loaded["half_spread_bps"] == pytest.approx(1.111)
    assert loaded["impact_coef_bps"] == pytest.approx(6.789)
    assert loaded["slippage_p50_bps"] == pytest.approx(2.5)
    assert loaded["slippage_p95_bps"] == pytest.approx(11.0)


def test_save_roundtrip_latency_nested(tmp_path):
    r = make_result(latency=make_preset(7.0, 8.0, 9.0, 1.5))
    loaded, _ = _roundtrip(r, tmp_path)
    lat = loaded["latency"]
    assert lat["submit_latency_ms"] == 7.0
    assert lat["ack_latency_ms"] == 8.0
    assert lat["fill_latency_ms"] == 9.0
    assert lat["jitter_ms"] == 1.5


def test_save_roundtrip_none_latency(tmp_path):
    r = make_result(latency=None)
    loaded, _ = _roundtrip(r, tmp_path)
    assert loaded["latency"] is None


def test_save_roundtrip_notes_list(tmp_path):
    notes = ["a", "b", "impact fitted via sqrt-participation OLS (intercept 1.23 bps)"]
    r = make_result(notes=list(notes))
    loaded, _ = _roundtrip(r, tmp_path)
    assert loaded["notes"] == notes


def test_save_roundtrip_empty_notes(tmp_path):
    r = make_result(notes=[])
    loaded, _ = _roundtrip(r, tmp_path)
    assert loaded["notes"] == []


def test_save_reconstruct_latency_preset_and_feed_adapter(tmp_path):
    # round-trip the latency, rebuild a LatencyPreset from the loaded JSON, and
    # confirm it still drives a SimAdapter identically.
    r = make_result(latency=make_preset(120.0, 128.0, 60.0, 25.0))
    loaded, _ = _roundtrip(r, tmp_path)
    p2 = LatencyPreset(**loaded["latency"])
    assert p2 == make_preset(120.0, 128.0, 60.0, 25.0)
    a = SimAdapter(submit_latency_ms=p2.submit_latency_ms,
                   ack_latency_ms=p2.ack_latency_ms,
                   fill_latency_ms=p2.fill_latency_ms,
                   latency_jitter_ms=p2.jitter_ms, sleep=False, seed=0)
    assert a.submit_latency_ms == 120.0
    assert a.fill_latency_ms == 60.0


def test_save_to_dict_equals_loaded_for_jsonable_result(tmp_path):
    # When every field is JSON-native (no NaN/inf), to_dict() must equal the
    # reloaded dict exactly (default=str only triggers on non-serializable).
    r = make_result(
        slippage_p50_bps=2.0, slippage_p95_bps=9.0,
        eta=0.5, beta=0.42, beta_fitted=True,
        impact_coef_low_vol=3.0, impact_coef_high_vol=9.0,
        vol_threshold=0.25, half_spread_measured=True,
        latency=make_preset(7.0, 8.0, 9.0, 1.5), notes=["x"],
    )
    loaded, _ = _roundtrip(r, tmp_path)
    assert loaded == r.to_dict()


def test_save_creates_parent_dir(tmp_path):
    nested = os.path.join(str(tmp_path), "deep", "sub", "calib.json")
    r = make_result()
    r.save(nested)
    assert os.path.exists(nested)


def test_save_overwrite_idempotent(tmp_path):
    r = make_result(half_spread_bps=1.0)
    path = os.path.join(str(tmp_path), "cal.json")
    r.save(path)
    with open(path) as fh:
        first = json.load(fh)
    # save again -> identical content
    r.save(path)
    with open(path) as fh:
        second = json.load(fh)
    assert first == second


def test_save_overwrite_reflects_new_values(tmp_path):
    path = os.path.join(str(tmp_path), "cal.json")
    make_result(half_spread_bps=1.0).save(path)
    make_result(half_spread_bps=99.0).save(path)
    with open(path) as fh:
        loaded = json.load(fh)
    assert loaded["half_spread_bps"] == 99.0


# --------------------------------------------------------------------------
# NaN / inf serialization behavior (json.dump emits NaN/Infinity tokens;
# Python's json.load accepts them back). Assert ACTUAL current behavior.
# --------------------------------------------------------------------------

def test_save_roundtrip_nan_half_spread(tmp_path):
    r = make_result(half_spread_bps=float("nan"))
    loaded, _ = _roundtrip(r, tmp_path)
    assert math.isnan(loaded["half_spread_bps"])


def test_save_roundtrip_inf_fields(tmp_path):
    r = make_result(half_spread_bps=float("inf"),
                    impact_coef_bps=float("-inf"))
    loaded, _ = _roundtrip(r, tmp_path)
    assert math.isinf(loaded["half_spread_bps"]) and loaded["half_spread_bps"] > 0
    assert math.isinf(loaded["impact_coef_bps"]) and loaded["impact_coef_bps"] < 0


def test_save_roundtrip_extreme_and_negative(tmp_path):
    r = make_result(half_spread_bps=-12345.678, impact_coef_bps=1e12,
                    n_records=0, n_buys=0, n_sells=0)
    loaded, _ = _roundtrip(r, tmp_path)
    assert loaded["half_spread_bps"] == pytest.approx(-12345.678)
    assert loaded["impact_coef_bps"] == pytest.approx(1e12)
    assert loaded["n_records"] == 0


# --------------------------------------------------------------------------
# calibrate() end-to-end -> save -> reload (the result is NOT hand-built here)
# --------------------------------------------------------------------------

def test_calibrate_empty_returns_prior_and_serializes(tmp_path):
    r = calibrate([])
    assert r.n_records == 0
    assert r.half_spread_bps == pytest.approx(1.0)
    assert r.impact_coef_bps == pytest.approx(8.0)
    assert r.impact_fitted is False
    assert r.slippage_p50_bps is None
    assert r.slippage_p95_bps is None
    assert r.latency is None
    loaded, _ = _roundtrip(r, tmp_path, "empty.json")
    assert loaded["n_records"] == 0
    assert loaded["latency"] is None
    assert "no valid records" in " ".join(loaded["notes"])


def test_calibrate_uniform_slippage_p50_p95(tmp_path):
    # All records have slippage exactly 3.0 bps -> p50 == p95 == 3.0
    recs = make_records(8, slip_bps=3.0, with_latency=True)
    r = calibrate(recs)
    assert r.n_records == 8
    assert r.n_buys == 4
    assert r.n_sells == 4
    assert r.slippage_p50_bps == pytest.approx(3.0, abs=1e-3)
    assert r.slippage_p95_bps == pytest.approx(3.0, abs=1e-3)
    # latency captured -> preset present, submit/ack are half of p50(submit->ack)
    assert r.latency is not None
    assert r.latency.submit_latency_ms == pytest.approx(100.0, abs=0.1)
    assert r.latency.ack_latency_ms == pytest.approx(100.0, abs=0.1)
    assert r.latency.fill_latency_ms == pytest.approx(60.0, abs=0.1)
    loaded, _ = _roundtrip(r, tmp_path, "uniform.json")
    assert loaded["slippage_p50_bps"] == pytest.approx(3.0, abs=1e-3)
    assert loaded["latency"]["fill_latency_ms"] == pytest.approx(60.0, abs=0.1)


def test_calibrate_p95_exceeds_p50_when_slippage_varies(tmp_path):
    # Build records with a spread of slippage values; p95 must be >= p50.
    recs = []
    for i, s in enumerate([1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 20.0, 20.0]):
        side = "BUY"
        recs.append(CalibrationRecord(
            symbol="X", side=side, qty=10.0,
            intended_price=100.0, fill_price=100.0 * (1 + s / 10_000.0),
            bar_volume=10_000.0, ts=float(i)))
    r = calibrate(recs)
    assert r.slippage_p95_bps >= r.slippage_p50_bps
    assert r.slippage_p50_bps == pytest.approx(3.5, abs=0.5)  # median of {1,1,2,2,5,5,20,20}=3.5
    loaded, _ = _roundtrip(r, tmp_path, "vary.json")
    assert loaded["slippage_p95_bps"] >= loaded["slippage_p50_bps"]


def test_calibrate_no_latency_records_have_none_preset(tmp_path):
    recs = make_records(6, with_latency=False)
    r = calibrate(recs)
    assert r.latency is None
    loaded, _ = _roundtrip(r, tmp_path, "nolat.json")
    assert loaded["latency"] is None
    assert any("no latency" in n for n in loaded["notes"])


def test_calibrate_low_record_count_flagged(tmp_path):
    recs = make_records(2, with_latency=True)
    r = calibrate(recs, min_records=6)
    assert any("low-confidence" in n for n in r.notes)
    loaded, _ = _roundtrip(r, tmp_path, "low.json")
    assert any("low-confidence" in n for n in loaded["notes"])


def test_calibrate_impact_fitted_with_varied_participation(tmp_path):
    # vary qty (hence participation) widely so the sqrt-OLS impact fit engages.
    recs = []
    for i in range(8):
        q = 10.0 * (i + 1)  # 10..80 -> participation 0.001..0.008 (8x spread)
        recs.append(CalibrationRecord(
            symbol="X", side="BUY", qty=q,
            intended_price=100.0,
            fill_price=100.0 * (1 + (2.0 + 0.5 * i) / 10_000.0),
            bar_volume=10_000.0, ts=float(i)))
    r = calibrate(recs)
    assert r.impact_fitted is True
    assert r.impact_coef_bps >= 0.0
    assert any("impact fitted" in n for n in r.notes)
    loaded, _ = _roundtrip(r, tmp_path, "impact.json")
    assert loaded["impact_fitted"] is True


def test_calibrate_narrow_participation_keeps_prior(tmp_path):
    # all identical participation -> impact NOT fitted, prior coef kept.
    recs = make_records(8, slip_bps=3.0, qty=10.0, bar_volume=10_000.0,
                        side_cycle=("BUY",), vary_part=False)
    r = calibrate(recs)
    assert r.impact_fitted is False
    assert r.impact_coef_bps == pytest.approx(8.0)
    assert any("too narrow" in n for n in r.notes)


def test_calibrate_quoted_half_spread_measured(tmp_path):
    # >=4 records carrying a real NBBO -> half_spread MEASURED, flag set true.
    recs = make_records(6, with_quote=True, slip_bps=3.0)
    r = calibrate(recs)
    assert r.half_spread_measured is True
    # bid=99.95 ask=100.05 mid=100 -> half spread = 0.05/100 *1e4 = 5 bps
    assert r.half_spread_bps == pytest.approx(5.0, abs=1e-6)
    loaded, _ = _roundtrip(r, tmp_path, "quoted.json")
    assert loaded["half_spread_measured"] is True
    assert loaded["half_spread_bps"] == pytest.approx(5.0, abs=1e-6)


def test_calibrate_drops_records_with_no_slippage(tmp_path):
    # intended_price <= 0 -> slippage_bps None -> dropped from valid set.
    good = make_records(4, slip_bps=3.0)
    bad = [CalibrationRecord(symbol="X", side="BUY", qty=1.0,
                             intended_price=0.0, fill_price=100.0, ts=0.0)]
    r = calibrate(good + bad)
    assert r.n_records == 4  # the bad one is excluded


def test_calibrate_result_is_deterministic():
    recs = make_records(8, slip_bps=4.0, with_latency=True)
    r1 = calibrate(recs)
    r2 = calibrate(recs)
    assert r1.to_dict() == r2.to_dict()


def test_calibrate_buys_sells_partition():
    recs = make_records(10, side_cycle=("BUY", "BUY", "SELL"))
    r = calibrate(recs)
    assert r.n_buys + r.n_sells == r.n_records
    assert r.n_records == 10


# --------------------------------------------------------------------------
# round-trip a calibrate() result fully (with phase-1 extras possibly set)
# --------------------------------------------------------------------------

def test_calibrate_full_roundtrip_equals_to_dict(tmp_path):
    recs = make_records(8, slip_bps=3.0, with_latency=True)
    r = calibrate(recs)
    loaded, _ = _roundtrip(r, tmp_path, "full.json")
    # no NaN/inf in a normal calibrate result -> exact equality
    assert loaded == r.to_dict()
