"""
Tests for the pure calibration fitter — feed synthetic "real fills" whose true
parameters are known, and assert the fit recovers them.
"""

import math

from alpca.calibration.fit import calibrate
from alpca.calibration.records import CalibrationRecord, CalibrationStore


def _rec(side, qty, intended, fill, vol=None, sa=None, af=None):
    return CalibrationRecord(symbol="SPY", side=side, qty=qty, intended_price=intended,
                             fill_price=fill, bar_volume=vol,
                             submit_to_ack_ms=sa, ack_to_fill_ms=af)


# ----------------------------------------------------------------- record math
def test_record_slippage_sign():
    # BUY filled above intended -> positive (worse)
    buy = _rec("BUY", 1, 100.0, 100.10)   # +10 bps
    assert abs(buy.slippage_bps - 10.0) < 1e-6
    # SELL filled below intended -> positive (worse)
    sell = _rec("SELL", 1, 100.0, 99.90)  # +10 bps
    assert abs(sell.slippage_bps - 10.0) < 1e-6


def test_record_participation():
    r = _rec("BUY", 100, 100.0, 100.0, vol=10_000)
    assert abs(r.participation - 0.01) < 1e-12


# ----------------------------------------------------------------- half-spread
def test_recovers_half_spread_from_tiny_trades():
    # all tiny trades (negligible impact), true half-spread = 3 bps
    recs = []
    for i in range(10):
        side = "BUY" if i % 2 == 0 else "SELL"
        intended = 100.0
        # +3 bps adverse regardless of side
        fill = intended * (1 + 3e-4) if side == "BUY" else intended * (1 - 3e-4)
        recs.append(_rec(side, 1, intended, round(fill, 4), vol=1e7))
    res = calibrate(recs)
    assert abs(res.half_spread_bps - 3.0) < 0.5
    assert res.n_records == 10 and res.n_buys == 5 and res.n_sells == 5


# ----------------------------------------------------------------- impact fit
def test_recovers_impact_coefficient():
    # true model: slippage = 2 + 20*sqrt(participation), BUY side, varying size
    recs = []
    for qty in (10, 50, 100, 500, 1000, 5000):
        vol = 100_000
        part = qty / vol
        slip_bps = 2.0 + 20.0 * math.sqrt(part)
        fill = 100.0 * (1 + slip_bps / 10_000.0)
        recs.append(_rec("BUY", qty, 100.0, round(fill, 6), vol=vol))
    res = calibrate(recs)
    assert res.impact_fitted
    # recover coef ~20 and half-spread ~2 (loose tolerance — small sample)
    assert abs(res.impact_coef_bps - 20.0) < 5.0
    assert abs(res.half_spread_bps - 2.0) < 2.0


def test_impact_not_fitted_when_size_uniform():
    # all same tiny size -> participation doesn't vary -> impact left at prior
    recs = [_rec("BUY", 1, 100.0, 100.02, vol=1e7) for _ in range(8)]
    res = calibrate(recs)
    assert not res.impact_fitted
    assert any("narrow" in n or "prior" in n for n in res.notes)


# ----------------------------------------------------------------- latency
def test_latency_preset_from_records():
    recs = [_rec("BUY", 1, 100.0, 100.02, vol=1e7, sa=240.0, af=60.0) for _ in range(6)]
    res = calibrate(recs)
    assert res.latency is not None
    # submit+ack ~= 240, fill ~= 60
    assert abs((res.latency.submit_latency_ms + res.latency.ack_latency_ms) - 240.0) < 1.0
    assert abs(res.latency.fill_latency_ms - 60.0) < 1.0


def test_empty_records_returns_priors():
    res = calibrate([])
    assert res.n_records == 0
    assert res.half_spread_bps == 1.0 and res.impact_coef_bps == 8.0
    assert not res.impact_fitted


def test_low_confidence_flag_under_min_records():
    recs = [_rec("BUY", 1, 100.0, 100.02, vol=1e7) for _ in range(3)]
    res = calibrate(recs, min_records=6)
    assert any("low-confidence" in n for n in res.notes)


# ----------------------------------------------------------------- store
def test_store_roundtrip_and_fit(tmp_path):
    store = CalibrationStore(str(tmp_path / "fills.jsonl"))
    for i in range(5):
        store.append(_rec("BUY", 1, 100.0, 100.03, vol=1e7, sa=240.0, af=55.0))
    assert store.count() == 5
    recs = store.read_all()
    assert all(r.symbol == "SPY" for r in recs)
    res = calibrate(recs)
    assert res.n_records == 5
    # the fitted model is constructible
    fm = res.to_fill_model()
    assert fm.half_spread_bps >= 0


def test_result_save_and_to_fill_model(tmp_path):
    recs = [_rec("BUY", q, 100.0, 100.0 * (1 + (2 + 15 * math.sqrt(q / 1e5)) / 1e4),
                 vol=1e5) for q in (10, 100, 1000, 5000)]
    res = calibrate(recs)
    p = res.save(str(tmp_path / "calib.json"))
    import json
    d = json.load(open(p))
    assert "half_spread_bps" in d and "impact_coef_bps" in d
    fm = res.to_fill_model(participation_cap=0.2)
    assert fm.participation_cap == 0.2
