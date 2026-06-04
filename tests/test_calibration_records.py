"""
Phase 0: CalibrationRecord gained quote-at-signal + σ fields. They must round-trip
through the JSONL store, and OLD rows (without the new keys) must still load.
"""

import json

from alpca.calibration.records import CalibrationRecord, CalibrationStore


def test_new_fields_round_trip(tmp_path):
    store = CalibrationStore(str(tmp_path / "fills.jsonl"))
    store.append(CalibrationRecord(
        symbol="SPY", side="BUY", qty=2.0, intended_price=100.0, fill_price=100.05,
        bar_volume=1e6, realized_vol=0.18,
        bid=99.95, ask=100.05, bid_size=300, ask_size=120, quote_ts=1_700_000_000.0))
    rec = store.read_all()[0]
    assert rec.realized_vol == 0.18
    assert rec.bid == 99.95 and rec.ask == 100.05
    assert rec.bid_size == 300 and rec.ask_size == 120
    assert rec.quote_ts == 1_700_000_000.0


def test_quoted_half_spread_bps():
    rec = CalibrationRecord("SPY", "BUY", 1.0, 100.0, 100.0, bid=99.95, ask=100.05)
    # half spread = (0.10/2)/100 * 1e4 = 5 bps
    assert abs(rec.quoted_half_spread_bps - 5.0) < 1e-9
    assert CalibrationRecord("SPY", "BUY", 1.0, 100.0, 100.0).quoted_half_spread_bps is None


def test_old_rows_without_new_keys_still_load(tmp_path):
    path = tmp_path / "old.jsonl"
    # an old record as previously written — no realized_vol/bid/ask/quoted_half_spread
    path.write_text(json.dumps({
        "symbol": "SPY", "side": "SELL", "qty": 1.0, "intended_price": 100.0,
        "fill_price": 99.96, "bar_volume": 500000.0, "submit_to_ack_ms": 240.0,
        "ack_to_fill_ms": 60.0, "ts": 1.0, "slippage_bps": 4.0, "participation": 2e-6,
    }) + "\n")
    store = CalibrationStore(str(path))
    rec = store.read_all()[0]
    assert rec.symbol == "SPY" and rec.side == "SELL"
    assert rec.realized_vol is None and rec.bid is None      # defaults applied
    assert abs(rec.slippage_bps - 4.0) < 1e-6                # still computes
