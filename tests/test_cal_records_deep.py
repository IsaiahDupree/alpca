"""
Deep, deterministic tests for alpca/calibration/records.py.

Covers CalibrationRecord field storage, derived properties
(slippage_bps signed BUY/SELL, participation, quoted_half_spread_bps),
to_dict including derived fields, and CalibrationStore append/read_all
round-trip including back-compat (rows missing newer keys) and None
latency fields.

All offline, no network, no mocks. Pure logic only.
"""

from __future__ import annotations

import json
import math

import pytest

from alpca.calibration.records import CalibrationRecord, CalibrationStore


# --------------------------------------------------------------------------
# tiny self-contained helpers
# --------------------------------------------------------------------------
def make_rec(**overrides) -> CalibrationRecord:
    """A fully-populated record; override any field per test."""
    base = dict(
        symbol="AAPL",
        side="BUY",
        qty=100.0,
        intended_price=100.0,
        fill_price=100.5,
        bar_volume=10_000.0,
        signal_to_submit_ms=5.0,
        submit_to_ack_ms=10.0,
        ack_to_fill_ms=20.0,
        signal_to_fill_ms=35.0,
        ts=1_700_000_000.0,
        broker_order_id="ord-1",
        realized_vol=0.25,
        bid=99.99,
        ask=100.01,
        bid_size=300.0,
        ask_size=400.0,
        quote_ts=1_699_999_999.0,
    )
    base.update(overrides)
    return CalibrationRecord(**base)


def expected_slippage_bps(intended, fill, side):
    if intended is None or fill is None or intended <= 0:
        return None
    diff = fill - intended
    if side == "SELL":
        diff = -diff
    return (diff / intended) * 10_000.0


# --------------------------------------------------------------------------
# Field storage / defaults
# --------------------------------------------------------------------------
def test_minimal_construction_required_fields():
    r = CalibrationRecord(
        symbol="MSFT", side="BUY", qty=10.0,
        intended_price=200.0, fill_price=200.0,
    )
    assert r.symbol == "MSFT"
    assert r.side == "BUY"
    assert r.qty == 10.0
    assert r.intended_price == 200.0
    assert r.fill_price == 200.0


def test_optional_field_defaults_are_none_or_zero():
    r = CalibrationRecord(
        symbol="X", side="BUY", qty=1.0,
        intended_price=1.0, fill_price=1.0,
    )
    assert r.bar_volume is None
    assert r.signal_to_submit_ms is None
    assert r.submit_to_ack_ms is None
    assert r.ack_to_fill_ms is None
    assert r.signal_to_fill_ms is None
    assert r.ts == 0.0
    assert r.broker_order_id is None
    assert r.realized_vol is None
    assert r.bid is None
    assert r.ask is None
    assert r.bid_size is None
    assert r.ask_size is None
    assert r.quote_ts is None


def test_all_fields_round_trip_values():
    r = make_rec()
    assert r.bar_volume == 10_000.0
    assert r.signal_to_fill_ms == 35.0
    assert r.broker_order_id == "ord-1"
    assert r.realized_vol == 0.25
    assert r.bid == 99.99
    assert r.ask == 100.01
    assert r.bid_size == 300.0
    assert r.ask_size == 400.0
    assert r.quote_ts == 1_699_999_999.0


# --------------------------------------------------------------------------
# slippage_bps — signed BUY/SELL above/below across a param grid
# --------------------------------------------------------------------------
@pytest.mark.parametrize("intended", [50.0, 100.0, 250.5, 1000.0])
@pytest.mark.parametrize("fill_delta", [-2.0, -0.5, 0.0, 0.5, 3.0])
@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_slippage_bps_grid(intended, fill_delta, side):
    fill = intended + fill_delta
    r = make_rec(intended_price=intended, fill_price=fill, side=side)
    got = r.slippage_bps
    exp = expected_slippage_bps(intended, fill, side)
    assert got == pytest.approx(exp)


def test_slippage_buy_above_intended_is_positive():
    # paid more on a buy => worse => positive
    r = make_rec(side="BUY", intended_price=100.0, fill_price=101.0)
    assert r.slippage_bps == pytest.approx(100.0)  # 1% = 100 bps


def test_slippage_buy_below_intended_is_negative():
    # paid less on a buy => better => negative (price improvement)
    r = make_rec(side="BUY", intended_price=100.0, fill_price=99.0)
    assert r.slippage_bps == pytest.approx(-100.0)


def test_slippage_sell_below_intended_is_positive():
    # received less on a sell => worse => positive
    r = make_rec(side="SELL", intended_price=100.0, fill_price=99.0)
    assert r.slippage_bps == pytest.approx(100.0)


def test_slippage_sell_above_intended_is_negative():
    # received more on a sell => better => negative
    r = make_rec(side="SELL", intended_price=100.0, fill_price=101.0)
    assert r.slippage_bps == pytest.approx(-100.0)


def test_slippage_zero_when_fill_equals_intended():
    for side in ("BUY", "SELL"):
        r = make_rec(side=side, intended_price=100.0, fill_price=100.0)
        assert r.slippage_bps == 0.0


def test_slippage_buy_and_sell_are_exact_negatives_at_same_prices():
    buy = make_rec(side="BUY", intended_price=100.0, fill_price=100.7)
    sell = make_rec(side="SELL", intended_price=100.0, fill_price=100.7)
    assert buy.slippage_bps == pytest.approx(-sell.slippage_bps)


@pytest.mark.parametrize("intended", [0.0, -1.0, -100.0])
def test_slippage_none_when_intended_nonpositive(intended):
    r = make_rec(intended_price=intended, fill_price=10.0)
    assert r.slippage_bps is None


def test_slippage_unknown_side_treated_as_buy():
    # side that is neither SELL nor BUY: branch only flips on == "SELL",
    # so any other string behaves like a BUY (diff kept as-is).
    r = make_rec(side="HOLD", intended_price=100.0, fill_price=101.0)
    assert r.slippage_bps == pytest.approx(100.0)


def test_slippage_extreme_magnitude():
    r = make_rec(side="BUY", intended_price=1e-6, fill_price=2e-6)
    # diff/intended = 1.0 => 10000 bps
    assert r.slippage_bps == pytest.approx(10_000.0)


def test_slippage_with_inf_fill_is_inf():
    r = make_rec(side="BUY", intended_price=100.0, fill_price=math.inf)
    assert math.isinf(r.slippage_bps)


def test_slippage_with_nan_fill_is_nan():
    r = make_rec(side="BUY", intended_price=100.0, fill_price=math.nan)
    assert math.isnan(r.slippage_bps)


# --------------------------------------------------------------------------
# participation
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "qty,vol,expected",
    [
        (100.0, 10_000.0, 0.01),
        (50.0, 100.0, 0.5),
        (1.0, 1.0, 1.0),
        (250.0, 1000.0, 0.25),
        (5000.0, 1000.0, 5.0),  # >1 allowed (oversized vs bar)
    ],
)
def test_participation_grid(qty, vol, expected):
    r = make_rec(qty=qty, bar_volume=vol)
    assert r.participation == pytest.approx(expected)


@pytest.mark.parametrize("vol", [None, 0.0, -100.0])
def test_participation_none_when_volume_missing_or_nonpositive(vol):
    r = make_rec(qty=100.0, bar_volume=vol)
    assert r.participation is None


def test_participation_zero_qty():
    r = make_rec(qty=0.0, bar_volume=1000.0)
    assert r.participation == 0.0


# --------------------------------------------------------------------------
# quoted_half_spread_bps
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "bid,ask,expected",
    [
        (99.99, 100.01, ((0.02 / 2.0) / 100.0) * 10_000.0),
        (100.0, 100.2, ((0.2 / 2.0) / 100.1) * 10_000.0),
        (10.0, 10.5, ((0.5 / 2.0) / 10.25) * 10_000.0),
        (1.0, 3.0, ((2.0 / 2.0) / 2.0) * 10_000.0),
    ],
)
def test_quoted_half_spread_bps_grid(bid, ask, expected):
    r = make_rec(bid=bid, ask=ask)
    assert r.quoted_half_spread_bps == pytest.approx(expected)


@pytest.mark.parametrize(
    "bid,ask",
    [
        (None, 100.0),    # missing bid
        (100.0, None),    # missing ask
        (None, None),     # both missing
        (0.0, 100.0),     # falsy bid
        (100.0, 0.0),     # falsy ask
        (100.0, 100.0),   # ask == bid (not > bid)
        (100.0, 99.0),    # crossed / inverted: ask < bid
    ],
)
def test_quoted_half_spread_none_cases(bid, ask):
    r = make_rec(bid=bid, ask=ask)
    assert r.quoted_half_spread_bps is None


def test_quoted_half_spread_positive_and_finite():
    r = make_rec(bid=50.0, ask=50.5)
    v = r.quoted_half_spread_bps
    assert v is not None and v > 0 and math.isfinite(v)


def test_quoted_half_spread_symmetric_in_mid():
    # tight vs wide spread at same mid level => wider has bigger half-spread bps
    tight = make_rec(bid=99.95, ask=100.05).quoted_half_spread_bps
    wide = make_rec(bid=99.5, ask=100.5).quoted_half_spread_bps
    assert wide > tight


# --------------------------------------------------------------------------
# to_dict — includes derived fields + all dataclass fields
# --------------------------------------------------------------------------
def test_to_dict_contains_all_fields_and_derived():
    r = make_rec()
    d = r.to_dict()
    # dataclass fields
    for f in (
        "symbol", "side", "qty", "intended_price", "fill_price",
        "bar_volume", "signal_to_submit_ms", "submit_to_ack_ms",
        "ack_to_fill_ms", "signal_to_fill_ms", "ts", "broker_order_id",
        "realized_vol", "bid", "ask", "bid_size", "ask_size", "quote_ts",
    ):
        assert f in d
    # derived
    assert "slippage_bps" in d
    assert "participation" in d
    assert "quoted_half_spread_bps" in d


def test_to_dict_derived_values_match_properties():
    r = make_rec(side="SELL", intended_price=200.0, fill_price=199.0,
                 qty=300.0, bar_volume=6000.0, bid=199.9, ask=200.1)
    d = r.to_dict()
    assert d["slippage_bps"] == pytest.approx(r.slippage_bps)
    assert d["participation"] == pytest.approx(r.participation)
    assert d["quoted_half_spread_bps"] == pytest.approx(r.quoted_half_spread_bps)


def test_to_dict_derived_none_when_inputs_missing():
    r = CalibrationRecord(symbol="Z", side="BUY", qty=1.0,
                          intended_price=0.0, fill_price=1.0)
    d = r.to_dict()
    assert d["slippage_bps"] is None        # intended <= 0
    assert d["participation"] is None       # no bar_volume
    assert d["quoted_half_spread_bps"] is None  # no quote


def test_to_dict_is_json_serializable():
    r = make_rec()
    s = json.dumps(r.to_dict(), default=str)
    back = json.loads(s)
    assert back["symbol"] == "AAPL"
    assert back["slippage_bps"] == pytest.approx(r.slippage_bps)


# --------------------------------------------------------------------------
# CalibrationStore — append + read_all over MANY records
# --------------------------------------------------------------------------
def test_store_empty_read_when_no_file(tmp_path):
    store = CalibrationStore(path=str(tmp_path / "nope.jsonl"))
    assert store.read_all() == []
    assert store.count() == 0


def test_store_append_and_read_single(tmp_path):
    store = CalibrationStore(path=str(tmp_path / "f.jsonl"))
    r = make_rec()
    store.append(r)
    got = store.read_all()
    assert len(got) == 1
    assert got[0].symbol == r.symbol
    assert got[0].fill_price == r.fill_price
    assert got[0].bid == r.bid


def test_store_append_many_and_read_all_preserves_order(tmp_path):
    store = CalibrationStore(path=str(tmp_path / "many.jsonl"))
    n = 50
    for i in range(n):
        store.append(make_rec(
            symbol=f"SYM{i}",
            side="BUY" if i % 2 == 0 else "SELL",
            qty=float(i + 1),
            intended_price=100.0 + i,
            fill_price=100.0 + i + (0.1 if i % 2 == 0 else -0.1),
            ts=1_700_000_000.0 + i,
        ))
    got = store.read_all()
    assert len(got) == n
    assert store.count() == n
    for i, rec in enumerate(got):
        assert rec.symbol == f"SYM{i}"
        assert rec.qty == float(i + 1)
        assert rec.intended_price == 100.0 + i


def test_store_round_trip_preserves_derived_recomputation(tmp_path):
    store = CalibrationStore(path=str(tmp_path / "rt.jsonl"))
    r = make_rec(side="SELL", intended_price=300.0, fill_price=299.4,
                 qty=120.0, bar_volume=4800.0, bid=299.8, ask=300.2)
    store.append(r)
    loaded = store.read_all()[0]
    # derived props recompute identically after round-trip
    assert loaded.slippage_bps == pytest.approx(r.slippage_bps)
    assert loaded.participation == pytest.approx(r.participation)
    assert loaded.quoted_half_spread_bps == pytest.approx(r.quoted_half_spread_bps)


def test_store_append_is_additive_across_calls(tmp_path):
    store = CalibrationStore(path=str(tmp_path / "add.jsonl"))
    store.append(make_rec(symbol="A"))
    store.append(make_rec(symbol="B"))
    # new store instance, same path -> still sees both
    store2 = CalibrationStore(path=str(tmp_path / "add.jsonl"))
    syms = [r.symbol for r in store2.read_all()]
    assert syms == ["A", "B"]


def test_store_skips_blank_lines(tmp_path):
    p = tmp_path / "blanks.jsonl"
    store = CalibrationStore(path=str(p))
    store.append(make_rec(symbol="A"))
    # inject blank lines manually
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write("   \n")
    store.append(make_rec(symbol="B"))
    got = store.read_all()
    assert [r.symbol for r in got] == ["A", "B"]


def test_store_creates_directory(tmp_path):
    nested = tmp_path / "sub" / "dir" / "fills.jsonl"
    store = CalibrationStore(path=str(nested))
    store.append(make_rec())
    assert nested.exists()
    assert len(store.read_all()) == 1


# --------------------------------------------------------------------------
# Back-compat: rows missing the newer quote/vol keys
# --------------------------------------------------------------------------
def test_read_all_back_compat_missing_newer_keys(tmp_path):
    p = tmp_path / "old.jsonl"
    # old-style row: no realized_vol/bid/ask/bid_size/ask_size/quote_ts,
    # and no broker_order_id; also includes derived keys that must be dropped.
    old_row = {
        "symbol": "OLD",
        "side": "BUY",
        "qty": 10.0,
        "intended_price": 50.0,
        "fill_price": 50.25,
        "bar_volume": 2000.0,
        "signal_to_submit_ms": 4.0,
        "submit_to_ack_ms": 8.0,
        "ack_to_fill_ms": 12.0,
        "signal_to_fill_ms": 24.0,
        "ts": 1_650_000_000.0,
        # derived keys present in persisted output:
        "slippage_bps": 50.0,
        "participation": 0.005,
        "quoted_half_spread_bps": None,
    }
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(old_row) + "\n")
    store = CalibrationStore(path=str(p))
    got = store.read_all()
    assert len(got) == 1
    rec = got[0]
    assert rec.symbol == "OLD"
    # newer fields defaulted to None
    assert rec.realized_vol is None
    assert rec.bid is None
    assert rec.ask is None
    assert rec.quote_ts is None
    assert rec.broker_order_id is None
    # derived recompute from real fields
    assert rec.slippage_bps == pytest.approx(50.0)
    assert rec.participation == pytest.approx(0.005)
    assert rec.quoted_half_spread_bps is None


def test_read_all_strips_only_known_derived_keys(tmp_path):
    # A row with derived keys plus all real fields should load fine.
    p = tmp_path / "derived.jsonl"
    r = make_rec(symbol="DER")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(r.to_dict(), default=str) + "\n")
    store = CalibrationStore(path=str(p))
    got = store.read_all()
    assert len(got) == 1
    assert got[0].symbol == "DER"
    # bid/ask survived round-trip
    assert got[0].bid == pytest.approx(99.99)


def test_read_all_mixed_old_and_new_rows(tmp_path):
    p = tmp_path / "mixed.jsonl"
    new_rec = make_rec(symbol="NEW")
    old_row = {
        "symbol": "OLD", "side": "SELL", "qty": 5.0,
        "intended_price": 80.0, "fill_price": 79.5,
    }
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(new_rec.to_dict(), default=str) + "\n")
        fh.write(json.dumps(old_row) + "\n")
    store = CalibrationStore(path=str(p))
    got = store.read_all()
    assert [r.symbol for r in got] == ["NEW", "OLD"]
    assert got[0].bid == pytest.approx(99.99)
    assert got[1].bid is None
    # OLD is a SELL filled below intended => positive slippage
    assert got[1].slippage_bps == pytest.approx((0.5 / 80.0) * 10_000.0)


# --------------------------------------------------------------------------
# None latency fields persistence
# --------------------------------------------------------------------------
def test_none_latency_fields_round_trip(tmp_path):
    store = CalibrationStore(path=str(tmp_path / "lat.jsonl"))
    r = make_rec(
        signal_to_submit_ms=None,
        submit_to_ack_ms=None,
        ack_to_fill_ms=None,
        signal_to_fill_ms=None,
    )
    store.append(r)
    loaded = store.read_all()[0]
    assert loaded.signal_to_submit_ms is None
    assert loaded.submit_to_ack_ms is None
    assert loaded.ack_to_fill_ms is None
    assert loaded.signal_to_fill_ms is None
    # other fields intact
    assert loaded.symbol == "AAPL"
    assert loaded.slippage_bps == pytest.approx(r.slippage_bps)


def test_partial_none_latency_fields(tmp_path):
    store = CalibrationStore(path=str(tmp_path / "partial.jsonl"))
    r = make_rec(submit_to_ack_ms=None, ack_to_fill_ms=15.0)
    store.append(r)
    loaded = store.read_all()[0]
    assert loaded.submit_to_ack_ms is None
    assert loaded.ack_to_fill_ms == 15.0
    assert loaded.signal_to_submit_ms == 5.0


def test_none_optional_market_fields_round_trip(tmp_path):
    store = CalibrationStore(path=str(tmp_path / "mkt.jsonl"))
    r = make_rec(bar_volume=None, realized_vol=None, bid=None, ask=None,
                 bid_size=None, ask_size=None, quote_ts=None,
                 broker_order_id=None)
    store.append(r)
    loaded = store.read_all()[0]
    assert loaded.bar_volume is None
    assert loaded.participation is None
    assert loaded.quoted_half_spread_bps is None
    assert loaded.broker_order_id is None


# --------------------------------------------------------------------------
# Degenerate / robustness
# --------------------------------------------------------------------------
def test_negative_qty_participation_negative(tmp_path):
    r = make_rec(qty=-50.0, bar_volume=1000.0)
    assert r.participation == pytest.approx(-0.05)


def test_nan_intended_price_slippage_not_none_but_nan():
    # intended_price NaN: NaN <= 0 is False, so it does not short-circuit;
    # computation yields NaN.
    r = make_rec(intended_price=math.nan, fill_price=100.0)
    s = r.slippage_bps
    assert s is not None
    assert math.isnan(s)


def test_idempotent_read_all(tmp_path):
    store = CalibrationStore(path=str(tmp_path / "idem.jsonl"))
    for i in range(5):
        store.append(make_rec(symbol=f"S{i}"))
    first = [r.symbol for r in store.read_all()]
    second = [r.symbol for r in store.read_all()]
    assert first == second == ["S0", "S1", "S2", "S3", "S4"]


def test_count_matches_read_all_len(tmp_path):
    store = CalibrationStore(path=str(tmp_path / "cnt.jsonl"))
    for i in range(7):
        store.append(make_rec(symbol=f"C{i}"))
    assert store.count() == len(store.read_all()) == 7
