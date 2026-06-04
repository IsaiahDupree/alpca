"""
Deep, deterministic tests for alpca/execution/order.py.

Covers: Side/OrderType/TimeInForce/OrderStatus enums, OrderStatus.is_terminal,
new_client_order_id formatting/length/prefix, Fill dataclass, mark_signal/
mark_submit/mark_ack/add_fill lifecycle helpers, per-stage latency computations
(ms), avg_fill_price/filled_qty accumulation, slippage_bps, notional, and
to_dict serialization. No network, no mocks. Timestamps are injected directly
to keep everything deterministic (no wall-clock reliance for value assertions).
"""

from __future__ import annotations

import math
import re

import pytest

from alpca.execution.order import (
    Fill,
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
    new_client_order_id,
)


# --------------------------------------------------------------------------- #
# tiny local helpers (self-contained — no imports from other tests/ files)
# --------------------------------------------------------------------------- #
def make_order(**kw) -> Order:
    """Build an Order with sensible defaults overridable by kwargs."""
    params = dict(symbol="AAPL", side=Side.BUY, qty=10.0)
    params.update(kw)
    return params and Order(**params)


def set_stage_timestamps(o: Order, signal=None, submit=None, ack=None, fill=None) -> Order:
    """Inject lifecycle timestamps directly for deterministic latency math."""
    if signal is not None:
        o.signal_ts = signal
    if submit is not None:
        o.submit_ts = submit
    if ack is not None:
        o.ack_ts = ack
    if fill is not None:
        o.fill_ts = fill
    return o


# --------------------------------------------------------------------------- #
# Enum value tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "member,value",
    [
        (Side.BUY, "BUY"),
        (Side.SELL, "SELL"),
    ],
)
def test_side_values(member, value):
    assert member.value == value
    # str-Enum: member compares/usable as its string value
    assert member == value
    assert isinstance(member, str)


@pytest.mark.parametrize(
    "member,value",
    [
        (OrderType.MARKET, "MARKET"),
        (OrderType.LIMIT, "LIMIT"),
        (OrderType.STOP, "STOP"),
        (OrderType.STOP_LIMIT, "STOP_LIMIT"),
    ],
)
def test_order_type_values(member, value):
    assert member.value == value
    assert OrderType(value) is member


@pytest.mark.parametrize(
    "member,value",
    [
        (TimeInForce.DAY, "DAY"),
        (TimeInForce.GTC, "GTC"),
        (TimeInForce.IOC, "IOC"),
        (TimeInForce.FOK, "FOK"),
    ],
)
def test_tif_values(member, value):
    assert member.value == value
    assert TimeInForce(value) is member


@pytest.mark.parametrize(
    "member,value",
    [
        (OrderStatus.NEW, "NEW"),
        (OrderStatus.SUBMITTED, "SUBMITTED"),
        (OrderStatus.ACCEPTED, "ACCEPTED"),
        (OrderStatus.PARTIALLY_FILLED, "PARTIALLY_FILLED"),
        (OrderStatus.FILLED, "FILLED"),
        (OrderStatus.CANCELED, "CANCELED"),
        (OrderStatus.REJECTED, "REJECTED"),
        (OrderStatus.EXPIRED, "EXPIRED"),
    ],
)
def test_order_status_values(member, value):
    assert member.value == value
    assert OrderStatus(value) is member


# --------------------------------------------------------------------------- #
# OrderStatus.is_terminal — every status, explicitly
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "status,expected",
    [
        (OrderStatus.NEW, False),
        (OrderStatus.SUBMITTED, False),
        (OrderStatus.ACCEPTED, False),
        (OrderStatus.PARTIALLY_FILLED, False),
        (OrderStatus.FILLED, True),
        (OrderStatus.CANCELED, True),
        (OrderStatus.REJECTED, True),
        (OrderStatus.EXPIRED, True),
    ],
)
def test_is_terminal(status, expected):
    assert status.is_terminal is expected


def test_is_terminal_partition_consistency():
    terminal = {s for s in OrderStatus if s.is_terminal}
    non_terminal = {s for s in OrderStatus if not s.is_terminal}
    # the two sets fully partition the enum, no overlap
    assert terminal | non_terminal == set(OrderStatus)
    assert terminal & non_terminal == set()
    assert terminal == {
        OrderStatus.FILLED,
        OrderStatus.CANCELED,
        OrderStatus.REJECTED,
        OrderStatus.EXPIRED,
    }


# --------------------------------------------------------------------------- #
# new_client_order_id
# --------------------------------------------------------------------------- #
def test_client_order_id_default_form():
    cid = new_client_order_id()
    # form: a-{uuid8}  (no strategy, no seq)
    assert cid.startswith("a-")
    assert len(cid) <= 48
    m = re.fullmatch(r"a-[0-9a-f]{8}", cid)
    assert m is not None, cid


def test_client_order_id_with_strategy_prefix():
    cid = new_client_order_id(strategy="momentum")
    assert cid.startswith("a-momentum-")
    assert len(cid) <= 48
    # tail is 8 hex chars
    assert re.fullmatch(r"a-momentum-[0-9a-f]{8}", cid) is not None


def test_client_order_id_with_strategy_and_seq():
    cid = new_client_order_id(strategy="rsi", seq=42)
    assert cid.startswith("a-rsi-42-")
    assert re.fullmatch(r"a-rsi-42-[0-9a-f]{8}", cid) is not None
    assert len(cid) <= 48


def test_client_order_id_strips_spaces_in_strategy():
    cid = new_client_order_id(strategy="gap fade mr")
    # spaces removed before truncation
    assert cid.startswith("a-gapfademr-")
    assert " " not in cid


def test_client_order_id_strategy_truncated_to_16():
    long_name = "abcdefghijklmnopqrstuvwxyz"  # 26 chars
    cid = new_client_order_id(strategy=long_name)
    # strategy segment truncated to first 16 chars
    assert cid.startswith("a-" + long_name[:16] + "-")
    assert "-" + long_name[:17] not in cid


@pytest.mark.parametrize("seq", [0, 1, 7, 999999])
def test_client_order_id_seq_zero_is_included(seq):
    # seq=0 is not None, so it must appear (guards against falsy-int bug)
    cid = new_client_order_id(strategy="s", seq=seq)
    assert cid.startswith(f"a-s-{seq}-")


def test_client_order_id_capped_at_48_with_extreme_inputs():
    cid = new_client_order_id(strategy="x" * 200, seq=10 ** 30)
    assert len(cid) <= 48


def test_client_order_id_uniqueness_across_calls():
    ids = {new_client_order_id(strategy="m", seq=1) for _ in range(200)}
    # uuid8 tail makes collisions astronomically unlikely
    assert len(ids) == 200


def test_order_default_client_order_id_is_generated():
    o = Order(symbol="AAPL", side=Side.BUY, qty=1.0)
    assert o.client_order_id.startswith("a-")
    assert len(o.client_order_id) <= 48
    o2 = Order(symbol="AAPL", side=Side.BUY, qty=1.0)
    assert o.client_order_id != o2.client_order_id


# --------------------------------------------------------------------------- #
# Fill dataclass
# --------------------------------------------------------------------------- #
def test_fill_defaults_fee_zero():
    f = Fill(ts=100.0, price=50.0, qty=3.0)
    assert f.fee == 0.0
    assert (f.ts, f.price, f.qty) == (100.0, 50.0, 3.0)


def test_fill_explicit_fee():
    f = Fill(ts=1.0, price=2.0, qty=4.0, fee=0.25)
    assert f.fee == 0.25


def test_fill_equality():
    a = Fill(ts=1.0, price=2.0, qty=3.0)
    b = Fill(ts=1.0, price=2.0, qty=3.0)
    assert a == b


# --------------------------------------------------------------------------- #
# Order defaults
# --------------------------------------------------------------------------- #
def test_order_defaults():
    o = Order(symbol="MSFT", side=Side.SELL, qty=5.0)
    assert o.order_type is OrderType.MARKET
    assert o.tif is TimeInForce.DAY
    assert o.extended_hours is False
    assert o.status is OrderStatus.NEW
    assert o.filled_qty == 0.0
    assert o.avg_fill_price is None
    assert o.fills == []
    assert o.limit_price is None
    assert o.stop_price is None
    assert o.signal_ts is None and o.submit_ts is None
    assert o.ack_ts is None and o.fill_ts is None
    assert o.intended_price is None
    assert o.metadata == {}


# --------------------------------------------------------------------------- #
# Lifecycle state transitions via mark_* helpers
# --------------------------------------------------------------------------- #
def test_mark_submit_transitions_new_to_submitted():
    o = make_order()
    assert o.status is OrderStatus.NEW
    ret = o.mark_submit()
    assert ret is o  # returns self for chaining
    assert o.status is OrderStatus.SUBMITTED
    assert o.submit_ts is not None


def test_mark_submit_does_not_override_terminal_status():
    o = make_order()
    o.status = OrderStatus.REJECTED
    o.mark_submit()
    # only flips NEW -> SUBMITTED; REJECTED is preserved
    assert o.status is OrderStatus.REJECTED
    assert o.submit_ts is not None


def test_mark_ack_from_new_and_submitted():
    o1 = make_order()
    o1.mark_ack()
    assert o1.status is OrderStatus.ACCEPTED  # NEW -> ACCEPTED

    o2 = make_order().mark_submit()
    o2.mark_ack()
    assert o2.status is OrderStatus.ACCEPTED  # SUBMITTED -> ACCEPTED


def test_mark_ack_preserves_filled_status():
    o = make_order()
    o.status = OrderStatus.FILLED
    o.mark_ack()
    assert o.status is OrderStatus.FILLED
    assert o.ack_ts is not None


def test_mark_signal_sets_intended_price():
    o = make_order()
    ret = o.mark_signal(intended_price=123.45)
    assert ret is o
    assert o.signal_ts is not None
    assert o.intended_price == 123.45


def test_mark_signal_without_price_keeps_intended_none():
    o = make_order()
    o.mark_signal()
    assert o.signal_ts is not None
    assert o.intended_price is None


def test_mark_signal_does_not_overwrite_with_none():
    o = make_order()
    o.mark_signal(intended_price=10.0)
    first = o.signal_ts
    o.mark_signal()  # intended_price None -> keep existing
    assert o.intended_price == 10.0
    assert o.signal_ts is not None
    # signal_ts may update; just ensure still set
    assert o.signal_ts >= first


def test_full_lifecycle_chaining_monotonic_timestamps():
    o = make_order(qty=10.0)
    o.mark_signal(intended_price=100.0).mark_submit().mark_ack()
    o.add_fill(Fill(ts=o.ack_ts + 0.01, price=100.0, qty=10.0))
    # timestamps recorded by real time.time() should be non-decreasing in order
    assert o.signal_ts <= o.submit_ts <= o.ack_ts <= o.fill_ts
    assert o.status is OrderStatus.FILLED


# --------------------------------------------------------------------------- #
# add_fill: filled_qty / avg_fill_price / status accumulation
# --------------------------------------------------------------------------- #
def test_add_fill_full_single():
    o = make_order(qty=10.0)
    o.add_fill(Fill(ts=5.0, price=99.0, qty=10.0))
    assert o.filled_qty == 10.0
    assert o.avg_fill_price == 99.0
    assert o.status is OrderStatus.FILLED
    assert o.fill_ts == 5.0


def test_add_fill_partial_then_complete():
    o = make_order(qty=10.0)
    o.add_fill(Fill(ts=1.0, price=100.0, qty=4.0))
    assert o.status is OrderStatus.PARTIALLY_FILLED
    assert o.filled_qty == 4.0
    assert o.avg_fill_price == 100.0
    assert o.fill_ts == 1.0

    o.add_fill(Fill(ts=2.0, price=110.0, qty=6.0))
    assert o.status is OrderStatus.FILLED
    assert o.filled_qty == 10.0
    # vwap = (100*4 + 110*6) / 10 = (400 + 660)/10 = 106.0
    assert o.avg_fill_price == pytest.approx(106.0)
    assert o.fill_ts == 2.0


def test_add_fill_vwap_weighted():
    o = make_order(qty=3.0)
    o.add_fill(Fill(ts=1.0, price=10.0, qty=1.0))
    o.add_fill(Fill(ts=2.0, price=20.0, qty=2.0))
    # (10*1 + 20*2)/3 = 50/3
    assert o.avg_fill_price == pytest.approx(50.0 / 3.0)
    assert o.filled_qty == 3.0
    assert o.status is OrderStatus.FILLED


def test_add_fill_epsilon_tolerance_fills():
    # total within 1e-9 of qty counts as FILLED
    o = make_order(qty=1.0)
    o.add_fill(Fill(ts=1.0, price=50.0, qty=1.0 - 5e-10))
    assert o.status is OrderStatus.FILLED


def test_add_fill_just_under_tolerance_is_partial():
    o = make_order(qty=1.0)
    o.add_fill(Fill(ts=1.0, price=50.0, qty=0.5))
    assert o.status is OrderStatus.PARTIALLY_FILLED


def test_add_fill_overfill_is_filled():
    o = make_order(qty=5.0)
    o.add_fill(Fill(ts=1.0, price=50.0, qty=7.0))
    assert o.filled_qty == 7.0
    assert o.status is OrderStatus.FILLED


def test_add_fill_returns_self_for_chaining():
    o = make_order(qty=2.0)
    ret = o.add_fill(Fill(ts=1.0, price=1.0, qty=1.0))
    assert ret is o


# --------------------------------------------------------------------------- #
# Latency computations — deterministic injected timestamps
# --------------------------------------------------------------------------- #
def test_latencies_all_stages_exact():
    o = make_order()
    # 0.0 -> 0.010s -> 0.025s -> 0.100s
    set_stage_timestamps(o, signal=0.0, submit=0.010, ack=0.025, fill=0.100)
    assert o.signal_to_submit_ms == pytest.approx(10.0)
    assert o.submit_to_ack_ms == pytest.approx(15.0)
    assert o.ack_to_fill_ms == pytest.approx(75.0)
    assert o.submit_to_fill_ms == pytest.approx(90.0)
    assert o.signal_to_fill_ms == pytest.approx(100.0)


@pytest.mark.parametrize(
    "prop",
    [
        "signal_to_submit_ms",
        "submit_to_ack_ms",
        "ack_to_fill_ms",
        "submit_to_fill_ms",
        "signal_to_fill_ms",
    ],
)
def test_latency_none_when_timestamps_missing(prop):
    o = make_order()  # all timestamps None
    assert getattr(o, prop) is None


def test_latency_partial_missing_endpoint():
    o = make_order()
    set_stage_timestamps(o, signal=0.0, submit=0.01)  # ack/fill missing
    assert o.signal_to_submit_ms == pytest.approx(10.0)
    assert o.submit_to_ack_ms is None
    assert o.ack_to_fill_ms is None
    assert o.signal_to_fill_ms is None


def test_latency_negative_out_of_order_timestamps():
    # out-of-order (clock skew / retro fill): computed, not clamped
    o = make_order()
    set_stage_timestamps(o, signal=1.0, submit=0.5)
    assert o.signal_to_submit_ms == pytest.approx(-500.0)


def test_latency_zero_duration():
    o = make_order()
    set_stage_timestamps(o, signal=2.0, submit=2.0, ack=2.0, fill=2.0)
    assert o.signal_to_submit_ms == 0.0
    assert o.submit_to_ack_ms == 0.0
    assert o.ack_to_fill_ms == 0.0
    assert o.signal_to_fill_ms == 0.0


def test_latency_large_magnitude():
    o = make_order()
    set_stage_timestamps(o, signal=0.0, fill=3600.0)  # one hour
    assert o.signal_to_fill_ms == pytest.approx(3_600_000.0)


def test_ms_static_method_directly():
    assert Order._ms(None, 5.0) is None
    assert Order._ms(5.0, None) is None
    assert Order._ms(None, None) is None
    assert Order._ms(1.0, 1.5) == pytest.approx(500.0)


# --------------------------------------------------------------------------- #
# slippage_bps
# --------------------------------------------------------------------------- #
def test_slippage_buy_worse_is_positive():
    o = make_order(side=Side.BUY)
    o.intended_price = 100.0
    o.avg_fill_price = 100.5  # paid more
    # diff=0.5 ; (0.5/100)*10000 = 50 bps
    assert o.slippage_bps == pytest.approx(50.0)


def test_slippage_buy_better_is_negative():
    o = make_order(side=Side.BUY)
    o.intended_price = 100.0
    o.avg_fill_price = 99.5  # paid less -> favorable
    assert o.slippage_bps == pytest.approx(-50.0)


def test_slippage_sell_received_less_is_positive():
    o = make_order(side=Side.SELL)
    o.intended_price = 100.0
    o.avg_fill_price = 99.5  # received less -> worse for a sell
    # diff = 99.5-100 = -0.5 ; sell negates -> +0.5 ; 50 bps
    assert o.slippage_bps == pytest.approx(50.0)


def test_slippage_sell_received_more_is_negative():
    o = make_order(side=Side.SELL)
    o.intended_price = 100.0
    o.avg_fill_price = 101.0  # received more -> favorable
    assert o.slippage_bps == pytest.approx(-100.0)


def test_slippage_none_without_intended():
    o = make_order()
    o.avg_fill_price = 100.0  # intended None
    assert o.slippage_bps is None


def test_slippage_none_without_fill():
    o = make_order()
    o.intended_price = 100.0  # avg_fill_price None
    assert o.slippage_bps is None


def test_slippage_none_when_intended_zero():
    o = make_order()
    o.intended_price = 0.0
    o.avg_fill_price = 5.0
    # guarded division-by-zero -> None
    assert o.slippage_bps is None


def test_slippage_exact_match_is_zero():
    o = make_order()
    o.intended_price = 250.0
    o.avg_fill_price = 250.0
    assert o.slippage_bps == 0.0


def test_slippage_through_full_lifecycle():
    o = make_order(qty=10.0)
    o.mark_signal(intended_price=200.0)
    o.add_fill(Fill(ts=1.0, price=201.0, qty=10.0))
    # buy paid 201 vs 200 -> (1/200)*10000 = 50 bps
    assert o.slippage_bps == pytest.approx(50.0)


# --------------------------------------------------------------------------- #
# notional
# --------------------------------------------------------------------------- #
def test_notional_uses_limit_price_first():
    o = make_order(qty=10.0, limit_price=50.0, intended_price=99.0)
    o.avg_fill_price = 12.0
    # limit_price takes precedence
    assert o.notional == pytest.approx(500.0)


def test_notional_falls_back_to_intended_price():
    o = make_order(qty=4.0, intended_price=25.0)
    assert o.notional == pytest.approx(100.0)


def test_notional_falls_back_to_avg_fill_price():
    o = make_order(qty=2.0)
    o.avg_fill_price = 30.0
    assert o.notional == pytest.approx(60.0)


def test_notional_zero_when_no_price():
    o = make_order(qty=10.0)
    assert o.notional == 0.0


def test_notional_uses_abs_qty():
    o = make_order(qty=-3.0, intended_price=10.0)
    assert o.notional == pytest.approx(30.0)


# --------------------------------------------------------------------------- #
# to_dict serialization
# --------------------------------------------------------------------------- #
def test_to_dict_enum_values_are_strings():
    o = make_order(side=Side.SELL, order_type=OrderType.LIMIT, tif=TimeInForce.GTC)
    d = o.to_dict()
    assert d["side"] == "SELL"
    assert d["order_type"] == "LIMIT"
    assert d["tif"] == "GTC"
    assert d["status"] == "NEW"
    assert isinstance(d["side"], str)


def test_to_dict_latency_block_present():
    o = make_order()
    set_stage_timestamps(o, signal=0.0, submit=0.01, ack=0.02, fill=0.05)
    o.intended_price = 100.0
    o.avg_fill_price = 100.0
    d = o.to_dict()
    lat = d["latency"]
    assert lat["signal_to_submit_ms"] == pytest.approx(10.0)
    assert lat["submit_to_ack_ms"] == pytest.approx(10.0)
    assert lat["ack_to_fill_ms"] == pytest.approx(30.0)
    assert lat["submit_to_fill_ms"] == pytest.approx(40.0)
    assert lat["signal_to_fill_ms"] == pytest.approx(50.0)
    assert lat["slippage_bps"] == 0.0


def test_to_dict_fills_serialized_as_dicts():
    o = make_order(qty=2.0)
    o.add_fill(Fill(ts=1.0, price=10.0, qty=1.0, fee=0.1))
    o.add_fill(Fill(ts=2.0, price=12.0, qty=1.0))
    d = o.to_dict()
    assert isinstance(d["fills"], list)
    assert all(isinstance(f, dict) for f in d["fills"])
    assert d["fills"][0] == {"ts": 1.0, "price": 10.0, "qty": 1.0, "fee": 0.1}
    assert d["filled_qty"] == pytest.approx(2.0)


def test_to_dict_latency_none_when_unfilled():
    o = make_order()
    d = o.to_dict()
    lat = d["latency"]
    assert lat["signal_to_submit_ms"] is None
    assert lat["signal_to_fill_ms"] is None
    assert lat["slippage_bps"] is None


# --------------------------------------------------------------------------- #
# Degenerate / edge inputs
# --------------------------------------------------------------------------- #
def test_add_fill_zero_total_qty_leaves_avg_none():
    # a fill with qty=0 -> total_qty == 0, avg stays None (branch guard)
    o = make_order(qty=5.0)
    o.add_fill(Fill(ts=1.0, price=50.0, qty=0.0))
    assert o.filled_qty == 0.0
    assert o.avg_fill_price is None
    assert o.status is OrderStatus.PARTIALLY_FILLED


def test_zero_qty_order_first_fill_marks_filled():
    # qty=0 order: any fill of 0 total >= 0 - eps -> FILLED
    o = make_order(qty=0.0)
    o.add_fill(Fill(ts=1.0, price=10.0, qty=0.0))
    # total_qty == 0, not > 0 so avg stays None; 0 >= 0-1e-9 -> FILLED
    assert o.status is OrderStatus.FILLED
    assert o.avg_fill_price is None


def test_slippage_nan_intended_propagates_nan():
    o = make_order(side=Side.BUY)
    o.intended_price = float("nan")
    o.avg_fill_price = 100.0
    # nan != 0 passes the guard; result is nan, not None
    result = o.slippage_bps
    assert result is not None
    assert math.isnan(result)


def test_slippage_inf_avg_price():
    o = make_order(side=Side.BUY)
    o.intended_price = 100.0
    o.avg_fill_price = float("inf")
    assert math.isinf(o.slippage_bps)


def test_negative_intended_price_slippage_defined():
    # negative price is degenerate but math is still well-defined
    o = make_order(side=Side.BUY)
    o.intended_price = -100.0
    o.avg_fill_price = -90.0
    # diff = -90 - (-100) = 10 ; (10 / -100)*10000 = -1000
    assert o.slippage_bps == pytest.approx(-1000.0)


def test_extreme_magnitude_notional():
    o = make_order(qty=1e9, intended_price=1e6)
    assert o.notional == pytest.approx(1e15)


def test_multiple_partial_fills_idempotent_status_progression():
    o = make_order(qty=9.0)
    for i in range(3):
        o.add_fill(Fill(ts=float(i), price=10.0 + i, qty=3.0))
    assert o.filled_qty == pytest.approx(9.0)
    assert o.status is OrderStatus.FILLED
    # vwap = (10*3 + 11*3 + 12*3)/9 = (30+33+36)/9 = 99/9 = 11.0
    assert o.avg_fill_price == pytest.approx(11.0)
    assert o.fill_ts == 2.0


def test_fill_ts_tracks_last_fill_even_if_earlier_ts():
    # add_fill always assigns fill_ts = the incoming fill's ts (last-write)
    o = make_order(qty=4.0)
    o.add_fill(Fill(ts=10.0, price=10.0, qty=2.0))
    o.add_fill(Fill(ts=5.0, price=10.0, qty=2.0))  # out-of-order ts
    assert o.fill_ts == 5.0  # current behavior: last fill wins regardless
