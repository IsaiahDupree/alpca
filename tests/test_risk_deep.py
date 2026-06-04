"""
Deep, deterministic tests for alpca/risk/risk_engine.py.

All tests are pure/offline: no network, no mocks, no live Alpaca. The rate
limiter is driven by an injectable `now` clock so the 60s sliding window is
fully deterministic. Each parametrize case is an independent test.

Source under test:
  - RiskEngine.check() gate ordering + every gate code
  - halt/resume, set_day_start_equity
  - RiskDecision truthiness (__bool__)
  - Position.notional
  - record_submission + sliding-window pruning
"""

from __future__ import annotations

import math

import pytest

from alpca.config import RiskConfig
from alpca.execution.order import Order, OrderType, Side
from alpca.risk.risk_engine import Position, RiskDecision, RiskEngine


# ---------------------------------------------------------------- helpers
def mk_order(
    symbol: str = "AAPL",
    side: Side = Side.BUY,
    qty: float = 1.0,
    *,
    limit_price=None,
    intended_price=None,
    order_type: OrderType = OrderType.MARKET,
) -> Order:
    return Order(
        symbol=symbol,
        side=side,
        qty=qty,
        order_type=order_type,
        limit_price=limit_price,
        intended_price=intended_price,
    )


def cfg(**overrides) -> RiskConfig:
    """RiskConfig with sane permissive-ish defaults, overridable per-test."""
    base = dict(
        max_order_notional=50_000.0,
        daily_loss_pct=0.02,
        max_concentration_pct=0.25,
        max_open_positions=20,
        max_orders_per_min=60,
        enforce_buying_power=True,
        allow_short=False,
        short_borrow_apr=0.03,
    )
    base.update(overrides)
    return RiskConfig(**base)


class Clock:
    """Deterministic monotonic-style clock; advance() to move time forward."""

    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def engine(config=None, **kw) -> RiskEngine:
    return RiskEngine(config or cfg(), **kw)


# ================================================================ Position
@pytest.mark.parametrize(
    "qty,avg_price,expected",
    [
        (10.0, 100.0, 1000.0),
        (-10.0, 100.0, 1000.0),   # short: notional is abs(qty)*price
        (0.0, 100.0, 0.0),
        (3.0, 0.0, 0.0),
        (-2.5, 4.0, 10.0),
    ],
)
def test_position_notional(qty, avg_price, expected):
    p = Position("XYZ", qty, avg_price)
    assert p.notional == pytest.approx(expected)


def test_position_notional_extreme_magnitude():
    p = Position("XYZ", -1e9, 1e6)
    assert p.notional == pytest.approx(1e15)


# ============================================================ RiskDecision
def test_riskdecision_truthiness_true():
    d = RiskDecision(True, "OK", "")
    assert bool(d) is True
    assert d  # truthy in if-context


def test_riskdecision_truthiness_false():
    d = RiskDecision(False, "HALTED", "x")
    assert bool(d) is False
    assert not d


def test_riskdecision_defaults():
    d = RiskDecision(True)
    assert d.code == "OK"
    assert d.reason == ""


# ============================================================ halt / resume
def test_halt_sets_state_and_blocks():
    e = engine()
    assert e.halted is False
    e.halt("kill")
    assert e.halted is True
    d = e.check(mk_order(), equity=100_000.0, ref_price=100.0)
    assert not d
    assert d.code == "HALTED"
    assert "kill" in d.reason


def test_resume_clears_halt():
    e = engine()
    e.halt("kill")
    e.resume()
    assert e.halted is False
    d = e.check(mk_order(), equity=100_000.0, ref_price=100.0)
    assert d
    assert d.code == "OK"


def test_halt_invokes_breach_handler():
    events = []
    e = engine(breach_handler=lambda code, msg: events.append((code, msg)))
    e.halt("boom")
    assert events == [("HALT", "boom")]


def test_default_halt_reason():
    e = engine()
    e.halt()  # default reason "manual"
    d = e.check(mk_order(), equity=100_000.0, ref_price=100.0)
    assert d.code == "HALTED"
    assert "manual" in d.reason


# ================================================================ BAD_QTY
@pytest.mark.parametrize("bad_qty", [0.0, -1.0, -0.0001, -1e9])
def test_bad_qty_rejected(bad_qty):
    e = engine()
    d = e.check(mk_order(qty=bad_qty), equity=100_000.0, ref_price=100.0)
    assert not d
    assert d.code == "BAD_QTY"


def test_halted_takes_precedence_over_bad_qty():
    e = engine()
    e.halt("x")
    d = e.check(mk_order(qty=-5), equity=100_000.0, ref_price=100.0)
    assert d.code == "HALTED"  # halt gate is checked first


# ================================================================ NO_PRICE
def test_no_price_when_nothing_supplied():
    e = engine()
    d = e.check(mk_order(), equity=100_000.0)  # no ref/limit/intended/position
    assert not d
    assert d.code == "NO_PRICE"


@pytest.mark.parametrize("bad_price", [0.0, -1.0, -100.0])
def test_no_price_when_nonpositive_ref(bad_price):
    e = engine()
    d = e.check(mk_order(), equity=100_000.0, ref_price=bad_price)
    assert not d
    assert d.code == "NO_PRICE"


def test_price_falls_back_to_limit_price():
    e = engine()
    d = e.check(mk_order(limit_price=120.0), equity=100_000.0)
    assert d.code == "OK"


def test_price_falls_back_to_intended_price():
    e = engine()
    d = e.check(mk_order(intended_price=120.0), equity=100_000.0)
    assert d.code == "OK"


def test_price_falls_back_to_position_avg_price():
    e = engine()
    pos = {"AAPL": Position("AAPL", 10.0, 150.0)}
    # SELL reducing the long so short gate is fine; relies on avg_price for sizing
    d = e.check(mk_order(side=Side.SELL, qty=1.0), equity=100_000.0, positions=pos)
    assert d.code == "OK"


def test_ref_price_takes_priority_over_limit():
    # ref_price 100 keeps notional under cap; limit 999999 would breach if used.
    e = engine(cfg(max_order_notional=1000.0))
    d = e.check(mk_order(qty=5, limit_price=999_999.0), equity=1e9, ref_price=100.0)
    assert d.code == "OK"  # uses ref_price=100 -> notional 500


# =============================================================== FORBIDDEN
@pytest.mark.parametrize("sym,forbidden", [("TSLA", ["TSLA"]), ("tsla", ["TSLA"]), ("TSLA", ["tsla"])])
def test_forbidden_symbol_case_insensitive(sym, forbidden):
    e = engine(forbidden_symbols=forbidden)
    d = e.check(mk_order(symbol=sym), equity=100_000.0, ref_price=100.0)
    assert not d
    assert d.code == "FORBIDDEN"


def test_non_forbidden_symbol_passes():
    e = engine(forbidden_symbols=["TSLA"])
    d = e.check(mk_order(symbol="AAPL"), equity=100_000.0, ref_price=100.0)
    assert d.code == "OK"


# ====================================================== MAX_ORDER_NOTIONAL
@pytest.mark.parametrize(
    "qty,price,cap,expect_ok",
    [
        (10, 100, 1000, True),    # exactly at cap -> not greater than -> OK
        (10, 100.01, 1000, False),  # just over
        (1, 50_001, 50_000, False),
        (1, 50_000, 50_000, True),  # exactly at default cap
    ],
)
def test_max_order_notional(qty, price, cap, expect_ok):
    e = engine(cfg(max_order_notional=cap))
    d = e.check(mk_order(qty=qty), equity=1e12, ref_price=price, cash=1e12)
    if expect_ok:
        assert d.code == "OK"
    else:
        assert d.code == "MAX_ORDER_NOTIONAL"


def test_notional_cap_boundary_is_strict_greater_than():
    e = engine(cfg(max_order_notional=1000.0, max_concentration_pct=1.0))
    d = e.check(mk_order(qty=10), equity=1e9, ref_price=100.0, cash=1e9)
    assert d.code == "OK"  # 1000 == cap, not > cap


# ================================================ INSUFFICIENT_BUYING_POWER
def test_buying_power_rejects_buy_over_cash():
    e = engine()
    d = e.check(mk_order(side=Side.BUY, qty=10), equity=1e9, ref_price=100.0, cash=999.0)
    assert not d
    assert d.code == "INSUFFICIENT_BUYING_POWER"


def test_buying_power_allows_buy_at_exactly_cash():
    e = engine(cfg(max_concentration_pct=1.0))
    d = e.check(mk_order(side=Side.BUY, qty=10), equity=1e9, ref_price=100.0, cash=1000.0)
    assert d.code == "OK"  # 1000 not > 1000


def test_buying_power_not_enforced_for_sell():
    # SELL is not gated by cash; reduce a long so short gate passes.
    e = engine()
    pos = {"AAPL": Position("AAPL", 10.0, 100.0)}
    d = e.check(mk_order(side=Side.SELL, qty=5), equity=1e9, ref_price=100.0,
                cash=0.0, positions=pos)
    assert d.code == "OK"


def test_buying_power_skipped_when_cash_none():
    e = engine()
    d = e.check(mk_order(side=Side.BUY, qty=10), equity=1e9, ref_price=100.0, cash=None)
    assert d.code == "OK"


def test_buying_power_disabled_by_config():
    e = engine(cfg(enforce_buying_power=False, max_concentration_pct=1.0))
    d = e.check(mk_order(side=Side.BUY, qty=10), equity=1e9, ref_price=100.0, cash=1.0)
    assert d.code == "OK"


# ========================================================= SHORT_NOT_ALLOWED
def test_short_sell_with_no_position_rejected():
    e = engine()  # allow_short False
    d = e.check(mk_order(side=Side.SELL, qty=5), equity=1e9, ref_price=100.0)
    assert not d
    assert d.code == "SHORT_NOT_ALLOWED"


def test_sell_reducing_long_allowed():
    e = engine()
    pos = {"AAPL": Position("AAPL", 10.0, 100.0)}
    d = e.check(mk_order(side=Side.SELL, qty=4), equity=1e9, ref_price=100.0, positions=pos)
    assert d.code == "OK"  # resulting +6, still long


def test_sell_closing_long_exactly_flat_allowed():
    e = engine()
    pos = {"AAPL": Position("AAPL", 10.0, 100.0)}
    d = e.check(mk_order(side=Side.SELL, qty=10), equity=1e9, ref_price=100.0, positions=pos)
    assert d.code == "OK"  # resulting 0 -> not < -1e-9


def test_sell_exceeding_long_flips_short_rejected():
    e = engine()
    pos = {"AAPL": Position("AAPL", 10.0, 100.0)}
    d = e.check(mk_order(side=Side.SELL, qty=15), equity=1e9, ref_price=100.0, positions=pos)
    assert not d
    assert d.code == "SHORT_NOT_ALLOWED"  # resulting -5


def test_short_allowed_when_config_enables_it():
    e = engine(cfg(allow_short=True, max_concentration_pct=1.0))
    d = e.check(mk_order(side=Side.SELL, qty=5), equity=1e9, ref_price=100.0)
    assert d.code == "OK"


def test_short_allowed_can_extend_existing_short():
    e = engine(cfg(allow_short=True, max_concentration_pct=1.0))
    pos = {"AAPL": Position("AAPL", -5.0, 100.0)}
    d = e.check(mk_order(side=Side.SELL, qty=5), equity=1e9, ref_price=100.0, positions=pos)
    assert d.code == "OK"  # resulting -10, shorting enabled


def test_buy_partially_covering_short_is_allowed():
    # The short gate is side-aware: it only blocks a SELL that leaves a net short.
    # A BUY can only reduce/cover a short, so a BUY of 4 against a -10 short
    # (leaving -6) is allowed even though the net position is still short.
    e = engine()
    pos = {"AAPL": Position("AAPL", -10.0, 100.0)}
    d = e.check(mk_order(side=Side.BUY, qty=4), equity=1e9, ref_price=100.0, cash=1e9, positions=pos)
    assert d
    assert d.code == "OK"


def test_buy_fully_covering_short_to_flat_allowed():
    # A BUY that brings the short exactly to flat (resulting 0) is allowed,
    # since 0 is not < -1e-9.
    e = engine(cfg(max_concentration_pct=1.0))
    pos = {"AAPL": Position("AAPL", -10.0, 100.0)}
    d = e.check(mk_order(side=Side.BUY, qty=10), equity=1e9, ref_price=100.0, cash=1e9, positions=pos)
    assert d.code == "OK"


# ================================================================ RATE_LIMIT
def test_rate_limit_blocks_at_cap():
    clk = Clock()
    e = engine(cfg(max_orders_per_min=3), now=clk)
    for _ in range(3):
        e.record_submission()
    d = e.check(mk_order(), equity=1e9, ref_price=100.0, cash=1e9)
    assert not d
    assert d.code == "RATE_LIMIT"


def test_rate_limit_below_cap_ok():
    clk = Clock()
    e = engine(cfg(max_orders_per_min=3), now=clk)
    e.record_submission()
    e.record_submission()
    d = e.check(mk_order(), equity=1e9, ref_price=100.0, cash=1e9)
    assert d.code == "OK"  # 2 < 3


def test_rate_limit_window_slides_after_60s():
    clk = Clock(start=1000.0)
    e = engine(cfg(max_orders_per_min=2), now=clk)
    e.record_submission()  # t=1000
    e.record_submission()  # t=1000
    blocked = e.check(mk_order(), equity=1e9, ref_price=100.0, cash=1e9)
    assert blocked.code == "RATE_LIMIT"
    clk.advance(60.001)  # both submissions now older than 60s -> pruned
    ok = e.check(mk_order(), equity=1e9, ref_price=100.0, cash=1e9)
    assert ok.code == "OK"


def test_rate_limit_partial_window_slide():
    clk = Clock(start=1000.0)
    e = engine(cfg(max_orders_per_min=2), now=clk)
    e.record_submission()       # t=1000
    clk.advance(30.0)
    e.record_submission()       # t=1030
    # at t=1030 both in window -> blocked
    assert e.check(mk_order(), equity=1e9, ref_price=100.0, cash=1e9).code == "RATE_LIMIT"
    clk.advance(31.0)           # t=1061: first (1000) is >60s old, pruned; second (1030) stays
    d = e.check(mk_order(), equity=1e9, ref_price=100.0, cash=1e9)
    assert d.code == "OK"  # only 1 in window now


def test_prune_boundary_exactly_60s_not_pruned():
    # cutoff = now - 60; pruned only if ts < cutoff (strict). ts == cutoff stays.
    clk = Clock(start=1000.0)
    e = engine(cfg(max_orders_per_min=1), now=clk)
    e.record_submission()  # ts=1000
    clk.advance(60.0)      # now=1060, cutoff=1000, 1000 < 1000 is False -> kept
    d = e.check(mk_order(), equity=1e9, ref_price=100.0, cash=1e9)
    assert d.code == "RATE_LIMIT"


def test_record_submission_grows_window():
    clk = Clock()
    e = engine(cfg(max_orders_per_min=100), now=clk)
    for _ in range(5):
        e.record_submission()
    assert len(e._submission_ts) == 5


# ================================================================ DAILY_LOSS
def test_daily_loss_halts_at_floor():
    e = engine(cfg(daily_loss_pct=0.02), day_start_equity=100_000.0)
    floor = 100_000.0 * 0.98  # 98_000
    d = e.check(mk_order(), equity=floor, ref_price=100.0, cash=1e9)  # equity <= floor
    assert not d
    assert d.code == "DAILY_LOSS"
    assert e.halted is True  # daily loss auto-halts the engine


def test_daily_loss_below_floor_halts():
    e = engine(cfg(daily_loss_pct=0.02), day_start_equity=100_000.0)
    d = e.check(mk_order(), equity=90_000.0, ref_price=100.0, cash=1e9)
    assert d.code == "DAILY_LOSS"


def test_daily_loss_above_floor_ok():
    e = engine(cfg(daily_loss_pct=0.02), day_start_equity=100_000.0)
    d = e.check(mk_order(), equity=99_000.0, ref_price=100.0, cash=1e9)
    assert d.code == "OK"
    assert e.halted is False


def test_daily_loss_skipped_when_no_day_start_equity():
    e = engine(cfg(max_concentration_pct=1.0))  # day_start_equity None
    d = e.check(mk_order(qty=1), equity=1e9, ref_price=100.0, cash=1e9)
    assert d.code == "OK"


def test_set_day_start_equity_then_loss_trips():
    e = engine()
    e.set_day_start_equity(50_000.0)
    floor = 50_000.0 * 0.98
    d = e.check(mk_order(), equity=floor - 1, ref_price=100.0, cash=1e9)
    assert d.code == "DAILY_LOSS"


def test_daily_loss_zero_day_start_equity_skipped():
    # day_start_equity == 0 is falsy -> gate skipped entirely.
    e = engine(day_start_equity=0.0)
    d = e.check(mk_order(), equity=-1e6, ref_price=100.0, cash=1e9)
    assert d.code == "OK"


# ============================================================== MAX_POSITIONS
def test_max_positions_blocks_new_symbol_buy():
    e = engine(cfg(max_open_positions=2, max_concentration_pct=1.0))
    pos = {"AAA": Position("AAA", 1, 10), "BBB": Position("BBB", 1, 10)}
    d = e.check(mk_order(symbol="CCC", side=Side.BUY), equity=1e9, ref_price=10.0,
                cash=1e9, positions=pos)
    assert not d
    assert d.code == "MAX_POSITIONS"


def test_max_positions_allows_buy_into_existing_symbol():
    e = engine(cfg(max_open_positions=2, max_concentration_pct=1.0))
    pos = {"AAPL": Position("AAPL", 1, 10), "BBB": Position("BBB", 1, 10)}
    d = e.check(mk_order(symbol="AAPL", side=Side.BUY), equity=1e9, ref_price=10.0,
                cash=1e9, positions=pos)
    assert d.code == "OK"  # not opening a NEW symbol


def test_max_positions_allows_sell_even_at_cap():
    e = engine(cfg(max_open_positions=2, max_concentration_pct=1.0))
    pos = {"AAA": Position("AAA", 5, 10), "BBB": Position("BBB", 5, 10)}
    # SELL of an existing symbol is not "opening_new"; reduce existing long
    d = e.check(mk_order(symbol="AAA", side=Side.SELL, qty=2), equity=1e9,
                ref_price=10.0, positions=pos)
    assert d.code == "OK"


def test_max_positions_under_cap_ok():
    e = engine(cfg(max_open_positions=3, max_concentration_pct=1.0))
    pos = {"AAA": Position("AAA", 1, 10)}
    d = e.check(mk_order(symbol="CCC", side=Side.BUY), equity=1e9, ref_price=10.0,
                cash=1e9, positions=pos)
    assert d.code == "OK"


# ============================================================== CONCENTRATION
def test_concentration_blocks_oversized_position():
    # projected/equity > 0.25 -> blocked
    e = engine(cfg(max_concentration_pct=0.25, max_order_notional=1e12))
    # qty 4000 * 100 = 400k; equity 1M -> 40% > 25%
    d = e.check(mk_order(qty=4000), equity=1_000_000.0, ref_price=100.0, cash=1e12)
    assert not d
    assert d.code == "CONCENTRATION"


def test_concentration_at_cap_ok():
    e = engine(cfg(max_concentration_pct=0.25, max_order_notional=1e12))
    # 2500 * 100 = 250k; equity 1M -> exactly 25% -> not > cap -> OK
    d = e.check(mk_order(qty=2500), equity=1_000_000.0, ref_price=100.0, cash=1e12)
    assert d.code == "OK"


def test_concentration_signed_aware_on_cover_reduces_exposure():
    # Long 10000 @ 100 = 1M position. A BUY would add; but a SELL covering
    # part reduces projected exposure so concentration passes.
    e = engine(cfg(max_concentration_pct=0.30, max_order_notional=1e12))
    pos = {"AAPL": Position("AAPL", 10_000.0, 100.0)}
    # SELL 8000 -> resulting 2000 -> projected 200k / 1M = 20% < 30% -> OK
    d = e.check(mk_order(side=Side.SELL, qty=8000), equity=1_000_000.0,
                ref_price=100.0, positions=pos)
    assert d.code == "OK"


def test_concentration_signed_aware_on_add():
    # Already long 2000 @ 100 = 200k; BUY 1000 more -> resulting 3000 -> 300k.
    e = engine(cfg(max_concentration_pct=0.25, max_order_notional=1e12))
    pos = {"AAPL": Position("AAPL", 2000.0, 100.0)}
    # 300k / 1M = 30% > 25% -> blocked
    d = e.check(mk_order(side=Side.BUY, qty=1000), equity=1_000_000.0,
                ref_price=100.0, cash=1e12, positions=pos)
    assert not d
    assert d.code == "CONCENTRATION"


def test_concentration_signed_aware_short_exposure():
    # allow_short: SELL extends a short; |resulting| drives concentration.
    e = engine(cfg(allow_short=True, max_concentration_pct=0.25, max_order_notional=1e12))
    pos = {"AAPL": Position("AAPL", -2000.0, 100.0)}
    # SELL 1000 -> resulting -3000 -> |3000|*100 = 300k / 1M = 30% > 25% -> blocked
    d = e.check(mk_order(side=Side.SELL, qty=1000), equity=1_000_000.0,
                ref_price=100.0, positions=pos)
    assert not d
    assert d.code == "CONCENTRATION"


def test_concentration_skipped_when_equity_nonpositive():
    # equity > 0 guard: with equity 0 the concentration gate is skipped.
    e = engine(cfg(max_concentration_pct=0.01, max_order_notional=1e12))
    d = e.check(mk_order(qty=1000), equity=0.0, ref_price=100.0, cash=1e12)
    assert d.code == "OK"


# ============================================================ gate ordering
def test_forbidden_before_notional():
    e = engine(cfg(max_order_notional=1.0), forbidden_symbols=["AAPL"])
    d = e.check(mk_order(symbol="AAPL", qty=1000), equity=1e9, ref_price=100.0)
    assert d.code == "FORBIDDEN"  # forbidden checked before notional cap


def test_notional_before_buying_power():
    e = engine(cfg(max_order_notional=100.0))
    d = e.check(mk_order(side=Side.BUY, qty=1000), equity=1e9, ref_price=100.0, cash=0.0)
    assert d.code == "MAX_ORDER_NOTIONAL"  # notional checked before buying power


def test_buying_power_before_short_gate():
    # BUY can't trip short gate, but confirms BP runs before the rate/short block.
    e = engine(cfg(max_order_notional=1e12))
    d = e.check(mk_order(side=Side.BUY, qty=10), equity=1e9, ref_price=100.0, cash=10.0)
    assert d.code == "INSUFFICIENT_BUYING_POWER"


def test_rate_limit_before_daily_loss():
    clk = Clock()
    e = engine(cfg(max_orders_per_min=1, daily_loss_pct=0.02),
               day_start_equity=100_000.0, now=clk)
    e.record_submission()
    # equity below floor would trip DAILY_LOSS, but rate limit is checked first
    d = e.check(mk_order(), equity=1.0, ref_price=100.0, cash=1e9)
    assert d.code == "RATE_LIMIT"
    assert e.halted is False  # daily-loss halt never reached


def test_daily_loss_before_max_positions():
    e = engine(cfg(max_open_positions=1, daily_loss_pct=0.02, max_concentration_pct=1.0),
               day_start_equity=100_000.0)
    pos = {"AAA": Position("AAA", 1, 10)}
    d = e.check(mk_order(symbol="CCC", side=Side.BUY), equity=1.0, ref_price=10.0,
                cash=1e9, positions=pos)
    assert d.code == "DAILY_LOSS"  # daily-loss checked before max positions


# ======================================================== degenerate inputs
def test_nan_ref_price_behavior():
    # A non-finite reference price (NaN) is rejected up front by the finite-check
    # that runs BEFORE the notional/concentration gates, so the order is blocked
    # with NO_PRICE rather than slipping through as a NaN notional.
    e = engine()
    d = e.check(mk_order(qty=1), equity=1e9, ref_price=float("nan"), cash=1e12)
    assert not d
    assert d.code == "NO_PRICE"
    assert math.isnan(1.0 * float("nan"))  # invariant the finite-check guards against


def test_inf_ref_price_rejected_as_no_price():
    # A non-finite reference price (inf) is caught by the finite-check before the
    # notional cap, so it yields NO_PRICE (not MAX_ORDER_NOTIONAL).
    e = engine()
    d = e.check(mk_order(qty=1), equity=1e9, ref_price=float("inf"), cash=1e12)
    assert not d
    assert d.code == "NO_PRICE"


def test_empty_positions_dict_and_none_equivalent():
    e = engine()
    d_none = e.check(mk_order(), equity=1e9, ref_price=100.0, cash=1e9, positions=None)
    d_empty = e.check(mk_order(), equity=1e9, ref_price=100.0, cash=1e9, positions={})
    assert d_none.code == d_empty.code == "OK"


def test_negative_equity_passes_concentration_guard():
    # equity > 0 is False for negative equity -> concentration gate skipped.
    e = engine(cfg(max_concentration_pct=0.01, daily_loss_pct=0.5))
    d = e.check(mk_order(qty=1), equity=-5.0, ref_price=100.0, cash=1e9)
    assert d.code == "OK"


def test_extreme_qty_trips_notional():
    e = engine()
    d = e.check(mk_order(qty=1e12), equity=1e30, ref_price=1.0, cash=1e30)
    assert d.code == "MAX_ORDER_NOTIONAL"


def test_idempotent_check_does_not_mutate_rate_window():
    # check() prunes but never appends; repeated checks keep the window stable.
    clk = Clock()
    e = engine(cfg(max_orders_per_min=10), now=clk)
    e.record_submission()
    before = len(e._submission_ts)
    for _ in range(5):
        e.check(mk_order(), equity=1e9, ref_price=100.0, cash=1e9)
    assert len(e._submission_ts) == before  # check() doesn't record


def test_full_pass_returns_ok_with_empty_reason():
    e = engine(cfg(max_concentration_pct=1.0))
    d = e.check(mk_order(qty=1), equity=1e9, ref_price=100.0, cash=1e9)
    assert d.allowed is True
    assert d.code == "OK"
    assert d.reason == ""
