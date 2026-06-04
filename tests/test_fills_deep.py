"""
Deep, deterministic tests for alpca/execution/fills.py (FillModel / FillResult).

Covers, with EXACT computed numbers and invariants:
  - FillModel.fill : impact_bps = impact_coef_bps * sqrt(qty/bar_volume),
    half-spread, buy adverse-up / sell adverse-down, volume-cap partials +
    `capped` flag, min_tick rounding, degenerate inputs.
  - FillModel.flat across a bps grid (pure flat-bps model).
  - FillModel.fill_limit : through-trade test (buy iff low<=limit /
    sell iff high>=limit), price improvement on gap-through, no-fill rests,
    legacy volume-cap proxy, FIFO queue_pos via a real QueuePosition.
  - FillResult fields.

Everything is offline and deterministic. No network, no mocks.
"""

from __future__ import annotations

import math

import pytest

from alpca.execution.fills import FillModel, FillResult
from alpca.execution.queue_prob import QueuePosition, PowerProbQueueFunc


# --------------------------------------------------------------------------
# tiny self-contained helpers
# --------------------------------------------------------------------------
def expected_market_price(side_buy, ref, half_spread_bps, impact_coef_bps,
                          qty, bar_volume, min_tick):
    """Re-derive the expected fill price independently of the source body."""
    impact_bps = 0.0
    if impact_coef_bps > 0 and bar_volume is not None and bar_volume > 0:
        impact_bps = impact_coef_bps * math.sqrt(qty / bar_volume)
    slip = half_spread_bps + impact_bps
    adj = slip / 10_000.0
    price = ref * (1 + adj) if side_buy else ref * (1 - adj)
    if min_tick and min_tick > 0:
        price = round(price / min_tick) * min_tick
    return price, slip


def approx(x, **kw):
    return pytest.approx(x, **kw)


# --------------------------------------------------------------------------
# FillResult dataclass
# --------------------------------------------------------------------------
def test_fillresult_fields_and_types():
    r = FillResult(price=10.5, filled_qty=3.0, slippage_bps=2.0, capped=True)
    assert r.price == 10.5
    assert r.filled_qty == 3.0
    assert r.slippage_bps == 2.0
    assert r.capped is True
    # dataclass equality
    assert r == FillResult(price=10.5, filled_qty=3.0, slippage_bps=2.0, capped=True)
    assert r != FillResult(price=10.5, filled_qty=3.0, slippage_bps=2.0, capped=False)


# --------------------------------------------------------------------------
# FillModel defaults / construction
# --------------------------------------------------------------------------
def test_default_construction_values():
    m = FillModel()
    assert m.half_spread_bps == 1.0
    assert m.impact_coef_bps == 8.0
    assert m.participation_cap == 1.0
    assert m.min_tick == 0.01


# --------------------------------------------------------------------------
# fill(): exact half-spread only (no impact because no bar_volume)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("side_buy", [True, False])
@pytest.mark.parametrize("half_spread_bps", [0.5, 1.0, 5.0, 25.0])
def test_fill_half_spread_only_no_volume(side_buy, half_spread_bps):
    m = FillModel(half_spread_bps=half_spread_bps, impact_coef_bps=8.0,
                  participation_cap=1.0, min_tick=0.0)
    ref = 100.0
    r = m.fill(side_buy, ref, qty=10.0, bar_volume=None)
    # impact is zero with no bar_volume -> slippage == half_spread only
    assert r.slippage_bps == approx(half_spread_bps)
    exp = ref * (1 + half_spread_bps / 1e4) if side_buy else ref * (1 - half_spread_bps / 1e4)
    assert r.price == approx(exp)
    assert r.filled_qty == 10.0
    assert r.capped is False


# --------------------------------------------------------------------------
# fill(): EXACT impact = coef * sqrt(qty/vol)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("qty,vol", [
    (100.0, 100.0),     # participation 1.0 -> sqrt 1.0
    (25.0, 100.0),      # 0.25 -> sqrt 0.5
    (4.0, 100.0),       # 0.04 -> sqrt 0.2
    (1.0, 10_000.0),    # 1e-4 -> sqrt 0.01
    (900.0, 100.0),     # 9.0 -> sqrt 3.0 (over-participation)
])
def test_fill_exact_impact_bps(qty, vol):
    coef = 8.0
    hs = 1.0
    m = FillModel(half_spread_bps=hs, impact_coef_bps=coef,
                  participation_cap=1.0, min_tick=0.0)
    r = m.fill(True, 100.0, qty=qty, bar_volume=vol)
    expected_impact = coef * math.sqrt(qty / vol)
    assert r.slippage_bps == approx(hs + expected_impact)


@pytest.mark.parametrize("side_buy", [True, False])
def test_fill_exact_price_with_impact(side_buy):
    m = FillModel(half_spread_bps=2.0, impact_coef_bps=10.0,
                  participation_cap=1.0, min_tick=0.0)
    ref, qty, vol = 50.0, 25.0, 100.0  # participation 0.25 -> sqrt 0.5
    r = m.fill(side_buy, ref, qty=qty, bar_volume=vol)
    exp_price, exp_slip = expected_market_price(
        side_buy, ref, 2.0, 10.0, qty, vol, 0.0)
    assert r.slippage_bps == approx(exp_slip)        # 2 + 10*0.5 = 7.0
    assert r.slippage_bps == approx(7.0)
    assert r.price == approx(exp_price)


# --------------------------------------------------------------------------
# fill(): buy adverse-up, sell adverse-down invariant
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ref,qty,vol", [
    (10.0, 50.0, 200.0),
    (250.0, 10.0, 1000.0),
    (1.23, 5.0, 50.0),
])
def test_fill_buy_above_sell_below(ref, qty, vol):
    m = FillModel(half_spread_bps=3.0, impact_coef_bps=8.0,
                  participation_cap=1.0, min_tick=0.0)
    buy = m.fill(True, ref, qty, vol)
    sell = m.fill(False, ref, qty, vol)
    assert buy.price > ref          # buyer pays up
    assert sell.price < ref         # seller receives less
    # symmetric magnitude about ref
    assert (buy.price - ref) == approx(ref - sell.price)
    assert buy.slippage_bps == approx(sell.slippage_bps)


# --------------------------------------------------------------------------
# fill(): volume-cap partial + capped flag
# --------------------------------------------------------------------------
def test_fill_volume_cap_partial_and_flag():
    m = FillModel(half_spread_bps=1.0, impact_coef_bps=0.0,
                  participation_cap=0.1, min_tick=0.0)
    # request 500 but cap is 10% of 1000 = 100
    r = m.fill(True, 100.0, qty=500.0, bar_volume=1000.0)
    assert r.filled_qty == approx(100.0)
    assert r.capped is True


def test_fill_no_cap_when_under_limit():
    m = FillModel(half_spread_bps=1.0, impact_coef_bps=0.0,
                  participation_cap=0.5, min_tick=0.0)
    # cap = 500, request 100 -> no cap
    r = m.fill(True, 100.0, qty=100.0, bar_volume=1000.0)
    assert r.filled_qty == approx(100.0)
    assert r.capped is False


def test_fill_cap_inactive_when_cap_is_one():
    m = FillModel(half_spread_bps=1.0, impact_coef_bps=0.0,
                  participation_cap=1.0, min_tick=0.0)
    r = m.fill(True, 100.0, qty=1e9, bar_volume=1000.0)
    assert r.filled_qty == approx(1e9)   # cap==1.0 disables capping
    assert r.capped is False


def test_fill_impact_uses_requested_qty_even_when_capped():
    # impact should use REQUESTED qty (you move the market trying to take it)
    coef = 8.0
    m = FillModel(half_spread_bps=0.0, impact_coef_bps=coef,
                  participation_cap=0.1, min_tick=0.0)
    qty, vol = 1000.0, 1000.0   # participation requested = 1.0 -> sqrt 1.0
    r = m.fill(True, 100.0, qty=qty, bar_volume=vol)
    assert r.capped is True
    assert r.filled_qty == approx(100.0)            # capped to 10%
    assert r.slippage_bps == approx(coef * 1.0)     # impact still from full qty


# --------------------------------------------------------------------------
# fill(): min_tick rounding
# --------------------------------------------------------------------------
@pytest.mark.parametrize("min_tick", [0.01, 0.05, 0.5, 1.0])
def test_fill_min_tick_rounding(min_tick):
    m = FillModel(half_spread_bps=7.3, impact_coef_bps=0.0,
                  participation_cap=1.0, min_tick=min_tick)
    r = m.fill(True, 99.97, qty=1.0, bar_volume=None)
    # result must be an exact multiple of the tick
    quotient = r.price / min_tick
    assert quotient == approx(round(quotient))


def test_fill_min_tick_exact_value():
    m = FillModel(half_spread_bps=10.0, impact_coef_bps=0.0,
                  participation_cap=1.0, min_tick=0.01)
    # 100 * (1 + 10/1e4) = 100.10 exactly -> rounds to 100.10
    r = m.fill(True, 100.0, qty=1.0, bar_volume=None)
    assert r.price == approx(100.10)


def test_fill_zero_min_tick_no_rounding():
    m = FillModel(half_spread_bps=3.0, impact_coef_bps=0.0,
                  participation_cap=1.0, min_tick=0.0)
    r = m.fill(True, 100.0, qty=1.0, bar_volume=None)
    assert r.price == approx(100.0 * (1 + 3.0 / 1e4))


# --------------------------------------------------------------------------
# fill(): degenerate / guard inputs
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ref,qty", [
    (0.0, 10.0),      # zero ref price
    (-5.0, 10.0),     # negative ref price
    (100.0, 0.0),     # zero qty
    (100.0, -3.0),    # negative qty
])
def test_fill_guard_returns_zero_fill(ref, qty):
    m = FillModel()
    r = m.fill(True, ref, qty, bar_volume=1000.0)
    assert r.filled_qty == 0.0
    assert r.slippage_bps == 0.0
    assert r.capped is False
    assert r.price == ref      # ref echoed unchanged


def test_fill_zero_bar_volume_no_impact_no_cap():
    m = FillModel(half_spread_bps=2.0, impact_coef_bps=8.0,
                  participation_cap=0.1, min_tick=0.0)
    # bar_volume == 0 disables both impact and cap branches
    r = m.fill(True, 100.0, qty=1e6, bar_volume=0.0)
    assert r.slippage_bps == approx(2.0)   # half-spread only
    assert r.capped is False
    assert r.filled_qty == approx(1e6)


def test_fill_negative_bar_volume_no_impact_no_cap():
    m = FillModel(half_spread_bps=2.0, impact_coef_bps=8.0,
                  participation_cap=0.1, min_tick=0.0)
    r = m.fill(True, 100.0, qty=1e6, bar_volume=-50.0)
    assert r.slippage_bps == approx(2.0)
    assert r.capped is False


def test_fill_extreme_magnitude_ref_price():
    m = FillModel(half_spread_bps=1.0, impact_coef_bps=0.0,
                  participation_cap=1.0, min_tick=0.0)
    r = m.fill(True, 1e9, qty=1.0, bar_volume=None)
    assert r.price == approx(1e9 * (1 + 1e-4))
    assert math.isfinite(r.price)


def test_fill_idempotent_repeated_calls():
    m = FillModel(half_spread_bps=2.5, impact_coef_bps=8.0,
                  participation_cap=0.5, min_tick=0.01)
    a = m.fill(True, 100.0, 50.0, 1000.0)
    b = m.fill(True, 100.0, 50.0, 1000.0)
    assert a == b


# --------------------------------------------------------------------------
# flat(): pure flat-bps model across a grid
# --------------------------------------------------------------------------
@pytest.mark.parametrize("bps", [0.0, 0.5, 1.0, 3.0, 7.5, 50.0, 100.0])
@pytest.mark.parametrize("side_buy", [True, False])
def test_flat_grid_exact(bps, side_buy):
    m = FillModel.flat(bps)
    # flat model: no impact, no cap, no tick rounding
    assert m.impact_coef_bps == 0.0
    assert m.participation_cap == 1.0
    assert m.min_tick == 0.0
    assert m.half_spread_bps == bps
    ref = 100.0
    # bar_volume present but impact_coef==0 so no impact contribution
    r = m.fill(side_buy, ref, qty=10.0, bar_volume=500.0)
    assert r.slippage_bps == approx(bps)
    exp = ref * (1 + bps / 1e4) if side_buy else ref * (1 - bps / 1e4)
    assert r.price == approx(exp)
    assert r.capped is False
    assert r.filled_qty == approx(10.0)


def test_flat_never_caps_or_rounds():
    m = FillModel.flat(2.0)
    r = m.fill(True, 33.333333, qty=1e9, bar_volume=10.0)
    assert r.capped is False
    assert r.filled_qty == approx(1e9)
    # no rounding -> raw float
    assert r.price == approx(33.333333 * (1 + 2.0 / 1e4))


# --------------------------------------------------------------------------
# fill_limit(): through-trade test (reached / not reached)
# --------------------------------------------------------------------------
def test_fill_limit_buy_reached_at_limit():
    m = FillModel(half_spread_bps=1.0, impact_coef_bps=8.0,
                  participation_cap=1.0, min_tick=0.0)
    # buy limit 100, bar low 99 (<=100) reached; open 101 (no gap-through)
    r = m.fill_limit(True, limit_price=100.0, bar_open=101.0,
                     bar_high=102.0, bar_low=99.0, qty=10.0, bar_volume=None)
    assert r.filled_qty == approx(10.0)
    assert r.price == approx(100.0)     # fills exactly at limit (open above limit)
    assert r.slippage_bps == approx(0.0)


def test_fill_limit_sell_reached_at_limit():
    m = FillModel(half_spread_bps=1.0, impact_coef_bps=8.0,
                  participation_cap=1.0, min_tick=0.0)
    # sell limit 100, bar high 101 (>=100) reached; open 99 (below limit)
    r = m.fill_limit(False, limit_price=100.0, bar_open=99.0,
                     bar_high=101.0, bar_low=98.0, qty=10.0, bar_volume=None)
    assert r.filled_qty == approx(10.0)
    assert r.price == approx(100.0)     # max(open=99, limit=100) -> 100
    assert r.slippage_bps == approx(0.0)


def test_fill_limit_buy_not_reached_rests():
    m = FillModel()
    # buy limit 100, low 100.5 never came down to bid -> no fill
    r = m.fill_limit(True, limit_price=100.0, bar_open=101.0,
                     bar_high=102.0, bar_low=100.5, qty=10.0)
    assert r.filled_qty == 0.0
    assert r.price == approx(100.0)
    assert r.slippage_bps == 0.0
    assert r.capped is False


def test_fill_limit_sell_not_reached_rests():
    m = FillModel()
    # sell limit 100, high 99.5 never rose to offer -> no fill
    r = m.fill_limit(False, limit_price=100.0, bar_open=99.0,
                     bar_high=99.5, bar_low=98.0, qty=10.0)
    assert r.filled_qty == 0.0
    assert r.price == approx(100.0)
    assert r.slippage_bps == 0.0


def test_fill_limit_buy_exact_touch_fills():
    m = FillModel(min_tick=0.0)
    # bar_low == limit (boundary) -> reached
    r = m.fill_limit(True, limit_price=100.0, bar_open=100.5,
                     bar_high=101.0, bar_low=100.0, qty=5.0)
    assert r.filled_qty == approx(5.0)
    assert r.price == approx(100.0)


def test_fill_limit_sell_exact_touch_fills():
    m = FillModel(min_tick=0.0)
    r = m.fill_limit(False, limit_price=100.0, bar_open=99.5,
                     bar_high=100.0, bar_low=99.0, qty=5.0)
    assert r.filled_qty == approx(5.0)
    assert r.price == approx(100.0)


# --------------------------------------------------------------------------
# fill_limit(): price improvement on gap-through (favorable slippage)
# --------------------------------------------------------------------------
def test_fill_limit_buy_gap_through_price_improvement():
    m = FillModel(min_tick=0.0)
    # buy limit 100 but bar OPENS at 98 (gapped below) -> fill at 98, improvement
    r = m.fill_limit(True, limit_price=100.0, bar_open=98.0,
                     bar_high=99.0, bar_low=97.0, qty=10.0)
    assert r.price == approx(98.0)            # min(open=98, limit=100)
    assert r.filled_qty == approx(10.0)
    # raw = (98-100)/100 * 1e4 = -200 ; buy slip = raw = -200 (favorable)
    assert r.slippage_bps == approx(-200.0)
    assert r.slippage_bps < 0


def test_fill_limit_sell_gap_through_price_improvement():
    m = FillModel(min_tick=0.0)
    # sell limit 100 but bar OPENS at 103 (gapped above) -> fill at 103
    r = m.fill_limit(False, limit_price=100.0, bar_open=103.0,
                     bar_high=104.0, bar_low=100.0, qty=10.0)
    assert r.price == approx(103.0)           # max(open=103, limit=100)
    # raw = (103-100)/100*1e4 = 300 ; sell slip = -raw = -300 (favorable)
    assert r.slippage_bps == approx(-300.0)
    assert r.slippage_bps < 0


def test_fill_limit_no_improvement_when_open_inside():
    m = FillModel(min_tick=0.0)
    # buy limit 100, open 99.9 (slightly below) -> tiny improvement to 99.9
    r = m.fill_limit(True, limit_price=100.0, bar_open=99.9,
                     bar_high=100.5, bar_low=99.0, qty=1.0)
    assert r.price == approx(99.9)
    assert r.slippage_bps == approx((99.9 - 100.0) / 100.0 * 1e4)  # -10.0


# --------------------------------------------------------------------------
# fill_limit(): guard inputs
# --------------------------------------------------------------------------
@pytest.mark.parametrize("limit,qty", [
    (0.0, 10.0),
    (-1.0, 10.0),
    (100.0, 0.0),
    (100.0, -5.0),
])
def test_fill_limit_guard_returns_zero(limit, qty):
    m = FillModel()
    r = m.fill_limit(True, limit_price=limit, bar_open=100.0,
                     bar_high=101.0, bar_low=99.0, qty=qty)
    assert r.filled_qty == 0.0
    assert r.slippage_bps == 0.0
    assert r.price == limit
    assert r.capped is False


# --------------------------------------------------------------------------
# fill_limit(): legacy volume-cap proxy
# --------------------------------------------------------------------------
def test_fill_limit_volume_cap_partial():
    m = FillModel(participation_cap=0.1, min_tick=0.0)
    # traded_at_level = 0.1 * 1000 = 100; request 500 -> capped to 100
    r = m.fill_limit(True, limit_price=100.0, bar_open=100.0,
                     bar_high=101.0, bar_low=99.0, qty=500.0, bar_volume=1000.0)
    assert r.filled_qty == approx(100.0)
    assert r.capped is True


def test_fill_limit_volume_cap_inactive_full_cap():
    m = FillModel(participation_cap=1.0, min_tick=0.0)
    r = m.fill_limit(True, limit_price=100.0, bar_open=100.0,
                     bar_high=101.0, bar_low=99.0, qty=500.0, bar_volume=1000.0)
    assert r.filled_qty == approx(500.0)   # cap==1.0 -> no proxy capping
    assert r.capped is False


def test_fill_limit_no_volume_fills_full_request():
    m = FillModel(participation_cap=0.1, min_tick=0.0)
    # bar_volume None -> assume enough traded, fill full request
    r = m.fill_limit(True, limit_price=100.0, bar_open=100.0,
                     bar_high=101.0, bar_low=99.0, qty=500.0, bar_volume=None)
    assert r.filled_qty == approx(500.0)
    assert r.capped is False


# --------------------------------------------------------------------------
# fill_limit(): FIFO queue_pos path with a real QueuePosition
# --------------------------------------------------------------------------
def test_fill_limit_queue_pos_deep_queue_no_fill():
    m = FillModel(participation_cap=1.0, min_tick=0.0)
    # front=1000 ahead; traded_at_level = 1.0*100 = 100 < 1000 -> 0 fills
    q = QueuePosition(front=1000.0)
    r = m.fill_limit(True, limit_price=100.0, bar_open=100.0,
                     bar_high=101.0, bar_low=99.0, qty=10.0,
                     bar_volume=100.0, queue_pos=q)
    assert r.filled_qty == 0.0
    assert r.capped is True            # filled (0) < qty (10)
    # queue advanced by the 100 traded
    assert q.front == approx(900.0)


def test_fill_limit_queue_pos_partial_then_full():
    m = FillModel(participation_cap=1.0, min_tick=0.0)
    q = QueuePosition(front=50.0)
    # bar1: traded 100 -> eat 50 of front, 50 leftover fills (qty 10) -> filled 10
    r1 = m.fill_limit(True, limit_price=100.0, bar_open=100.0,
                      bar_high=101.0, bar_low=99.0, qty=10.0,
                      bar_volume=100.0, queue_pos=q)
    # fillable from queue = 100-50 = 50, min(qty=10, 50) = 10
    assert r1.filled_qty == approx(10.0)
    assert r1.capped is False          # 10 == qty
    assert q.front == approx(0.0)


def test_fill_limit_queue_pos_exact_front_consumption():
    m = FillModel(participation_cap=1.0, min_tick=0.0)
    q = QueuePosition(front=80.0)
    # traded 100 -> eat 80, 20 leftover; request 30 -> min(30,20)=20 filled, capped
    r = m.fill_limit(True, limit_price=100.0, bar_open=100.0,
                     bar_high=101.0, bar_low=99.0, qty=30.0,
                     bar_volume=100.0, queue_pos=q)
    assert r.filled_qty == approx(20.0)
    assert r.capped is True
    assert q.front == approx(0.0)


def test_fill_limit_queue_pos_respects_participation_cap():
    # traded_at_level uses participation_cap * bar_volume
    m = FillModel(participation_cap=0.5, min_tick=0.0)
    q = QueuePosition(front=0.0)   # at front, nothing ahead
    # traded_at_level = 0.5 * 100 = 50 ; fillable = 50 ; qty 10 -> 10 filled
    r = m.fill_limit(True, limit_price=100.0, bar_open=100.0,
                     bar_high=101.0, bar_low=99.0, qty=10.0,
                     bar_volume=100.0, queue_pos=q)
    assert r.filled_qty == approx(10.0)
    assert r.capped is False


# --------------------------------------------------------------------------
# QueuePosition primitive behavior (used by fill_limit) — direct invariants
# --------------------------------------------------------------------------
def test_queue_position_at_front_initial():
    q = QueuePosition(front=0.0)
    assert q.at_front is True
    q2 = QueuePosition(front=5.0)
    assert q2.at_front is False


def test_queue_position_negative_front_clamped():
    q = QueuePosition(front=-100.0)
    assert q.front0 == 0.0
    assert q.at_front is True


def test_queue_position_fifo_trade_eats_then_fills():
    q = QueuePosition(front=30.0)
    filled = q.advance(traded_qty=100.0)
    assert filled == approx(70.0)      # 100 - eat(30)
    assert q.front == approx(0.0)
    assert q.filled == approx(70.0)


def test_queue_position_cancellation_advances_no_fill():
    # PowerProbQueueFunc(2): front=10,back=0 -> prob_ahead = 1.0
    q = QueuePosition(front=10.0, prob_func=PowerProbQueueFunc(2.0))
    filled = q.advance(depth_reduction=4.0, back=0.0)
    assert filled == 0.0               # cancellation never fills
    assert q.front == approx(6.0)      # advanced by prob(1.0)*4 = 4


def test_queue_position_cancellation_with_back_partial_advance():
    # front=back=10 -> prob_ahead = 10^2/(10^2+10^2)=0.5
    q = QueuePosition(front=10.0, prob_func=PowerProbQueueFunc(2.0))
    q.advance(depth_reduction=4.0, back=10.0)
    assert q.front == approx(10.0 - 0.5 * 4.0)   # 8.0


def test_queue_position_negative_traded_clamped():
    q = QueuePosition(front=10.0)
    filled = q.advance(traded_qty=-50.0)
    assert filled == 0.0
    assert q.front == approx(10.0)


def test_queue_position_accumulates_filled_across_bars():
    q = QueuePosition(front=0.0)
    f1 = q.advance(traded_qty=5.0)
    f2 = q.advance(traded_qty=7.0)
    assert f1 == approx(5.0)
    assert f2 == approx(7.0)
    assert q.filled == approx(12.0)


# --------------------------------------------------------------------------
# PowerProbQueueFunc edge cases (documented monotonicity / boundaries)
# --------------------------------------------------------------------------
def test_prob_func_front_zero_gives_zero():
    f = PowerProbQueueFunc(2.0)
    assert f(0.0, 10.0) == 0.0


def test_prob_func_back_zero_gives_one():
    f = PowerProbQueueFunc(2.0)
    assert f(10.0, 0.0) == approx(1.0)


def test_prob_func_both_zero_symmetric():
    f = PowerProbQueueFunc(2.0)
    assert f(0.0, 0.0) == approx(0.5)


def test_prob_func_negative_inputs_clamped():
    f = PowerProbQueueFunc(2.0)
    # negative front clamped to 0 -> behaves like front=0 -> 0
    assert f(-5.0, 10.0) == 0.0


@pytest.mark.parametrize("n", [1.0, 2.0, 3.0])
def test_prob_func_monotone_in_front(n):
    f = PowerProbQueueFunc(n)
    lo = f(5.0, 10.0)
    hi = f(50.0, 10.0)
    assert hi > lo            # more size ahead -> higher prob it's ahead
    assert 0.0 <= lo <= 1.0
    assert 0.0 <= hi <= 1.0
