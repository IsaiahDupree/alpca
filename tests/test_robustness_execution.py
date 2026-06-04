"""
Robustness / edge-case tests for the OFFLINE execution + microstructure logic.

Covers (pure, deterministic, no network/mocks):
  - alpca.execution.fills.FillModel           (market + limit fills)
  - alpca.execution.queue_prob                (prob funcs + FIFO QueuePosition)
  - alpca.risk.risk_engine.RiskEngine         (pre-trade gates)
  - alpca.strategies.microstructure           (microprice / tilt / signal kernels)
  - alpca.strategies.order_flow.ofi_event     (L1 OFI increment)

Each parametrized case is one test. We feed malformed / extreme inputs and assert
GRACEFUL handling (finite, sane, non-adversely-wrong output OR the module's
documented guard) — never an unhandled crash — EXCEPT where the real source today
raises on a degenerate input; those cases assert the ACTUAL current behavior and
are documented as known guard gaps (see real_bug_found in the run report).
"""

from __future__ import annotations

import math

import pytest

from alpca.execution.fills import FillModel, FillResult
from alpca.execution.queue_prob import (
    QueuePosition,
    PROB_FUNCS,
    PowerProbQueueFunc,
    LogProbQueueFunc,
    SqrtProbQueueFunc,
    power_prob,
    _combine,
)
from alpca.config import RiskConfig
from alpca.risk.risk_engine import RiskEngine, Position, RiskDecision
from alpca.execution.order import Order, Side
from alpca.strategies.microstructure import (
    microprice,
    microprice_tilt,
    microprice_signal,
)
from alpca.strategies.order_flow import ofi_event


# --------------------------------------------------------------------------- helpers
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def _mk_order(side=Side.BUY, qty=10.0, price=100.0, symbol="SPY"):
    return Order(symbol=symbol, side=side, qty=qty, intended_price=price)


def _fixed_clock():
    """A monotonic clock we control explicitly (no wall-clock dependence)."""
    state = {"t": 1000.0}

    def now():
        return state["t"]

    return state, now


# =========================================================================== FillModel.fill
# ----- normal / well-formed: invariants ------------------------------------
@pytest.mark.parametrize("side_buy", [True, False])
def test_fill_market_basic_direction_and_finiteness(side_buy):
    m = FillModel()  # half_spread=1, impact=8, cap=1.0, min_tick=0.01
    ref = 100.0
    r = m.fill(side_buy, ref, qty=100.0, bar_volume=10_000.0)
    assert isinstance(r, FillResult)
    assert _finite(r.price) and _finite(r.slippage_bps) and _finite(r.filled_qty)
    assert r.filled_qty == 100.0  # cap=1.0 -> no partial
    assert r.capped is False
    # adverse: buy fills at/above ref, sell at/below ref
    if side_buy:
        assert r.price >= ref
    else:
        assert r.price <= ref
    assert r.slippage_bps > 0  # always adverse vs ref for a market fill


def test_fill_flat_model_reproduces_flat_bps():
    m = FillModel.flat(5.0)  # half_spread=5, no impact, no cap, no tick
    r = m.fill(True, 200.0, qty=10.0, bar_volume=1_000.0)
    # 5 bps adverse, no impact term, no rounding
    assert r.slippage_bps == pytest.approx(5.0)
    assert r.price == pytest.approx(200.0 * (1 + 5.0 / 10_000.0))
    assert r.capped is False


# ----- degenerate ref/qty that hit the documented guard (ref<=0 or qty<=0) --
@pytest.mark.parametrize(
    "ref,qty",
    [
        (0.0, 100.0),      # zero ref
        (-50.0, 100.0),    # negative ref
        (100.0, 0.0),      # zero qty
        (100.0, -5.0),     # negative qty
        (-1.0, -1.0),      # both negative
    ],
)
def test_fill_nonpositive_ref_or_qty_returns_zero_fill(ref, qty):
    m = FillModel()
    r = m.fill(True, ref, qty=qty, bar_volume=1_000.0)
    # documented guard: no fill, ref echoed back, zero slippage, not capped
    assert r.filled_qty == 0.0
    assert r.price == ref
    assert r.slippage_bps == 0.0
    assert r.capped is False


# ----- NaN / inf ref or qty: now GUARDED gracefully ------------------------
# Source fix: fill() guards non-finite ref_price / qty (NaN or inf) the same way
# it guards non-positive inputs -> a no-fill FillResult with a finite price.
# Previously these slipped past `ref<=0 or qty<=0` and crashed in round() for the
# default (min_tick>0) model; now they return gracefully instead of raising.
@pytest.mark.parametrize(
    "ref,qty",
    [
        (float("nan"), 100.0),     # NaN ref
        (float("inf"), 100.0),     # inf ref
        (100.0, float("nan")),     # NaN qty
    ],
)
def test_fill_nonfinite_inputs_default_tick_graceful(ref, qty):
    m = FillModel()  # min_tick=0.01 (default-tick model)
    r = m.fill(True, ref, qty=qty, bar_volume=1_000.0)  # no raise
    assert isinstance(r, FillResult)
    assert r.filled_qty == 0.0
    assert r.slippage_bps == 0.0
    assert r.capped is False
    assert _finite(r.price)  # finite price, no NaN/inf propagation


@pytest.mark.parametrize(
    "ref,qty",
    [
        (float("nan"), 100.0),  # NaN ref -> no fill, finite price
        (float("inf"), 100.0),  # inf ref -> no fill, finite price
        (100.0, float("nan")),  # NaN qty -> no fill, finite price
    ],
)
def test_fill_nonfinite_inputs_flat_model_no_crash(ref, qty):
    # flat model (min_tick=0) also guards non-finite inputs gracefully
    m = FillModel.flat(5.0)
    r = m.fill(True, ref, qty=qty)  # no bar_volume -> no impact branch
    assert isinstance(r, FillResult)
    assert r.filled_qty == 0.0
    assert r.slippage_bps == 0.0
    assert r.capped is False
    assert _finite(r.price)  # finite price, no NaN/inf propagation


# ----- volume cap / participation > 1 -------------------------------------
def test_fill_volume_cap_partial_and_flag():
    m = FillModel(half_spread_bps=1.0, impact_coef_bps=8.0,
                  participation_cap=0.1, min_tick=0.01)
    r = m.fill(True, 100.0, qty=1_000.0, bar_volume=2_000.0)  # cap = 200
    assert r.capped is True
    assert r.filled_qty == pytest.approx(200.0)
    assert r.filled_qty < 1_000.0
    assert _finite(r.price) and r.price >= 100.0


def test_fill_gigantic_qty_no_cap_fills_all_finite():
    # cap defaults to 1.0 (no cap): participation 1e12/1000 huge but finite sqrt
    m = FillModel()
    r = m.fill(True, 100.0, qty=1e12, bar_volume=1_000.0)
    assert r.filled_qty == 1e12
    assert r.capped is False
    assert _finite(r.price) and _finite(r.slippage_bps)
    assert r.slippage_bps > 0  # impact dominates but stays finite


@pytest.mark.parametrize("bar_volume", [None, 0.0, -10.0])
def test_fill_empty_or_zero_bar_volume_no_impact_no_cap(bar_volume):
    # impact + cap branches both require bar_volume > 0; otherwise plain spread
    m = FillModel(half_spread_bps=2.0, impact_coef_bps=8.0,
                  participation_cap=0.1, min_tick=0.0)
    r = m.fill(True, 100.0, qty=5_000.0, bar_volume=bar_volume)
    assert r.filled_qty == 5_000.0       # no cap applied
    assert r.capped is False
    assert r.slippage_bps == pytest.approx(2.0)  # half-spread only, no impact


def test_fill_participation_cap_unbounded_still_caps_to_volume():
    # cap > 1.0 is allowed by the dataclass but only triggers when cap<1.0;
    # cap>=1.0 means "take all" -> never capped even if qty >> volume
    m = FillModel(half_spread_bps=1.0, impact_coef_bps=0.0,
                  participation_cap=1.0, min_tick=0.0)
    r = m.fill(True, 100.0, qty=1e9, bar_volume=10.0)
    assert r.capped is False
    assert r.filled_qty == 1e9


# =========================================================================== FillModel.fill_limit
@pytest.mark.parametrize(
    "side_buy,limit,bar_low,bar_high,reached",
    [
        (True, 100.0, 98.0, 101.0, True),    # buy: low<=limit -> fills
        (True, 100.0, 100.5, 102.0, False),  # buy: never came down -> rest
        (False, 100.0, 99.0, 100.5, True),   # sell: high>=limit -> fills
        (False, 100.0, 97.0, 99.5, False),   # sell: never came up -> rest
    ],
)
def test_fill_limit_through_trade_test(side_buy, limit, bar_low, bar_high, reached):
    m = FillModel()
    bar_open = 99.5
    r = m.fill_limit(side_buy, limit, bar_open, bar_high, bar_low, qty=50.0,
                     bar_volume=10_000.0)
    if reached:
        assert r.filled_qty == 50.0
        # never pay worse than the limit: slippage favorable or zero
        assert r.slippage_bps <= 1e-9
    else:
        assert r.filled_qty == 0.0
        assert r.price == limit
        assert r.slippage_bps == 0.0


def test_fill_limit_price_improvement_on_gap_through():
    # buy limit 100, bar opens at 99 (gapped below): fill at the better open price
    m = FillModel(min_tick=0.0)
    r = m.fill_limit(True, 100.0, bar_open=99.0, bar_high=99.5, bar_low=98.0,
                     qty=10.0, bar_volume=1_000.0)
    assert r.filled_qty == 10.0
    assert r.price == pytest.approx(99.0)        # improved
    assert r.slippage_bps < 0                    # favorable (negative)


@pytest.mark.parametrize("limit,qty", [(0.0, 10.0), (-5.0, 10.0), (100.0, 0.0), (100.0, -3.0)])
def test_fill_limit_nonpositive_inputs_guarded(limit, qty):
    m = FillModel()
    r = m.fill_limit(True, limit, 99.0, 101.0, 98.0, qty, bar_volume=1_000.0)
    assert r.filled_qty == 0.0
    assert r.price == limit
    assert r.slippage_bps == 0.0
    assert r.capped is False


@pytest.mark.parametrize("bar_volume", [None, 0.0])
def test_fill_limit_no_volume_context_fills_request(bar_volume):
    # with no usable volume, assume enough traded to fill the whole request
    m = FillModel(participation_cap=0.1)
    r = m.fill_limit(True, 100.0, 99.0, 101.0, 98.0, qty=500.0, bar_volume=bar_volume)
    assert r.filled_qty == 500.0
    assert r.capped is False


def test_fill_limit_volume_cap_proxy_partial():
    m = FillModel(participation_cap=0.1, min_tick=0.0)
    r = m.fill_limit(True, 100.0, 99.0, 101.0, 98.0, qty=1_000.0, bar_volume=2_000.0)
    # only 0.1*2000 = 200 can fill at our level
    assert r.capped is True
    assert r.filled_qty == pytest.approx(200.0)


def test_fill_limit_with_queue_pos_fifo_across_bars():
    # 500 shares ahead; first bar trades 300 (no fill), second trades 400 (fills 100)
    m = FillModel(participation_cap=1.0, min_tick=0.0)
    qp = QueuePosition(500.0)
    r1 = m.fill_limit(True, 100.0, 99.0, 101.0, 98.0, qty=100.0,
                      bar_volume=300.0, queue_pos=qp)
    assert r1.filled_qty == 0.0
    assert r1.capped is True
    assert qp.front == pytest.approx(200.0)
    r2 = m.fill_limit(True, 100.0, 99.0, 101.0, 98.0, qty=100.0,
                      bar_volume=400.0, queue_pos=qp)
    assert r2.filled_qty == pytest.approx(100.0)
    assert r2.capped is False
    assert qp.front == pytest.approx(0.0)


# =========================================================================== queue_prob: _combine + prob funcs
@pytest.mark.parametrize(
    "ffront,fback,expected",
    [
        (10.0, 0.0, 1.0),     # everything ahead
        (0.0, 10.0, 0.0),     # nothing ahead
        (0.0, 0.0, 0.5),      # both empty -> symmetric
        (5.0, 5.0, 0.5),      # equal
        (30.0, 10.0, 0.75),   # 30/40
    ],
)
def test_combine_edge_and_normal(ffront, fback, expected):
    assert _combine(ffront, fback) == pytest.approx(expected)


@pytest.mark.parametrize("name", list(PROB_FUNCS.keys()))
def test_prob_funcs_bounded_and_edge(name):
    f = PROB_FUNCS[name]
    # output always in [0,1]
    for front, back in [(0.0, 0.0), (0.0, 100.0), (100.0, 0.0), (37.0, 13.0), (1e9, 1.0)]:
        p = f(front, back)
        assert 0.0 <= p <= 1.0
        assert _finite(p)
    # documented edge behaviors
    assert f(0.0, 50.0) == 0.0     # front=0 -> nothing ahead
    assert f(50.0, 0.0) == 1.0     # back=0 -> everything ahead


@pytest.mark.parametrize(
    "name",
    ["power1", "power2", "power3", "log", "sqrt"],
)
def test_prob_funcs_negative_inputs_clamped(name):
    f = PROB_FUNCS[name]
    # negative front/back are clamped to 0 inside each func -> still in [0,1]
    p = f(-10.0, 50.0)
    assert p == 0.0  # clamped front=0 -> nothing ahead
    p2 = f(50.0, -10.0)
    assert p2 == 1.0  # clamped back=0 -> everything ahead


def test_power_prob_monotone_in_n():
    # higher n weights a larger queue-ahead more strongly when front>back
    front, back = 30.0, 10.0
    p1 = PowerProbQueueFunc(1.0)(front, back)
    p2 = PowerProbQueueFunc(2.0)(front, back)
    p3 = PowerProbQueueFunc(3.0)(front, back)
    assert p1 < p2 < p3
    assert power_prob(2.0)(front, back) == pytest.approx(p2)


def test_prob_func_variants_ordering():
    # at front>back: power(n=2) >= sqrt >= log (gentleness order documented)
    front, back = 30.0, 10.0
    pw = PowerProbQueueFunc(2.0)(front, back)
    sq = SqrtProbQueueFunc()(front, back)
    lg = LogProbQueueFunc()(front, back)
    assert pw >= sq >= lg


# =========================================================================== QueuePosition
def test_queue_advance_zero_is_noop():
    q = QueuePosition(100.0)
    filled = q.advance(0.0)
    assert filled == 0.0
    assert q.front == 100.0
    assert q.filled == 0.0
    assert q.at_front is False


def test_queue_advance_huge_trade_fills_overflow():
    q = QueuePosition(100.0)
    filled = q.advance(1e9)
    assert filled == pytest.approx(1e9 - 100.0)  # ate 100 ahead, rest fills us
    assert q.front == 0.0
    assert q.at_front is True
    assert q.filled == pytest.approx(1e9 - 100.0)


def test_queue_front_zero_fills_immediately():
    q = QueuePosition(0.0)
    assert q.at_front is True
    filled = q.advance(50.0)
    assert filled == 50.0  # nothing ahead -> all trade volume fills us


@pytest.mark.parametrize("bad_front", [-100.0, float("-inf") if False else -1.0])
def test_queue_negative_front_clamped_to_zero(bad_front):
    q = QueuePosition(bad_front)  # __init__ does max(0.0, front)
    assert q.front0 == 0.0
    assert q.front == 0.0
    assert q.at_front is True


def test_queue_negative_traded_qty_is_noop():
    q = QueuePosition(50.0)
    filled = q.advance(traded_qty=-100.0)  # max(0.0, ...) -> 0
    assert filled == 0.0
    assert q.front == 50.0


def test_queue_cancellation_advances_without_filling():
    # depth_reduction shrinks the queue ahead but never fills us
    q = QueuePosition(100.0)
    # back=0 -> prob_ahead=1.0 -> full reduction applied to front
    filled = q.advance(traded_qty=0.0, depth_reduction=40.0, back=0.0)
    assert filled == 0.0
    assert q.front == pytest.approx(60.0)
    assert q.filled == 0.0


def test_queue_huge_cancellation_floors_front_at_zero():
    q = QueuePosition(100.0)
    filled = q.advance(traded_qty=0.0, depth_reduction=1e12, back=0.0)
    assert filled == 0.0
    assert q.front == 0.0  # floored, never negative
    assert q.at_front is True


def test_queue_cancellation_then_trade_combined():
    q = QueuePosition(100.0)
    # cancel 50 ahead (back=0 -> full), leaving 50; then trade 80 -> eat 50, fill 30
    filled = q.advance(traded_qty=80.0, depth_reduction=50.0, back=0.0)
    assert q.front == 0.0
    assert filled == pytest.approx(30.0)
    assert q.filled == pytest.approx(30.0)


def test_queue_idempotent_fill_accumulation():
    q = QueuePosition(10.0)
    q.advance(5.0)   # eats 5 ahead, no fill
    q.advance(5.0)   # eats remaining 5 ahead, no fill
    q.advance(7.0)   # front=0 -> fills 7
    assert q.front == 0.0
    assert q.filled == pytest.approx(7.0)


# =========================================================================== RiskEngine
def _engine(cfg=None, **kw):
    state, now = _fixed_clock()
    eng = RiskEngine(cfg or RiskConfig(), now=now, **kw)
    return eng, state


def test_risk_basic_allow():
    eng, _ = _engine()
    d = eng.check(_mk_order(), equity=1_000_000.0)
    assert bool(d) is True
    assert d.code == "OK"


@pytest.mark.parametrize("positions", [None, {}])
def test_risk_none_or_empty_positions_ok(positions):
    eng, _ = _engine()
    d = eng.check(_mk_order(), equity=1_000_000.0, positions=positions)
    assert d.allowed is True


@pytest.mark.parametrize("qty", [0.0, -5.0, -1e-3])
def test_risk_nonpositive_qty_rejected(qty):
    eng, _ = _engine()
    d = eng.check(_mk_order(qty=qty), equity=1_000_000.0)
    assert d.allowed is False
    assert d.code == "BAD_QTY"


def test_risk_missing_ref_price_rejected():
    eng, _ = _engine()
    o = Order(symbol="SPY", side=Side.BUY, qty=10.0)  # no intended/limit price
    d = eng.check(o, equity=1_000_000.0)  # no ref_price, no positions
    assert d.allowed is False
    assert d.code == "NO_PRICE"


@pytest.mark.parametrize("price", [0.0, -10.0])
def test_risk_nonpositive_ref_price_rejected(price):
    eng, _ = _engine()
    d = eng.check(_mk_order(price=None), equity=1_000_000.0, ref_price=price)
    assert d.allowed is False
    assert d.code == "NO_PRICE"


def test_risk_ref_price_from_position_avg_when_no_other():
    # documented fallback: position avg_price used as ref when nothing else given
    eng, _ = _engine()
    o = Order(symbol="SPY", side=Side.BUY, qty=1.0)
    pos = {"SPY": Position("SPY", qty=5.0, avg_price=123.0)}
    d = eng.check(o, equity=1_000_000.0, positions=pos)
    assert d.allowed is True  # 1*123 well under caps


def test_risk_nan_ref_price_rejected():
    # Source fix: a non-finite (NaN) reference price is no longer treated as a
    # valid price. It is rejected up front rather than slipping past every cap.
    eng, _ = _engine()
    d = eng.check(_mk_order(price=float("nan")), equity=1_000_000.0)
    assert d.allowed is False
    assert d.code == "NO_PRICE"


def test_risk_huge_notional_rejected():
    eng, _ = _engine()
    d = eng.check(_mk_order(qty=1e9, price=100.0), equity=1e15, cash=1e18)
    assert d.allowed is False
    assert d.code == "MAX_ORDER_NOTIONAL"


def test_risk_halt_blocks_everything():
    eng, _ = _engine()
    eng.halt("manual stop")
    assert eng.halted is True
    d = eng.check(_mk_order(), equity=1_000_000.0)
    assert d.allowed is False
    assert d.code == "HALTED"
    eng.resume()
    assert eng.halted is False
    assert eng.check(_mk_order(), equity=1_000_000.0).allowed is True


def test_risk_forbidden_symbol():
    eng, _ = _engine(forbidden_symbols=["tsla"])  # lowercased input
    d = eng.check(_mk_order(symbol="TSLA"), equity=1_000_000.0)
    assert d.allowed is False
    assert d.code == "FORBIDDEN"


def test_risk_insufficient_buying_power():
    eng, _ = _engine(RiskConfig(max_order_notional=1e12))
    # buy 100 * 100 = 10_000 notional but only 5_000 cash
    d = eng.check(_mk_order(qty=100.0, price=100.0), equity=1e9, cash=5_000.0)
    assert d.allowed is False
    assert d.code == "INSUFFICIENT_BUYING_POWER"


def test_risk_short_not_allowed_when_flat():
    eng, _ = _engine(RiskConfig(max_order_notional=1e12))
    d = eng.check(_mk_order(side=Side.SELL, qty=10.0, price=100.0), equity=1e9)
    assert d.allowed is False
    assert d.code == "SHORT_NOT_ALLOWED"


def test_risk_sell_reducing_long_allowed():
    eng, _ = _engine(RiskConfig(max_order_notional=1e12))
    pos = {"SPY": Position("SPY", qty=50.0, avg_price=100.0)}
    d = eng.check(_mk_order(side=Side.SELL, qty=10.0, price=100.0),
                  equity=1e9, positions=pos)
    assert d.allowed is True  # 50 - 10 = +40 still long


def test_risk_rate_limit_sliding_window():
    cfg = RiskConfig(max_order_notional=1e12, max_orders_per_min=3)
    state, now = _fixed_clock()
    eng = RiskEngine(cfg, now=now)
    for _ in range(3):
        eng.record_submission()
    d = eng.check(_mk_order(price=1.0, qty=1.0), equity=1e9)
    assert d.allowed is False
    assert d.code == "RATE_LIMIT"
    # advance the clock past the 60s window -> entries pruned -> allowed again
    state["t"] += 61.0
    d2 = eng.check(_mk_order(price=1.0, qty=1.0), equity=1e9)
    assert d2.allowed is True


def test_risk_daily_loss_autohalt():
    cfg = RiskConfig(max_order_notional=1e12, daily_loss_pct=0.02)
    eng, _ = _engine(cfg, day_start_equity=100_000.0)
    # equity below floor (98_000) -> halt + reject
    d = eng.check(_mk_order(price=1.0, qty=1.0), equity=97_000.0)
    assert d.allowed is False
    assert d.code == "DAILY_LOSS"
    assert eng.halted is True


def test_risk_max_open_positions():
    cfg = RiskConfig(max_order_notional=1e12, max_open_positions=2)
    eng, _ = _engine(cfg)
    pos = {
        "AAA": Position("AAA", 1.0, 10.0),
        "BBB": Position("BBB", 1.0, 10.0),
    }
    # opening a NEW symbol when already at cap
    d = eng.check(_mk_order(symbol="CCC", side=Side.BUY, qty=1.0, price=10.0),
                  equity=1e9, positions=pos)
    assert d.allowed is False
    assert d.code == "MAX_POSITIONS"


def test_risk_concentration_cap():
    cfg = RiskConfig(max_order_notional=1e12, max_concentration_pct=0.25)
    eng, _ = _engine(cfg)
    # 100 * 1000 = 100k projected vs 200k equity = 50% > 25%
    d = eng.check(_mk_order(qty=100.0, price=1_000.0), equity=200_000.0)
    assert d.allowed is False
    assert d.code == "CONCENTRATION"


def test_risk_decision_truthiness():
    assert bool(RiskDecision(True)) is True
    assert bool(RiskDecision(False, "X")) is False


# =========================================================================== microprice kernels
@pytest.mark.parametrize(
    "bid,ask,bs,az,expected",
    [
        (99.0, 101.0, 100.0, 100.0, 100.0),      # balanced -> mid
        (99.0, 101.0, 300.0, 100.0, 100.5),      # heavier bid -> tilts up toward ask
    ],
)
def test_microprice_value(bid, ask, bs, az, expected):
    mp = microprice(bid, ask, bs, az)
    assert mp == pytest.approx(expected)


@pytest.mark.parametrize(
    "bid,ask,bs,az",
    [
        (None, 101.0, 100.0, 100.0),   # missing field
        (99.0, None, 100.0, 100.0),
        (99.0, 101.0, None, 100.0),
        (99.0, 101.0, 100.0, None),
        (99.0, 101.0, 0.0, 0.0),       # zero total size -> degenerate
        (99.0, 101.0, -50.0, 10.0),    # negative total
    ],
)
def test_microprice_degenerate_returns_none(bid, ask, bs, az):
    assert microprice(bid, ask, bs, az) is None


@pytest.mark.parametrize(
    "bid,ask,bs,az",
    [
        (100.0, 100.0, 50.0, 50.0),    # locked (bid==ask) -> half<=0
        (101.0, 99.0, 50.0, 50.0),     # crossed (bid>ask) -> half<0
        (99.0, 101.0, 0.0, 0.0),       # no size -> microprice None
        (None, 101.0, 10.0, 10.0),     # missing field
    ],
)
def test_microprice_tilt_degenerate_returns_none(bid, ask, bs, az):
    assert microprice_tilt(bid, ask, bs, az) is None


@pytest.mark.parametrize(
    "bs,az,sign",
    [
        (900.0, 100.0, 1.0),   # heavy bid -> tilt > 0 (toward ask)
        (100.0, 900.0, -1.0),  # heavy ask -> tilt < 0
    ],
)
def test_microprice_tilt_bounded_and_signed(bs, az, sign):
    t = microprice_tilt(99.0, 101.0, bs, az)
    assert t is not None
    assert -1.0 <= t <= 1.0
    assert math.copysign(1.0, t) == sign


def test_microprice_tilt_balanced_is_zero():
    assert microprice_tilt(99.0, 101.0, 100.0, 100.0) == pytest.approx(0.0)


@pytest.mark.parametrize(
    "bs,az,k,expected",
    [
        (100.0, 100.0, 0.5, "flat"),   # balanced inside deadband
        (900.0, 100.0, 0.0, "bull"),   # any positive tilt with k=0
        (100.0, 900.0, 0.0, "bear"),
        (110.0, 100.0, 0.5, "flat"),   # small tilt under k=0.5 deadband
    ],
)
def test_microprice_signal_deadband(bs, az, k, expected):
    assert microprice_signal(99.0, 101.0, bs, az, k=k) == expected


@pytest.mark.parametrize(
    "bid,ask,bs,az",
    [
        (None, 101.0, 100.0, 100.0),   # no usable quote
        (100.0, 100.0, 50.0, 50.0),    # locked
        (99.0, 101.0, 0.0, 0.0),       # zero size
    ],
)
def test_microprice_signal_no_quote_returns_none(bid, ask, bs, az):
    assert microprice_signal(bid, ask, bs, az, k=0.0) is None


# =========================================================================== OFI kernel
@pytest.mark.parametrize(
    "args,expected",
    [
        # bid up, ask up: dW=bid_size(50), dV=-prev_ask_size(-60) -> e=50-(-60)=110
        ((101, 50, 103, 40, 100, 30, 102, 60), 110),
        # bid unchanged, ask unchanged: dW=50-30=20, dV=40-60=-20 -> e=20-(-20)=40
        ((100, 50, 102, 40, 100, 30, 102, 60), 40),
        # bid down, ask down: dW=-prev_bid(-30), dV=ask_size(40) -> e=-30-40=-70
        ((99, 50, 101, 40, 100, 30, 102, 60), -70),
    ],
)
def test_ofi_event_branches(args, expected):
    assert ofi_event(*args) == expected


def test_ofi_event_finite_and_symmetric_zero():
    # identical snapshots: bid unchanged (50-50=0), ask unchanged (40-40=0) -> 0
    e = ofi_event(100, 50, 102, 40, 100, 50, 102, 40)
    assert e == 0
    assert _finite(e)


def test_ofi_event_extreme_sizes_finite():
    e = ofi_event(101, 1e9, 103, 1e9, 100, 1.0, 102, 1.0)
    assert _finite(e)
    # bid up -> dW=1e9 ; ask up -> dV=-1 ; e = 1e9 + 1
    assert e == pytest.approx(1e9 + 1.0)
