"""
Cross-module property/invariant tests: fill model, FIFO queue, microstructure
kernels, risk-engine gates, and the NYSE calendar. Parametrized sweeps; each case
is a distinct check of a real function.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from alpca.config import RiskConfig
from alpca.data.calendar import (
    Session,
    calendar_covers,
    is_regular_hours,
    is_tradeable,
    session_at,
    session_date,
)
from alpca.execution.fills import FillModel
from alpca.execution.order import Order, Side
from alpca.execution.queue_prob import PROB_FUNCS, QueuePosition
from alpca.risk.risk_engine import Position, RiskEngine
from alpca.strategies.microstructure import microprice, microprice_signal, microprice_tilt
from alpca.strategies.order_flow import ofi_event

ET = ZoneInfo("America/New_York")


# =========================================================== fill model (.fill)
_FM = FillModel(half_spread_bps=2.0, impact_coef_bps=8.0, participation_cap=0.1, min_tick=0.01)


@pytest.mark.parametrize("ref", [10.0, 100.0, 575.25])
@pytest.mark.parametrize("qty", [1.0, 100.0])
def test_buy_fill_is_adverse_up(ref, qty):
    r = _FM.fill(True, ref, qty, bar_volume=1e6)
    assert r.price >= ref
    assert r.slippage_bps >= _FM.half_spread_bps - 1e-9


@pytest.mark.parametrize("ref", [10.0, 100.0, 575.25])
@pytest.mark.parametrize("qty", [1.0, 100.0])
def test_sell_fill_is_adverse_down(ref, qty):
    r = _FM.fill(False, ref, qty, bar_volume=1e6)
    assert r.price <= ref


def test_flat_model_reproduces_flat_bps():
    fm = FillModel.flat(5.0)
    r = fm.fill(True, 100.0, 10.0)
    assert r.slippage_bps == pytest.approx(5.0)
    assert r.price == pytest.approx(100.0 * 1.0005)


@pytest.mark.parametrize("vol,cap,qty,exp_filled", [
    (1000.0, 0.1, 500.0, 100.0),   # capped to 10% of 1000
    (1000.0, 0.1, 50.0, 50.0),     # under cap -> full
    (2000.0, 0.25, 1000.0, 500.0),
])
def test_volume_cap_partial_fill(vol, cap, qty, exp_filled):
    fm = FillModel(half_spread_bps=1.0, impact_coef_bps=0.0, participation_cap=cap, min_tick=0.0)
    r = fm.fill(True, 100.0, qty, bar_volume=vol)
    assert r.filled_qty == pytest.approx(exp_filled)
    assert r.capped == (qty > cap * vol)


def test_impact_grows_with_participation():
    small = _FM.fill(True, 100.0, 100.0, bar_volume=1e6).slippage_bps
    big = _FM.fill(True, 100.0, 100_000.0, bar_volume=1e6).slippage_bps
    assert big > small


@pytest.mark.parametrize("tick", [0.01, 0.05])
def test_fill_price_rounds_to_tick(tick):
    fm = FillModel(half_spread_bps=3.3, impact_coef_bps=0.0, participation_cap=1.0, min_tick=tick)
    p = fm.fill(True, 100.0, 1.0).price
    assert abs((p / tick) - round(p / tick)) < 1e-9


# ===================================================== fill model (.fill_limit)
def test_buy_limit_fills_when_touched():
    r = _FM.fill_limit(True, 100.0, bar_open=100.0, bar_high=101.0, bar_low=99.5, qty=10.0)
    assert r.filled_qty == 10.0
    assert r.price <= 100.0          # never worse than the limit


def test_buy_limit_no_fill_when_not_touched():
    r = _FM.fill_limit(True, 99.0, bar_open=100.0, bar_high=101.0, bar_low=99.5, qty=10.0)
    assert r.filled_qty == 0.0       # low 99.5 never reached 99.0


def test_buy_limit_price_improvement_on_gap_down_open():
    # opens at 98 (below the 100 limit) -> fill at 98, favorable
    r = _FM.fill_limit(True, 100.0, bar_open=98.0, bar_high=99.0, bar_low=97.0, qty=10.0)
    assert r.price == pytest.approx(98.0)
    assert r.slippage_bps <= 0


def test_sell_limit_fills_when_high_reaches():
    r = _FM.fill_limit(False, 100.0, bar_open=100.0, bar_high=100.5, bar_low=99.0, qty=10.0)
    assert r.filled_qty == 10.0
    assert r.price >= 100.0


def test_sell_limit_no_fill_when_high_below():
    r = _FM.fill_limit(False, 101.0, bar_open=100.0, bar_high=100.5, bar_low=99.0, qty=10.0)
    assert r.filled_qty == 0.0


# ============================================================== FIFO queue model
@pytest.mark.parametrize("name", list(PROB_FUNCS))
@pytest.mark.parametrize("front,back", [(10.0, 5.0), (1.0, 100.0), (50.0, 50.0)])
def test_prob_func_in_unit_interval(name, front, back):
    p = PROB_FUNCS[name](front, back)
    assert 0.0 <= p <= 1.0


@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_prob_func_edges(name):
    f = PROB_FUNCS[name]
    assert f(10.0, 0.0) == pytest.approx(1.0)   # all ahead
    assert f(0.0, 10.0) == pytest.approx(0.0)   # none ahead


@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_prob_func_monotone_in_front(name):
    f = PROB_FUNCS[name]
    assert f(5.0, 10.0) <= f(50.0, 10.0)


def test_queue_fifo_eats_front_then_fills():
    q = QueuePosition(100.0, PROB_FUNCS["power2"])
    assert q.advance(traded_qty=30.0) == 0.0      # all 30 eats the queue ahead
    assert q.front == pytest.approx(70.0)
    filled = q.advance(traded_qty=100.0)          # 70 eats rest, 30 fills us
    assert filled == pytest.approx(30.0)
    assert q.at_front


def test_queue_cancellation_advances_without_filling():
    q = QueuePosition(100.0, PROB_FUNCS["power2"])
    filled = q.advance(depth_reduction=40.0, back=0.0)  # back=0 -> all ahead -> advance 40
    assert filled == 0.0
    assert q.front < 100.0


# ============================================================== microprice / tilt
_Q = [(100.0, 100.10, 300.0, 100.0), (100.0, 100.10, 100.0, 300.0),
      (50.0, 50.02, 500.0, 500.0), (10.0, 10.50, 10.0, 1000.0)]


@pytest.mark.parametrize("bid,ask,bs,az", _Q)
def test_microprice_within_quote(bid, ask, bs, az):
    mp = microprice(bid, ask, bs, az)
    assert bid <= mp <= ask


@pytest.mark.parametrize("bid,ask,bs,az", _Q)
def test_tilt_in_unit_band(bid, ask, bs, az):
    t = microprice_tilt(bid, ask, bs, az)
    assert -1.0 <= t <= 1.0


def test_tilt_sign_follows_size_imbalance():
    assert microprice_tilt(100.0, 100.1, 300.0, 100.0) > 0   # heavy bid -> up
    assert microprice_tilt(100.0, 100.1, 100.0, 300.0) < 0   # heavy ask -> down


@pytest.mark.parametrize("bad", [(None, 100.0, 1.0, 1.0), (100.0, None, 1.0, 1.0),
                                 (100.0, 100.0, 0.0, 0.0)])
def test_microprice_none_on_bad_quote(bad):
    assert microprice(*bad) is None


def test_tilt_none_on_locked_quote():
    assert microprice_tilt(100.0, 100.0, 100.0, 100.0) is None  # half-spread 0


@pytest.mark.parametrize("k,expect", [(0.0, "bull"), (0.9, "flat")])
def test_microprice_signal_deadband(k, expect):
    # heavy bid -> tilt ~ +0.5; k=0 -> bull, k=0.9 -> flat (below deadband)
    assert microprice_signal(100.0, 100.1, 300.0, 100.0, k=k) == expect


# ====================================================================== OFI event
def test_ofi_bid_rise_is_bullish():
    e = ofi_event(100.01, 200.0, 100.05, 200.0, 100.00, 200.0, 100.05, 200.0)
    assert e > 0


def test_ofi_ask_rise_is_bullish():
    # ask retreats upward (supply pulls back) -> bullish
    e = ofi_event(100.00, 200.0, 100.06, 200.0, 100.00, 200.0, 100.05, 200.0)
    assert e > 0


def test_ofi_bid_fall_is_bearish():
    e = ofi_event(99.99, 200.0, 100.05, 200.0, 100.00, 200.0, 100.05, 200.0)
    assert e < 0


def test_ofi_ask_fall_is_bearish():
    e = ofi_event(100.00, 200.0, 100.04, 200.0, 100.00, 200.0, 100.05, 200.0)
    assert e < 0


# ================================================================== risk engine
def _order(side, qty, sym="SPY"):
    return Order(symbol=sym, side=side, qty=qty)


def test_risk_allows_normal_buy():
    eng = RiskEngine(RiskConfig())
    d = eng.check(_order(Side.BUY, 10.0), equity=100_000, ref_price=100.0, cash=100_000)
    assert d.allowed and d.code == "OK"


def test_risk_blocks_bad_qty():
    d = RiskEngine(RiskConfig()).check(_order(Side.BUY, 0.0), equity=100_000, ref_price=100.0)
    assert not d.allowed and d.code == "BAD_QTY"


def test_risk_blocks_over_notional():
    eng = RiskEngine(RiskConfig(max_order_notional=1000.0))
    d = eng.check(_order(Side.BUY, 100.0), equity=1e6, ref_price=50.0)
    assert not d.allowed and d.code == "MAX_ORDER_NOTIONAL"


def test_risk_blocks_insufficient_buying_power():
    eng = RiskEngine(RiskConfig())
    d = eng.check(_order(Side.BUY, 10.0), equity=1e6, ref_price=50.0, cash=100.0)
    assert not d.allowed and d.code == "INSUFFICIENT_BUYING_POWER"


def test_risk_blocks_naked_short_by_default():
    eng = RiskEngine(RiskConfig())
    d = eng.check(_order(Side.SELL, 10.0), equity=1e6, ref_price=50.0)
    assert not d.allowed and d.code == "SHORT_NOT_ALLOWED"


def test_risk_allows_short_when_enabled():
    eng = RiskEngine(RiskConfig(allow_short=True))
    d = eng.check(_order(Side.SELL, 10.0), equity=1e6, ref_price=50.0)
    assert d.allowed


def test_risk_allows_sell_that_only_reduces_long():
    eng = RiskEngine(RiskConfig())
    pos = {"SPY": Position("SPY", 100.0, 50.0)}
    d = eng.check(_order(Side.SELL, 50.0), equity=1e6, positions=pos, ref_price=50.0)
    assert d.allowed


def test_risk_blocks_concentration():
    eng = RiskEngine(RiskConfig(max_concentration_pct=0.1))
    d = eng.check(_order(Side.BUY, 100.0), equity=10_000, ref_price=50.0, cash=1e9)
    assert not d.allowed and d.code == "CONCENTRATION"


def test_risk_blocks_max_positions():
    eng = RiskEngine(RiskConfig(max_open_positions=1))
    pos = {"AAA": Position("AAA", 10.0, 50.0)}
    d = eng.check(_order(Side.BUY, 1.0, sym="BBB"), equity=1e6, positions=pos, ref_price=50.0)
    assert not d.allowed and d.code == "MAX_POSITIONS"


def test_risk_blocks_forbidden_symbol():
    eng = RiskEngine(RiskConfig(), forbidden_symbols=["BAD"])
    d = eng.check(_order(Side.BUY, 1.0, sym="BAD"), equity=1e6, ref_price=50.0)
    assert not d.allowed and d.code == "FORBIDDEN"


def test_risk_blocks_when_halted():
    eng = RiskEngine(RiskConfig())
    eng.halt("test")
    d = eng.check(_order(Side.BUY, 1.0), equity=1e6, ref_price=50.0)
    assert not d.allowed and d.code == "HALTED"


def test_risk_daily_loss_halt():
    eng = RiskEngine(RiskConfig(daily_loss_pct=0.02), day_start_equity=100_000)
    d = eng.check(_order(Side.BUY, 1.0), equity=90_000, ref_price=50.0, cash=1e9)
    assert not d.allowed and d.code == "DAILY_LOSS"
    assert eng.halted


# =================================================================== NYSE calendar
def _ts(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=ET).timestamp()


@pytest.mark.parametrize("hh,mm,expect", [
    (9, 30, Session.REGULAR), (10, 0, Session.REGULAR), (15, 59, Session.REGULAR),
    (8, 0, Session.PRE_MARKET), (4, 0, Session.PRE_MARKET),
    (16, 30, Session.AFTER_HOURS), (19, 59, Session.AFTER_HOURS),
    (21, 0, Session.CLOSED), (2, 0, Session.CLOSED),
])
def test_session_classification_regular_weekday(hh, mm, expect):
    # 2026-06-02 is a normal Tuesday
    assert session_at(_ts(2026, 6, 2, hh, mm)) == expect


def test_weekend_is_closed():
    assert session_at(_ts(2026, 6, 6, 12, 0)) == Session.CLOSED_WEEKEND  # Saturday


def test_holiday_is_closed():
    # 2026-07-03 is an observed NYSE holiday
    assert session_at(_ts(2026, 7, 3, 12, 0)) == Session.CLOSED_HOLIDAY


def test_early_close_after_1pm_is_after_hours():
    # 2026-11-27 is a half-day (closes 13:00 ET)
    assert session_at(_ts(2026, 11, 27, 14, 0)) == Session.AFTER_HOURS
    assert session_at(_ts(2026, 11, 27, 12, 0)) == Session.REGULAR


@pytest.mark.parametrize("hh,reg", [(10, True), (8, False), (17, False)])
def test_is_regular_hours(hh, reg):
    assert is_regular_hours(_ts(2026, 6, 2, hh, 0)) is reg


def test_is_tradeable_respects_extended_flag():
    pre = _ts(2026, 6, 2, 8, 0)
    assert is_tradeable(pre) is False
    assert is_tradeable(pre, allow_extended=True) is True
    assert is_tradeable(_ts(2026, 6, 2, 10, 0)) is True


def test_session_date_is_et():
    assert session_date(_ts(2026, 6, 2, 10, 0)) == "2026-06-02"


def test_calendar_covers_known_years():
    assert calendar_covers(_ts(2026, 6, 2, 10, 0)) is True
    assert calendar_covers(_ts(2030, 6, 2, 10, 0)) is False
