"""
Order-lifecycle tests: resting LIMIT/STOP across bars, DAY/GTC expiry,
through-trade fills, partial fills, STOP triggering, cancel/replace.
"""

import pytest

from alpca.execution.fills import FillModel
from alpca.execution.open_orders import OpenOrderBook
from alpca.execution.order import Order, OrderStatus, OrderType, Side, TimeInForce


FM = FillModel(half_spread_bps=0.0, impact_coef_bps=0.0, participation_cap=1.0, min_tick=0.0)


def _bar(o, h, l, c, ts=0, vol=1e7):
    return {"open": o, "high": h, "low": l, "close": c, "volume": vol, "timestamp": ts}


def _limit(side, qty, limit, tif=TimeInForce.DAY):
    return Order(symbol="T", side=side, qty=qty, order_type=OrderType.LIMIT,
                 limit_price=limit, tif=tif)


# ----------------------------------------------------------------- resting / fill
def test_limit_rests_until_through_traded():
    book = OpenOrderBook(FM)
    o = _limit(Side.BUY, 10, 99.0)
    book.add(o, session_index=0)
    assert len(book) == 1
    # bar stays above 99 -> no fill, still resting
    ev = book.on_bar(_bar(100, 101, 99.5, 100), session_index=0)
    assert ev == []
    assert len(book) == 1
    assert o.status == OrderStatus.ACCEPTED
    # next bar dips to 98 (<=99) -> fills, leaves the book
    ev = book.on_bar(_bar(100, 100.5, 98.0, 99), session_index=0)
    assert len(ev) == 1 and ev[0].kind == "fill"
    assert o.status == OrderStatus.FILLED
    assert abs(o.avg_fill_price - 99.0) < 1e-9
    assert len(book) == 0


def test_day_order_expires_next_session():
    book = OpenOrderBook(FM)
    o = _limit(Side.BUY, 10, 90.0, tif=TimeInForce.DAY)
    book.add(o, session_index=0)
    # same session, never reached -> still resting
    book.on_bar(_bar(100, 101, 99, 100), session_index=0)
    assert len(book) == 1
    # next session opens -> DAY order expires before any fill
    ev = book.on_bar(_bar(100, 101, 80, 85), session_index=1)
    assert len(ev) == 1 and ev[0].kind == "expire"
    assert o.status == OrderStatus.EXPIRED
    assert len(book) == 0


def test_gtc_order_persists_across_sessions():
    book = OpenOrderBook(FM)
    o = _limit(Side.BUY, 10, 90.0, tif=TimeInForce.GTC)
    book.add(o, session_index=0)
    book.on_bar(_bar(100, 101, 95, 100), session_index=0)
    book.on_bar(_bar(100, 101, 95, 100), session_index=1)
    book.on_bar(_bar(100, 101, 95, 100), session_index=5)
    assert len(book) == 1  # GTC never expires
    assert o.status == OrderStatus.ACCEPTED
    # eventually fills when price reaches it, even sessions later
    ev = book.on_bar(_bar(95, 96, 89, 90), session_index=5)
    assert any(e.kind == "fill" for e in ev)
    assert o.status == OrderStatus.FILLED


def test_partial_fill_leaves_remainder_resting():
    # volume cap forces a partial: 10% of 1000 = 100 per bar
    fm = FillModel(half_spread_bps=0.0, impact_coef_bps=0.0, participation_cap=0.10, min_tick=0.0)
    book = OpenOrderBook(fm)
    o = _limit(Side.BUY, 1000, 99.0, tif=TimeInForce.GTC)
    book.add(o, session_index=0)
    ev = book.on_bar(_bar(100, 100.5, 98.0, 99, vol=1000), session_index=0)
    assert len(ev) == 1 and ev[0].kind == "partial_fill"
    assert o.status == OrderStatus.PARTIALLY_FILLED
    assert o.filled_qty == 100
    assert len(book) == 1  # remainder still resting
    # next bar fills another 100
    book.on_bar(_bar(100, 100.5, 98.0, 99, vol=1000), session_index=0)
    assert o.filled_qty == 200
    assert o.status == OrderStatus.PARTIALLY_FILLED


# ----------------------------------------------------------------- stop triggers
def test_buy_stop_triggers_and_fills_on_breakout():
    book = OpenOrderBook(FM)
    o = Order(symbol="T", side=Side.BUY, qty=10, order_type=OrderType.STOP,
              stop_price=105.0, tif=TimeInForce.GTC)
    book.add(o, session_index=0)
    # bar below the stop -> no trigger
    ev = book.on_bar(_bar(100, 104, 99, 103), session_index=0)
    assert ev == []
    assert o.status == OrderStatus.ACCEPTED
    # bar trades up through 105 -> trigger + fill
    ev = book.on_bar(_bar(104, 108, 103, 107), session_index=0)
    kinds = [e.kind for e in ev]
    assert "trigger" in kinds and "fill" in kinds
    assert o.status == OrderStatus.FILLED
    # filled at >= stop (max of stop/open)
    assert o.avg_fill_price >= 105.0


def test_sell_stop_triggers_on_breakdown():
    book = OpenOrderBook(FM)
    o = Order(symbol="T", side=Side.SELL, qty=10, order_type=OrderType.STOP,
              stop_price=95.0, tif=TimeInForce.GTC)
    book.add(o, session_index=0)
    ev = book.on_bar(_bar(100, 101, 96, 100), session_index=0)  # never reaches 95
    assert ev == []
    ev = book.on_bar(_bar(96, 97, 92, 93), session_index=0)     # breaks 95
    assert "trigger" in [e.kind for e in ev]
    assert o.status == OrderStatus.FILLED
    assert o.avg_fill_price <= 95.0  # sell stop fills at <= stop


def test_stop_limit_triggers_then_rests_as_limit():
    book = OpenOrderBook(FM)
    # buy stop-limit: trigger at 105, but only buy up to 105.50
    o = Order(symbol="T", side=Side.BUY, qty=10, order_type=OrderType.STOP_LIMIT,
              stop_price=105.0, limit_price=105.50, tif=TimeInForce.GTC)
    book.add(o, session_index=0)
    # bar trades to 106 high (triggers) but opens at 107 (above the 105.50 limit)
    # -> triggers, but the limit can't fill at 107; rests as a limit
    ev = book.on_bar(_bar(107, 108, 106, 107.5), session_index=0)
    assert "trigger" in [e.kind for e in ev]
    assert o.status == OrderStatus.ACCEPTED  # triggered but unfilled, still resting
    assert len(book) == 1
    # later bar dips to 105 (<= limit 105.50) -> fills at the limit
    ev = book.on_bar(_bar(105.4, 106, 105.0, 105.5), session_index=1)  # GTC, survives session roll
    assert o.status == OrderStatus.FILLED
    assert o.avg_fill_price <= 105.50


# ----------------------------------------------------------------- cancel/replace
def test_cancel_removes_resting_order():
    book = OpenOrderBook(FM)
    o = _limit(Side.BUY, 10, 90.0)
    book.add(o, session_index=0)
    canceled = book.cancel(o.client_order_id)
    assert canceled is o
    assert o.status == OrderStatus.CANCELED
    assert len(book) == 0
    # cancelling an unknown id is a no-op
    assert book.cancel("nope") is None


def test_replace_amends_price_and_keeps_remaining():
    book = OpenOrderBook(FM)
    o = _limit(Side.BUY, 10, 90.0, tif=TimeInForce.GTC)
    book.add(o, session_index=0)
    new = book.replace(o.client_order_id, limit_price=95.0)
    assert new is not None
    assert o.status == OrderStatus.CANCELED          # old terminated
    assert new.client_order_id != o.client_order_id  # fresh id
    assert new.limit_price == 95.0
    assert new.qty == 10
    assert len(book) == 1
    assert book.get(new.client_order_id) is new
    # the amended (higher) limit now fills when price dips to 95
    ev = book.on_bar(_bar(96, 97, 94, 95), session_index=0)
    assert any(e.kind == "fill" for e in ev)
    assert new.status == OrderStatus.FILLED


def test_market_and_ioc_cannot_rest():
    book = OpenOrderBook(FM)
    with pytest.raises(ValueError):
        book.add(Order(symbol="T", side=Side.BUY, qty=1, order_type=OrderType.MARKET),
                 session_index=0)
    with pytest.raises(ValueError):
        book.add(_limit(Side.BUY, 1, 99.0, tif=TimeInForce.IOC), session_index=0)


def test_expire_all_day_orders():
    book = OpenOrderBook(FM)
    a = _limit(Side.BUY, 1, 90.0, tif=TimeInForce.DAY)
    b = _limit(Side.BUY, 1, 90.0, tif=TimeInForce.GTC)
    book.add(a, session_index=0)
    book.add(b, session_index=0)
    ev = book.expire_all_day_orders(session_index=0)
    assert len(ev) == 1 and ev[0].order is a
    assert a.status == OrderStatus.EXPIRED
    assert b.status == OrderStatus.ACCEPTED  # GTC untouched
    assert len(book) == 1
