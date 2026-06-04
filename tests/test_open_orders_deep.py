"""
Deep, deterministic tests for alpca/execution/open_orders.py (OpenOrderBook).

Covers the resting-order lifecycle entirely offline (no network, no live Alpaca):
  - add / get / cancel / replace input validation and bookkeeping
  - DAY expiry at the next session open; GTC persistence
  - STOP and STOP_LIMIT trigger-on-touch then fill
  - partial fills leaving a resting remainder
  - cancel-replace preserving already-filled qty (re-rests only remaining)
  - IOC / FOK / MARKET never rest (add rejects)
  - on_bar -> List[BookEvent] with the correct event kinds
  - edge / degenerate inputs (None, zero/negative qty, gaps, idempotency)

All tests use real Order/FillModel objects with hand-chosen deterministic bars.
"""

from __future__ import annotations

import math

import pytest

from alpca.execution.open_orders import OpenOrderBook, BookEvent
from alpca.execution.fills import FillModel
from alpca.execution.order import (
    Order,
    OrderStatus,
    OrderType,
    Side,
    TimeInForce,
)


# ----------------------------------------------------------------- tiny helpers
def make_bar(open_=100.0, high=101.0, low=99.0, close=100.0, volume=10_000.0,
             **extra):
    bar = {"open": open_, "high": high, "low": low, "close": close,
           "volume": volume}
    bar.update(extra)
    return bar


def make_order(side=Side.BUY, qty=100.0, order_type=OrderType.LIMIT,
               limit_price=99.0, stop_price=None, tif=TimeInForce.DAY,
               strategy="t"):
    return Order(
        symbol="TEST",
        side=side,
        qty=qty,
        order_type=order_type,
        limit_price=limit_price,
        stop_price=stop_price,
        tif=tif,
        strategy=strategy,
    )


def fresh_book(**kw):
    # default fill model: no cap, no impact, 1bp half-spread, penny tick
    return OpenOrderBook(**kw)


def kinds(events):
    return [e.kind for e in events]


# =============================================================================
# add() validation + restability
# =============================================================================
class TestAddValidation:
    def test_market_never_rests(self):
        book = fresh_book()
        o = make_order(order_type=OrderType.MARKET, limit_price=None)
        with pytest.raises(ValueError, match="MARKET"):
            book.add(o, 0)
        assert len(book) == 0

    @pytest.mark.parametrize("tif", [TimeInForce.IOC, TimeInForce.FOK])
    def test_ioc_fok_never_rest(self, tif):
        book = fresh_book()
        o = make_order(tif=tif)
        with pytest.raises(ValueError, match="immediate-or-cancel"):
            book.add(o, 0)
        assert len(book) == 0

    @pytest.mark.parametrize("ot", [OrderType.LIMIT, OrderType.STOP_LIMIT])
    def test_limit_requires_limit_price(self, ot):
        book = fresh_book()
        # supply a stop_price so the STOP_LIMIT path reaches the limit_price check
        o = make_order(order_type=ot, limit_price=None, stop_price=100.0)
        with pytest.raises(ValueError, match="requires a limit_price"):
            book.add(o, 0)

    @pytest.mark.parametrize("ot", [OrderType.STOP, OrderType.STOP_LIMIT])
    def test_stop_requires_stop_price(self, ot):
        book = fresh_book()
        # STOP_LIMIT needs a limit_price to get past that check first
        lp = 100.0 if ot == OrderType.STOP_LIMIT else None
        o = make_order(order_type=ot, limit_price=lp, stop_price=None)
        with pytest.raises(ValueError, match="requires a stop_price"):
            book.add(o, 0)

    @pytest.mark.parametrize("tif", [TimeInForce.DAY, TimeInForce.GTC])
    def test_restable_tifs_accepted(self, tif):
        book = fresh_book()
        o = make_order(tif=tif)
        ret = book.add(o, 3)
        assert ret is o
        assert len(book) == 1
        assert book.get(o.client_order_id) is o

    def test_add_stamps_submit_session(self):
        book = fresh_book()
        o = make_order()
        book.add(o, 7)
        assert o.metadata["submit_session"] == 7

    def test_add_limit_marks_already_triggered(self):
        book = fresh_book()
        o = make_order(order_type=OrderType.LIMIT)
        book.add(o, 0)
        # LIMIT/MARKET are not stops -> considered already "triggered"
        assert o.metadata["stop_triggered"] is True

    @pytest.mark.parametrize("ot", [OrderType.STOP, OrderType.STOP_LIMIT])
    def test_add_stop_marks_untriggered(self, ot):
        book = fresh_book()
        o = make_order(order_type=ot, stop_price=105.0, limit_price=104.0)
        book.add(o, 0)
        assert o.metadata["stop_triggered"] is False

    def test_add_sets_ack_and_status(self):
        book = fresh_book()
        o = make_order()
        assert o.ack_ts is None
        book.add(o, 0)
        assert o.ack_ts is not None
        assert o.status == OrderStatus.ACCEPTED

    def test_add_existing_ack_sets_accepted_status(self):
        book = fresh_book()
        o = make_order()
        o.mark_ack()  # ack_ts now set, status ACCEPTED already
        prior_ack = o.ack_ts
        book.add(o, 0)
        # branch: ack_ts is not None -> status forced ACCEPTED, ack_ts unchanged
        assert o.ack_ts == prior_ack
        assert o.status == OrderStatus.ACCEPTED


# =============================================================================
# queries: working, len, get, _remaining
# =============================================================================
class TestQueries:
    def test_get_unknown_returns_none(self):
        assert fresh_book().get("nope") is None

    def test_working_lists_all_resting(self):
        book = fresh_book()
        a = make_order(strategy="a")
        b = make_order(strategy="b")
        book.add(a, 0)
        book.add(b, 0)
        working_ids = {o.client_order_id for o in book.working}
        assert working_ids == {a.client_order_id, b.client_order_id}
        assert len(book) == 2

    def test_remaining_full_unfilled(self):
        o = make_order(qty=50.0)
        assert OpenOrderBook._remaining(o) == 50.0

    def test_remaining_after_partial(self):
        o = make_order(qty=50.0)
        from alpca.execution.order import Fill
        o.add_fill(Fill(ts=0.0, price=99.0, qty=20.0))
        assert OpenOrderBook._remaining(o) == pytest.approx(30.0)

    def test_remaining_never_negative(self):
        o = make_order(qty=10.0)
        from alpca.execution.order import Fill
        o.add_fill(Fill(ts=0.0, price=99.0, qty=10.0))  # fully filled
        assert OpenOrderBook._remaining(o) == 0.0


# =============================================================================
# cancel()
# =============================================================================
class TestCancel:
    def test_cancel_unknown_returns_none(self):
        assert fresh_book().cancel("missing") is None

    def test_cancel_removes_and_marks_terminal(self):
        book = fresh_book()
        o = make_order()
        book.add(o, 0)
        ret = book.cancel(o.client_order_id)
        assert ret is o
        assert o.status == OrderStatus.CANCELED
        assert o.status.is_terminal
        assert len(book) == 0
        assert book.get(o.client_order_id) is None

    def test_cancel_idempotent_second_call_none(self):
        book = fresh_book()
        o = make_order()
        book.add(o, 0)
        book.cancel(o.client_order_id)
        assert book.cancel(o.client_order_id) is None

    def test_cancel_partially_filled_marks_canceled(self):
        # partial fill via volume cap, then cancel the resting remainder
        book = fresh_book(fill_model=FillModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                               participation_cap=0.5, min_tick=0.0))
        o = make_order(side=Side.BUY, qty=1000.0, limit_price=100.0)
        book.add(o, 0)
        bar = make_bar(open_=99.0, high=100.0, low=98.0, volume=1000.0)
        book.on_bar(bar, 0)  # fills 500 (cap), 500 remains resting
        assert o.status == OrderStatus.PARTIALLY_FILLED
        assert o.filled_qty == pytest.approx(500.0)
        ret = book.cancel(o.client_order_id)
        assert ret is o
        assert o.status == OrderStatus.CANCELED


# =============================================================================
# replace() cancel-replace semantics
# =============================================================================
class TestReplace:
    def test_replace_unknown_returns_none(self):
        assert fresh_book().replace("missing", qty=5.0) is None

    def test_replace_creates_new_coid_and_supersedes_old(self):
        book = fresh_book()
        old = make_order(qty=100.0, limit_price=99.0)
        book.add(old, 2)
        new = book.replace(old.client_order_id, limit_price=98.0)
        assert new is not None
        assert new.client_order_id != old.client_order_id
        # old is terminal + tagged; only the new order rests
        assert old.status == OrderStatus.CANCELED
        assert old.metadata["replaced_by"] == new.client_order_id
        assert book.get(old.client_order_id) is None
        assert book.get(new.client_order_id) is new
        assert len(book) == 1

    def test_replace_default_qty_is_remaining(self):
        # after a 500-share partial, replace() with no qty re-rests only 500
        book = fresh_book(fill_model=FillModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                               participation_cap=0.5, min_tick=0.0))
        old = make_order(side=Side.BUY, qty=1000.0, limit_price=100.0)
        book.add(old, 0)
        book.on_bar(make_bar(open_=99.0, high=100.0, low=98.0, volume=1000.0), 0)
        assert old.filled_qty == pytest.approx(500.0)
        new = book.replace(old.client_order_id, limit_price=97.0)
        assert new.qty == pytest.approx(500.0)  # remaining only
        assert new.filled_qty == 0.0            # fresh order, no carried fills

    def test_replace_explicit_qty_overrides(self):
        book = fresh_book()
        old = make_order(qty=100.0, limit_price=99.0)
        book.add(old, 0)
        new = book.replace(old.client_order_id, qty=42.0)
        assert new.qty == 42.0

    def test_replace_inherits_unspecified_fields(self):
        book = fresh_book()
        old = make_order(side=Side.SELL, qty=80.0, order_type=OrderType.LIMIT,
                         limit_price=101.0, tif=TimeInForce.GTC, strategy="momo")
        book.add(old, 5)
        new = book.replace(old.client_order_id, qty=80.0)
        assert new.side == Side.SELL
        assert new.order_type == OrderType.LIMIT
        assert new.limit_price == 101.0
        assert new.tif == TimeInForce.GTC
        assert new.strategy == "momo"
        assert new.symbol == "TEST"

    def test_replace_amends_stop_price(self):
        book = fresh_book()
        old = make_order(side=Side.BUY, order_type=OrderType.STOP,
                         limit_price=None, stop_price=105.0)
        book.add(old, 0)
        new = book.replace(old.client_order_id, stop_price=110.0)
        assert new.stop_price == 110.0
        assert new.order_type == OrderType.STOP

    def test_replace_uses_old_session_when_unspecified(self):
        book = fresh_book()
        old = make_order(tif=TimeInForce.DAY)
        book.add(old, 9)
        new = book.replace(old.client_order_id, qty=10.0)
        assert new.metadata["submit_session"] == 9

    def test_replace_explicit_session_overrides(self):
        book = fresh_book()
        old = make_order(tif=TimeInForce.DAY)
        book.add(old, 9)
        new = book.replace(old.client_order_id, qty=10.0, session_index=20)
        assert new.metadata["submit_session"] == 20


# =============================================================================
# DAY expiry / GTC persistence
# =============================================================================
class TestExpiry:
    def test_day_order_expires_next_session(self):
        book = fresh_book()
        o = make_order(tif=TimeInForce.DAY, limit_price=90.0)  # never fills
        book.add(o, 0)
        evs = book.on_bar(make_bar(low=95.0), 1)  # session advanced
        assert kinds(evs) == ["expire"]
        assert o.status == OrderStatus.EXPIRED
        assert o.status.is_terminal
        assert len(book) == 0

    def test_day_order_survives_same_session(self):
        book = fresh_book()
        o = make_order(tif=TimeInForce.DAY, limit_price=90.0)
        book.add(o, 4)
        evs = book.on_bar(make_bar(low=95.0), 4)  # same session, no fill
        assert evs == []
        assert o.status == OrderStatus.ACCEPTED
        assert len(book) == 1

    def test_gtc_persists_across_sessions(self):
        book = fresh_book()
        o = make_order(tif=TimeInForce.GTC, limit_price=90.0)
        book.add(o, 0)
        for sess in range(1, 5):
            evs = book.on_bar(make_bar(low=95.0), sess)
            assert evs == []
        assert o.status == OrderStatus.ACCEPTED
        assert len(book) == 1

    def test_expire_all_day_orders_only_day(self):
        book = fresh_book()
        day = make_order(tif=TimeInForce.DAY, strategy="d", limit_price=90.0)
        gtc = make_order(tif=TimeInForce.GTC, strategy="g", limit_price=90.0)
        book.add(day, 0)
        book.add(gtc, 0)
        evs = book.expire_all_day_orders(0)
        assert kinds(evs) == ["expire"]
        assert evs[0].order is day
        assert day.status == OrderStatus.EXPIRED
        assert gtc.status == OrderStatus.ACCEPTED
        assert len(book) == 1
        assert book.get(gtc.client_order_id) is gtc

    def test_expire_all_empty_book(self):
        assert fresh_book().expire_all_day_orders(0) == []

    def test_day_expiry_precedes_fill(self):
        # a DAY order whose bar WOULD fill it but the session advanced -> expires,
        # never fills (expiry is step 1).
        book = fresh_book()
        o = make_order(side=Side.BUY, tif=TimeInForce.DAY, limit_price=100.0)
        book.add(o, 0)
        bar = make_bar(open_=100.0, high=101.0, low=98.0)  # would fill the buy
        evs = book.on_bar(bar, 5)
        assert kinds(evs) == ["expire"]
        assert o.filled_qty == 0.0
        assert o.status == OrderStatus.EXPIRED


# =============================================================================
# LIMIT fills via on_bar
# =============================================================================
class TestLimitFill:
    def test_buy_limit_fills_when_low_touches(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, limit_price=99.0)
        book.add(o, 0)
        bar = make_bar(open_=100.0, high=101.0, low=98.5)  # low < limit -> fill
        evs = book.on_bar(bar, 0)
        assert kinds(evs) == ["fill"]
        assert o.status == OrderStatus.FILLED
        assert o.filled_qty == 100.0
        # filled exactly at the limit (open above limit -> no improvement)
        assert o.avg_fill_price == pytest.approx(99.0)
        assert len(book) == 0

    def test_buy_limit_no_fill_when_low_above_limit(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, limit_price=99.0)
        book.add(o, 0)
        bar = make_bar(open_=100.0, high=101.0, low=99.5)  # never reaches 99
        evs = book.on_bar(bar, 0)
        assert evs == []
        assert o.status == OrderStatus.ACCEPTED
        assert o.filled_qty == 0.0

    def test_sell_limit_fills_when_high_touches(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.SELL, qty=100.0, limit_price=101.0)
        book.add(o, 0)
        bar = make_bar(open_=100.0, high=101.5, low=99.0)  # high > limit -> fill
        evs = book.on_bar(bar, 0)
        assert kinds(evs) == ["fill"]
        assert o.avg_fill_price == pytest.approx(101.0)

    def test_buy_limit_price_improvement_on_gap_down(self):
        # bar opens BELOW the buy limit -> fill at the better (open) price
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, limit_price=99.0)
        book.add(o, 0)
        bar = make_bar(open_=98.0, high=98.5, low=97.0)  # gapped through
        evs = book.on_bar(bar, 0)
        assert kinds(evs) == ["fill"]
        assert o.avg_fill_price == pytest.approx(98.0)  # min(open, limit)

    def test_sell_limit_price_improvement_on_gap_up(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.SELL, qty=100.0, limit_price=101.0)
        book.add(o, 0)
        bar = make_bar(open_=102.0, high=103.0, low=101.5)  # gapped up through
        evs = book.on_bar(bar, 0)
        assert o.avg_fill_price == pytest.approx(102.0)  # max(open, limit)


# =============================================================================
# Partial fills + remainder resting + multi-bar completion
# =============================================================================
class TestPartialFill:
    def test_partial_fill_leaves_remainder_resting(self):
        book = fresh_book(fill_model=FillModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                               participation_cap=0.4, min_tick=0.0))
        o = make_order(side=Side.BUY, qty=1000.0, limit_price=100.0)
        book.add(o, 0)
        bar = make_bar(open_=99.0, high=100.0, low=98.0, volume=1000.0)
        evs = book.on_bar(bar, 0)  # cap = 0.4*1000 = 400
        assert kinds(evs) == ["partial_fill"]
        assert o.status == OrderStatus.PARTIALLY_FILLED
        assert o.filled_qty == pytest.approx(400.0)
        assert OpenOrderBook._remaining(o) == pytest.approx(600.0)
        assert len(book) == 1  # remainder still resting

    def test_partial_completes_over_multiple_bars(self):
        book = fresh_book(fill_model=FillModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                               participation_cap=0.5, min_tick=0.0))
        o = make_order(side=Side.BUY, qty=1000.0, limit_price=100.0, tif=TimeInForce.GTC)
        book.add(o, 0)
        bar = make_bar(open_=99.0, high=100.0, low=98.0, volume=1000.0)
        e1 = book.on_bar(bar, 0)  # 500 filled
        assert kinds(e1) == ["partial_fill"]
        assert o.filled_qty == pytest.approx(500.0)
        e2 = book.on_bar(bar, 1)  # remaining 500: cap=500 -> fills the rest
        assert kinds(e2) == ["fill"]
        assert o.status == OrderStatus.FILLED
        assert o.filled_qty == pytest.approx(1000.0)
        assert len(book) == 0

    def test_no_event_when_limit_untouched_across_bars(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, limit_price=90.0, tif=TimeInForce.GTC)
        book.add(o, 0)
        for s in range(3):
            assert book.on_bar(make_bar(low=95.0), s) == []
        assert o.filled_qty == 0.0


# =============================================================================
# STOP trigger -> market fill
# =============================================================================
class TestStop:
    def test_buy_stop_triggers_on_high_touch_then_fills(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, order_type=OrderType.STOP,
                       limit_price=None, stop_price=105.0)
        book.add(o, 0)
        # high reaches stop -> trigger; ref = max(stop, open) for a buy
        bar = make_bar(open_=104.0, high=106.0, low=103.0)
        evs = book.on_bar(bar, 0)
        assert kinds(evs) == ["trigger", "fill"]
        assert o.status == OrderStatus.FILLED
        # ref = max(105, 104) = 105, flat 0bps -> 105
        assert o.avg_fill_price == pytest.approx(105.0)
        assert len(book) == 0

    def test_sell_stop_triggers_on_low_touch(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.SELL, qty=100.0, order_type=OrderType.STOP,
                       limit_price=None, stop_price=95.0)
        book.add(o, 0)
        bar = make_bar(open_=96.0, high=97.0, low=94.0)  # low <= stop
        evs = book.on_bar(bar, 0)
        assert kinds(evs) == ["trigger", "fill"]
        # ref = min(stop, open) = min(95, 96) = 95
        assert o.avg_fill_price == pytest.approx(95.0)

    def test_buy_stop_gap_fills_at_open_worse_than_stop(self):
        # gap up: open ABOVE the stop -> filled at the open (worse), gap-aware
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, order_type=OrderType.STOP,
                       limit_price=None, stop_price=105.0)
        book.add(o, 0)
        bar = make_bar(open_=108.0, high=109.0, low=107.0)
        evs = book.on_bar(bar, 0)
        assert kinds(evs) == ["trigger", "fill"]
        assert o.avg_fill_price == pytest.approx(108.0)  # max(105, 108)

    def test_stop_not_touched_keeps_resting(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, order_type=OrderType.STOP,
                       limit_price=None, stop_price=105.0)
        book.add(o, 0)
        bar = make_bar(open_=100.0, high=104.0, low=99.0)  # high < stop
        evs = book.on_bar(bar, 0)
        assert evs == []
        assert o.metadata["stop_triggered"] is False
        assert len(book) == 1

    def test_stop_triggers_once_persists_flag(self):
        book = fresh_book(fill_model=FillModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                               participation_cap=0.3, min_tick=0.0))
        o = make_order(side=Side.BUY, qty=1000.0, order_type=OrderType.STOP,
                       limit_price=None, stop_price=105.0, tif=TimeInForce.GTC)
        book.add(o, 0)
        bar = make_bar(open_=105.0, high=106.0, low=104.0, volume=1000.0)
        e1 = book.on_bar(bar, 0)  # trigger + partial (cap 300)
        assert kinds(e1) == ["trigger", "partial_fill"]
        assert o.metadata["stop_triggered"] is True
        # next bar: already triggered, no second "trigger" event
        e2 = book.on_bar(bar, 1)
        assert "trigger" not in kinds(e2)


# =============================================================================
# STOP_LIMIT: trigger then resting LIMIT
# =============================================================================
class TestStopLimit:
    def test_stop_limit_triggers_then_fills_as_limit(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        # buy stop-limit: stop 105, limit 106
        o = make_order(side=Side.BUY, qty=100.0, order_type=OrderType.STOP_LIMIT,
                       limit_price=106.0, stop_price=105.0)
        book.add(o, 0)
        # high >= 105 triggers; then as a LIMIT, buy fills iff low <= 106
        bar = make_bar(open_=105.0, high=106.5, low=104.5)
        evs = book.on_bar(bar, 0)
        assert kinds(evs) == ["trigger", "fill"]
        assert o.status == OrderStatus.FILLED
        # fill = min(open, limit) = min(105, 106) = 105
        assert o.avg_fill_price == pytest.approx(105.0)

    def test_stop_limit_triggers_but_limit_not_reached_rests(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        # buy stop-limit: stop 105, limit 105 (tight). After trigger the bar must
        # have low <= 105 to fill as a limit.
        o = make_order(side=Side.BUY, qty=100.0, order_type=OrderType.STOP_LIMIT,
                       limit_price=104.0, stop_price=105.0, tif=TimeInForce.GTC)
        book.add(o, 0)
        # high 106 triggers, but low 105.5 never reaches the 104 buy limit -> rests
        bar = make_bar(open_=105.5, high=106.0, low=105.5)
        evs = book.on_bar(bar, 0)
        assert kinds(evs) == ["trigger"]
        assert o.metadata["stop_triggered"] is True
        assert o.filled_qty == 0.0
        assert len(book) == 1  # still resting as a limit

    def test_stop_limit_fills_on_later_bar_after_trigger(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, order_type=OrderType.STOP_LIMIT,
                       limit_price=104.0, stop_price=105.0, tif=TimeInForce.GTC)
        book.add(o, 0)
        e1 = book.on_bar(make_bar(open_=105.5, high=106.0, low=105.5), 0)
        assert kinds(e1) == ["trigger"]
        # later bar dips to the limit -> fills, no second trigger
        e2 = book.on_bar(make_bar(open_=104.5, high=105.0, low=103.5), 1)
        assert kinds(e2) == ["fill"]
        assert o.avg_fill_price == pytest.approx(104.0)  # fill at limit


# =============================================================================
# on_bar event-list correctness across multiple orders
# =============================================================================
class TestOnBarMultiOrder:
    def test_events_returned_for_each_affected_order(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        filler = make_order(side=Side.BUY, qty=100.0, limit_price=99.0, strategy="f")
        rester = make_order(side=Side.BUY, qty=100.0, limit_price=80.0, strategy="r")
        book.add(filler, 0)
        book.add(rester, 0)
        bar = make_bar(open_=100.0, high=101.0, low=98.0)  # touches 99 not 80
        evs = book.on_bar(bar, 0)
        assert kinds(evs) == ["fill"]
        assert evs[0].order is filler
        assert book.get(rester.client_order_id) is rester  # still resting
        assert book.get(filler.client_order_id) is None

    def test_all_book_events_have_valid_kinds(self):
        valid = {"trigger", "partial_fill", "fill", "expire", "cancel"}
        book = fresh_book(fill_model=FillModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                               participation_cap=0.3, min_tick=0.0))
        stop = make_order(side=Side.BUY, qty=1000.0, order_type=OrderType.STOP,
                          limit_price=None, stop_price=105.0, tif=TimeInForce.GTC)
        book.add(stop, 0)
        evs = book.on_bar(make_bar(open_=105.0, high=106.0, low=104.0, volume=1000.0), 0)
        for e in evs:
            assert isinstance(e, BookEvent)
            assert e.kind in valid
            assert e.order is stop

    def test_empty_book_on_bar_returns_empty(self):
        assert fresh_book().on_bar(make_bar(), 0) == []


# =============================================================================
# edge / degenerate inputs
# =============================================================================
class TestEdgeCases:
    def test_zero_qty_order_no_fill(self):
        # qty<=0 -> fill model returns filled_qty 0, order rests untouched
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=0.0, limit_price=99.0)
        book.add(o, 0)
        evs = book.on_bar(make_bar(open_=100.0, high=101.0, low=98.0), 0)
        assert evs == []
        assert o.filled_qty == 0.0

    def test_negative_qty_no_fill(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=-50.0, limit_price=99.0)
        book.add(o, 0)
        evs = book.on_bar(make_bar(open_=100.0, high=101.0, low=98.0), 0)
        assert evs == []
        assert o.filled_qty == 0.0

    def test_missing_volume_key_does_not_crash(self):
        # bar without "volume": .get returns None -> fill model treats as enough
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, limit_price=99.0)
        book.add(o, 0)
        bar = {"open": 100.0, "high": 101.0, "low": 98.0, "close": 100.0}
        evs = book.on_bar(bar, 0)
        assert kinds(evs) == ["fill"]
        assert o.filled_qty == 100.0

    def test_zero_volume_treated_as_enough(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, limit_price=99.0)
        book.add(o, 0)
        # bar_volume=0 -> fill_limit's "no volume context" branch -> fills all
        evs = book.on_bar(make_bar(open_=100.0, high=101.0, low=98.0, volume=0.0), 0)
        assert kinds(evs) == ["fill"]
        assert o.filled_qty == 100.0

    def test_buy_limit_exactly_at_low_fills(self):
        # boundary: bar_low == limit -> reached is True (<=)
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, limit_price=99.0)
        book.add(o, 0)
        evs = book.on_bar(make_bar(open_=100.0, high=101.0, low=99.0), 0)
        assert kinds(evs) == ["fill"]

    def test_buy_stop_exactly_at_high_triggers(self):
        # boundary: high == stop -> triggered True (>=)
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, order_type=OrderType.STOP,
                       limit_price=None, stop_price=105.0)
        book.add(o, 0)
        evs = book.on_bar(make_bar(open_=104.0, high=105.0, low=103.0), 0)
        assert "trigger" in kinds(evs)

    def test_extreme_magnitude_qty_and_price(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=1e9, limit_price=1e6)
        book.add(o, 0)
        evs = book.on_bar(make_bar(open_=2e6, high=2e6, low=5e5), 0)
        assert kinds(evs) == ["fill"]
        assert o.filled_qty == pytest.approx(1e9)
        # open 2e6 > limit 1e6 -> fills at limit
        assert o.avg_fill_price == pytest.approx(1e6)

    def test_nan_low_does_not_fill_buy_limit(self):
        # NaN comparisons are always False -> (nan <= limit) False -> no fill, no crash
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, limit_price=99.0)
        book.add(o, 0)
        evs = book.on_bar(make_bar(open_=100.0, high=101.0, low=float("nan")), 0)
        assert evs == []
        assert o.filled_qty == 0.0

    def test_inf_high_triggers_buy_stop(self):
        book = fresh_book(fill_model=FillModel.flat(0.0))
        o = make_order(side=Side.BUY, qty=100.0, order_type=OrderType.STOP,
                       limit_price=None, stop_price=105.0)
        book.add(o, 0)
        # high inf >= stop True -> triggers; ref = max(105, open=104)=105
        evs = book.on_bar(make_bar(open_=104.0, high=float("inf"), low=103.0), 0)
        assert "trigger" in kinds(evs)
        assert o.status == OrderStatus.FILLED
        assert o.avg_fill_price == pytest.approx(105.0)

    def test_on_bar_safe_iteration_when_orders_terminate(self):
        # several fills + an expiry in the same bar must not raise (snapshot keys)
        book = fresh_book(fill_model=FillModel.flat(0.0))
        a = make_order(side=Side.BUY, qty=10.0, limit_price=99.0, strategy="a", tif=TimeInForce.GTC)
        b = make_order(side=Side.BUY, qty=10.0, limit_price=99.0, strategy="b", tif=TimeInForce.GTC)
        c = make_order(side=Side.BUY, qty=10.0, limit_price=90.0, strategy="c", tif=TimeInForce.DAY)
        book.add(a, 0)
        book.add(b, 0)
        book.add(c, 0)
        evs = book.on_bar(make_bar(open_=100.0, high=101.0, low=98.0), 1)
        ks = kinds(evs)
        assert ks.count("fill") == 2   # a and b
        assert ks.count("expire") == 1  # c (DAY, next session)
        assert len(book) == 0


# =============================================================================
# fill price reflects spread (default model) + tick rounding
# =============================================================================
class TestFillPriceMechanics:
    def test_default_model_applies_half_spread_to_stop(self):
        # default OpenOrderBook model: half_spread_bps=1.0
        book = fresh_book()
        o = make_order(side=Side.BUY, qty=100.0, order_type=OrderType.STOP,
                       limit_price=None, stop_price=100.0)
        book.add(o, 0)
        # ref = max(100, open=100) = 100; buy pays +1bp -> 100.01, tick 0.01
        book.on_bar(make_bar(open_=100.0, high=100.5, low=99.5), 0)
        assert o.status == OrderStatus.FILLED
        assert o.avg_fill_price == pytest.approx(100.01)

    def test_limit_fill_price_is_rounded_to_tick(self):
        book = fresh_book(fill_model=FillModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                               participation_cap=1.0, min_tick=0.01))
        o = make_order(side=Side.BUY, qty=100.0, limit_price=99.005)
        book.add(o, 0)
        # gap down: open 98.333 -> min(open, limit)=98.333 rounded to 98.33
        book.on_bar(make_bar(open_=98.333, high=98.5, low=98.0), 0)
        assert o.avg_fill_price == pytest.approx(98.33)


# =============================================================================
# queue model opt-in path
# =============================================================================
class TestQueueModel:
    def test_queue_model_delays_fill_behind_displayed_size(self):
        # FIFO: front = bid_size ahead of us; trades eat the front first, then us
        book = fresh_book(use_queue_model=True,
                          fill_model=FillModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                               participation_cap=1.0, min_tick=0.0))
        o = make_order(side=Side.BUY, qty=100.0, limit_price=99.0, tif=TimeInForce.GTC)
        book.add(o, 0)
        # 500 shares ahead; bar trades 500 at level -> all eaten, 0 fills us
        bar = make_bar(open_=99.0, high=99.5, low=98.5, volume=500.0, bid_size=500.0)
        evs = book.on_bar(bar, 0)
        assert evs == []  # queue ahead fully consumed, nothing left to fill us
        assert o.filled_qty == 0.0
        assert len(book) == 1
        # next bar: front now 0, trades fill us
        bar2 = make_bar(open_=99.0, high=99.5, low=98.5, volume=500.0, bid_size=500.0)
        evs2 = book.on_bar(bar2, 1)
        assert kinds(evs2) == ["fill"]
        assert o.filled_qty == pytest.approx(100.0)

    def test_queue_position_created_once_and_persists(self):
        book = fresh_book(use_queue_model=True,
                          fill_model=FillModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                                               participation_cap=1.0, min_tick=0.0))
        o = make_order(side=Side.BUY, qty=100.0, limit_price=99.0, tif=TimeInForce.GTC)
        book.add(o, 0)
        bar = make_bar(open_=99.0, high=99.5, low=98.5, volume=200.0, bid_size=300.0)
        book.on_bar(bar, 0)
        qp1 = o.metadata["queue_pos"]
        book.on_bar(bar, 1)
        qp2 = o.metadata["queue_pos"]
        assert qp1 is qp2  # same QueuePosition reused across bars
