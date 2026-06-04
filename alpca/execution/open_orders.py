"""
Open-order book — resting LIMIT/STOP orders that live ACROSS bars.

The single-bar SimAdapter fills (or rejects) an order against one bar. Real venues
keep an unfilled LIMIT/STOP order WORKING until it fills, expires, or is canceled.
This book models that lifecycle (audit gap: "no order lifecycle — no DAY-order
expiry, cancel/replace, GTC resting, open-order book across bars"):

  - rest LIMIT / STOP / STOP_LIMIT orders until they fill or terminate
  - DAY orders EXPIRE at the next session open; GTC orders persist
  - STOP / STOP_LIMIT TRIGGER when the bar touches the stop price, then fill
    (STOP -> marketable; STOP_LIMIT -> a resting LIMIT at limit_price)
  - LIMIT orders fill only on a real through-trade (delegated to FillModel),
    with partial fills leaving the remainder resting
  - cancel() and replace() (cancel-old + add-amended)

Session-indexed by the caller's integer trading-session counter, so DAY expiry
and "next session" are counted in trading days (correct across weekends/holidays).
IOC/FOK never rest — add() rejects them (they are immediate-or-cancel).

The book is the sim venue for resting orders; it does NOT go through a broker
adapter. on_bar() returns the lifecycle events that occurred so the caller can
journal them to the OrderEventLog.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from alpca.execution.fills import FillModel
from alpca.execution.order import Fill, Order, OrderStatus, OrderType, Side, TimeInForce


# metadata keys the book stamps on each resting order
_K_SESSION = "submit_session"
_K_TRIGGERED = "stop_triggered"
_K_QUEUE = "queue_pos"


@dataclass
class BookEvent:
    order: Order
    kind: str  # "trigger" | "partial_fill" | "fill" | "expire" | "cancel"


class OpenOrderBook:
    def __init__(self, fill_model: Optional[FillModel] = None, *,
                 use_queue_model: bool = False, prob_func=None) -> None:
        # default model: pure marketable behavior (1bp spread, no impact/cap)
        self.fill_model = fill_model or FillModel(half_spread_bps=1.0, impact_coef_bps=0.0,
                                                  participation_cap=1.0, min_tick=0.01)
        # opt-in FIFO queue-position model for resting LIMIT fills (default off ->
        # legacy volume-cap proxy, so all existing behavior is unchanged).
        self.use_queue_model = use_queue_model
        self.prob_func = prob_func
        self._resting: Dict[str, Order] = {}

    # ----------------------------------------------------------------- queries
    @property
    def working(self) -> List[Order]:
        return list(self._resting.values())

    def __len__(self) -> int:
        return len(self._resting)

    def get(self, client_order_id: str) -> Optional[Order]:
        return self._resting.get(client_order_id)

    @staticmethod
    def _remaining(order: Order) -> float:
        return max(0.0, order.qty - order.filled_qty)

    # ----------------------------------------------------------------- mutators
    def add(self, order: Order, session_index: int) -> Order:
        """Register a resting order. MARKET and IOC/FOK are not restable."""
        if order.order_type == OrderType.MARKET:
            raise ValueError("MARKET orders do not rest; submit them to the adapter")
        if order.tif in (TimeInForce.IOC, TimeInForce.FOK):
            raise ValueError(f"{order.tif.value} orders are immediate-or-cancel; they never rest")
        if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT) and order.limit_price is None:
            raise ValueError(f"{order.order_type.value} requires a limit_price")
        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) and order.stop_price is None:
            raise ValueError(f"{order.order_type.value} requires a stop_price")
        order.metadata[_K_SESSION] = session_index
        order.metadata[_K_TRIGGERED] = order.order_type not in (OrderType.STOP, OrderType.STOP_LIMIT)
        if order.ack_ts is None:
            order.mark_ack()
        else:
            order.status = OrderStatus.ACCEPTED
        self._resting[order.client_order_id] = order
        return order

    def cancel(self, client_order_id: str) -> Optional[Order]:
        order = self._resting.pop(client_order_id, None)
        if order is None:
            return None
        # a partially-filled order stays PARTIALLY_FILLED conceptually, but its
        # working remainder is canceled; we mark CANCELED (terminal) either way.
        order.status = OrderStatus.CANCELED
        return order

    def replace(self, client_order_id: str, *, qty: Optional[float] = None,
                limit_price: Optional[float] = None, stop_price: Optional[float] = None,
                session_index: Optional[int] = None) -> Optional[Order]:
        """
        Cancel-replace: terminate the old order and rest an amended copy. The new
        order keeps any fills already taken (remaining qty only is re-rested) and
        gets a fresh client_order_id. Returns the new order (or None if unknown).
        """
        old = self._resting.get(client_order_id)
        if old is None:
            return None
        from alpca.execution.order import new_client_order_id
        new = Order(
            symbol=old.symbol, side=old.side,
            qty=qty if qty is not None else self._remaining(old),
            order_type=old.order_type,
            limit_price=limit_price if limit_price is not None else old.limit_price,
            stop_price=stop_price if stop_price is not None else old.stop_price,
            tif=old.tif, extended_hours=old.extended_hours,
            client_order_id=new_client_order_id(old.strategy),
            strategy=old.strategy, intended_price=old.intended_price,
        )
        old.status = OrderStatus.CANCELED  # superseded by the replacement
        old.metadata["replaced_by"] = new.client_order_id
        self._resting.pop(client_order_id, None)
        sess = session_index if session_index is not None else old.metadata.get(_K_SESSION, 0)
        self.add(new, sess)
        return new

    # ----------------------------------------------------------------- the tick
    def on_bar(self, bar: Dict[str, float], session_index: int) -> List[BookEvent]:
        """
        Advance every resting order against one bar. Order of operations per order:
          1) DAY expiry  (a DAY order from an earlier session expires at open)
          2) STOP trigger (if the bar touches the stop price)
          3) fill        (market-style for triggered stops; through-trade for limits)
        Returns the lifecycle events that occurred this bar.
        """
        events: List[BookEvent] = []
        high = bar["high"]
        low = bar["low"]
        open_px = bar["open"]
        vol = bar.get("volume")

        for coid in list(self._resting.keys()):
            order = self._resting[coid]

            # 1) DAY expiry — a DAY order that survived its submit session expires
            if order.tif == TimeInForce.DAY and session_index > order.metadata.get(_K_SESSION, session_index):
                order.status = OrderStatus.EXPIRED
                self._resting.pop(coid, None)
                events.append(BookEvent(order, "expire"))
                continue

            buy = order.side == Side.BUY

            # 2) STOP / STOP_LIMIT trigger
            if not order.metadata.get(_K_TRIGGERED, True):
                stop = order.stop_price
                triggered = (high >= stop) if buy else (low <= stop)
                if not triggered:
                    continue  # stop not touched this bar; keep resting
                order.metadata[_K_TRIGGERED] = True
                events.append(BookEvent(order, "trigger"))
                if order.order_type == OrderType.STOP:
                    # becomes marketable: fill at the worse of stop/open (gap-aware)
                    ref = max(stop, open_px) if buy else min(stop, open_px)
                    res = self.fill_model.fill(buy, ref, self._remaining(order), bar_volume=vol)
                    self._apply(order, res.price, res.filled_qty, events)
                    if order.status.is_terminal:
                        self._resting.pop(coid, None)
                    continue
                # STOP_LIMIT: now a resting LIMIT; fall through to limit fill below

            # 3) LIMIT fill (resting limit, or a just-triggered stop-limit)
            if order.order_type in (OrderType.LIMIT, OrderType.STOP_LIMIT):
                qpos = self._queue_pos(order, buy, bar, vol) if self.use_queue_model else None
                res = self.fill_model.fill_limit(buy, order.limit_price, open_px, high, low,
                                                 self._remaining(order), bar_volume=vol,
                                                 queue_pos=qpos)
                if res.filled_qty > 0:
                    self._apply(order, res.price, res.filled_qty, events)
                    if order.status.is_terminal:
                        self._resting.pop(coid, None)

        return events

    def _queue_pos(self, order: Order, buy: bool, bar: Dict[str, float], vol):
        """Get-or-create this order's FIFO QueuePosition. `front` (shares ahead at
        join) = the displayed size on our side of the book if the bar carries a
        quote, else a bar-volume proxy."""
        qpos = order.metadata.get(_K_QUEUE)
        if qpos is None:
            from alpca.execution.queue_prob import QueuePosition
            front0 = bar.get("bid_size") if buy else bar.get("ask_size")
            if not front0:
                front0 = (vol or 0.0) * self.fill_model.participation_cap
            qpos = QueuePosition(front0 or 0.0, self.prob_func)
            order.metadata[_K_QUEUE] = qpos
        return qpos

    def _apply(self, order: Order, price: float, qty: float, events: List[BookEvent]) -> None:
        if qty <= 0:
            return
        order.add_fill(Fill(ts=time.time(), price=round(price, 4), qty=qty))
        events.append(BookEvent(order, "fill" if order.status == OrderStatus.FILLED else "partial_fill"))

    def expire_all_day_orders(self, session_index: int) -> List[BookEvent]:
        """Force-expire every resting DAY order (e.g. at end of a session)."""
        events: List[BookEvent] = []
        for coid in list(self._resting.keys()):
            order = self._resting[coid]
            if order.tif == TimeInForce.DAY:
                order.status = OrderStatus.EXPIRED
                self._resting.pop(coid, None)
                events.append(BookEvent(order, "expire"))
        return events
