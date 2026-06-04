"""
Alpaca broker adapter — paper or live US equities via alpaca-py.

The sync alpaca-py SDK is run off the event loop with asyncio.to_thread so a slow
network call never stalls the strategy loop (which matters for latency accuracy).

Lifecycle stamping (the whole point of this bot):
  - submit_ts is set by the router right before calling us
  - we call mark_ack() the instant the SDK returns the accepted order
  - we poll get_order_by_id until terminal, calling add_fill() on fills

Safety: paper endpoint by default (TradingClient(paper=cfg.paper));
config.require_credentials() refuses live unless explicitly confirmed.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

from alpca.config import AlpacaConfig
from alpca.execution.adapters.base import BaseAdapter
from alpca.execution.order import Fill, Order, OrderStatus, OrderType, Side, TimeInForce


# Map Alpaca order status strings -> our OrderStatus enum.
_STATUS_MAP = {
    "new": OrderStatus.ACCEPTED,
    "accepted": OrderStatus.ACCEPTED,
    "pending_new": OrderStatus.SUBMITTED,
    "accepted_for_bidding": OrderStatus.ACCEPTED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "filled": OrderStatus.FILLED,
    "done_for_day": OrderStatus.ACCEPTED,
    "canceled": OrderStatus.CANCELED,
    "expired": OrderStatus.EXPIRED,
    "replaced": OrderStatus.ACCEPTED,
    "pending_cancel": OrderStatus.ACCEPTED,
    "pending_replace": OrderStatus.ACCEPTED,
    "rejected": OrderStatus.REJECTED,
    "suspended": OrderStatus.SUBMITTED,
    "calculated": OrderStatus.ACCEPTED,
    "stopped": OrderStatus.ACCEPTED,
}


def _map_status(raw: str) -> OrderStatus:
    return _STATUS_MAP.get(str(raw).lower(), OrderStatus.SUBMITTED)


class AlpacaAdapter(BaseAdapter):
    name = "alpaca"
    supports_modes = ("paper", "live")

    def __init__(self, config: AlpacaConfig, *, clock_cache_ttl_s: float = 30.0) -> None:
        config.require_credentials()
        self.cfg = config
        self._clock_cache_ttl_s = clock_cache_ttl_s
        self._clock_cache: tuple[float, bool] | None = None  # (fetched_at, is_open)
        self._client = None  # lazy

    # --------------------------------------------------------------- client
    @property
    def client(self):
        if self._client is None:
            from alpaca.trading.client import TradingClient

            self._client = TradingClient(
                self.cfg.api_key,
                self.cfg.secret_key,
                paper=self.cfg.paper,
            )
        return self._client

    # ----------------------------------------------------------- request build
    def _build_request(self, order: Order):
        from alpaca.trading.enums import OrderSide, TimeInForce as ATIF
        from alpaca.trading.requests import (
            LimitOrderRequest,
            MarketOrderRequest,
            StopLimitOrderRequest,
            StopOrderRequest,
        )

        side = OrderSide.BUY if order.side == Side.BUY else OrderSide.SELL
        tif_map = {
            TimeInForce.DAY: ATIF.DAY,
            TimeInForce.GTC: ATIF.GTC,
            TimeInForce.IOC: ATIF.IOC,
            TimeInForce.FOK: ATIF.FOK,
        }
        tif = tif_map.get(order.tif, ATIF.DAY)

        common = dict(
            symbol=order.symbol,
            qty=order.qty,
            side=side,
            time_in_force=tif,
            client_order_id=order.client_order_id,
        )
        if order.order_type == OrderType.MARKET:
            return MarketOrderRequest(**common)
        if order.order_type == OrderType.LIMIT:
            # extended_hours is only valid for LIMIT + DAY
            return LimitOrderRequest(
                limit_price=order.limit_price,
                extended_hours=order.extended_hours,
                **common,
            )
        if order.order_type == OrderType.STOP:
            return StopOrderRequest(stop_price=order.stop_price, **common)
        if order.order_type == OrderType.STOP_LIMIT:
            return StopLimitOrderRequest(
                stop_price=order.stop_price, limit_price=order.limit_price, **common
            )
        raise ValueError(f"unsupported order type {order.order_type}")

    # --------------------------------------------------------------- market open
    async def is_market_open(self) -> bool:
        now = time.monotonic()
        if self._clock_cache and (now - self._clock_cache[0]) < self._clock_cache_ttl_s:
            return self._clock_cache[1]
        clock = await asyncio.to_thread(self.client.get_clock)
        is_open = bool(getattr(clock, "is_open", False))
        self._clock_cache = (now, is_open)
        return is_open

    # --------------------------------------------------------------- submit
    async def submit(self, order: Order, *, ref_price: Optional[float] = None) -> Order:
        try:
            req = self._build_request(order)
            broker_order = await asyncio.to_thread(self.client.submit_order, req)
        except Exception as e:  # SDK raises on reject/validation
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"alpaca:submit:{type(e).__name__}:{e}"
            order.mark_ack()
            return order

        order.mark_ack()  # broker acknowledged
        order.broker_order_id = str(getattr(broker_order, "id", "") or "")
        self._apply_broker_order(order, broker_order)
        return order

    # --------------------------------------------------------------- poll
    async def poll(self, order: Order) -> Order:
        if not order.broker_order_id:
            return order
        try:
            bo = await asyncio.to_thread(self.client.get_order_by_id, order.broker_order_id)
        except Exception:
            return order
        self._apply_broker_order(order, bo)
        return order

    def _apply_broker_order(self, order: Order, bo) -> None:
        status = _map_status(getattr(bo, "status", "new"))
        filled_qty = float(getattr(bo, "filled_qty", 0) or 0)
        avg_price = getattr(bo, "filled_avg_price", None)

        # Record a synthetic fill that brings our running total up to filled_qty.
        delta = filled_qty - order.filled_qty
        if delta > 1e-9 and avg_price:
            order.add_fill(Fill(ts=time.time(), price=float(avg_price), qty=delta))
        else:
            if status in (OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED):
                order.status = status
                if status == OrderStatus.REJECTED and not order.reject_reason:
                    order.reject_reason = "alpaca: rejected"
            elif not order.status.is_terminal:
                order.status = status

    # --------------------------------------------------------------- cancel
    async def cancel(self, order: Order) -> None:
        if order.broker_order_id:
            try:
                await asyncio.to_thread(self.client.cancel_order_by_id, order.broker_order_id)
            except Exception:
                pass

    async def is_available(self) -> bool:
        try:
            await asyncio.to_thread(self.client.get_account)
            return True
        except Exception:
            return False
