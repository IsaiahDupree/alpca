"""
Simulated broker adapter — offline fills with injected latency + slippage.

Lets the whole stack (strategy -> router -> risk -> latency ledger -> metrics)
run end-to-end with ZERO credentials, while producing realistic-looking latency
numbers and slippage so the metrics pipeline has meaningful data in tests/CI.

Determinism: pass a seed for reproducible latency/slippage draws.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Optional

from alpca.execution.adapters.base import BaseAdapter
from alpca.execution.order import Fill, Order, OrderStatus, OrderType, Side


class SimAdapter(BaseAdapter):
    name = "sim"
    supports_modes = ("sim",)

    def __init__(
        self,
        *,
        submit_latency_ms: float = 5.0,
        ack_latency_ms: float = 8.0,
        fill_latency_ms: float = 20.0,
        latency_jitter_ms: float = 4.0,
        slippage_bps_mean: float = 1.5,
        slippage_bps_std: float = 1.0,
        reject_prob: float = 0.0,
        seed: Optional[int] = None,
        sleep: bool = True,
        fill_model=None,
    ) -> None:
        self.submit_latency_ms = submit_latency_ms
        self.ack_latency_ms = ack_latency_ms
        self.fill_latency_ms = fill_latency_ms
        self.jitter = latency_jitter_ms
        self.slip_mean = slippage_bps_mean
        self.slip_std = slippage_bps_std
        self.reject_prob = reject_prob
        self._rng = random.Random(seed)
        self._sleep = sleep
        # Optional richer fill model (spread + size impact + volume cap). When
        # None, the legacy gaussian-bps slippage draw is used (back-compat).
        self.fill_model = fill_model  # if False, don't actually sleep (fast tests) but still stamp latencies

    @classmethod
    def paper_calibrated(cls, *, seed=None, sleep: bool = True, fill_model=None,
                         reject_prob: float = 0.0):
        """
        SimAdapter with latencies calibrated to MEASURED Alpaca paper REST
        (docs/BASELINE.md): submit->ack ~248 ms (120 + 128). fill_latency is a
        placeholder (~60 ms) — real ack->fill needs a market-hours fill to
        calibrate. Use this preset for realistic offline timing; the bare
        constructor keeps fast defaults for unit tests.
        """
        return cls(submit_latency_ms=120.0, ack_latency_ms=128.0,
                   fill_latency_ms=60.0, latency_jitter_ms=25.0,
                   seed=seed, sleep=sleep, fill_model=fill_model,
                   reject_prob=reject_prob)

    def _draw_latency(self, base_ms: float) -> float:
        return max(0.0, base_ms + self._rng.uniform(-self.jitter, self.jitter))

    async def _delay(self, ms: float) -> None:
        if self._sleep and ms > 0:
            await asyncio.sleep(ms / 1000.0)

    async def submit(self, order: Order, *, ref_price: Optional[float] = None) -> Order:
        price = ref_price or order.limit_price or order.intended_price
        if price is None or price <= 0:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "sim: no reference price"
            order.mark_ack()
            return order

        order.broker_order_id = f"sim-{self._rng.getrandbits(32):08x}"

        # submit -> ack
        await self._delay(self._draw_latency(self.submit_latency_ms))
        await self._delay(self._draw_latency(self.ack_latency_ms))
        order.mark_ack()

        # random reject (e.g. modeling broker rejects)
        if self._rng.random() < self.reject_prob:
            order.status = OrderStatus.REJECTED
            order.reject_reason = "sim: random reject"
            return order

        # ack -> fill
        await self._delay(self._draw_latency(self.fill_latency_ms))

        buy = order.side == Side.BUY
        meta = order.metadata
        bar_high = meta.get("bar_high")
        bar_low = meta.get("bar_low")
        bar_open = meta.get("bar_open", price)
        bar_volume = meta.get("bar_volume")

        # LIMIT with bar context + a fill model: real through-trade test. If the
        # bar never traded through the limit, the order does NOT fill (it stays
        # ACCEPTED/working) rather than always filling at the clamp.
        if (order.order_type == OrderType.LIMIT and order.limit_price is not None
                and self.fill_model is not None
                and bar_high is not None and bar_low is not None):
            res = self.fill_model.fill_limit(buy, order.limit_price, bar_open,
                                             bar_high, bar_low, order.qty,
                                             bar_volume=bar_volume)
            if res.filled_qty <= 0:
                return order  # resting, no fill this bar
            order.add_fill(Fill(ts=time.time(), price=round(res.price, 4), qty=res.filled_qty))
            return order

        # Otherwise: market-style fill — richer fill model if provided, else the
        # legacy gaussian slippage draw.
        if self.fill_model is not None:
            res = self.fill_model.fill(buy, price, order.qty, bar_volume=bar_volume)
            fill_price = res.price
        else:
            slip_bps = self._rng.gauss(self.slip_mean, self.slip_std)
            fill_price = price * (1.0 + slip_bps / 10_000.0) if buy else price * (1.0 - slip_bps / 10_000.0)

        # marketable-limit clamp (no bar context): never fill worse than the limit
        if order.order_type == OrderType.LIMIT and order.limit_price is not None:
            if buy and fill_price > order.limit_price:
                fill_price = order.limit_price
            if not buy and fill_price < order.limit_price:
                fill_price = order.limit_price

        order.add_fill(Fill(ts=time.time(), price=round(fill_price, 4), qty=order.qty))
        return order
