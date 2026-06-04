"""
ExecutionRouter — the single chokepoint every order flows through.

Responsibilities (in order):
  1. idempotency — drop a duplicate of an already-working client_order_id
  2. SIGNAL — stamp signal_ts (if unset) and log it
  3. RISK — RiskEngine.check(); on deny, log RISK_BLOCK and return rejected
  4. SUBMIT — stamp submit_ts, log, start a perf_counter, record the submission
  5. adapter.submit(...) then poll until terminal or fill_timeout
  6. record submit->terminal rtt_ms; log ACK / FILL / REJECT
  7. keep the order for the aggregate latency report

This is where latency is captured at every stage, so no strategy or adapter has
to think about it.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Deque, Dict, List, Optional

from alpca.execution.adapters.base import BaseAdapter
from alpca.execution.order import Order, OrderStatus
from alpca.execution.order_event_log import EventType, OrderEventLog
from alpca.metrics.latency import LatencyReport, build_latency_report, percentile
from alpca.risk.risk_engine import Position, RiskEngine


class ExecutionRouter:
    def __init__(
        self,
        adapter: BaseAdapter,
        risk_engine: RiskEngine,
        event_log: Optional[OrderEventLog] = None,
        *,
        rtt_window: int = 200,
        poll_interval_s: float = 0.05,
        fill_timeout_s: float = 10.0,
    ) -> None:
        self.adapter = adapter
        self.risk = risk_engine
        self.log = event_log
        self.poll_interval_s = poll_interval_s
        self.fill_timeout_s = fill_timeout_s
        self._rtt_samples: Deque[float] = deque(maxlen=rtt_window)
        self._orders: List[Order] = []
        self._working: Dict[str, Order] = {}

    def _emit(self, order: Order, event: EventType) -> None:
        if self.log is not None:
            self.log.append(order, event)

    async def submit(
        self,
        order: Order,
        *,
        equity: float,
        positions: Optional[Dict[str, Position]] = None,
        ref_price: Optional[float] = None,
        cash: Optional[float] = None,
        fill_timeout_s: Optional[float] = None,
    ) -> Order:
        # 1. idempotency
        if order.client_order_id in self._working:
            return self._working[order.client_order_id]

        # 2. SIGNAL
        if order.signal_ts is None:
            order.mark_signal(ref_price)
        self._emit(order, EventType.SIGNAL)

        # 3. RISK
        decision = self.risk.check(order, equity=equity, positions=positions,
                                   ref_price=ref_price, cash=cash)
        if not decision.allowed:
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"risk:{decision.code}:{decision.reason}"
            self._emit(order, EventType.RISK_BLOCK)
            self._orders.append(order)
            return order

        # 4. SUBMIT (timed)
        self._working[order.client_order_id] = order
        t0 = time.perf_counter()
        order.mark_submit()
        self._emit(order, EventType.SUBMIT)
        self.risk.record_submission()

        try:
            order = await self.adapter.submit(order, ref_price=ref_price)
            if order.ack_ts is not None:
                self._emit(order, EventType.ACK)

            # 5. poll until terminal or timeout
            deadline = time.perf_counter() + (fill_timeout_s or self.fill_timeout_s)
            while not order.status.is_terminal and time.perf_counter() < deadline:
                await asyncio.sleep(self.poll_interval_s)
                order = await self.adapter.poll(order)
        finally:
            # 6. rtt + terminal logging
            rtt_ms = (time.perf_counter() - t0) * 1000.0
            self._rtt_samples.append(rtt_ms)
            order.metadata["rtt_ms"] = rtt_ms
            self._working.pop(order.client_order_id, None)

        if order.status == OrderStatus.FILLED:
            self._emit(order, EventType.FILL)
        elif order.status == OrderStatus.PARTIALLY_FILLED:
            self._emit(order, EventType.PARTIAL_FILL)
        elif order.status == OrderStatus.REJECTED:
            self._emit(order, EventType.REJECT)

        self._orders.append(order)
        return order

    # --------------------------------------------------------------- telemetry
    def rtt_stats(self) -> Dict[str, float]:
        vals = sorted(self._rtt_samples)
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "p50_ms": percentile(vals, 0.50),
            "p95_ms": percentile(vals, 0.95),
            "max_ms": vals[-1],
        }

    def latency_report(self) -> LatencyReport:
        return build_latency_report(self._orders)

    @property
    def orders(self) -> List[Order]:
        return list(self._orders)
