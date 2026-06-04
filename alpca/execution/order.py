"""
Core order/fill data contract — latency-first.

Unlike a typical bot that records a single `latency_ms`, every Order here carries
its full lifecycle as wall-clock timestamps:

    signal_ts  -> the strategy emitted the signal
    submit_ts  -> we called the broker's submit API
    ack_ts     -> the broker acknowledged/accepted the order
    fill_ts    -> the order reached a terminal filled state

From those, per-stage latencies and slippage-vs-intended are computed on demand.
All timestamps are epoch seconds from time.time() (UTC, comparable across stages).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class TimeInForce(str, Enum):
    DAY = "DAY"
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class OrderStatus(str, Enum):
    NEW = "NEW"                      # created locally, not yet submitted
    SUBMITTED = "SUBMITTED"          # submit() called
    ACCEPTED = "ACCEPTED"            # broker acknowledged
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

    @property
    def is_terminal(self) -> bool:
        return self in (
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
        )


@dataclass
class Fill:
    """A single (partial) fill event."""
    ts: float
    price: float
    qty: float
    fee: float = 0.0


def _now() -> float:
    return time.time()


def new_client_order_id(strategy: str = "", seq: Optional[int] = None) -> str:
    """
    Runner-attributable client order id, capped at 48 chars (safe for Alpaca).
    Form: a-{strategy}-{seq?}-{uuid8}
    """
    base = "a"
    if strategy:
        base += "-" + strategy.replace(" ", "")[:16]
    if seq is not None:
        base += f"-{seq}"
    base += "-" + uuid.uuid4().hex[:8]
    return base[:48]


@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: TimeInForce = TimeInForce.DAY
    extended_hours: bool = False

    # identity / provenance
    client_order_id: str = field(default_factory=new_client_order_id)
    broker_order_id: Optional[str] = None
    strategy: str = ""

    # lifecycle timestamps (epoch seconds) — the heart of latency measurement
    signal_ts: Optional[float] = None
    submit_ts: Optional[float] = None
    ack_ts: Optional[float] = None
    fill_ts: Optional[float] = None

    # results
    status: OrderStatus = OrderStatus.NEW
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    fills: List[Fill] = field(default_factory=list)
    reject_reason: Optional[str] = None

    # slippage reference: the price the strategy *expected* at signal time
    intended_price: Optional[float] = None

    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def mark_signal(self, intended_price: Optional[float] = None) -> "Order":
        self.signal_ts = _now()
        if intended_price is not None:
            self.intended_price = intended_price
        return self

    def mark_submit(self) -> "Order":
        self.submit_ts = _now()
        if self.status == OrderStatus.NEW:
            self.status = OrderStatus.SUBMITTED
        return self

    def mark_ack(self) -> "Order":
        self.ack_ts = _now()
        if self.status in (OrderStatus.NEW, OrderStatus.SUBMITTED):
            self.status = OrderStatus.ACCEPTED
        return self

    def add_fill(self, fill: Fill) -> "Order":
        self.fills.append(fill)
        total_qty = sum(f.qty for f in self.fills)
        if total_qty > 0:
            self.avg_fill_price = sum(f.price * f.qty for f in self.fills) / total_qty
        self.filled_qty = total_qty
        self.fill_ts = fill.ts
        if total_qty >= self.qty - 1e-9:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
        return self

    # --------------------------------------------------------------- latencies
    @staticmethod
    def _ms(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            return None
        return (b - a) * 1000.0

    @property
    def signal_to_submit_ms(self) -> Optional[float]:
        return self._ms(self.signal_ts, self.submit_ts)

    @property
    def submit_to_ack_ms(self) -> Optional[float]:
        return self._ms(self.submit_ts, self.ack_ts)

    @property
    def ack_to_fill_ms(self) -> Optional[float]:
        return self._ms(self.ack_ts, self.fill_ts)

    @property
    def submit_to_fill_ms(self) -> Optional[float]:
        return self._ms(self.submit_ts, self.fill_ts)

    @property
    def signal_to_fill_ms(self) -> Optional[float]:
        return self._ms(self.signal_ts, self.fill_ts)

    @property
    def notional(self) -> float:
        ref = self.limit_price or self.intended_price or self.avg_fill_price or 0.0
        return abs(self.qty) * float(ref)

    @property
    def slippage_bps(self) -> Optional[float]:
        """
        Signed slippage in basis points vs the strategy's intended price.
        Positive = worse than intended (paid more on a buy / received less on a sell).
        """
        if self.intended_price is None or self.avg_fill_price is None or self.intended_price == 0:
            return None
        diff = self.avg_fill_price - self.intended_price
        if self.side == Side.SELL:
            diff = -diff
        return (diff / self.intended_price) * 10_000.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["side"] = self.side.value
        d["order_type"] = self.order_type.value
        d["tif"] = self.tif.value
        d["status"] = self.status.value
        d["fills"] = [asdict(f) for f in self.fills]
        d["latency"] = {
            "signal_to_submit_ms": self.signal_to_submit_ms,
            "submit_to_ack_ms": self.submit_to_ack_ms,
            "ack_to_fill_ms": self.ack_to_fill_ms,
            "submit_to_fill_ms": self.submit_to_fill_ms,
            "signal_to_fill_ms": self.signal_to_fill_ms,
            "slippage_bps": self.slippage_bps,
        }
        return d
