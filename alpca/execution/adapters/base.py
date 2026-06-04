"""
Broker adapter contract.

An adapter only translates a unified Order into a broker-specific submission and
reports fills back. The ExecutionRouter owns risk checks, idempotency, latency
timing, and the event ledger — adapters stay thin.

Adapters are async so a slow network call never blocks the event loop (the
Alpaca SDK is sync, so its adapter runs calls via asyncio.to_thread).
"""

from __future__ import annotations

import abc
from typing import Optional

from alpca.execution.order import Order


class BaseAdapter(abc.ABC):
    name: str = "base"
    supports_modes: tuple = ()  # subset of ("sim", "paper", "live")

    @abc.abstractmethod
    async def submit(self, order: Order, *, ref_price: Optional[float] = None) -> Order:
        """
        Submit the order to the venue. Must:
          - set order.broker_order_id (if the venue assigns one)
          - call order.mark_ack() when the venue acknowledges
          - add fills via order.add_fill(...) if the fill is known synchronously
        Returns the same order, mutated.
        """
        raise NotImplementedError

    async def poll(self, order: Order) -> Order:
        """Refresh the order from the venue (status + any new fills). Default: no-op."""
        return order

    async def cancel(self, order: Order) -> None:
        """Cancel a working order. Default: no-op."""
        return None

    async def is_available(self) -> bool:
        return True
