"""
Strategy contract.

A strategy consumes OHLCV bars one at a time via on_bar(bar) and returns a
Signal. The runner/backtester turns a BUY/SELL/EXIT Signal into an Order
(stamping signal_ts at emit and intended_price = the reference price the strategy
saw), so strategies themselves stay free of any broker/latency concerns.

Bar schema (dict): {open, high, low, close, volume, timestamp, symbol}
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Signal sides. EXIT means "flatten whatever position this strategy holds".
BUY = "BUY"
SELL = "SELL"
EXIT = "EXIT"
HOLD = "HOLD"


@dataclass
class Signal:
    side: str = HOLD
    strength: float = 0.0          # [0,1], used for position sizing
    reason: str = ""
    price: Optional[float] = None  # reference price acted on (becomes intended_price)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Resting-order intent. Default MARKET = the legacy "act at next open" path.
    # LIMIT/STOP/STOP_LIMIT make the runner REST the order in the open-order book
    # so it works across bars (fills on a real through-trade / stop trigger).
    order_type: str = "MARKET"     # MARKET | LIMIT | STOP | STOP_LIMIT
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tif: str = "DAY"               # DAY | GTC | IOC | FOK

    @property
    def is_actionable(self) -> bool:
        return self.side in (BUY, SELL, EXIT) and (self.side == EXIT or self.strength > 0)

    @property
    def is_resting(self) -> bool:
        """True if this signal asks for a resting (working) order, not a market order."""
        return self.order_type in ("LIMIT", "STOP", "STOP_LIMIT")


def hold(reason: str = "") -> Signal:
    return Signal(side=HOLD, strength=0.0, reason=reason)


class Strategy(abc.ABC):
    """Base class. Concrete strategies implement on_bar(bar) -> Signal."""

    #: short id used in client_order_ids, metrics, the registry
    name: str = "base"

    def __init__(self) -> None:
        self._in_position: bool = False
        self._entry_price: float = 0.0

    @abc.abstractmethod
    def on_bar(self, bar: Dict[str, float]) -> Signal:
        ...

    def reset(self) -> None:
        """Clear internal state (between backtests / re-arms)."""
        self._in_position = False
        self._entry_price = 0.0

    # convenience for subclasses ------------------------------------------------
    def _enter(self, price: float, strength: float, reason: str, **meta) -> Signal:
        self._in_position = True
        self._entry_price = price
        return Signal(side=BUY, strength=strength, reason=reason, price=price, metadata=meta)

    def _exit(self, price: float, reason: str, **meta) -> Signal:
        self._in_position = False
        self._entry_price = 0.0
        return Signal(side=EXIT, strength=1.0, reason=reason, price=price, metadata=meta)

    def _short(self, price: float, strength: float, reason: str, **meta) -> Signal:
        """Open a SHORT (a SELL from flat). Requires RiskConfig.allow_short on the
        runner; otherwise the SELL is rejected SHORT_NOT_ALLOWED and _in_position
        stays False. The runner's on_fill() confirms the actual fill."""
        self._in_position = True
        self._entry_price = price
        return Signal(side=SELL, strength=strength, reason=reason, price=price, metadata=meta)

    # resting-order intents -----------------------------------------------------
    # These do NOT flip _in_position (the order may rest unfilled); the runner
    # calls on_fill() when/if it actually fills so the strategy can sync state.
    def _rest_buy_stop(self, stop_price: float, strength: float = 1.0, reason: str = "",
                       tif: str = "GTC", **meta) -> Signal:
        return Signal(side=BUY, strength=strength, reason=reason, price=stop_price,
                      order_type="STOP", stop_price=stop_price, tif=tif, metadata=meta)

    def _rest_buy_limit(self, limit_price: float, strength: float = 1.0, reason: str = "",
                        tif: str = "GTC", **meta) -> Signal:
        return Signal(side=BUY, strength=strength, reason=reason, price=limit_price,
                      order_type="LIMIT", limit_price=limit_price, tif=tif, metadata=meta)

    # fill hook -----------------------------------------------------------------
    def on_fill(self, side: str, qty: float, price: float) -> None:
        """
        Called by the runner when one of THIS strategy's orders fills — including
        a resting limit/stop order that fills bars after it was placed. Default
        no-op; strategies that use resting entries override this to sync their
        own _in_position / stop state with reality (a resting order may never
        fill, so the strategy can't assume it did). `side` is "BUY"/"SELL".
        """
        return None

    # session hook --------------------------------------------------------------
    def on_session_start(self) -> None:
        """
        Called by the runner at the start of each NEW trading session (ET-date
        change), before that session's first on_bar. Default no-op. Strategies
        whose rolling state must NOT cross an overnight gap (e.g. an intraday
        microstructure window) override this to reset that state — otherwise a
        rolling window would straddle the close, mixing yesterday's late bars
        with today's open. Price-based indicators (RSI/z-score) legitimately use
        overnight gaps, so they leave this a no-op on purpose.
        """
        return None
