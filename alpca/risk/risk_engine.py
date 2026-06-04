"""
Pre-trade risk engine.

Every order MUST pass RiskEngine.check() before it reaches any broker/sim. The
check is pure and side-effect-light; the caller records the order via
record_submission() only once it actually submits, so the rate limiter reflects
real submissions.

Gates (all configurable via RiskConfig / env):
  - global kill-switch HALT
  - positive qty + price sanity
  - forbidden-symbol list
  - per-order notional cap
  - sliding-window orders/min rate limit
  - daily-loss auto-halt (vs. day-start equity)
  - max open positions
  - post-trade concentration cap (per symbol, vs. equity)
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Dict, Optional, Sequence

from alpca.config import RiskConfig
from alpca.execution.order import Order, Side


@dataclass
class RiskDecision:
    allowed: bool
    code: str = "OK"
    reason: str = ""

    def __bool__(self) -> bool:  # allow `if decision:`
        return self.allowed


@dataclass
class Position:
    symbol: str
    qty: float          # signed (negative = short)
    avg_price: float

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.avg_price


# handler(code, message) called when a critical limit trips (wire to kill switch)
BreachHandler = Callable[[str, str], None]


class RiskEngine:
    def __init__(
        self,
        config: RiskConfig,
        *,
        forbidden_symbols: Optional[Sequence[str]] = None,
        day_start_equity: Optional[float] = None,
        breach_handler: Optional[BreachHandler] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cfg = config
        self.forbidden = {s.upper() for s in (forbidden_symbols or ())}
        self.day_start_equity = day_start_equity
        self.breach_handler = breach_handler
        self._now = now
        self._halted = False
        self._halt_reason = ""
        self._submission_ts: Deque[float] = deque()

    # ---------------------------------------------------------------- halting
    @property
    def halted(self) -> bool:
        return self._halted

    def halt(self, reason: str = "manual") -> None:
        self._halted = True
        self._halt_reason = reason
        if self.breach_handler:
            self.breach_handler("HALT", reason)

    def resume(self) -> None:
        self._halted = False
        self._halt_reason = ""

    def set_day_start_equity(self, equity: float) -> None:
        self.day_start_equity = equity

    # ----------------------------------------------------------------- rate
    def _prune_rate_window(self) -> None:
        cutoff = self._now() - 60.0
        while self._submission_ts and self._submission_ts[0] < cutoff:
            self._submission_ts.popleft()

    def record_submission(self) -> None:
        """Call right after a real submit so the rate limiter is accurate."""
        self._submission_ts.append(self._now())

    # ----------------------------------------------------------------- check
    def check(
        self,
        order: Order,
        *,
        equity: float,
        positions: Optional[Dict[str, Position]] = None,
        ref_price: Optional[float] = None,
        cash: Optional[float] = None,
    ) -> RiskDecision:
        positions = positions or {}

        if self._halted:
            return RiskDecision(False, "HALTED", f"trading halted: {self._halt_reason}")

        if order.qty <= 0 or not math.isfinite(order.qty):
            return RiskDecision(False, "BAD_QTY", "order qty must be positive and finite")

        price = (
            ref_price
            or order.limit_price
            or order.intended_price
            or (positions.get(order.symbol).avg_price if positions.get(order.symbol) else None)
        )
        if price is None or not math.isfinite(price) or price <= 0:
            return RiskDecision(False, "NO_PRICE", "no positive finite reference price for risk sizing")

        sym = order.symbol.upper()
        if sym in self.forbidden:
            return RiskDecision(False, "FORBIDDEN", f"{sym} is on the forbidden list")

        order_notional = order.qty * price
        if order_notional > self.cfg.max_order_notional:
            return RiskDecision(
                False, "MAX_ORDER_NOTIONAL",
                f"order notional ${order_notional:,.0f} > cap ${self.cfg.max_order_notional:,.0f}",
            )

        # buying power: a BUY cannot cost more than available cash (no margin).
        # Only enforced when cash is supplied by the caller.
        if (order.side == Side.BUY and cash is not None
                and self.cfg.enforce_buying_power and order_notional > cash):
            return RiskDecision(
                False, "INSUFFICIENT_BUYING_POWER",
                f"buy notional ${order_notional:,.0f} > available cash ${cash:,.0f}",
            )

        # short-sale gate: unless shorting is enabled, a SELL may only reduce or
        # close an existing long — never open or extend a short. Gate ONLY SELLs:
        # a BUY can only reduce/cover a short (risk-decreasing), so a partial-cover
        # buy that leaves a smaller short must NOT be blocked here.
        existing_pos = positions.get(sym)
        existing_qty = existing_pos.qty if existing_pos else 0.0
        signed = order.qty if order.side == Side.BUY else -order.qty
        resulting_qty = existing_qty + signed
        if (not self.cfg.allow_short and order.side == Side.SELL
                and resulting_qty < -1e-9):
            return RiskDecision(
                False, "SHORT_NOT_ALLOWED",
                f"SELL of {order.qty} would leave {resulting_qty:+.0f} {sym} "
                f"(short); shorting disabled",
            )

        # rate limit (sliding 60s window)
        self._prune_rate_window()
        if len(self._submission_ts) >= self.cfg.max_orders_per_min:
            return RiskDecision(
                False, "RATE_LIMIT",
                f"{len(self._submission_ts)} orders in last 60s >= {self.cfg.max_orders_per_min}/min",
            )

        # daily-loss auto-halt
        if self.day_start_equity:
            floor = self.day_start_equity * (1.0 - self.cfg.daily_loss_pct)
            if equity <= floor:
                self.halt(f"daily loss breached: equity ${equity:,.0f} <= floor ${floor:,.0f}")
                return RiskDecision(False, "DAILY_LOSS", self._halt_reason)

        # max open positions (only matters if this opens a NEW symbol)
        opening_new = sym not in positions and order.side == Side.BUY
        if opening_new and len(positions) >= self.cfg.max_open_positions:
            return RiskDecision(
                False, "MAX_POSITIONS",
                f"{len(positions)} open positions >= cap {self.cfg.max_open_positions}",
            )

        # post-trade concentration (per symbol) — signed-aware: the projected
        # exposure is |resulting signed qty| * price, which is correct whether
        # the order opens, adds to, reduces, flips, or covers a long/short.
        projected = abs(resulting_qty) * price
        if equity > 0 and (projected / equity) > self.cfg.max_concentration_pct:
            return RiskDecision(
                False, "CONCENTRATION",
                f"{sym} would be {projected/equity:.0%} of equity > cap {self.cfg.max_concentration_pct:.0%}",
            )

        return RiskDecision(True, "OK", "")
