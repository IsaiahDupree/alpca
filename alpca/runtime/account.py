"""
Cash-account realism: T+1 settlement, the PDT (pattern-day-trader) rule, and
short-borrow fee accrual.

Opt-in components a real account must respect — a naive backtest ignores them and
overstates how freely (and cheaply) it can trade:

  SettlementLedger — US equities settle **T+1** (next trading session). Sale
    proceeds are NOT immediately available to buy with; spending unsettled cash
    in a cash account is a good-faith / free-riding violation. The ledger tracks
    settled vs. pending cash and credits pending proceeds when the session
    advances by one.

  PdtGuard — a margin account under **$25,000** equity may not make a 4th
    *day trade* (open+close the same symbol in one session) within any rolling
    5-session window. Above $25k the rule doesn't apply.

  BorrowFeeLedger — a short position pays a daily borrow fee (annual_rate / 252
    of the short market value) for every session it is held open. Ignoring it
    overstates short-strategy returns.

Session-indexed: the caller advances an integer session counter (one per distinct
trading date), so "T+1", "rolling 5 sessions", and per-session borrow accrual are
counted in trading days, not wall-clock days (correct across weekends/holidays).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Set, Tuple


# --------------------------------------------------------------- T+1 settlement
class SettlementLedger:
    """
    Tracks settled vs. unsettled cash for a CASH account (T+1).

    Invariant: settled + sum(pending) == total cash held by the runner. Buys draw
    down settled cash on the fill date; sale proceeds enter `pending` and become
    settled when the session advances by `settle_lag` (default 1 = T+1).
    """

    def __init__(self, starting_cash: float, *, settle_lag: int = 1) -> None:
        self.settled = float(starting_cash)
        self.settle_lag = settle_lag
        # pending proceeds keyed by the session index on which they settle
        self._pending: Dict[int, float] = {}

    @property
    def pending_total(self) -> float:
        return sum(self._pending.values())

    @property
    def total(self) -> float:
        return self.settled + self.pending_total

    def available(self) -> float:
        """Cash that can fund a new BUY right now (settled only)."""
        return self.settled

    def advance_to(self, session_index: int) -> float:
        """Settle any pending proceeds whose settle-session is <= session_index.
        Returns the amount newly settled."""
        matured = [s for s in self._pending if s <= session_index]
        amt = 0.0
        for s in matured:
            amt += self._pending.pop(s)
        self.settled += amt
        return amt

    def record_buy(self, cost: float) -> None:
        self.settled -= cost

    def record_sell(self, proceeds: float, current_session: int) -> None:
        settle_at = current_session + self.settle_lag
        self._pending[settle_at] = self._pending.get(settle_at, 0.0) + proceeds


# --------------------------------------------------------------- PDT guard
@dataclass
class PdtGuard:
    min_equity: float = 25_000.0      # rule applies only BELOW this equity
    max_day_trades: int = 3           # 4th in the window is the violation
    window_sessions: int = 5
    # session_index -> {symbol -> set of sides executed that session}
    _sides: Dict[int, Dict[str, Set[str]]] = field(default_factory=dict)
    _day_trades: Deque[int] = field(default_factory=deque)  # one entry per day-trade

    def _prune(self, current_session: int) -> None:
        cutoff = current_session - self.window_sessions + 1
        while self._day_trades and self._day_trades[0] < cutoff:
            self._day_trades.popleft()

    def day_trade_count(self, current_session: int) -> int:
        self._prune(current_session)
        return len(self._day_trades)

    def would_be_day_trade(self, session_index: int, symbol: str, side: str) -> bool:
        """True if executing `side` on `symbol` now closes a same-session round
        trip (the opposite side already executed this session)."""
        sides = self._sides.get(session_index, {}).get(symbol.upper(), set())
        opp = "SELL" if side == "BUY" else "BUY"
        return opp in sides

    def check(self, session_index: int, symbol: str, side: str, equity: float) -> Tuple[bool, str]:
        """Pre-trade: block the order if it would be a day-trade that exceeds the
        rolling cap while equity is under the PDT threshold."""
        if equity >= self.min_equity:
            return True, ""  # PDT only restricts small accounts
        if self.would_be_day_trade(session_index, symbol, side):
            if self.day_trade_count(session_index) >= self.max_day_trades:
                return (False,
                        f"PDT: would be day-trade #{self.day_trade_count(session_index)+1} "
                        f"in {self.window_sessions} sessions with equity "
                        f"${equity:,.0f} < ${self.min_equity:,.0f}")
        return True, ""

    def record_fill(self, session_index: int, symbol: str, side: str) -> None:
        sym = symbol.upper()
        sides = self._sides.setdefault(session_index, {}).setdefault(sym, set())
        opp = "SELL" if side == "BUY" else "BUY"
        if opp in sides:
            # completing the opposite side in the same session = one day trade
            self._day_trades.append(session_index)
        sides.add(side)


# --------------------------------------------------------------- borrow fees
@dataclass
class BorrowFeeLedger:
    """
    Accrues the daily borrow fee on an open SHORT position.

    Per session held, a short pays `annual_rate / trading_days` of its short
    market value (|short qty| * price). The caller invokes accrue_for_session()
    once per new trading session with the current short notional; the fee is
    debited from cash by the runner.
    """
    annual_rate: float = 0.03
    trading_days: int = 252
    total_accrued: float = 0.0

    @property
    def daily_rate(self) -> float:
        return self.annual_rate / self.trading_days

    def accrue_for_session(self, short_market_value: float) -> float:
        """Return (and tally) the borrow fee for one session on a short of
        `short_market_value` dollars (>= 0). Returns 0 if flat/long."""
        if short_market_value <= 0:
            return 0.0
        fee = short_market_value * self.daily_rate
        self.total_accrued += fee
        return fee
