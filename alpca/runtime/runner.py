"""
Live (or replay) trading runner — the actual bot loop.

For each incoming bar it:
  1. advances any RESTING orders (open-order book) against the bar — fills,
     stop triggers, and DAY expiries that happened intrabar
  2. feeds the bar to the strategy -> Signal
  3. on an actionable signal, either:
       - rests a LIMIT/STOP order in the open-order book (works across bars), or
       - submits a MARKET order through the ExecutionRouter (risk + latency + ledger)
  4. updates internal cash/position accounting (so equity + concentration are real)

Positions are SIGNED (positive long, negative short). All fill accounting flows
through the verified `apply_fill` (alpca.runtime.position_math), so long, short,
add, reduce, close, and FLIP are all handled by one tested function.

Broker-agnostic: pass a SimAdapter-backed router for offline replay, or an
AlpacaAdapter-backed router for real paper.

Opt-in realism layers (all default off): require_regular_hours (calendar gate),
settlement (T+1), pdt (PDT guard), open_order_book (resting orders), borrow
(short-borrow fee). Shorting itself is gated by RiskConfig.allow_short.

The runner records a per-bar equity curve + signed flat<->position round-trip
trades, so a runner-driven backtest produces the same BacktestResult analytics as
run_backtest (see to_result()).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from alpca.execution.order import Order, OrderType, Side, TimeInForce
from alpca.execution.order_event_log import EventType
from alpca.execution.router import ExecutionRouter
from alpca.metrics.latency import LatencyReport
from alpca.risk.risk_engine import Position
from alpca.runtime.position_math import apply_fill
from alpca.strategies.base import BUY, EXIT, SELL, Strategy


@dataclass
class RunnerStats:
    bars_seen: int = 0
    signals: int = 0
    orders_submitted: int = 0
    fills: int = 0
    rejects: int = 0
    skipped_off_session: int = 0
    skipped_unsettled: int = 0   # BUY blocked: not enough SETTLED cash (T+1)
    skipped_pdt: int = 0         # order blocked by the PDT day-trade guard
    resting_added: int = 0       # resting limit/stop orders placed in the book
    resting_filled: int = 0      # resting orders that filled (incl. partials)
    resting_expired: int = 0     # DAY resting orders that expired
    resting_canceled: int = 0    # resting orders canceled (e.g. on exit)
    shorts_opened: int = 0       # short positions opened
    borrow_paid: float = 0.0     # cumulative short-borrow fees debited
    realized_pnl: float = 0.0
    feed_latency: Dict = field(default_factory=dict)


class LiveRunner:
    def __init__(
        self,
        strategy: Strategy,
        symbol: str,
        router: ExecutionRouter,
        *,
        starting_equity: float = 100_000.0,
        target_notional_pct: float = 0.20,
        require_regular_hours: bool = False,
        allow_extended: bool = False,
        settlement=None,       # SettlementLedger | None  (T+1 settled-cash gating)
        pdt=None,              # PdtGuard | None          (PDT day-trade gating)
        open_order_book=None,  # OpenOrderBook | None     (resting limit/stop orders)
        borrow=None,           # BorrowFeeLedger | None   (short-borrow fee accrual)
    ) -> None:
        self.strategy = strategy
        self.symbol = symbol
        self.router = router
        self.target_notional_pct = target_notional_pct
        self.require_regular_hours = require_regular_hours
        self.allow_extended = allow_extended

        # Opt-in cash-account realism. Session-indexed by trading DATE, so they
        # need real epoch timestamps on bars to advance correctly.
        self.settlement = settlement
        self.pdt = pdt
        self.borrow = borrow
        self._session_index = -1
        self._session_date = None

        # Opt-in resting-order book: LIMIT/STOP signals rest here across bars.
        self.book = open_order_book
        self._working_coid: Optional[str] = None  # the symbol's resting order id

        self.cash = starting_equity
        self.starting_equity = starting_equity
        self._positions: Dict[str, Position] = {}
        self._last_price: float = 0.0
        self.stats = RunnerStats()
        self._stop = False

        # analytics: per-bar equity curve + signed round-trip trades.
        self.equity_curve: List[float] = []
        self._trades: list = []
        self._cur_trade = None
        self._bar_ts: float = 0.0
        # integer UTC-day of the last bar, for the strategy.on_session_start() hook.
        # RTH bars within a session share a UTC day; it only changes across the
        # overnight gap, so this is a cheap, calendar-free session-boundary proxy.
        # Synthetic integer-ts test bars all map to day 0 -> hook never re-fires.
        self._session_day = None

    def _advance_session(self, ts: float) -> None:
        """Bump the trading-session counter when the ET date changes; settle T+1
        proceeds and accrue the short-borrow fee for the new session."""
        from alpca.data.calendar import session_date
        d = session_date(ts)
        if d != self._session_date:
            self._session_date = d
            self._session_index += 1
            if self.settlement is not None:
                self.settlement.advance_to(self._session_index)
            # short-borrow fee on the position held INTO this session (no fills
            # have happened yet this bar, so the position is the overnight one).
            if self.borrow is not None:
                pos = self._positions.get(self.symbol)
                if pos is not None and pos.qty < 0 and self._last_price > 0:
                    smv = abs(pos.qty) * self._last_price
                    fee = self.borrow.accrue_for_session(smv)
                    self.cash -= fee
                    self.stats.borrow_paid += fee

    def _journal(self, order: Order, event: EventType) -> None:
        log = getattr(self.router, "log", None)
        if log is not None:
            log.append(order, event)

    # ------------------------------------------------------------ accounting
    @property
    def position_qty(self) -> float:
        p = self._positions.get(self.symbol)
        return p.qty if p else 0.0

    @property
    def equity(self) -> float:
        # signed qty * price is correct for shorts: a price rise lowers equity.
        held = sum(p.qty * self._last_price for p in self._positions.values())
        return self.cash + held

    def _account_fill(self, side: Side, qty: float, price: float) -> None:
        """Apply a single fill DELTA (qty @ price) to cash/positions/ledgers via
        the verified signed-position math. Handles long/short/add/reduce/close/
        flip. Used by both the market path and per-bar resting partial fills."""
        if qty <= 0 or price <= 0:
            return
        if self.settlement is not None:
            if side == Side.BUY:
                self.settlement.record_buy(qty * price)
            else:
                self.settlement.record_sell(qty * price, self._session_index)
        if self.pdt is not None:
            self.pdt.record_fill(self._session_index, self.symbol, side.value)

        existing = self._positions.get(self.symbol)
        pre_qty = existing.qty if existing else 0.0
        pre_avg = existing.avg_price if existing else 0.0

        self._record_trade(pre_qty, side, qty, price)

        eff = apply_fill(pre_qty, pre_avg, side.value, qty, price)
        self.cash += eff.cash_delta
        self.stats.realized_pnl += eff.realized
        if eff.opened_qty > 0 and pre_qty * eff.new_qty <= 0 and eff.new_qty < 0:
            # newly opened (or flipped into) a short
            self.stats.shorts_opened += 1
        if abs(eff.new_qty) < 1e-9:
            self._positions.pop(self.symbol, None)
        else:
            self._positions[self.symbol] = Position(self.symbol, eff.new_qty, eff.new_avg)

    def _record_trade(self, pre_qty: float, side: Side, qty: float, price: float) -> None:
        """Record signed flat<->position round trips as Trades. Trade.qty is
        stored SIGNED (negative = short) so Trade.pnl = (exit-entry)*qty is
        correct for both directions."""
        from alpca.backtest.engine import Trade
        signed = qty if side == Side.BUY else -qty
        new_qty = pre_qty + signed
        pre_flat = abs(pre_qty) < 1e-9
        new_flat = abs(new_qty) < 1e-9
        same_dir = (not pre_flat) and ((pre_qty > 0) == (signed > 0))

        if pre_flat:
            self._cur_trade = Trade(
                symbol=self.symbol, entry_ts=self._bar_ts, entry_price=price,
                entry_ref=price, qty=signed, reason_in="")
            return
        if same_dir:
            t = self._cur_trade
            if t is not None:
                tot = abs(t.qty) + qty
                t.entry_price = (t.entry_price * abs(t.qty) + price * qty) / tot
                t.qty = new_qty
            return
        # opposite direction: reduce / close / flip
        if new_flat:
            self._close_cur_trade(price)
            return
        if (pre_qty > 0) != (new_qty > 0):  # flip
            self._close_cur_trade(price)
            self._cur_trade = Trade(
                symbol=self.symbol, entry_ts=self._bar_ts, entry_price=price,
                entry_ref=price, qty=new_qty, reason_in="flip")
            return
        # partial reduce — keep open, shrink magnitude (keep sign + entry)
        if self._cur_trade is not None:
            self._cur_trade.qty = new_qty

    def _close_cur_trade(self, price: float) -> None:
        if self._cur_trade is not None:
            t = self._cur_trade
            t.exit_ts = self._bar_ts
            t.exit_price = price
            t.exit_ref = price
            self._trades.append(t)
            self._cur_trade = None

    def _apply_fill(self, order: Order) -> None:
        """Market-path fill: a single full fill, so the delta is the whole order."""
        self._account_fill(order.side, order.filled_qty, order.avg_fill_price or 0.0)

    # ---------------------------------------------------------------- loop
    def stop(self) -> None:
        self._stop = True

    async def run(self, bar_source) -> RunnerStats:
        """Consume bars from any async-iterable bar source until exhausted/stopped."""
        async for bar in bar_source:
            if self._stop:
                break
            await self._on_bar(bar)
            self.equity_curve.append(self.equity)
        lat = getattr(bar_source, "latency", None)
        if lat is not None and hasattr(lat, "stats"):
            self.stats.feed_latency = lat.stats()
        return self.stats

    # ------------------------------------------------------------ book tick
    def _process_book(self, bar: Dict[str, float]) -> None:
        """Advance resting orders against THIS bar; apply fills/expiries."""
        for ev in self.book.on_bar(bar, self._session_index):
            order = ev.order
            if ev.kind == "trigger":
                self._journal(order, EventType.TRIGGER)
            elif ev.kind in ("fill", "partial_fill"):
                f = order.fills[-1]  # the delta filled on THIS bar
                self._account_fill(order.side, f.qty, f.price)
                self.stats.fills += 1
                self.stats.resting_filled += 1
                self.strategy.on_fill(order.side.value, f.qty, f.price)
                self._journal(order, EventType.FILL if ev.kind == "fill"
                              else EventType.PARTIAL_FILL)
                if ev.kind == "fill" and order.client_order_id == self._working_coid:
                    self._working_coid = None
            elif ev.kind == "expire":
                self.stats.resting_expired += 1
                self._journal(order, EventType.EXPIRE)
                if order.client_order_id == self._working_coid:
                    self._working_coid = None

    # ------------------------------------------------------------ gates
    def _gate_blocked(self, order: Order, ref: float, side_str: str) -> bool:
        """PDT + T+1 settled-cash gates shared by the market and resting paths.
        Returns True (and bumps a stat) if the order must be blocked."""
        if self.pdt is not None:
            ok, _reason = self.pdt.check(self._session_index, self.symbol, side_str, self.equity)
            if not ok:
                self.stats.skipped_pdt += 1
                return True
        # settled-cash gate only constrains a BUY that OPENS/ADDS to a long (cash
        # outflow). A BUY that covers a short isn't constrained here.
        if self.settlement is not None and order.side == Side.BUY:
            if order.qty * ref > self.settlement.available() + 1e-9:
                self.stats.skipped_unsettled += 1
                return True
        return False

    # ------------------------------------------------------------ the tick
    async def _on_bar(self, bar: Dict[str, float]) -> None:
        self.stats.bars_seen += 1
        close = bar["close"]
        self._last_price = close
        ts = float(bar.get("timestamp", 0) or 0)
        self._bar_ts = ts
        if (self.settlement is not None or self.pdt is not None
                or self.book is not None or self.borrow is not None):
            self._advance_session(ts)

        # session-start hook: reset strategies' intraday-only rolling state at the
        # overnight boundary (always on, independent of the opt-in ledgers above).
        day = int(ts // 86400)
        if self._session_day is not None and day != self._session_day:
            self.strategy.on_session_start()
        self._session_day = day

        # 1) resting orders fill/trigger/expire against THIS bar (intrabar), before
        #    the strategy acts on the close — no look-ahead.
        if self.book is not None:
            self._process_book(bar)

        sig = self.strategy.on_bar(bar)  # always consume (indicator continuity)
        if not sig.is_actionable:
            return

        # Calendar gate: never act outside a tradeable session.
        if self.require_regular_hours:
            from alpca.data.calendar import is_tradeable
            if not is_tradeable(ts, allow_extended=self.allow_extended):
                self.stats.skipped_off_session += 1
                return

        self.stats.signals += 1
        ref = sig.price if sig.price is not None else close

        # Guard a degenerate reference price: a zero/negative/NaN ref would either
        # divide-by-zero in position sizing or submit a nonsense order. Skip it.
        if not math.isfinite(ref) or ref <= 0:
            return

        # 2a) RESTING-ORDER intent (LIMIT/STOP) -> open-order book
        if self.book is not None and sig.is_resting:
            side_str = BUY if sig.side == BUY else "SELL"
            self._handle_resting_signal(sig, ref, side_str)
            return

        # 2b) MARKET path — translate the signal + current SIGNED position into a
        #     concrete order. Cancel any working resting setup first.
        if self.book is not None and self._working_coid:
            canceled = self.book.cancel(self._working_coid)
            if canceled is not None:
                self.stats.resting_canceled += 1
                self._journal(canceled, EventType.CANCEL)
            self._working_coid = None

        pos_qty = self.position_qty
        flat = abs(pos_qty) < 1e-9
        size = max(1, int((self.equity * self.target_notional_pct) / ref))

        order = None
        if sig.side == BUY:
            if flat:
                order = self._mk(Side.BUY, size)          # open long
            elif pos_qty < 0:
                order = self._mk(Side.BUY, abs(pos_qty))  # cover short
            # already long -> no pyramiding in the market path
        elif sig.side == SELL:
            if flat:
                order = self._mk(Side.SELL, size)         # open short (risk-gated)
            elif pos_qty > 0:
                order = self._mk(Side.SELL, pos_qty)      # sell the long
            # already short -> ignore
        elif sig.side == EXIT:
            if pos_qty > 0:
                order = self._mk(Side.SELL, pos_qty)      # flatten long
            elif pos_qty < 0:
                order = self._mk(Side.BUY, abs(pos_qty))  # flatten short
        if order is None:
            return

        side_str = order.side.value
        if self._gate_blocked(order, ref, side_str):
            return

        order.mark_signal(intended_price=ref)
        self.stats.orders_submitted += 1
        res = await self.router.submit(order, equity=self.equity,
                                       positions=dict(self._positions), ref_price=ref,
                                       cash=self.cash)
        if res.status.value == "FILLED":
            self.stats.fills += 1
            self._apply_fill(res)
            self.strategy.on_fill(res.side.value, res.filled_qty, res.avg_fill_price or 0.0)
        elif res.status.value == "REJECTED":
            self.stats.rejects += 1

    def _mk(self, side: Side, qty: float) -> Order:
        return Order(symbol=self.symbol, side=side, qty=qty,
                     order_type=OrderType.MARKET, strategy=self.strategy.name)

    def _handle_resting_signal(self, sig, ref: float, side_str: str) -> None:
        """Place / amend a resting LIMIT/STOP order in the open-order book."""
        qty = max(1, int((self.equity * self.target_notional_pct) / ref))
        side = Side.BUY if sig.side == BUY else Side.SELL
        order = Order(symbol=self.symbol, side=side, qty=qty,
                      order_type=OrderType(sig.order_type),
                      limit_price=sig.limit_price, stop_price=sig.stop_price,
                      tif=TimeInForce(sig.tif), strategy=self.strategy.name)
        order.mark_signal(intended_price=ref)

        decision = self.router.risk.check(order, equity=self.equity,
                                          positions=dict(self._positions),
                                          ref_price=ref, cash=self.cash)
        if not decision.allowed:
            from alpca.execution.order import OrderStatus
            order.status = OrderStatus.REJECTED
            order.reject_reason = f"risk:{decision.code}:{decision.reason}"
            self.stats.rejects += 1
            self._journal(order, EventType.RISK_BLOCK)
            return
        if self._gate_blocked(order, ref, side_str):
            return

        if self._working_coid and self.book.get(self._working_coid) is not None:
            new = self.book.replace(self._working_coid, qty=qty,
                                    limit_price=sig.limit_price, stop_price=sig.stop_price,
                                    session_index=self._session_index)
            if new is not None:
                self._working_coid = new.client_order_id
                self._journal(new, EventType.REPLACE)
        else:
            self.book.add(order, self._session_index)
            self._working_coid = order.client_order_id
            self.stats.resting_added += 1
            self.router.risk.record_submission()
            self._journal(order, EventType.SIGNAL)

    # ---------------------------------------------------------------- report
    def latency_report(self) -> LatencyReport:
        return self.router.latency_report()

    def to_result(self, *, commission_bps: float = 0.0, slippage_bps: float = 0.0):
        """
        Build a BacktestResult from the equity curve + recorded trades. A still-
        open position is marked-to-market at the last price as a synthetic closing
        trade (signed qty makes its PnL correct for shorts too).
        """
        from alpca.backtest.engine import BacktestResult, Trade
        trades = list(self._trades)
        if self._cur_trade is not None and self._last_price > 0:
            t: Trade = self._cur_trade
            t.exit_ts = self._bar_ts
            t.exit_price = self._last_price
            t.exit_ref = self._last_price
            t.reason_out = "EOD mark-to-market"
            trades.append(t)
        return BacktestResult(
            symbol=self.symbol,
            strategy=self.strategy.name,
            starting_equity=self.starting_equity,
            ending_equity=self.equity_curve[-1] if self.equity_curve else self.equity,
            trades=trades,
            equity_curve=list(self.equity_curve),
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
        )

    def summary(self) -> Dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy.name,
            "bars_seen": self.stats.bars_seen,
            "orders_submitted": self.stats.orders_submitted,
            "fills": self.stats.fills,
            "rejects": self.stats.rejects,
            "shorts_opened": self.stats.shorts_opened,
            "borrow_paid": round(self.stats.borrow_paid, 2),
            "resting_added": self.stats.resting_added,
            "resting_filled": self.stats.resting_filled,
            "resting_expired": self.stats.resting_expired,
            "resting_canceled": self.stats.resting_canceled,
            "realized_pnl": round(self.stats.realized_pnl, 2),
            "ending_equity": round(self.equity, 2),
            "total_return": round((self.equity - self.starting_equity) / self.starting_equity, 4),
            "rtt_stats": self.router.rtt_stats(),
            "feed_latency": self.stats.feed_latency,
        }
