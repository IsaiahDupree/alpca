"""
Event-driven backtester with an explicit cost model.

Its second job (beyond PnL) is to record, per trade, the *modeled* fill price —
close * (1 +/- slippage_bps) plus commission — so the live bot can later compare
its REAL Alpaca fills against what the backtest assumed. That backtest-vs-live
slippage gap is the headline number this whole project exists to measure.

One position per symbol, long-only (matches the ported strategies' BUY/EXIT
contract). Fills happen at the signal bar's close (the price the strategy saw),
adjusted by the cost model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from alpca.strategies.base import BUY, EXIT, SELL, Signal, Strategy

if TYPE_CHECKING:  # type-only; runtime imports are local to avoid cycles
    from alpca.execution.fees import AlpacaFeeModel
    from alpca.execution.fills import FillModel
    from alpca.data.corporate_actions import DividendSchedule


@dataclass
class Trade:
    symbol: str
    entry_ts: float
    entry_price: float          # modeled fill (incl. slippage)
    entry_ref: float            # the close the strategy acted on (intended)
    qty: float
    exit_ts: Optional[float] = None
    exit_price: Optional[float] = None
    exit_ref: Optional[float] = None
    reason_in: str = ""
    reason_out: str = ""

    @property
    def pnl(self) -> Optional[float]:
        if self.exit_price is None:
            return None
        return (self.exit_price - self.entry_price) * self.qty

    @property
    def return_pct(self) -> Optional[float]:
        if self.exit_price is None or self.entry_price == 0:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    starting_equity: float
    ending_equity: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)

    # cost model echoed back for provenance
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    # cumulative cash dividends credited (set by run_backtest; 0 for runner-built
    # results). A real field so summary() works regardless of how it was built.
    dividend_income: float = 0.0

    @property
    def total_return(self) -> float:
        if self.starting_equity == 0:
            return 0.0
        return (self.ending_equity - self.starting_equity) / self.starting_equity

    @property
    def closed_trades(self) -> List[Trade]:
        return [t for t in self.trades if t.exit_price is not None]

    @property
    def n_trades(self) -> int:
        return len(self.closed_trades)

    @property
    def win_rate(self) -> Optional[float]:
        ct = self.closed_trades
        if not ct:
            return None
        wins = sum(1 for t in ct if (t.pnl or 0) > 0)
        return wins / len(ct)

    def _returns(self) -> List[float]:
        ec = self.equity_curve
        out = []
        for i in range(1, len(ec)):
            if ec[i - 1] != 0:
                out.append((ec[i] - ec[i - 1]) / ec[i - 1])
        return out

    @property
    def sharpe(self) -> Optional[float]:
        """Per-bar Sharpe (not annualized), simple and dependency-free."""
        rs = self._returns()
        if len(rs) < 2:
            return None
        mean = sum(rs) / len(rs)
        var = sum((r - mean) ** 2 for r in rs) / (len(rs) - 1)
        std = math.sqrt(var)
        if std == 0:
            return None
        return mean / std

    @property
    def max_drawdown(self) -> float:
        peak = -math.inf
        mdd = 0.0
        for v in self.equity_curve:
            peak = max(peak, v)
            if peak > 0:
                mdd = min(mdd, (v - peak) / peak)
        return mdd

    def summary(self) -> Dict:
        return {
            "symbol": self.symbol,
            "strategy": self.strategy,
            "starting_equity": self.starting_equity,
            "ending_equity": round(self.ending_equity, 2),
            "total_return": round(self.total_return, 4),
            "n_trades": self.n_trades,
            "win_rate": None if self.win_rate is None else round(self.win_rate, 3),
            "sharpe_per_bar": None if self.sharpe is None else round(self.sharpe, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "commission_bps": self.commission_bps,
            "slippage_bps": self.slippage_bps,
            "dividend_income": round(self.dividend_income, 2),
        }


def _apply_cost(ref_price: float, side_buy: bool, slippage_bps: float) -> float:
    adj = slippage_bps / 10_000.0
    return ref_price * (1 + adj) if side_buy else ref_price * (1 - adj)


def run_backtest(
    strategy: Strategy,
    bars: List[Dict[str, float]],
    *,
    starting_equity: float = 100_000.0,
    position_size_pct: float = 0.95,
    commission_bps: float = 1.0,
    slippage_bps: float = 2.0,
    fill_model: "Optional[FillModel]" = None,
    fee_model: "Optional[AlpacaFeeModel]" = None,
    require_regular_hours: bool = False,
    allow_extended: bool = False,
    dividends: "Optional[DividendSchedule]" = None,
) -> BacktestResult:
    """
    Event-driven backtest with next-bar-open execution.

    Cost model:
      - `fill_model` (FillModel) controls the FILL PRICE — spread + size-dependent
        impact + volume-cap/partial fills. Defaults to FillModel.flat(slippage_bps),
        which reproduces the legacy flat-bps behavior exactly.
      - `fee_model` (AlpacaFeeModel) controls FEES charged to cash — when given,
        it supersedes commission_bps (Alpaca equities are commission-free but pay
        SEC/TAF on sells). When None, the legacy commission_bps is used.

    Calendar:
      - `require_regular_hours` (default False): when True, a fill may only happen
        on a bar whose timestamp falls in a tradeable NYSE session. A pending
        signal landing on an off-session bar is CARRIED FORWARD to the next
        tradeable bar (a signal generated after the close fills at the next
        open — never look-ahead, never an impossible off-hours fill). Requires
        REAL epoch-second timestamps; leave False for synthetic/integer-ts bars.
      - `allow_extended`: count pre/after-market as tradeable when gating.

    Dividends:
      - `dividends` (DividendSchedule): when a held position crosses a cash
        dividend's ex-date, credit qty*amount to cash. USE ONLY WITH RAW/SPLIT
        bars — dividend-adjusted bars ("all"/"dividend") already include the
        dividend in price continuity, so crediting would double-count.
    """
    from alpca.execution.fills import FillModel  # local import: avoid cycles
    if require_regular_hours:
        from alpca.data.calendar import is_tradeable
    if fill_model is None:
        fill_model = FillModel.flat(slippage_bps)

    def _commission(side_buy: bool, q: float, px: float, notional: float) -> float:
        if fee_model is not None:
            return fee_model.fee(side_buy, q, px)
        return notional * (commission_bps / 10_000.0)

    strategy.reset()
    cash = starting_equity
    qty = 0.0
    open_trade: Optional[Trade] = None
    trades: List[Trade] = []
    equity_curve: List[float] = []

    # NO LOOK-AHEAD: a signal derived from bar i is only known at bar i's CLOSE,
    # so it cannot execute until bar i+1. We carry it in `pending` and fill it at
    # the NEXT bar's OPEN (the first realistically tradeable price). This is the
    # single most important realism property of the backtester — filling on the
    # same bar's close trades on information that doesn't exist yet and inflates
    # returns. Ported from TradingBot's pending_signal deferral.
    pending: Optional[Signal] = None
    prev_ts: Optional[float] = None
    div_income = 0.0  # cumulative dividend cash credited (for reporting)

    for bar in bars:
        open_px = bar["open"]
        close = bar["close"]
        ts = float(bar.get("timestamp", 0) or 0)

        # Dividend cash-flow: credit on shares held coming INTO this bar when an
        # ex-date falls in (prev_ts, ts]. Crediting uses qty before this bar's
        # fill (holders of record earn the dividend). RAW/SPLIT bars only.
        if dividends is not None and prev_ts is not None and qty > 0:
            dps = dividends.per_share_between(prev_ts, ts)
            if dps:
                credit = qty * dps
                cash += credit
                div_income += credit
        prev_ts = ts

        vol = bar.get("volume")

        # Calendar gate: only fill on a tradeable NYSE bar. If enforcing and this
        # bar is off-session, the pending signal is NOT executed — it carries
        # forward to the next tradeable bar (models "submitted after close ->
        # fills at next regular open"; never an off-hours fill, never look-ahead).
        bar_tradeable = True
        if require_regular_hours:
            bar_tradeable = is_tradeable(ts, allow_extended=allow_extended)

        # 1) Execute the PREVIOUS bar's signal at THIS bar's open (if tradeable).
        if pending is not None and bar_tradeable:
            if pending.side == BUY and open_trade is None:
                budget = cash * position_size_pct
                # provisional size off the raw open; the fill model then applies
                # spread+impact (and any volume cap) to that size.
                provisional_qty = budget / open_px if open_px > 0 else 0.0
                res = fill_model.fill(True, open_px, provisional_qty, bar_volume=vol)
                fill = res.price
                # re-solve qty so spend stays within budget at the realized price
                qty = min(res.filled_qty, budget / fill) if fill > 0 else 0.0
                commission = _commission(True, qty, fill, qty * fill)
                cash -= qty * fill + commission
                open_trade = Trade(
                    symbol=bar.get("symbol", strategy.name), entry_ts=ts,
                    entry_price=fill, entry_ref=open_px, qty=qty,
                    reason_in=pending.reason,
                )
            elif pending.side in (EXIT, SELL) and open_trade is not None:
                res = fill_model.fill(False, open_px, qty, bar_volume=vol)
                fill = res.price
                commission = _commission(False, qty, fill, qty * fill)
                cash += qty * fill - commission
                open_trade.exit_ts = ts
                open_trade.exit_price = fill
                open_trade.exit_ref = open_px
                open_trade.reason_out = pending.reason
                trades.append(open_trade)
                open_trade = None
                qty = 0.0
            pending = None  # consumed (only reached when bar_tradeable)

        # 2) Generate THIS bar's signal (only actionable ones become pending);
        #    it can fire no earlier than the next bar's open.
        sig: Signal = strategy.on_bar(bar)
        if sig.is_actionable:
            pending = sig

        # 3) Mark-to-market at this bar's close.
        equity = cash + qty * close
        equity_curve.append(equity)

    # Liquidate any still-open position at the last close. This pays the SAME
    # fill costs (spread/impact) and fees as a normal exit — a costless
    # end-of-data liquidation would understate the true cost of round trips
    # (audit gap: "costless EOD liquidation").
    if open_trade is not None and bars:
        last_bar = bars[-1]
        last_close = last_bar["close"]
        res = fill_model.fill(False, last_close, qty, bar_volume=last_bar.get("volume"))
        fill = res.price
        commission = _commission(False, qty, fill, qty * fill)
        cash += qty * fill - commission
        open_trade.exit_ts = float(last_bar.get("timestamp", 0) or 0)
        open_trade.exit_price = fill
        open_trade.exit_ref = last_close
        open_trade.reason_out = "EOD liquidation"
        trades.append(open_trade)
        equity_curve[-1] = cash

    result = BacktestResult(
        symbol=bars[0].get("symbol", "?") if bars else "?",
        strategy=strategy.name,
        starting_equity=starting_equity,
        ending_equity=equity_curve[-1] if equity_curve else starting_equity,
        trades=trades,
        equity_curve=equity_curve,
        commission_bps=commission_bps,
        slippage_bps=slippage_bps,
    )
    result.dividend_income = div_income
    return result
