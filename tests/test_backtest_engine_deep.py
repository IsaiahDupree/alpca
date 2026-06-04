"""
Deep, deterministic tests for alpca/backtest/engine.py.

Covers:
  - run_backtest NO LOOK-AHEAD: a signal from bar i fills at bar i+1's OPEN.
  - BacktestResult metrics: total_return, sharpe, max_drawdown, win_rate,
    n_trades, equity_curve.
  - Trade.pnl signed for long & short, return_pct, closed_trades.
  - dividend_income crediting via a DividendSchedule on RAW bars.
  - FillModel / fee_model / commission_bps wiring.

All inputs are deterministic (fixed seeds, crafted series). No network, no
mocks, no live-Alpaca calls.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import pytest

from alpca.backtest.engine import (
    BacktestResult,
    Trade,
    run_backtest,
    _apply_cost,
)
from alpca.strategies.base import (
    BUY,
    EXIT,
    SELL,
    HOLD,
    Signal,
    Strategy,
    hold,
)
from alpca.execution.fills import FillModel, FillResult
from alpca.execution.fees import AlpacaFeeModel, ZERO_FEES
from alpca.data.corporate_actions import Dividend, DividendSchedule
from alpca.data.bars import synthetic_bars


# --------------------------------------------------------------------------- #
# Local helpers (self-contained — no imports from other tests/ files).
# --------------------------------------------------------------------------- #

def mk_bar(o, h=None, l=None, c=None, v=1_000_000.0, ts=0.0, symbol="TEST") -> Dict[str, float]:
    """Build one OHLCV bar; high/low default to span open..close."""
    if c is None:
        c = o
    if h is None:
        h = max(o, c)
    if l is None:
        l = min(o, c)
    return {
        "open": float(o), "high": float(h), "low": float(l),
        "close": float(c), "volume": float(v), "timestamp": float(ts),
        "symbol": symbol,
    }


def flat_bars(prices: List[float], *, start_ts: float = 0.0, vol: float = 1_000_000.0,
              symbol="TEST") -> List[Dict[str, float]]:
    """Bars where open == close == price for each price (no intrabar range).

    timestamp increments by 1 starting at start_ts.
    """
    out = []
    for i, p in enumerate(prices):
        out.append(mk_bar(p, c=p, v=vol, ts=start_ts + i, symbol=symbol))
    return out


def oc_bars(pairs: List[tuple], *, start_ts: float = 0.0, vol: float = 1_000_000.0,
            symbol="TEST") -> List[Dict[str, float]]:
    """Bars from (open, close) pairs; high/low span open..close."""
    out = []
    for i, (o, c) in enumerate(pairs):
        out.append(mk_bar(o, c=c, v=vol, ts=start_ts + i, symbol=symbol))
    return out


class ScriptedStrategy(Strategy):
    """Emits a pre-scripted Signal per bar by index. Past the script -> HOLD.

    The backtester calls reset() before the run, so we re-zero the index there.
    """
    name = "scripted"

    def __init__(self, script: List[Optional[Signal]]):
        super().__init__()
        self._script = script
        self._i = 0

    def reset(self) -> None:
        super().reset()
        self._i = 0

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        sig = hold()
        if self._i < len(self._script):
            s = self._script[self._i]
            if s is not None:
                sig = s
        self._i += 1
        return sig


class NeverTrades(Strategy):
    name = "never"

    def on_bar(self, bar):
        return hold("flat")


def buy(strength=1.0, reason="buy"):
    return Signal(side=BUY, strength=strength, reason=reason)


def exit_sig(reason="exit"):
    return Signal(side=EXIT, strength=1.0, reason=reason)


def sell(strength=1.0, reason="sell"):
    return Signal(side=SELL, strength=strength, reason=reason)


# A no-cost run: zero slippage, zero commission. Makes fills equal the raw open.
NOCOST = dict(commission_bps=0.0, slippage_bps=0.0)


# --------------------------------------------------------------------------- #
# 1) Trade dataclass: pnl / return_pct semantics (long & short, edge cases).
# --------------------------------------------------------------------------- #

class TestTradePnl:
    def test_long_profit_pnl(self):
        t = Trade("X", 0.0, entry_price=100.0, entry_ref=100.0, qty=10.0,
                  exit_price=110.0)
        assert t.pnl == pytest.approx(100.0)
        assert t.return_pct == pytest.approx(0.10)

    def test_long_loss_pnl(self):
        t = Trade("X", 0.0, entry_price=100.0, entry_ref=100.0, qty=10.0,
                  exit_price=90.0)
        assert t.pnl == pytest.approx(-100.0)
        assert t.return_pct == pytest.approx(-0.10)

    def test_short_profit_negative_qty(self):
        # Short = negative qty. Price falls -> profit: (exit-entry)*qty.
        t = Trade("X", 0.0, entry_price=100.0, entry_ref=100.0, qty=-10.0,
                  exit_price=90.0)
        # (90-100)*-10 = 100 profit
        assert t.pnl == pytest.approx(100.0)

    def test_short_loss_negative_qty(self):
        t = Trade("X", 0.0, entry_price=100.0, entry_ref=100.0, qty=-10.0,
                  exit_price=110.0)
        # (110-100)*-10 = -100 loss
        assert t.pnl == pytest.approx(-100.0)

    def test_pnl_none_when_open(self):
        t = Trade("X", 0.0, entry_price=100.0, entry_ref=100.0, qty=10.0)
        assert t.pnl is None
        assert t.return_pct is None

    def test_return_pct_none_when_entry_zero(self):
        t = Trade("X", 0.0, entry_price=0.0, entry_ref=0.0, qty=10.0,
                  exit_price=5.0)
        assert t.return_pct is None
        # pnl is still defined (exit_price present)
        assert t.pnl == pytest.approx(50.0)

    @pytest.mark.parametrize("entry,exit_,qty,expected", [
        (100.0, 105.0, 10.0, 50.0),
        (50.0, 25.0, 4.0, -100.0),
        (1e6, 1e6 + 1.0, 2.0, 2.0),     # extreme magnitude, tiny move
        (100.0, 100.0, 10.0, 0.0),      # flat round trip
    ])
    def test_pnl_parametrized(self, entry, exit_, qty, expected):
        t = Trade("X", 0.0, entry_price=entry, entry_ref=entry, qty=qty,
                  exit_price=exit_)
        assert t.pnl == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 2) BacktestResult metrics computed directly on crafted equity curves.
# --------------------------------------------------------------------------- #

def mk_result(curve: List[float], *, start=None, trades=None, divs=0.0) -> BacktestResult:
    start = curve[0] if (start is None and curve) else (start or 0.0)
    end = curve[-1] if curve else start
    r = BacktestResult(
        symbol="X", strategy="s", starting_equity=start, ending_equity=end,
        trades=trades or [], equity_curve=list(curve),
    )
    r.dividend_income = divs
    return r


class TestResultMetrics:
    def test_total_return_basic(self):
        r = BacktestResult("X", "s", 100.0, 130.0)
        assert r.total_return == pytest.approx(0.30)

    def test_total_return_zero_start(self):
        r = BacktestResult("X", "s", 0.0, 50.0)
        assert r.total_return == 0.0  # guarded division

    def test_total_return_loss(self):
        r = BacktestResult("X", "s", 100.0, 40.0)
        assert r.total_return == pytest.approx(-0.60)

    def test_max_drawdown_known_curve(self):
        # peak 120 then trough 90 -> dd = (90-120)/120 = -0.25
        r = mk_result([100.0, 120.0, 90.0, 110.0])
        assert r.max_drawdown == pytest.approx(-0.25)

    def test_max_drawdown_monotonic_up_is_zero(self):
        r = mk_result([100.0, 101.0, 102.0, 200.0])
        assert r.max_drawdown == 0.0

    def test_max_drawdown_empty_curve(self):
        r = mk_result([])
        assert r.max_drawdown == 0.0

    def test_max_drawdown_single_point(self):
        r = mk_result([100.0])
        assert r.max_drawdown == 0.0

    def test_sharpe_none_too_few_returns(self):
        # 2 points -> 1 return -> need >=2 returns -> None
        r = mk_result([100.0, 101.0])
        assert r.sharpe is None

    def test_sharpe_none_when_constant(self):
        # constant curve -> all returns 0 -> std 0 -> None
        r = mk_result([100.0, 100.0, 100.0, 100.0])
        assert r.sharpe is None

    def test_sharpe_exact_value(self):
        # curve -> returns: from [100,110,99,108.9]
        # r1 = 0.10, r2 = (99-110)/110 = -0.1, r3 = (108.9-99)/99 = 0.1
        curve = [100.0, 110.0, 99.0, 108.9]
        r = mk_result(curve)
        rs = [0.10, -0.10, 0.10]
        mean = sum(rs) / 3
        var = sum((x - mean) ** 2 for x in rs) / 2  # sample (n-1)
        expected = mean / math.sqrt(var)
        assert r.sharpe == pytest.approx(expected)

    def test_returns_skips_zero_denominator(self):
        # A zero in the curve (other than last) makes that step's denom zero ->
        # that return is skipped, not a ZeroDivision.
        r = mk_result([100.0, 0.0, 50.0])
        # i=1: prev 100 != 0 -> (0-100)/100 = -1.0
        # i=2: prev 0 == 0   -> skipped
        assert r._returns() == pytest.approx([-1.0])

    def test_win_rate_none_no_closed_trades(self):
        r = mk_result([100.0, 101.0])
        assert r.win_rate is None
        assert r.n_trades == 0

    def test_win_rate_and_n_trades(self):
        trades = [
            Trade("X", 0, 100, 100, 1, exit_price=110),  # win
            Trade("X", 0, 100, 100, 1, exit_price=90),   # loss
            Trade("X", 0, 100, 100, 1, exit_price=105),  # win
            Trade("X", 0, 100, 100, 1),                  # OPEN -> excluded
        ]
        r = mk_result([100.0, 101.0], trades=trades)
        assert r.n_trades == 3
        assert r.win_rate == pytest.approx(2 / 3)
        assert len(r.closed_trades) == 3

    def test_win_rate_flat_trade_not_a_win(self):
        # pnl == 0 is NOT a win (strict > 0).
        trades = [Trade("X", 0, 100, 100, 1, exit_price=100)]
        r = mk_result([100.0, 101.0], trades=trades)
        assert r.win_rate == 0.0
        assert r.n_trades == 1

    def test_summary_shape_and_rounding(self):
        trades = [Trade("X", 0, 100, 100, 1, exit_price=110)]
        r = mk_result([100.0, 110.0, 120.0], trades=trades, divs=12.345)
        r.commission_bps = 1.5
        r.slippage_bps = 2.5
        s = r.summary()
        assert set(s.keys()) == {
            "symbol", "strategy", "starting_equity", "ending_equity",
            "total_return", "n_trades", "win_rate", "sharpe_per_bar",
            "max_drawdown", "commission_bps", "slippage_bps", "dividend_income",
        }
        assert s["n_trades"] == 1
        assert s["win_rate"] == 1.0
        assert s["dividend_income"] == 12.35  # round(.,2)
        assert s["commission_bps"] == 1.5

    def test_summary_none_metrics_pass_through(self):
        r = mk_result([100.0, 100.0])  # sharpe None, win_rate None
        s = r.summary()
        assert s["sharpe_per_bar"] is None
        assert s["win_rate"] is None


# --------------------------------------------------------------------------- #
# 3) _apply_cost helper (legacy flat-bps direction).
# --------------------------------------------------------------------------- #

class TestApplyCost:
    @pytest.mark.parametrize("ref,buy,bps,expected", [
        (100.0, True, 10.0, 100.0 * 1.001),    # buy pays up
        (100.0, False, 10.0, 100.0 * 0.999),   # sell receives less
        (100.0, True, 0.0, 100.0),             # zero slippage = ref
        (50.0, False, 100.0, 50.0 * 0.99),
    ])
    def test_apply_cost(self, ref, buy, bps, expected):
        assert _apply_cost(ref, buy, bps) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# 4) run_backtest: NO LOOK-AHEAD — signal at bar i fills at bar i+1 OPEN.
# --------------------------------------------------------------------------- #

class TestNoLookAhead:
    def test_buy_fills_at_next_bar_open_not_signal_bar(self):
        # Bars: opens 10, 20, 30, 40 ; closes equal opens.
        # Strategy BUYs on bar 0. It must NOT fill at bar0's open (10) — it fills
        # at bar1's open (20).
        bars = flat_bars([10.0, 20.0, 30.0, 40.0])
        strat = ScriptedStrategy([buy()])  # only bar0 emits BUY
        r = run_backtest(strat, bars, **NOCOST)
        assert r.n_trades == 1  # entered then EOD-liquidated
        t = r.trades[0]
        assert t.entry_ref == 20.0  # filled at bar1's OPEN, not bar0's (10)
        assert t.entry_ts == bars[1]["timestamp"]

    def test_signal_on_last_bar_never_fills(self):
        # A BUY emitted on the FINAL bar has no next bar to fill at -> no trade.
        bars = flat_bars([10.0, 20.0, 30.0])
        strat = ScriptedStrategy([None, None, buy()])  # buy on last bar only
        r = run_backtest(strat, bars, **NOCOST)
        assert r.n_trades == 0
        assert r.trades == []
        # No position ever taken -> equity stays flat at starting_equity.
        assert r.equity_curve[-1] == pytest.approx(100_000.0)

    def test_exit_fills_at_bar_after_exit_signal_open(self):
        # BUY on bar0 -> fill bar1 open (20). EXIT on bar2 -> fill bar3 open (40).
        bars = flat_bars([10.0, 20.0, 30.0, 40.0, 50.0])
        strat = ScriptedStrategy([buy(), None, exit_sig()])
        r = run_backtest(strat, bars, **NOCOST)
        assert r.n_trades == 1
        t = r.trades[0]
        assert t.entry_ref == 20.0
        assert t.exit_ref == 40.0  # bar3 open, the bar AFTER the exit signal
        assert t.reason_out == "exit"

    def test_no_double_entry_while_in_position(self):
        # Repeated BUYs while already long must not open a second trade.
        bars = flat_bars([10.0, 20.0, 30.0, 40.0])
        strat = ScriptedStrategy([buy(), buy(), buy(), buy()])
        r = run_backtest(strat, bars, **NOCOST)
        # one open position, liquidated at EOD -> exactly 1 trade
        assert r.n_trades == 1

    def test_exit_with_no_position_is_noop(self):
        bars = flat_bars([10.0, 20.0, 30.0])
        strat = ScriptedStrategy([exit_sig(), exit_sig(), exit_sig()])
        r = run_backtest(strat, bars, **NOCOST)
        assert r.n_trades == 0
        assert r.equity_curve[-1] == pytest.approx(100_000.0)


# --------------------------------------------------------------------------- #
# 5) run_backtest: exact PnL / equity accounting under no cost.
# --------------------------------------------------------------------------- #

class TestExactAccounting:
    def test_long_roundtrip_exact_equity_and_pnl(self):
        # Enter at 100 (bar1 open), exit at 120 (bar3 open). No cost.
        # budget = 100_000 * 0.95 = 95_000. qty = 95000/100 = 950.
        # cash after buy = 100000 - 950*100 = 5000.
        # exit: cash = 5000 + 950*120 = 119000. PnL = (120-100)*950 = 19000.
        bars = flat_bars([10.0, 100.0, 110.0, 120.0, 130.0])
        strat = ScriptedStrategy([buy(), None, exit_sig()])
        r = run_backtest(strat, bars, **NOCOST)
        t = r.trades[0]
        assert t.qty == pytest.approx(950.0)
        assert t.entry_price == pytest.approx(100.0)
        assert t.exit_price == pytest.approx(120.0)
        assert t.pnl == pytest.approx(19_000.0)
        # ending equity = starting + pnl (all cash after exit)
        assert r.ending_equity == pytest.approx(119_000.0)
        assert r.total_return == pytest.approx(0.19)

    def test_eod_liquidation_closes_open_position(self):
        # BUY, never exit -> liquidated at last close. equity_curve[-1] == cash.
        bars = flat_bars([10.0, 100.0, 150.0])
        strat = ScriptedStrategy([buy()])
        r = run_backtest(strat, bars, **NOCOST)
        assert r.n_trades == 1
        t = r.trades[0]
        assert t.reason_out == "EOD liquidation"
        assert t.exit_price == pytest.approx(150.0)  # last close
        # last equity point equals cash (no position left)
        assert r.equity_curve[-1] == pytest.approx(r.ending_equity)

    def test_equity_curve_length_equals_n_bars(self):
        bars = flat_bars([10.0, 11.0, 12.0, 13.0, 14.0])
        r = run_backtest(NeverTrades(), bars, **NOCOST)
        assert len(r.equity_curve) == len(bars)

    def test_no_trades_equity_flat(self):
        bars = flat_bars([10.0, 11.0, 12.0])
        r = run_backtest(NeverTrades(), bars, **NOCOST)
        assert all(e == pytest.approx(100_000.0) for e in r.equity_curve)
        assert r.total_return == 0.0

    def test_mtm_unrealized_in_equity_curve(self):
        # While long, equity marks to bar close. Enter at 100 (bar1 open),
        # bar1 close is also 100 (flat bars) -> equity at bar1 == starting.
        # bar2 close 110 -> equity = cash + qty*110.
        bars = flat_bars([10.0, 100.0, 110.0])
        strat = ScriptedStrategy([buy()])
        r = run_backtest(strat, bars, **NOCOST)
        # qty=950, cash after buy=5000; bar2 marks at last close but bar2 is the
        # LAST bar so EOD liquidation overwrites equity_curve[-1] with cash.
        # Check bar1 (index 1) mark = 5000 + 950*100 = 100000.
        assert r.equity_curve[1] == pytest.approx(100_000.0)


# --------------------------------------------------------------------------- #
# 6) Cost model wiring: slippage_bps, commission_bps, fill_model, fee_model.
# --------------------------------------------------------------------------- #

class TestCostWiring:
    def test_slippage_default_flat_model_applies_bps(self):
        # buy pays open*(1+slip), sell receives open*(1-slip). slip=20bps=0.002.
        bars = flat_bars([10.0, 100.0, 200.0])
        strat = ScriptedStrategy([buy()])
        r = run_backtest(strat, bars, commission_bps=0.0, slippage_bps=20.0)
        t = r.trades[0]
        assert t.entry_price == pytest.approx(100.0 * 1.002)  # buy at bar1 open
        # EOD liquidation at last close 200 with sell slippage
        assert t.exit_price == pytest.approx(200.0 * 0.998)

    def test_commission_bps_reduces_cash(self):
        # commission charged on notional both sides. 10 bps = 0.001.
        # Compare ending equity with vs without commission, same path.
        bars = flat_bars([10.0, 100.0, 100.0])  # flat price -> no price pnl
        strat0 = ScriptedStrategy([buy()])
        r0 = run_backtest(strat0, bars, commission_bps=0.0, slippage_bps=0.0)
        strat1 = ScriptedStrategy([buy()])
        r1 = run_backtest(strat1, bars, commission_bps=10.0, slippage_bps=0.0)
        # With commission, ending equity strictly lower (two commissions paid).
        assert r1.ending_equity < r0.ending_equity
        assert r0.ending_equity == pytest.approx(100_000.0)

    def test_fee_model_supersedes_commission_bps(self):
        # When fee_model given, commission_bps must be ignored.
        bars = flat_bars([10.0, 100.0, 100.0])
        s_a = ScriptedStrategy([buy()])
        r_fee = run_backtest(s_a, bars, commission_bps=999.0, slippage_bps=0.0,
                             fee_model=ZERO_FEES)
        # ZERO_FEES + zero slippage on flat price -> ending equity unchanged.
        assert r_fee.ending_equity == pytest.approx(100_000.0)

    def test_fee_model_charges_sell_regulatory(self):
        bars = flat_bars([10.0, 100.0, 100.0])
        strat = ScriptedStrategy([buy()])
        fm = AlpacaFeeModel()  # default 2024 rates
        r = run_backtest(strat, bars, slippage_bps=0.0, fee_model=fm)
        # Buy has zero regulatory; sell (EOD liq) pays SEC+TAF -> equity < start.
        assert r.ending_equity < 100_000.0
        # Magnitude is small (bps-scale on ~95k notional).
        assert r.ending_equity > 99_900.0

    def test_custom_fill_model_volume_cap_partial(self):
        # participation_cap < 1.0 + a small bar volume -> partial fill -> qty
        # smaller than the uncapped case.
        small_vol_bars = flat_bars([10.0, 100.0, 100.0], vol=100.0)
        capped = FillModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                           participation_cap=0.5, min_tick=0.0)
        s = ScriptedStrategy([buy()])
        r = run_backtest(s, small_vol_bars, commission_bps=0.0,
                         fill_model=capped)
        t = r.trades[0]
        # cap = 0.5 * 100 vol = 50 shares max; budget would want 950 -> capped 50
        assert t.qty == pytest.approx(50.0)

    def test_impact_fill_model_worsens_buy_price(self):
        # impact_coef_bps>0 with finite volume -> buy fill above the open.
        bars = flat_bars([10.0, 100.0, 100.0], vol=10_000.0)
        fm = FillModel(half_spread_bps=0.0, impact_coef_bps=8.0,
                       participation_cap=1.0, min_tick=0.0)
        s = ScriptedStrategy([buy()])
        r = run_backtest(s, bars, commission_bps=0.0, fill_model=fm)
        t = r.trades[0]
        assert t.entry_price > 100.0  # impact pushed the buy up


# --------------------------------------------------------------------------- #
# 7) Dividends: crediting on RAW bars while a position is held.
# --------------------------------------------------------------------------- #

class TestDividends:
    def test_dividend_credited_while_long(self):
        # Hold a position across an ex-date; cash dividend credited to equity.
        # Bars ts: 0,1,2,3. Enter bar1 (ts=1). Ex-date at ts=2.5 -> credited
        # when crossing into ts=3 (prev_ts=2, cur_ts=3) on shares held.
        bars = flat_bars([10.0, 100.0, 100.0, 100.0])
        strat = ScriptedStrategy([buy()])
        sched = DividendSchedule([Dividend(ex_date_ts=2.5, amount=1.0)])
        r = run_backtest(strat, bars, commission_bps=0.0, slippage_bps=0.0,
                         dividends=sched)
        # qty = 950 shares; dividend 1.0/share -> +950 cash.
        assert r.dividend_income == pytest.approx(950.0)
        # ending equity reflects the credit (flat price, no other pnl)
        assert r.ending_equity == pytest.approx(100_000.0 + 950.0)

    def test_dividend_not_credited_when_flat(self):
        # No position held -> no dividend credited even if ex-date crossed.
        bars = flat_bars([10.0, 100.0, 100.0])
        sched = DividendSchedule([Dividend(ex_date_ts=1.5, amount=5.0)])
        r = run_backtest(NeverTrades(), bars, **NOCOST, dividends=sched)
        assert r.dividend_income == 0.0

    def test_dividend_credited_once_per_exdate(self):
        # Two bars cross a single ex-date region but credit happens exactly once
        # (half-open interval keying).
        bars = flat_bars([10.0, 100.0, 100.0, 100.0, 100.0])
        strat = ScriptedStrategy([buy()])
        sched = DividendSchedule([Dividend(ex_date_ts=2.5, amount=2.0)])
        r = run_backtest(strat, bars, **NOCOST, dividends=sched)
        assert r.dividend_income == pytest.approx(950.0 * 2.0)

    def test_multiple_dividends_summed(self):
        bars = flat_bars([10.0, 100.0, 100.0, 100.0, 100.0])
        strat = ScriptedStrategy([buy()])
        sched = DividendSchedule([
            Dividend(ex_date_ts=2.5, amount=1.0),
            Dividend(ex_date_ts=3.5, amount=0.5),
        ])
        r = run_backtest(strat, bars, **NOCOST, dividends=sched)
        # 950 shares * (1.0 + 0.5) = 1425
        assert r.dividend_income == pytest.approx(1425.0)

    def test_no_dividends_schedule_zero_income(self):
        bars = flat_bars([10.0, 100.0, 100.0])
        strat = ScriptedStrategy([buy()])
        r = run_backtest(strat, bars, **NOCOST)
        assert r.dividend_income == 0.0


# --------------------------------------------------------------------------- #
# 8) Edge / degenerate inputs: graceful handling, not crashes.
# --------------------------------------------------------------------------- #

class TestEdgeCases:
    def test_empty_bars(self):
        r = run_backtest(NeverTrades(), [], **NOCOST)
        assert r.equity_curve == []
        assert r.n_trades == 0
        assert r.ending_equity == pytest.approx(100_000.0)
        assert r.symbol == "?"
        assert r.total_return == 0.0

    def test_single_bar_signal_never_fills(self):
        # One bar: a BUY signal has no next bar -> never fills.
        bars = flat_bars([100.0])
        strat = ScriptedStrategy([buy()])
        r = run_backtest(strat, bars, **NOCOST)
        assert r.n_trades == 0
        assert len(r.equity_curve) == 1

    def test_zero_open_price_skips_buy(self):
        # provisional_qty guarded when open_px <= 0; FillModel also returns 0 qty.
        bars = [mk_bar(10.0, c=10.0, ts=0), mk_bar(0.0, c=0.0, ts=1),
                mk_bar(50.0, c=50.0, ts=2)]
        strat = ScriptedStrategy([buy()])  # buy fills at bar1 open == 0
        r = run_backtest(strat, bars, **NOCOST)
        # zero open -> qty 0 -> trade opened with qty 0, liquidated at EOD.
        t = r.trades[0] if r.trades else None
        if t is not None:
            assert t.qty == pytest.approx(0.0)
        # Equity must not be NaN/inf and stays at starting capital.
        assert all(math.isfinite(e) for e in r.equity_curve)
        assert r.ending_equity == pytest.approx(100_000.0)

    def test_negative_strength_buy_not_actionable(self):
        # is_actionable requires strength>0 for BUY; strength 0 -> HOLD-equiv.
        bars = flat_bars([10.0, 100.0, 100.0])
        strat = ScriptedStrategy([buy(strength=0.0)])
        r = run_backtest(strat, bars, **NOCOST)
        assert r.n_trades == 0

    def test_missing_volume_key_handled(self):
        # Bars without a "volume" key -> bar.get("volume") is None; flat fill
        # model ignores volume, so trade still executes.
        bars = [
            {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "timestamp": 0.0},
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "timestamp": 1.0},
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "timestamp": 2.0},
        ]
        strat = ScriptedStrategy([buy()])
        r = run_backtest(strat, bars, **NOCOST)
        assert r.n_trades == 1
        assert r.trades[0].qty == pytest.approx(950.0)

    def test_missing_timestamp_defaults_zero(self):
        bars = [
            {"open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0, "volume": 1e6},
            {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1e6},
        ]
        strat = ScriptedStrategy([buy()])
        r = run_backtest(strat, bars, **NOCOST)
        # ts default 0 for both; trade fills at bar1 open. No dividends so fine.
        assert r.n_trades == 1
        assert r.trades[0].entry_ts == 0.0

    def test_symbol_inferred_from_first_bar(self):
        bars = flat_bars([10.0, 11.0], symbol="ZZZ")
        r = run_backtest(NeverTrades(), bars, **NOCOST)
        assert r.symbol == "ZZZ"


# --------------------------------------------------------------------------- #
# 9) Determinism & reset semantics on synthetic_bars.
# --------------------------------------------------------------------------- #

class TestDeterminismAndReset:
    def test_synthetic_run_is_deterministic(self):
        bars = synthetic_bars("DET", n=120, seed=7)
        strat = ScriptedStrategy([buy(), None, None, exit_sig()] + [None] * 200)
        r1 = run_backtest(strat, bars, **NOCOST)
        # reuse the SAME strategy obj -> run_backtest calls reset() internally.
        r2 = run_backtest(strat, bars, **NOCOST)
        assert r1.equity_curve == r2.equity_curve
        assert r1.ending_equity == pytest.approx(r2.ending_equity)
        assert r1.n_trades == r2.n_trades

    def test_reset_clears_strategy_index(self):
        # ScriptedStrategy.reset re-zeros _i; second run must reproduce the first.
        bars = flat_bars([10.0, 100.0, 110.0, 120.0])
        strat = ScriptedStrategy([buy(), None, exit_sig()])
        a = run_backtest(strat, bars, **NOCOST)
        b = run_backtest(strat, bars, **NOCOST)
        assert a.trades[0].entry_ref == b.trades[0].entry_ref == 100.0
        assert a.trades[0].exit_ref == b.trades[0].exit_ref == 120.0

    def test_provenance_fields_echoed(self):
        bars = flat_bars([10.0, 11.0])
        r = run_backtest(NeverTrades(), bars, commission_bps=3.0, slippage_bps=4.0)
        assert r.commission_bps == 3.0
        assert r.slippage_bps == 4.0
        assert r.strategy == "never"


# --------------------------------------------------------------------------- #
# 10) Sharpe / drawdown computed from an actual run (integration of metrics).
# --------------------------------------------------------------------------- #

class TestRunMetricsIntegration:
    def test_run_metrics_finite_on_synthetic(self):
        bars = synthetic_bars("MIX", n=200, seed=3, drift=0.0008, vol=0.012)
        # Alternate buy/exit to generate several closed trades deterministically.
        script: List[Optional[Signal]] = []
        for i in range(200):
            if i % 20 == 0:
                script.append(buy())
            elif i % 20 == 10:
                script.append(exit_sig())
            else:
                script.append(None)
        strat = ScriptedStrategy(script)
        r = run_backtest(strat, bars, commission_bps=1.0, slippage_bps=2.0)
        assert r.n_trades >= 1
        # all metrics finite / well-typed
        assert math.isfinite(r.total_return)
        assert math.isfinite(r.max_drawdown)
        assert r.max_drawdown <= 0.0
        assert r.win_rate is None or 0.0 <= r.win_rate <= 1.0
        assert r.sharpe is None or math.isfinite(r.sharpe)
        # equity curve has one point per bar
        assert len(r.equity_curve) == len(bars)

    def test_winning_trade_marks_win_rate_one(self):
        # Single profitable round trip -> win_rate 1.0.
        bars = flat_bars([10.0, 100.0, 105.0, 110.0, 110.0])
        strat = ScriptedStrategy([buy(), None, exit_sig()])
        r = run_backtest(strat, bars, **NOCOST)
        assert r.n_trades == 1
        assert r.trades[0].pnl > 0
        assert r.win_rate == 1.0

    def test_losing_trade_marks_win_rate_zero(self):
        bars = flat_bars([10.0, 100.0, 95.0, 90.0, 90.0])
        strat = ScriptedStrategy([buy(), None, exit_sig()])
        r = run_backtest(strat, bars, **NOCOST)
        assert r.n_trades == 1
        assert r.trades[0].pnl < 0
        assert r.win_rate == 0.0
