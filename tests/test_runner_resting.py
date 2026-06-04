"""
Runner wired to the OpenOrderBook: a strategy that emits a resting buy-STOP gets
that order rested, triggered intrabar, filled, and accounted — across bars.
"""

import asyncio

from alpca.config import RiskConfig
from alpca.data.feed import ReplayBarSource
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.fills import FillModel
from alpca.execution.open_orders import OpenOrderBook
from alpca.execution.order_event_log import OrderEventLog
from alpca.execution.router import ExecutionRouter
from alpca.risk.risk_engine import RiskEngine
from alpca.runtime.runner import LiveRunner
from alpca.strategies.breakout import DonchianBreakout


_FM = FillModel(half_spread_bps=0.0, impact_coef_bps=0.0, participation_cap=1.0, min_tick=0.0)


def _bar(o, h, l, c, ts):
    return {"open": o, "high": h, "low": l, "close": c, "volume": 1e7,
            "timestamp": ts, "symbol": "T"}


def _router(log=None):
    risk = RiskEngine(RiskConfig(max_order_notional=1e9, max_concentration_pct=1.0))
    return ExecutionRouter(SimAdapter(seed=1, sleep=False), risk, log, fill_timeout_s=1.0)


def test_resting_buy_stop_triggers_and_fills_across_bars():
    async def go():
        # flat channel (5 bars) so Donchian(5) arms; channel high ~100.5. With
        # entry="stop" the strategy rests a buy-stop at the channel high every bar.
        bars = [_bar(100, 100.5, 99.5, 100, i) for i in range(6)]
        # this bar does NOT exceed 100.5 -> stop stays resting, no fill
        bars.append(_bar(100, 100.4, 99.6, 100, 6))
        # this bar trades UP through 100.5 -> the resting buy-stop triggers + fills
        bars.append(_bar(101, 105, 100.5, 104, 7))
        # a later break-down would exit (close < prior low)
        bars.append(_bar(96, 97, 80, 82, 8))
        book = OpenOrderBook(_FM)
        runner = LiveRunner(DonchianBreakout(period=5, atr_period=3, entry="stop"),
                            "T", _router(), starting_equity=100_000,
                            target_notional_pct=0.5, open_order_book=book)
        await runner.run(ReplayBarSource(bars))
        return runner

    runner = asyncio.run(go())
    s = runner.stats
    assert s.resting_added >= 1            # a buy-stop was placed
    assert s.resting_filled >= 1           # and it filled when price broke through
    assert runner.position_qty > 0 or s.realized_pnl != 0.0  # we got into the trade
    # the strategy was told it filled (on_fill flipped it in-position at some point)


def test_resting_stop_not_filled_when_level_never_reached():
    async def go():
        bars = [_bar(100, 100.5, 99.5, 100, i) for i in range(5)]
        # price stays strictly below the 100.5 channel high forever -> never fills
        for i in range(5, 12):
            bars.append(_bar(100, 100.3, 99.7, 100, i))
        book = OpenOrderBook(_FM)
        runner = LiveRunner(DonchianBreakout(period=5, atr_period=3, entry="stop"),
                            "T", _router(), starting_equity=100_000,
                            target_notional_pct=0.5, open_order_book=book)
        await runner.run(ReplayBarSource(bars))
        return runner

    runner = asyncio.run(go())
    assert runner.stats.resting_added >= 1
    assert runner.stats.resting_filled == 0
    assert runner.position_qty == 0
    assert runner.cash == 100_000  # never spent — nothing filled


def test_resting_orders_journaled_to_ledger(tmp_path):
    async def go():
        log = OrderEventLog(str(tmp_path / "ev.jsonl"))
        # 6 channel bars: the strategy arms at bar 5 and rests the buy-stop @100.5;
        # bar 6 then trades up through it (book is advanced at the START of a bar,
        # so the stop must already be resting from a PRIOR bar — no look-ahead).
        bars = [_bar(100, 100.5, 99.5, 100, i) for i in range(6)]
        bars.append(_bar(101, 105, 100.5, 104, 6))   # triggers + fills the buy-stop
        book = OpenOrderBook(_FM)
        runner = LiveRunner(DonchianBreakout(period=5, atr_period=3, entry="stop"),
                            "T", _router(log), starting_equity=100_000,
                            target_notional_pct=0.5, open_order_book=book)
        await runner.run(ReplayBarSource(bars))
        return runner, log

    runner, log = asyncio.run(go())
    chk = log.verify_chain()
    assert chk.ok
    events = {row["event"] for row in log.read_all()}
    # a resting signal was journaled, and a trigger+fill occurred
    assert "SIGNAL" in events
    assert "TRIGGER" in events
    assert "FILL" in events


def test_market_entry_mode_unchanged_by_default():
    """entry='market' (default) must behave exactly as before — no book needed."""
    async def go():
        bars = [_bar(100, 100.5, 99.5, 100, i) for i in range(6)]
        bars.append(_bar(105, 112, 104, 112, 6))   # breakout signal
        bars.append(_bar(120, 122, 119, 121, 7))   # market entry fills next open
        runner = LiveRunner(DonchianBreakout(period=5, atr_period=3),  # default market
                            "T", _router(), starting_equity=100_000,
                            target_notional_pct=0.5)
        await runner.run(ReplayBarSource(bars))
        return runner

    runner = asyncio.run(go())
    assert runner.stats.orders_submitted >= 1
    assert runner.stats.resting_added == 0  # no book in use
    assert runner.stats.fills >= 1
