"""
Runner integration of T+1 settlement + PDT, driven by real ET-dated bars so the
session counter advances correctly.
"""

import asyncio
from zoneinfo import ZoneInfo
from datetime import datetime

from alpca.config import RiskConfig
from alpca.data.feed import ReplayBarSource
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.router import ExecutionRouter
from alpca.risk.risk_engine import RiskEngine
from alpca.runtime.account import PdtGuard, SettlementLedger
from alpca.runtime.runner import LiveRunner
from alpca.strategies.breakout import DonchianBreakout

_ET = ZoneInfo("America/New_York")


def _ts(y, mo, d, h=11, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=_ET).timestamp()


def _bar(o, h, l, c, ts):
    return {"open": o, "high": h, "low": l, "close": c, "volume": 1e7,
            "timestamp": ts, "symbol": "T"}


def _router(day_start_equity=None):
    # Generous caps + NO daily-loss gate (day_start_equity left None) so only the
    # feature under test can block an order. Daily-loss is covered in test_risk.py.
    risk = RiskEngine(RiskConfig(max_order_notional=1e9, max_concentration_pct=1.0),
                      day_start_equity=day_start_equity)
    return ExecutionRouter(SimAdapter(seed=1, sleep=False), risk, None, fill_timeout_s=1.0)


def _channel_then_breakout(entry_day_volume=1e7):
    """5-session flat channel (one bar per trading date) + a breakout signal bar,
    so Donchian(5) emits a BUY that fills on the following bar."""
    days = [(2025, 6, 9), (2025, 6, 10), (2025, 6, 11), (2025, 6, 12), (2025, 6, 13)]
    bars = [_bar(100, 100.5, 99.5, 100, _ts(*d)) for d in days]
    bars.append(_bar(105, 112, 104, 112, _ts(2025, 6, 16)))   # breakout signal
    bars.append(_bar(120, 122, 119, 121, _ts(2025, 6, 17)))   # ENTRY fills here
    return bars


def test_settlement_blocks_buy_when_cash_is_unsettled():
    """Account whose cash is almost entirely UNSETTLED (pending T+1) can't fund a
    new BUY until it settles."""
    async def go():
        bars = _channel_then_breakout()
        # $5k settled + $95k pending that settles FAR in the future (session 999),
        # so it never settles during this run. The entry BUY wants ~95% of equity,
        # far more than the $5k settled -> blocked for the whole run.
        led = SettlementLedger(5_000.0)
        led._pending[999] = 95_000.0  # pending, settles only at session 999
        runner = LiveRunner(DonchianBreakout(period=5, atr_period=3), "T", _router(),
                            starting_equity=100_000, target_notional_pct=0.95,
                            settlement=led)
        await runner.run(ReplayBarSource(bars))
        return runner

    runner = asyncio.run(go())
    # every entry attempt is blocked: only $5k settled, never enough
    assert runner.stats.skipped_unsettled >= 1
    assert runner.stats.fills == 0


def test_settlement_allows_buy_once_cash_settles():
    """Same pending cash, but the entry happens AFTER the T+1 settlement session
    -> the BUY is funded and fills."""
    async def go():
        bars = _channel_then_breakout()
        led = SettlementLedger(5_000.0)
        # proceeds recorded in the deep past so they've long since settled by the
        # time any bar plays (advance_to settles everything <= current session).
        led.record_sell(95_000.0, current_session=-10)
        runner = LiveRunner(DonchianBreakout(period=5, atr_period=3), "T", _router(),
                            starting_equity=100_000, target_notional_pct=0.95,
                            settlement=led)
        await runner.run(ReplayBarSource(bars))
        return runner

    runner = asyncio.run(go())
    assert runner.stats.skipped_unsettled == 0
    assert runner.stats.fills >= 1


def test_settlement_off_by_default_no_gate():
    async def go():
        bars = _channel_then_breakout()
        runner = LiveRunner(DonchianBreakout(period=5, atr_period=3), "T", _router(),
                            starting_equity=100_000, target_notional_pct=0.95)
        await runner.run(ReplayBarSource(bars))
        return runner
    runner = asyncio.run(go())
    assert runner.stats.skipped_unsettled == 0
    assert runner.stats.fills >= 1


def test_pdt_blocks_intraday_roundtrip_when_small_and_capped():
    """A <$25k account that already used its 3 day-trades cannot intraday
    round-trip a 4th time: the same-session closing SELL is blocked."""
    async def go():
        # all bars share ONE ET date -> one session; channel, breakout, entry,
        # then an intraday exit signal (would complete a day trade).
        h = lambda m: _ts(2025, 6, 17, 10, m)
        bars = [_bar(100, 100.5, 99.5, 100, h(m)) for m in range(0, 5)]
        bars += [_bar(105, 112, 104, 112, h(5)),    # breakout signal
                 _bar(120, 122, 119, 121, h(6)),    # ENTRY fills (BUY this session)
                 _bar(118, 119, 90, 95, h(7)),      # exit signal (low break)
                 _bar(94, 95, 93, 94, h(8))]        # would-be 4th day-trade SELL
        pdt = PdtGuard(min_equity=25_000, max_day_trades=3, window_sessions=5)
        for sym in ("AAA", "BBB", "CCC"):           # pre-use the 3-day-trade budget
            pdt.record_fill(0, sym, "BUY")
            pdt.record_fill(0, sym, "SELL")
        runner = LiveRunner(DonchianBreakout(period=5, atr_period=3), "T",
                            _router(),
                            starting_equity=20_000, target_notional_pct=0.5, pdt=pdt)
        await runner.run(ReplayBarSource(bars))
        return runner

    runner = asyncio.run(go())
    assert runner.stats.fills >= 1            # entry BUY filled (not a day trade)
    assert runner.stats.skipped_pdt >= 1      # the intraday closing SELL was blocked


def test_pdt_inactive_for_large_account():
    """Same intraday round-trip, but a >$25k account is unaffected by PDT."""
    async def go():
        h = lambda m: _ts(2025, 6, 17, 10, m)
        bars = [_bar(100, 100.5, 99.5, 100, h(m)) for m in range(0, 5)]
        bars += [_bar(105, 112, 104, 112, h(5)),
                 _bar(120, 122, 119, 121, h(6)),
                 _bar(118, 119, 90, 95, h(7)),
                 _bar(94, 95, 93, 94, h(8))]
        pdt = PdtGuard(min_equity=25_000, max_day_trades=3, window_sessions=5)
        for sym in ("AAA", "BBB", "CCC"):
            pdt.record_fill(0, sym, "BUY")
            pdt.record_fill(0, sym, "SELL")
        runner = LiveRunner(DonchianBreakout(period=5, atr_period=3), "T",
                            _router(),
                            starting_equity=100_000, target_notional_pct=0.5, pdt=pdt)
        await runner.run(ReplayBarSource(bars))
        return runner

    runner = asyncio.run(go())
    assert runner.stats.skipped_pdt == 0  # PDT doesn't restrict accounts >= $25k
