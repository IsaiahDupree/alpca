"""
Calendar enforcement in the fill path (backtester + runner).

Uses REAL America/New_York epoch timestamps so the NYSE session classifier is
exercised for real. Verifies:
  - a signal whose execution bar is off-session is CARRIED FORWARD and fills at
    the next regular-hours bar (no off-hours fill, no look-ahead),
  - with enforcement OFF, behavior is unchanged,
  - the runner SKIPS submission on off-session bars.
"""

import asyncio
from zoneinfo import ZoneInfo

from datetime import datetime

from alpca.backtest.engine import run_backtest
from alpca.config import RiskConfig
from alpca.data.feed import ReplayBarSource
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.order_event_log import OrderEventLog
from alpca.execution.router import ExecutionRouter
from alpca.risk.risk_engine import RiskEngine
from alpca.runtime.runner import LiveRunner
from alpca.strategies.breakout import DonchianBreakout

_ET = ZoneInfo("America/New_York")


def _ts(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=_ET).timestamp()


def _bar(o, h, l, c, ts):
    return {"open": o, "high": h, "low": l, "close": c, "volume": 1e7,
            "timestamp": ts, "symbol": "T"}


def _session_boundary_bars():
    """
    Donchian(period=5) channel built during regular hours on 2025-06-17, the
    breakout signal lands on the LAST regular bar (15:59), and the immediately
    following bars are after-hours (16:30, 17:30) then the next regular open
    (2025-06-18 09:30). With enforcement ON, the entry must fill at the 06-18
    09:30 open, NOT at the 16:30 after-hours bar.
    """
    bars = []
    # 6 regular-hours channel bars 10:00..15:00
    for i, h in enumerate(range(10, 16)):
        bars.append(_bar(100, 100.5, 99.5, 100, _ts(2025, 6, 17, h, 0)))
    # breakout on the last regular bar 15:59 (signal generated here)
    bars.append(_bar(105, 112, 104, 112, _ts(2025, 6, 17, 15, 59)))
    # after-hours bars — must NOT fill here
    bars.append(_bar(130, 131, 129, 130, _ts(2025, 6, 17, 16, 30)))
    bars.append(_bar(140, 141, 139, 140, _ts(2025, 6, 17, 17, 30)))
    # next regular session open — entry should fill at THIS open (120)
    bars.append(_bar(120, 122, 119, 121, _ts(2025, 6, 18, 9, 30)))
    bars.append(_bar(121, 123, 120, 122, _ts(2025, 6, 18, 10, 30)))
    return bars


def test_enforced_entry_fills_at_next_regular_open_not_afterhours():
    bars = _session_boundary_bars()
    res = run_backtest(DonchianBreakout(period=5, atr_period=3), bars,
                       slippage_bps=0.0, commission_bps=0.0,
                       require_regular_hours=True)
    assert res.n_trades >= 1
    e = res.trades[0]
    # filled at the 06-18 09:30 regular open (120), never the 16:30 AH bar (130/140)
    assert abs(e.entry_ref - 120.0) < 1e-9, e.entry_ref


def test_unenforced_fills_at_immediate_next_bar():
    bars = _session_boundary_bars()
    res = run_backtest(DonchianBreakout(period=5, atr_period=3), bars,
                       slippage_bps=0.0, commission_bps=0.0,
                       require_regular_hours=False)
    e = res.trades[0]
    # without the gate, the very next bar (16:30 after-hours, open 130) fills
    assert abs(e.entry_ref - 130.0) < 1e-9, e.entry_ref


def _breakout_on_afterhours_bars():
    """Channel built in regular hours, but the breakout close lands on an
    AFTER-HOURS bar (16:30) — so the BUY signal fires off-session."""
    bars = [_bar(100, 100.5, 99.5, 100, _ts(2025, 6, 17, h, 0)) for h in range(10, 16)]
    # breakout fires here, but it's after-hours -> must be skipped when enforcing
    bars.append(_bar(105, 112, 104, 112, _ts(2025, 6, 17, 16, 30)))
    bars.append(_bar(113, 114, 112, 113, _ts(2025, 6, 17, 17, 30)))
    return bars


def test_runner_skips_off_session_submissions():
    async def run_with(enforce):
        bars = _breakout_on_afterhours_bars()
        risk = RiskEngine(RiskConfig(), day_start_equity=100_000)
        adapter = SimAdapter(seed=1, sleep=False)
        router = ExecutionRouter(adapter, risk, None, fill_timeout_s=1.0)
        runner = LiveRunner(DonchianBreakout(period=5, atr_period=3), "T", router,
                            require_regular_hours=enforce)
        await runner.run(ReplayBarSource(bars))
        return runner

    enforced = asyncio.run(run_with(True))
    unenforced = asyncio.run(run_with(False))

    # With enforcement: the after-hours breakout is skipped, not submitted.
    assert enforced.stats.orders_submitted == 0
    assert enforced.stats.skipped_off_session >= 1
    # Without enforcement: the same signal IS submitted (proves the gate is the
    # only thing suppressing it).
    assert unenforced.stats.orders_submitted >= 1
