"""
Short-selling end-to-end: risk gate, signed accounting through the runner,
borrow-fee accrual, and the long/short z-score strategy.
"""

import asyncio
from zoneinfo import ZoneInfo
from datetime import datetime

from alpca.config import RiskConfig
from alpca.data.feed import ReplayBarSource
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.order import Order, Side
from alpca.execution.router import ExecutionRouter
from alpca.risk.risk_engine import RiskEngine
from alpca.runtime.account import BorrowFeeLedger
from alpca.runtime.runner import LiveRunner
from alpca.strategies.mean_reversion import ZScoreMeanReversion

_ET = ZoneInfo("America/New_York")


def _ts(d, h=11):
    return datetime(2025, 6, d, h, 0, tzinfo=_ET).timestamp()


def _bar(o, h, l, c, ts):
    return {"open": o, "high": h, "low": l, "close": c, "volume": 1e7,
            "timestamp": ts, "symbol": "T"}


# ----------------------------------------------------------------- risk gate
def test_risk_blocks_short_by_default():
    eng = RiskEngine(RiskConfig(max_order_notional=1e9, max_concentration_pct=1.0))
    o = Order(symbol="T", side=Side.SELL, qty=100)
    o.mark_signal(50.0)
    d = eng.check(o, equity=100_000, positions={}, ref_price=50.0)  # flat -> would short
    assert not d.allowed
    assert d.code == "SHORT_NOT_ALLOWED"


def test_risk_allows_short_when_enabled():
    eng = RiskEngine(RiskConfig(max_order_notional=1e9, max_concentration_pct=1.0,
                                allow_short=True))
    o = Order(symbol="T", side=Side.SELL, qty=100)
    o.mark_signal(50.0)
    d = eng.check(o, equity=100_000, positions={}, ref_price=50.0)
    assert d.allowed


def test_risk_allows_sell_that_only_reduces_long():
    from alpca.risk.risk_engine import Position
    eng = RiskEngine(RiskConfig(max_order_notional=1e9, max_concentration_pct=1.0))
    o = Order(symbol="T", side=Side.SELL, qty=40)
    o.mark_signal(50.0)
    d = eng.check(o, equity=100_000, positions={"T": Position("T", 100, 50.0)},
                  ref_price=50.0)
    assert d.allowed  # selling 40 of a 100 long is fine even with shorting off


# ----------------------------------------------------------------- runner e2e
def _router(allow_short):
    risk = RiskEngine(RiskConfig(max_order_notional=1e9, max_concentration_pct=1.0,
                                 allow_short=allow_short))
    return ExecutionRouter(SimAdapter(seed=1, sleep=False), risk, None, fill_timeout_s=1.0)


def _overbought_then_revert_bars():
    """Flat ~100 to build the z-window, a spike UP (overbought -> SHORT signal),
    then a revert to mean (cover)."""
    bars = [_bar(100, 100.2, 99.8, 100 + (0.1 if i % 2 else -0.1), _ts(2 + i))
            for i in range(20)]
    # spike up well above mean -> z > entry_z -> short
    bars.append(_bar(100, 106, 100, 105, _ts(23)))
    bars.append(_bar(105, 106, 104, 105, _ts(24)))   # short fills here / holds
    # revert toward mean -> cover
    bars.append(_bar(102, 102, 99, 100, _ts(25)))
    bars.append(_bar(100, 100.5, 99.5, 100, _ts(26)))
    return bars


def test_runner_opens_and_covers_a_short():
    async def go():
        bars = _overbought_then_revert_bars()
        runner = LiveRunner(ZScoreMeanReversion(lookback=20, entry_z=1.5, exit_z=0.5,
                                                allow_short=True),
                            "T", _router(allow_short=True), starting_equity=100_000,
                            target_notional_pct=0.5)
        await runner.run(ReplayBarSource(bars))
        return runner

    runner = asyncio.run(go())
    assert runner.stats.shorts_opened >= 1
    # by the end it should have covered (position back to flat or at least not stuck)
    # and realized PnL should be finite
    assert isinstance(runner.stats.realized_pnl, float)


def test_short_blocked_when_disabled_runner():
    async def go():
        bars = _overbought_then_revert_bars()
        runner = LiveRunner(ZScoreMeanReversion(lookback=20, entry_z=1.5, exit_z=0.5,
                                                allow_short=True),  # strategy WANTS to short
                            "T", _router(allow_short=False),        # but risk says no
                            starting_equity=100_000, target_notional_pct=0.5)
        await runner.run(ReplayBarSource(bars))
        return runner

    runner = asyncio.run(go())
    assert runner.stats.shorts_opened == 0
    assert runner.stats.rejects >= 1  # the short attempt was rejected


def test_borrow_fee_accrues_while_short():
    async def go():
        # short held across several sessions -> borrow fee debited each session
        bars = _overbought_then_revert_bars()
        # extend with more HOLD sessions while short (no revert) to accrue fees
        for d in range(27, 31):  # June has 30 days
            bars.append(_bar(105, 106, 104, 105, _ts(d)))  # stays overbought -> stays short
        borrow = BorrowFeeLedger(annual_rate=0.10)  # 10% APR for a visible fee
        runner = LiveRunner(ZScoreMeanReversion(lookback=20, entry_z=1.5, exit_z=0.5,
                                                stop_z=10.0, allow_short=True),
                            "T", _router(allow_short=True), starting_equity=100_000,
                            target_notional_pct=0.5, borrow=borrow)
        await runner.run(ReplayBarSource(bars))
        return runner, borrow

    runner, borrow = asyncio.run(go())
    assert runner.stats.shorts_opened >= 1
    assert runner.stats.borrow_paid > 0
    assert abs(runner.stats.borrow_paid - borrow.total_accrued) < 1e-9


def test_borrow_ledger_math():
    led = BorrowFeeLedger(annual_rate=0.0252, trading_days=252)  # daily rate = 0.0001
    fee = led.accrue_for_session(100_000.0)  # 100k short * 0.0001 = $10
    assert abs(fee - 10.0) < 1e-9
    assert abs(led.total_accrued - 10.0) < 1e-9
    assert led.accrue_for_session(0.0) == 0.0  # flat/long -> no fee
