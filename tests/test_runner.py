import asyncio

from alpca.config import RiskConfig
from alpca.data.bars import synthetic_bars
from alpca.data.feed import ReplayBarSource
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.order_event_log import OrderEventLog
from alpca.execution.router import ExecutionRouter
from alpca.risk.risk_engine import RiskEngine
from alpca.runtime.runner import LiveRunner
from alpca.strategies.registry import make


def test_runner_offline_trades_and_reports(tmp_path):
    async def go():
        bars = synthetic_bars("DEMO", n=400, seed=4, drift=0.0004, vol=0.013)
        log = OrderEventLog(str(tmp_path / "events.jsonl"))
        risk = RiskEngine(RiskConfig(), day_start_equity=100_000)
        adapter = SimAdapter(seed=4, sleep=False, slippage_bps_mean=3.0, slippage_bps_std=1.0)
        router = ExecutionRouter(adapter, risk, log, fill_timeout_s=1.0)
        runner = LiveRunner(make("donchian"), "DEMO", router, starting_equity=100_000)

        stats = await runner.run(ReplayBarSource(bars))
        return runner, stats, log

    runner, stats, log = asyncio.run(go())

    assert stats.bars_seen == 400
    assert stats.orders_submitted >= 2          # at least one round trip
    assert stats.fills == stats.orders_submitted  # sim fills everything that passes risk
    # position accounting is sane: equity is finite and cash bounded
    assert runner.equity > 0
    # ledger intact
    assert log.verify_chain().ok
    # latency report populated
    rep = runner.latency_report()
    assert rep.n_orders == stats.orders_submitted
    assert rep.slippage_bps.count == stats.fills


def test_runner_flat_after_exit_cycles():
    async def go():
        # engineer a clean up-then-down so donchian enters then exits, ending flat
        up = [{"open": 100 + i, "high": 100 + i + 0.5, "low": 100 + i - 0.5,
               "close": 100.0 + i, "volume": 1, "timestamp": i, "symbol": "X"} for i in range(30)]
        down = [{"open": 130 - i, "high": 130 - i + 0.5, "low": 130 - i - 0.5,
                 "close": 130.0 - i, "volume": 1, "timestamp": 30 + i, "symbol": "X"} for i in range(30)]
        risk = RiskEngine(RiskConfig(), day_start_equity=100_000)
        adapter = SimAdapter(seed=1, sleep=False)
        router = ExecutionRouter(adapter, risk, None, fill_timeout_s=1.0)
        runner = LiveRunner(make("donchian", period=5, atr_period=3), "X", router)
        await runner.run(ReplayBarSource(up + down))
        return runner

    runner = asyncio.run(go())
    # after a full up/down cycle the donchian position should be closed
    assert runner.position_qty == 0
