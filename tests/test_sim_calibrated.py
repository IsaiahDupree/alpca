"""The paper-calibrated SimAdapter preset matches the measured ~248ms baseline."""

import asyncio

from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.order import Order, OrderType, Side


def test_calibrated_preset_latency_profile():
    a = SimAdapter.paper_calibrated(seed=1, sleep=False)
    # submit + ack ~= 248ms (matches docs/BASELINE.md measured submit->ack)
    assert abs((a.submit_latency_ms + a.ack_latency_ms) - 248.0) < 1.0
    assert a.fill_latency_ms > 0


def test_calibrated_adapter_still_fills():
    async def go():
        a = SimAdapter.paper_calibrated(seed=2, sleep=False)
        o = Order(symbol="SPY", side=Side.BUY, qty=1, order_type=OrderType.MARKET)
        o.mark_signal(100.0)
        return await a.submit(o, ref_price=100.0)
    o = asyncio.run(go())
    assert o.status.value == "FILLED"


def test_calibrated_sleep_true_actually_delays():
    import time

    async def go():
        a = SimAdapter.paper_calibrated(seed=3, sleep=True)
        o = Order(symbol="SPY", side=Side.BUY, qty=1, order_type=OrderType.MARKET)
        o.mark_signal(100.0)
        t0 = time.perf_counter()
        await a.submit(o, ref_price=100.0)
        return (time.perf_counter() - t0) * 1000.0
    elapsed_ms = asyncio.run(go())
    # submit+ack+fill ~= 308ms; allow generous lower bound for jitter/CI
    assert elapsed_ms > 150.0
