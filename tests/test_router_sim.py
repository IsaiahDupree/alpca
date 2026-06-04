import asyncio

from alpca.config import RiskConfig
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.order import Order, OrderType, Side
from alpca.execution.order_event_log import OrderEventLog
from alpca.execution.router import ExecutionRouter
from alpca.metrics.latency import build_latency_report
from alpca.risk.risk_engine import RiskEngine


def test_end_to_end_offline(tmp_path):
    async def run():
        path = str(tmp_path / "events.jsonl")
        log = OrderEventLog(path)
        risk = RiskEngine(RiskConfig(), day_start_equity=100_000)
        # fast tests: don't actually sleep, but latencies are still stamped (>=0)
        adapter = SimAdapter(seed=7, sleep=False, slippage_bps_mean=2.0, slippage_bps_std=0.5)
        router = ExecutionRouter(adapter, risk, log, fill_timeout_s=1.0)

        for _ in range(12):
            o = Order(symbol="SPY", side=Side.BUY, qty=2, order_type=OrderType.MARKET, strategy="orb")
            o.mark_signal(intended_price=500.0)
            res = await router.submit(o, equity=100_000, positions={}, ref_price=500.0)
            assert res.status.value == "FILLED"

        big = Order(symbol="SPY", side=Side.BUY, qty=1000, strategy="orb")
        big.mark_signal(500.0)
        rb = await router.submit(big, equity=100_000, positions={}, ref_price=500.0)
        assert rb.status.value == "REJECTED"
        assert "MAX_ORDER_NOTIONAL" in (rb.reject_reason or "")

        # ledger integrity: 12 filled * 4 events + 1 blocked * 2 events = 50
        chk = log.verify_chain()
        assert chk.ok
        assert chk.total == 12 * 4 + 2

        # rtt counts only orders that actually reached the broker (submitted),
        # NOT the risk-blocked one — it never made a round trip.
        rtt = router.rtt_stats()
        assert rtt["n"] == 12
        assert "p95_ms" in rtt

        rep = build_latency_report(router.orders)
        assert rep.n_orders == 13
        assert rep.n_filled == 12
        assert rep.slippage_bps.count == 12
        assert rep.slippage_bps.mean is not None

    asyncio.run(run())


def test_idempotency_dedup(tmp_path):
    async def run():
        risk = RiskEngine(RiskConfig(), day_start_equity=100_000)
        adapter = SimAdapter(seed=1, sleep=False)
        router = ExecutionRouter(adapter, risk, None, fill_timeout_s=1.0)

        o = Order(symbol="SPY", side=Side.BUY, qty=1, strategy="orb")
        o.mark_signal(500.0)
        r1 = await router.submit(o, equity=100_000, positions={}, ref_price=500.0)
        r2 = await router.submit(o, equity=100_000, positions={}, ref_price=500.0)
        assert r2.client_order_id == r1.client_order_id

    asyncio.run(run())
