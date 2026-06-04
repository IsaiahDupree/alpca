"""
Pin the SimAdapter slippage semantics for BOTH sides.

When intended_price == ref_price, realized slippage_bps must be small and
centered on the adapter's configured slippage_bps_mean — for buys AND sells. A
regression here (e.g. sell-side sign/scale bug) is the kind of thing that
produces absurd slippage numbers.
"""

import asyncio
import statistics

from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.order import Order, OrderType, Side


def _run_side(side: Side, n: int = 200, mean_bps: float = 3.5):
    async def go():
        adapter = SimAdapter(seed=11, sleep=False,
                             slippage_bps_mean=mean_bps, slippage_bps_std=1.0)
        slips = []
        ref = 100.0
        for _ in range(n):
            o = Order(symbol="SPY", side=side, qty=1, order_type=OrderType.MARKET)
            o.mark_signal(intended_price=ref)           # intended == ref
            o = await adapter.submit(o, ref_price=ref)  # adapter fills vs same ref
            assert o.status.value == "FILLED"
            slips.append(o.slippage_bps)
        return slips
    return asyncio.run(go())


def test_buy_slippage_matches_config():
    slips = _run_side(Side.BUY, mean_bps=3.5)
    m = statistics.fmean(slips)
    assert 2.5 < m < 4.5, f"buy slippage mean {m:.2f} off target"
    assert max(slips) < 12.0


def test_sell_slippage_matches_config():
    slips = _run_side(Side.SELL, mean_bps=3.5)
    m = statistics.fmean(slips)
    assert 2.5 < m < 4.5, f"sell slippage mean {m:.2f} off target (sign/scale bug?)"
    assert max(slips) < 12.0


def test_sell_and_buy_symmetric_magnitude():
    buy = statistics.fmean(_run_side(Side.BUY))
    sell = statistics.fmean(_run_side(Side.SELL))
    assert abs(buy - sell) < 1.5, f"asymmetric slippage buy={buy:.2f} sell={sell:.2f}"
