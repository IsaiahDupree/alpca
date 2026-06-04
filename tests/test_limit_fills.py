"""
Limit-fill realism: through-trade test, gap price-improvement, volume partial.
"""

import asyncio

from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.fills import FillModel
from alpca.execution.order import Order, OrderType, Side


FM = FillModel(half_spread_bps=1.0, impact_coef_bps=5.0, participation_cap=1.0, min_tick=0.01)


# --------------------------------------------------------------- FillModel unit
def test_buy_limit_no_fill_when_price_never_drops():
    # limit 99, but the bar's low is 100 -> price never came down -> no fill
    r = FM.fill_limit(True, 99.0, bar_open=100.0, bar_high=102.0, bar_low=100.0, qty=10)
    assert r.filled_qty == 0.0


def test_buy_limit_fills_at_limit_when_touched():
    # low dips to 98 (<=99), open 100 -> fills at the limit 99 (not the open)
    r = FM.fill_limit(True, 99.0, bar_open=100.0, bar_high=101.0, bar_low=98.0, qty=10)
    assert r.filled_qty == 10
    assert abs(r.price - 99.0) < 1e-9
    assert r.slippage_bps <= 1e-9  # never worse than the limit


def test_buy_limit_gap_through_gives_price_improvement():
    # bar OPENS at 97, below the 99 limit -> you fill at the better open (97)
    r = FM.fill_limit(True, 99.0, bar_open=97.0, bar_high=98.0, bar_low=96.0, qty=10)
    assert r.filled_qty == 10
    assert abs(r.price - 97.0) < 1e-9
    assert r.slippage_bps < 0  # favorable


def test_sell_limit_no_fill_when_price_never_rises():
    r = FM.fill_limit(False, 101.0, bar_open=100.0, bar_high=100.5, bar_low=99.0, qty=10)
    assert r.filled_qty == 0.0


def test_sell_limit_fills_and_gap_up_improves():
    touched = FM.fill_limit(False, 101.0, bar_open=100.0, bar_high=102.0, bar_low=99.0, qty=10)
    assert touched.filled_qty == 10
    assert abs(touched.price - 101.0) < 1e-9
    gap = FM.fill_limit(False, 101.0, bar_open=103.0, bar_high=104.0, bar_low=102.0, qty=10)
    assert abs(gap.price - 103.0) < 1e-9   # sell fills at the higher open
    assert gap.slippage_bps < 0            # favorable


def test_limit_volume_cap_partial():
    fm = FillModel(half_spread_bps=1.0, impact_coef_bps=0.0, participation_cap=0.10, min_tick=0.0)
    r = fm.fill_limit(True, 99.0, bar_open=100.0, bar_high=101.0, bar_low=98.0,
                      qty=50_000, bar_volume=100_000)
    assert r.capped
    assert r.filled_qty == 10_000


# --------------------------------------------------------------- SimAdapter wire
def test_sim_limit_no_fill_when_not_reached():
    async def go():
        adapter = SimAdapter(seed=1, sleep=False, fill_model=FM)
        o = Order(symbol="SPY", side=Side.BUY, qty=10, order_type=OrderType.LIMIT,
                  limit_price=99.0)
        o.mark_signal(100.0)
        # bar never trades down to 99
        o.metadata.update(bar_open=100.0, bar_high=101.0, bar_low=100.0, bar_volume=1e7)
        o = await adapter.submit(o, ref_price=100.0)
        return o
    o = asyncio.run(go())
    assert o.filled_qty == 0.0
    assert o.status.value != "FILLED"   # rests, not filled


def test_sim_limit_fills_when_reached():
    async def go():
        adapter = SimAdapter(seed=1, sleep=False, fill_model=FM)
        o = Order(symbol="SPY", side=Side.BUY, qty=10, order_type=OrderType.LIMIT,
                  limit_price=99.0)
        o.mark_signal(100.0)
        o.metadata.update(bar_open=100.0, bar_high=101.0, bar_low=98.0, bar_volume=1e7)
        o = await adapter.submit(o, ref_price=100.0)
        return o
    o = asyncio.run(go())
    assert o.status.value == "FILLED"
    assert abs(o.avg_fill_price - 99.0) < 1e-9


def test_sim_limit_without_bar_context_keeps_clamp_behavior():
    # no bar_high/low in metadata -> falls back to marketable clamp (always fills)
    async def go():
        adapter = SimAdapter(seed=1, sleep=False, fill_model=FM)
        o = Order(symbol="SPY", side=Side.BUY, qty=10, order_type=OrderType.LIMIT,
                  limit_price=99.0)
        o.mark_signal(100.0)
        o = await adapter.submit(o, ref_price=100.0)
        return o
    o = asyncio.run(go())
    assert o.status.value == "FILLED"
    assert o.avg_fill_price <= 99.0 + 1e-9  # clamped to the limit
