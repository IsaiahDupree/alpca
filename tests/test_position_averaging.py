"""
Runner position accounting: a 2nd BUY must AVERAGE into the existing position
(weighted-average cost), not overwrite it; partial SELLs keep the remainder at
the same cost basis. Direct unit tests of LiveRunner._apply_fill.
"""

import time

from alpca.config import RiskConfig
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.order import Fill, Order, Side
from alpca.execution.router import ExecutionRouter
from alpca.risk.risk_engine import RiskEngine
from alpca.runtime.runner import LiveRunner
from alpca.strategies.registry import make


def _runner():
    risk = RiskEngine(RiskConfig(), day_start_equity=100_000)
    router = ExecutionRouter(SimAdapter(seed=1, sleep=False), risk, None, fill_timeout_s=1.0)
    return LiveRunner(make("donchian"), "SPY", router, starting_equity=100_000)


def _filled(side, qty, price):
    o = Order(symbol="SPY", side=side, qty=qty)
    o.mark_signal(price)
    o.add_fill(Fill(ts=time.time(), price=price, qty=qty))
    return o


def test_second_buy_averages_not_overwrites():
    r = _runner()
    r._apply_fill(_filled(Side.BUY, 100, 50.0))   # 100 @ 50
    r._apply_fill(_filled(Side.BUY, 100, 60.0))   # +100 @ 60
    pos = r._positions["SPY"]
    assert pos.qty == 200                          # shares ADDED, not replaced
    assert abs(pos.avg_price - 55.0) < 1e-9        # weighted avg (50*100+60*100)/200
    # cash debited for both buys
    assert abs(r.cash - (100_000 - 100 * 50.0 - 100 * 60.0)) < 1e-6


def test_three_buys_weighted_average():
    r = _runner()
    r._apply_fill(_filled(Side.BUY, 10, 100.0))
    r._apply_fill(_filled(Side.BUY, 20, 130.0))
    r._apply_fill(_filled(Side.BUY, 70, 200.0))
    pos = r._positions["SPY"]
    assert pos.qty == 100
    expected = (10 * 100 + 20 * 130 + 70 * 200) / 100
    assert abs(pos.avg_price - expected) < 1e-9


def test_partial_sell_keeps_remainder_at_cost():
    r = _runner()
    r._apply_fill(_filled(Side.BUY, 100, 50.0))    # 100 @ 50
    r._apply_fill(_filled(Side.SELL, 40, 60.0))    # sell 40 @ 60
    pos = r._positions["SPY"]
    assert pos.qty == 60                           # 60 remain
    assert abs(pos.avg_price - 50.0) < 1e-9        # remainder keeps original cost
    # realized PnL on the 40 closed: (60-50)*40 = 400
    assert abs(r.stats.realized_pnl - 400.0) < 1e-9


def test_full_sell_closes_position():
    r = _runner()
    r._apply_fill(_filled(Side.BUY, 100, 50.0))
    r._apply_fill(_filled(Side.SELL, 100, 55.0))
    assert "SPY" not in r._positions
    assert abs(r.stats.realized_pnl - (55.0 - 50.0) * 100) < 1e-9


def test_oversell_flips_long_to_short():
    # _apply_fill now uses signed position math: selling MORE than held closes the
    # long (realizing PnL on the 100 held) and FLIPS into a short with the
    # remainder (50 @ 60). The runner's market path never emits this (it sizes
    # exactly), but the accounting must be correct if it happens.
    r = _runner()
    r._apply_fill(_filled(Side.BUY, 100, 50.0))
    r._apply_fill(_filled(Side.SELL, 150, 60.0))
    pos = r._positions["SPY"]
    assert pos.qty == -50            # flipped to a 50-share short
    assert abs(pos.avg_price - 60.0) < 1e-9
    assert abs(r.stats.realized_pnl - (60.0 - 50.0) * 100) < 1e-9  # realized on the 100 closed
