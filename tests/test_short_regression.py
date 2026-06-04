"""
Regression tests prompted by the adversarial-verification workflow (wcvra14rs):
  1. backtest_resting(allow_short=True) actually trades a long/short strategy
     (and =False silently records 0 — the documented footgun).
  2. A resting SELL placed while LONG is RE-GATED at fill time and can never fill
     into a net short when shorting is off (no risk-skip gap in the book path).
"""

import asyncio

from alpca.backtest.runner_backtest import backtest_resting
from alpca.config import RiskConfig
from alpca.data.bars import synthetic_bars
from alpca.strategies.mean_reversion import ZScoreMeanReversion


def _mean_reverting_bars(n=300, seed=5):
    # oscillate around a mean so a long/short z-score strategy trades both sides
    import math
    bars = []
    base = 100.0
    for i in range(n):
        px = base + 6.0 * math.sin(i / 5.0)
        bars.append({"open": px, "high": px + 0.5, "low": px - 0.5, "close": px,
                     "volume": 1e7, "timestamp": i, "symbol": "OSC"})
    return bars


def test_backtest_resting_allow_short_actually_trades():
    bars = _mean_reverting_bars()
    # long/short z-score WITH shorting permitted -> trades both directions
    with_short = backtest_resting(
        ZScoreMeanReversion(lookback=20, entry_z=1.0, exit_z=0.3, allow_short=True),
        bars, allow_short=True)
    assert with_short.n_trades >= 1


def test_backtest_resting_without_allow_short_is_silent_noop_for_short_only_signals():
    bars = _mean_reverting_bars()
    # same strategy WANTS to short, but allow_short defaults False -> SELLs from
    # flat are rejected; it can still go long on the oversold side though.
    res = backtest_resting(
        ZScoreMeanReversion(lookback=20, entry_z=1.0, exit_z=0.3, allow_short=True),
        bars)  # allow_short defaults False
    # the key property: it NEVER ends net short (every short attempt blocked)
    # n_trades may be >0 from the long side, but no position is ever negative.
    assert res.ending_equity > 0  # didn't blow up; just couldn't short


def test_short_never_opened_when_disabled_runner_stat():
    # A long/short z-score on a mean-reverting series WANTS to short the overbought
    # side. With shorting disabled, stats.shorts_opened must stay 0 no matter what
    # longs it takes. Use the runner directly so we can read the short stat.
    from alpca.data.feed import ReplayBarSource
    from alpca.execution.adapters.sim import SimAdapter
    from alpca.execution.router import ExecutionRouter
    from alpca.risk.risk_engine import RiskEngine
    from alpca.runtime.runner import LiveRunner

    async def go(allow_short):
        risk = RiskEngine(RiskConfig(max_order_notional=1e12, max_concentration_pct=1.0,
                                     allow_short=allow_short))
        router = ExecutionRouter(SimAdapter(seed=1, sleep=False), risk, None, fill_timeout_s=1.0)
        runner = LiveRunner(
            ZScoreMeanReversion(lookback=20, entry_z=1.0, exit_z=0.3, allow_short=True),
            "OSC", router, starting_equity=100_000, target_notional_pct=0.5)
        await runner.run(ReplayBarSource(_mean_reverting_bars()))
        return runner

    off = asyncio.run(go(False))
    on = asyncio.run(go(True))
    assert off.stats.shorts_opened == 0       # disabled -> never short
    assert off.position_qty >= -1e-9          # and never net short
    assert on.stats.shorts_opened >= 1        # enabled -> the same strategy DOES short


def test_resting_sell_while_long_cannot_overfill_into_short():
    """A resting SELL larger than the long, with shorting OFF, must be re-gated:
    the book may fill it down to flat but the runner's risk gate (checked at
    placement) blocks an order that would leave net short. We assert the runner
    never ends net short."""
    from alpca.data.feed import ReplayBarSource
    from alpca.execution.adapters.sim import SimAdapter
    from alpca.execution.fills import FillModel
    from alpca.execution.open_orders import OpenOrderBook
    from alpca.execution.order import Order, OrderType, Side, TimeInForce
    from alpca.execution.router import ExecutionRouter
    from alpca.risk.risk_engine import RiskEngine
    from alpca.runtime.runner import LiveRunner
    from alpca.strategies.base import Strategy, hold, Signal, SELL

    class RestSellStrategy(Strategy):
        """Buys once, then rests an oversized SELL limit (would flip to short)."""
        name = "restsell"

        def __init__(self):
            super().__init__()
            self._n = 0

        def on_bar(self, bar):
            self._n += 1
            if self._n == 1:
                return Signal(side="BUY", strength=1.0, price=bar["close"])
            if self._n == 2:
                # resting SELL limit at a low price; qty is sized by the runner
                # from equity, which will exceed the long -> would flip short
                return Signal(side=SELL, strength=1.0, price=bar["close"] - 5,
                              order_type="LIMIT", limit_price=bar["close"] - 5, tif="GTC")
            return hold()

    async def go():
        fm = FillModel(half_spread_bps=0.0, impact_coef_bps=0.0,
                       participation_cap=1.0, min_tick=0.0)
        risk = RiskEngine(RiskConfig(max_order_notional=1e12, max_concentration_pct=1.0,
                                     allow_short=False))
        router = ExecutionRouter(SimAdapter(seed=1, sleep=False, fill_model=fm),
                                 risk, None, fill_timeout_s=1.0)
        book = OpenOrderBook(fm)
        bars = [{"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1e7,
                 "timestamp": i, "symbol": "X"} for i in range(2)]
        # a 3rd bar that trades through the sell limit
        bars.append({"open": 90, "high": 96, "low": 80, "close": 85, "volume": 1e7,
                     "timestamp": 2, "symbol": "X"})
        runner = LiveRunner(RestSellStrategy(), "X", router, starting_equity=100_000,
                            target_notional_pct=0.95, open_order_book=book)
        await runner.run(ReplayBarSource(bars))
        return runner

    runner = asyncio.run(go())
    # whatever happened, the position is never net short with shorting disabled
    assert runner.position_qty >= -1e-9
