"""
Deep, deterministic, offline coverage of alpca.runtime.runner.LiveRunner.

All tests are pure/offline: SimAdapter (sleep=False, seeded) + ReplayBarSource +
asyncio.run. No network, no mocks. A tiny scripted strategy drives the runner's
market-path translation (open-long / cover-short / open-short / sell-long / flip /
flatten) deterministically so the SIGNED accounting, equity, stats counters,
to_result() and _record_trade can be asserted with exact computed values.

Covers:
  - _account_fill via apply_fill (signed cash/pnl/position)
  - equity property (short loses on price rise; long-only marks to market)
  - position_qty
  - market-path translation branches
  - RunnerStats counters
  - to_result() -> BacktestResult (incl. EOD mark-to-market of an open trade)
  - _record_trade signed round trips
  - edge/degenerate inputs (zero/negative/None price, empty source, idempotency,
    flip-through, repeated EXIT)
"""

import asyncio
import math

import pytest

from alpca.config import RiskConfig
from alpca.data.feed import ReplayBarSource
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.order import Side
from alpca.execution.router import ExecutionRouter
from alpca.risk.risk_engine import RiskEngine, Position
from alpca.runtime.position_math import apply_fill
from alpca.runtime.runner import LiveRunner, RunnerStats
from alpca.strategies.base import BUY, SELL, EXIT, HOLD, Signal, Strategy


# --------------------------------------------------------------------------- helpers
def _router(allow_short=True, *, max_notional=1e12, max_conc=1.0, seed=7):
    """A SimAdapter-backed ExecutionRouter that fills deterministically and never
    actually sleeps. slippage std 0 / mean 0 keeps fills near the reference price
    so position math is predictable (the only non-determinism is removed)."""
    risk = RiskEngine(RiskConfig(max_order_notional=max_notional,
                                 max_concentration_pct=max_conc,
                                 max_open_positions=50,
                                 allow_short=allow_short))
    adapter = SimAdapter(seed=seed, sleep=False,
                         slippage_bps_mean=0.0, slippage_bps_std=0.0,
                         submit_latency_ms=0.0, ack_latency_ms=0.0,
                         fill_latency_ms=0.0, latency_jitter_ms=0.0)
    return ExecutionRouter(adapter, risk, None, fill_timeout_s=1.0)


def _bar(c, ts, *, o=None, h=None, l=None, v=1e7):
    o = c if o is None else o
    return {"open": o, "high": (h if h is not None else c),
            "low": (l if l is not None else c), "close": c,
            "volume": v, "timestamp": float(ts), "symbol": "T"}


class ScriptedStrategy(Strategy):
    """Emits a pre-baked Signal per bar (cycling/holding past the end). Lets a test
    deterministically script BUY/SELL/EXIT/HOLD sequences to exercise every
    market-path translation branch in the runner without depending on indicators."""
    name = "scripted"

    def __init__(self, sides):
        super().__init__()
        self._sides = list(sides)
        self._i = 0
        self.fills = []  # (side, qty, price) recorded via on_fill

    def on_bar(self, bar):
        if self._i < len(self._sides):
            side = self._sides[self._i]
        else:
            side = HOLD
        self._i += 1
        if side == HOLD:
            return Signal(side=HOLD, strength=0.0)
        if side == EXIT:
            return Signal(side=EXIT, strength=1.0, price=bar["close"])
        return Signal(side=side, strength=1.0, price=bar["close"])

    def on_fill(self, side, qty, price):
        self.fills.append((side, qty, price))


def _run(sides, bars, *, router=None, **kw):
    """Drive a LiveRunner through `bars` with a scripted-signal strategy."""
    r = router if router is not None else _router()
    runner = LiveRunner(ScriptedStrategy(sides), "T", r, **kw)
    asyncio.run(runner.run(ReplayBarSource(bars)))
    return runner


# A monotone price ramp; fills land at ~close because slippage is zeroed.
def _ramp(prices, t0=1_700_000_000):
    return [_bar(p, t0 + i) for i, p in enumerate(prices)]


# ============================================================================
# apply_fill (signed) — the math the runner delegates to. Invariants + exacts.
# ============================================================================
@pytest.mark.parametrize("side,qty,price,exp_qty,exp_avg,exp_realized,exp_cash", [
    ("BUY", 10, 50.0, 10.0, 50.0, 0.0, -500.0),     # open long
    ("SELL", 10, 50.0, -10.0, 50.0, 0.0, 500.0),    # open short (receive proceeds)
])
def test_apply_fill_open_from_flat(side, qty, price, exp_qty, exp_avg, exp_realized, exp_cash):
    eff = apply_fill(0.0, 0.0, side, qty, price)
    assert eff.new_qty == exp_qty
    assert eff.new_avg == exp_avg
    assert eff.realized == exp_realized
    assert eff.cash_delta == exp_cash
    assert eff.opened_qty == qty
    assert eff.closed_qty == 0.0


def test_apply_fill_add_to_long_blends_avg():
    eff = apply_fill(10.0, 50.0, "BUY", 10, 60.0)
    assert eff.new_qty == 20.0
    assert eff.new_avg == pytest.approx(55.0)   # (10*50 + 10*60)/20
    assert eff.realized == 0.0
    assert eff.cash_delta == -600.0


def test_apply_fill_close_long_realizes_pnl():
    eff = apply_fill(10.0, 50.0, "SELL", 10, 60.0)
    assert abs(eff.new_qty) < 1e-9
    assert eff.new_avg == 0.0
    assert eff.realized == pytest.approx(100.0)  # (60-50)*10 long
    assert eff.cash_delta == 600.0
    assert eff.closed_qty == 10.0


def test_apply_fill_close_short_realizes_pnl():
    # short entry 50, cover at 45 -> profit (avg - exit)*qty = 5*10
    eff = apply_fill(-10.0, 50.0, "BUY", 10, 45.0)
    assert abs(eff.new_qty) < 1e-9
    assert eff.realized == pytest.approx(50.0)
    assert eff.cash_delta == -450.0


def test_apply_fill_flip_long_to_short():
    # long 10 @50, SELL 30 @60 -> close 10 (+100 realized) then open short 20 @60
    eff = apply_fill(10.0, 50.0, "SELL", 30, 60.0)
    assert eff.new_qty == -20.0
    assert eff.new_avg == 60.0
    assert eff.realized == pytest.approx(100.0)
    assert eff.closed_qty == 10.0
    assert eff.opened_qty == 20.0


@pytest.mark.parametrize("bad", [
    (0, 50.0), (-5, 50.0), (10, 0.0), (10, -1.0),
])
def test_apply_fill_degenerate_qty_or_price_is_noop(bad):
    qty, price = bad
    eff = apply_fill(7.0, 33.0, "BUY", qty, price)
    assert eff.new_qty == 7.0 and eff.new_avg == 33.0
    assert eff.realized == 0.0 and eff.cash_delta == 0.0


# ============================================================================
# LiveRunner construction / defaults
# ============================================================================
def test_default_construction_state():
    runner = LiveRunner(ScriptedStrategy([]), "T", _router())
    assert runner.cash == 100_000.0
    assert runner.starting_equity == 100_000.0
    assert runner.target_notional_pct == 0.20
    assert runner.position_qty == 0.0
    assert runner.equity == 100_000.0  # no position, last_price 0
    assert isinstance(runner.stats, RunnerStats)
    assert runner.equity_curve == []


def test_empty_bar_source_no_crash():
    runner = _run([], [])
    assert runner.stats.bars_seen == 0
    assert runner.equity_curve == []
    assert runner.position_qty == 0.0


def test_hold_only_never_trades():
    bars = _ramp([100, 101, 102, 103])
    runner = _run([HOLD, HOLD, HOLD, HOLD], bars)
    assert runner.stats.bars_seen == 4
    assert runner.stats.signals == 0
    assert runner.stats.orders_submitted == 0
    assert runner.stats.fills == 0
    assert runner.position_qty == 0.0
    assert len(runner.equity_curve) == 4
    assert all(e == 100_000.0 for e in runner.equity_curve)


# ============================================================================
# Market path: open long
# ============================================================================
def test_open_long_sets_signed_position_and_cash():
    bars = _ramp([100, 100, 100])
    # size = max(1, int(equity*0.20/ref)) = int(100000*0.2/100)=200
    runner = _run([BUY, HOLD, HOLD], bars, target_notional_pct=0.20)
    assert runner.position_qty == 200.0
    assert runner.stats.orders_submitted == 1
    assert runner.stats.fills == 1
    assert runner.stats.signals == 1
    # cash decreased by ~200*100 (slippage zeroed -> exactly 100)
    assert runner.cash == pytest.approx(100_000.0 - 200 * 100.0)
    # equity unchanged at entry price (cash + 200*100)
    assert runner.equity == pytest.approx(100_000.0)


def test_already_long_buy_does_not_pyramid():
    bars = _ramp([100, 100, 100])
    runner = _run([BUY, BUY, HOLD], bars)
    # second BUY while long -> no order
    assert runner.stats.orders_submitted == 1
    assert runner.position_qty == 200.0


def test_long_equity_rises_with_price():
    bars = _ramp([100, 110, 120])
    runner = _run([BUY, HOLD, HOLD], bars)
    # 200 sh long, last price 120 -> equity = cash + 200*120
    assert runner.position_qty == 200.0
    assert runner.equity == pytest.approx(100_000.0 - 200 * 100 + 200 * 120)
    assert runner.equity_curve[-1] == pytest.approx(runner.equity)
    assert runner.equity_curve[-1] > runner.equity_curve[0]


# ============================================================================
# Market path: open short + equity falls on price RISE (the key short invariant)
# ============================================================================
def test_open_short_signed_negative_and_cash_credited():
    bars = _ramp([100, 100, 100])
    runner = _run([SELL, HOLD, HOLD], bars, target_notional_pct=0.20)
    assert runner.position_qty == -200.0
    assert runner.stats.shorts_opened == 1
    # opening a short credits cash by proceeds
    assert runner.cash == pytest.approx(100_000.0 + 200 * 100.0)
    assert runner.equity == pytest.approx(100_000.0)


def test_short_loses_on_price_rise():
    bars = _ramp([100, 130])  # price rises after the short
    runner = _run([SELL, HOLD], bars)
    assert runner.position_qty == -200.0
    # equity = cash(+200*100) + (-200)*130 = 100000 + 20000 - 26000 = 94000
    assert runner.equity == pytest.approx(94_000.0)
    assert runner.equity < runner.starting_equity
    # equity strictly fell from the entry bar (100) to the rise bar (130)
    assert runner.equity_curve[1] < runner.equity_curve[0]


def test_short_blocked_when_disabled_no_position():
    bars = _ramp([100, 100])
    runner = _run([SELL, HOLD], bars, router=_router(allow_short=False))
    assert runner.position_qty == 0.0
    assert runner.stats.shorts_opened == 0
    assert runner.stats.rejects == 1
    assert runner.stats.fills == 0
    # equity unchanged: never traded
    assert runner.equity == pytest.approx(100_000.0)


# ============================================================================
# Market path: cover short, sell long, flatten via EXIT, flip
# ============================================================================
def test_cover_short_with_buy_returns_to_flat():
    bars = _ramp([100, 90])  # short at 100, cover (BUY) at 90 -> profit
    runner = _run([SELL, BUY], bars)
    assert runner.position_qty == 0.0   # cover buys back abs(pos)
    # realized: (avg - exit)*qty short = (100-90)*200 = 2000
    assert runner.stats.realized_pnl == pytest.approx(2000.0)
    assert runner.equity == pytest.approx(102_000.0)


def test_sell_long_returns_to_flat_and_realizes():
    bars = _ramp([100, 115])  # long at 100, SELL the long at 115
    runner = _run([BUY, SELL], bars)
    assert runner.position_qty == 0.0
    assert runner.stats.realized_pnl == pytest.approx((115 - 100) * 200)
    assert runner.equity == pytest.approx(100_000.0 + (115 - 100) * 200)


def test_exit_flattens_long():
    bars = _ramp([100, 100, 108])
    runner = _run([BUY, HOLD, EXIT], bars)
    assert runner.position_qty == 0.0
    assert runner.stats.realized_pnl == pytest.approx((108 - 100) * 200)


def test_exit_flattens_short():
    bars = _ramp([100, 100, 92])
    runner = _run([SELL, HOLD, EXIT], bars)
    assert runner.position_qty == 0.0
    # short profit (100-92)*200
    assert runner.stats.realized_pnl == pytest.approx((100 - 92) * 200)


def test_exit_while_flat_is_noop():
    bars = _ramp([100, 100])
    runner = _run([EXIT, EXIT], bars)
    assert runner.position_qty == 0.0
    assert runner.stats.orders_submitted == 0
    # EXIT is actionable so it counts as a signal, but no order is produced
    assert runner.stats.signals == 2
    assert runner.stats.fills == 0


def test_already_short_sell_is_ignored():
    bars = _ramp([100, 100, 100])
    runner = _run([SELL, SELL, HOLD], bars)
    assert runner.position_qty == -200.0
    assert runner.stats.orders_submitted == 1  # 2nd SELL while short -> ignored
    assert runner.stats.shorts_opened == 1


# ============================================================================
# Market-path FLIP: SELL while long sells only the long (no flip), and the
# reverse. The runner sizes the SELL to abs(pos_qty), so it flattens, not flips.
# A genuine flip needs a subsequent open. Verify the two-step long->flat->short.
# ============================================================================
def test_long_then_sell_then_short_two_steps():
    bars = _ramp([100, 100, 100, 100])
    # BUY (long 200), SELL (-> flat, sells the long), SELL again (-> open short)
    runner = _run([BUY, SELL, SELL, HOLD], bars)
    assert runner.position_qty == -200.0
    assert runner.stats.shorts_opened == 1


# ============================================================================
# RunnerStats counters aggregate correctly across a multi-trade run
# ============================================================================
def test_stats_counters_full_cycle():
    bars = _ramp([100, 110, 90, 95, 95])
    # BUY(open long), SELL(flatten long), SELL(open short), BUY(cover), HOLD
    runner = _run([BUY, SELL, SELL, BUY, HOLD], bars)
    assert runner.stats.bars_seen == 5
    assert runner.stats.signals == 4
    assert runner.stats.orders_submitted == 4
    assert runner.stats.fills == 4
    assert runner.stats.rejects == 0
    assert runner.stats.shorts_opened == 1
    assert runner.position_qty == 0.0


# ============================================================================
# to_result() -> BacktestResult
# ============================================================================
def test_to_result_closed_long_trade():
    bars = _ramp([100, 100, 120])
    runner = _run([BUY, HOLD, EXIT], bars)
    res = runner.to_result()
    assert res.symbol == "T"
    assert res.strategy == "scripted"
    assert res.starting_equity == 100_000.0
    assert res.ending_equity == pytest.approx(runner.equity_curve[-1])
    assert res.n_trades == 1
    t = res.closed_trades[0]
    assert t.qty == 200.0          # signed positive (long)
    assert t.entry_price == pytest.approx(100.0)
    assert t.exit_price == pytest.approx(120.0)
    assert t.pnl == pytest.approx((120 - 100) * 200)


def test_to_result_marks_open_position_to_market():
    bars = _ramp([100, 100, 130])
    runner = _run([BUY, HOLD, HOLD], bars)  # still long at end
    res = runner.to_result()
    # the open trade is synthetically closed at last price 130
    assert res.n_trades == 1
    t = res.closed_trades[0]
    assert t.exit_price == pytest.approx(130.0)
    assert t.reason_out == "EOD mark-to-market"
    assert t.pnl == pytest.approx((130 - 100) * 200)


def test_to_result_short_trade_signed_qty_negative():
    bars = _ramp([100, 100, 80])
    runner = _run([SELL, HOLD, HOLD], bars)  # short, held to end
    res = runner.to_result()
    t = res.closed_trades[0]
    assert t.qty == -200.0       # signed negative for shorts
    assert t.entry_price == pytest.approx(100.0)
    assert t.exit_price == pytest.approx(80.0)
    # short profit: (exit-entry)*qty = (80-100)*(-200) = +4000
    assert t.pnl == pytest.approx(4000.0)
    assert t.pnl > 0


def test_to_result_empty_equity_curve_uses_live_equity():
    runner = LiveRunner(ScriptedStrategy([]), "T", _router())
    res = runner.to_result()
    assert res.equity_curve == []
    assert res.ending_equity == runner.equity  # falls back to live equity
    assert res.n_trades == 0


def test_to_result_passes_through_cost_model_fields():
    bars = _ramp([100, 110])
    runner = _run([BUY, HOLD], bars)
    res = runner.to_result(commission_bps=2.5, slippage_bps=1.0)
    assert res.commission_bps == 2.5
    assert res.slippage_bps == 1.0


# ============================================================================
# _record_trade: signed round trips, average-in on adds (via resting? no — market
# path doesn't pyramid). Test the signed behaviour through full cycles instead.
# ============================================================================
def test_record_trade_two_separate_round_trips():
    bars = _ramp([100, 110, 100, 80, 60])
    # long round trip (100->110), then short round trip (100->80? cover at 80...)
    # BUY@100, EXIT@110 (close long), SELL@100 (open short), EXIT@80 (cover short)
    runner = _run([BUY, EXIT, SELL, EXIT, HOLD], bars)
    res = runner.to_result()
    assert res.n_trades == 2
    long_t, short_t = res.closed_trades
    assert long_t.qty > 0 and short_t.qty < 0
    # long opens flat: size = int(100000*0.2/100) = 200
    assert long_t.qty == 200.0
    assert long_t.pnl == pytest.approx((110 - 100) * 200)
    # after the long +2000, equity is 102000 when the short opens, so the short
    # is sized from that grown equity: size = int(102000*0.2/100) = 204
    assert short_t.qty == -204.0
    assert short_t.pnl == pytest.approx((80 - 100) * -204.0)  # +4080


def test_realized_pnl_accumulates_across_trades():
    bars = _ramp([100, 110, 100, 80, 80])
    runner = _run([BUY, EXIT, SELL, EXIT, HOLD], bars)
    # long: 200 sh @100 -> exit 110 = +2000. equity now 102000.
    # short: int(102000*0.2/100)=204 sh @100 -> cover 80 = (100-80)*204 = +4080.
    assert runner.stats.realized_pnl == pytest.approx(2000.0 + 4080.0)


# ============================================================================
# Idempotency / robustness on the bar stream
# ============================================================================
def test_zero_close_bar_skipped_no_order():
    # FIXED BEHAVIOR: an actionable market signal on a bar whose reference price
    # is 0 (or NaN/inf) is now skipped gracefully — the runner returns before
    # sizing instead of raising ZeroDivisionError at `int(equity*pct / ref)`.
    # The run completes, no order is submitted, and the position stays flat.
    bars = [_bar(0.0, 1_700_000_000)]
    runner = _run([BUY], bars)
    assert runner.stats.orders_submitted == 0
    assert runner.stats.fills == 0
    assert runner.position_qty == 0.0


def test_negative_close_bar_also_raises_or_no_fill():
    # A negative close ref: sizing yields a negative size, max(1, neg)=1, the sim
    # then rejects a non-positive reference price -> no position, no crash.
    bars = [_bar(-5.0, 1_700_000_000), _bar(100, 1_700_000_001)]
    runner = _run([BUY, HOLD], bars)
    assert runner.position_qty == 0.0
    assert runner.stats.bars_seen == 2


def test_stop_halts_consumption_midstream():
    async def go():
        bars = _ramp([100, 101, 102, 103, 104])
        runner = LiveRunner(ScriptedStrategy([HOLD] * 5), "T", _router())

        class StoppingSource:
            def __init__(self, inner, runner):
                self._inner = inner
                self._runner = runner
                self.latency = inner.latency

            async def __aiter__(self):
                n = 0
                async for bar in self._inner:
                    yield bar
                    n += 1
                    if n == 2:
                        self._runner.stop()

        await runner.run(StoppingSource(ReplayBarSource(bars), runner))
        return runner

    runner = asyncio.run(go())
    # run() breaks at top of loop AFTER stop set on bar 2 -> 2 bars processed,
    # then a 3rd is pulled but loop breaks before processing it.
    assert runner.stats.bars_seen == 2


def test_equity_curve_length_matches_bars_processed():
    bars = _ramp([100, 101, 102])
    runner = _run([HOLD, HOLD, HOLD], bars)
    assert len(runner.equity_curve) == 3
    assert runner.stats.bars_seen == 3


def test_feed_latency_populated_from_source():
    # ReplayBarSource records latency only when bars carry recv_ts; synthetic bars
    # don't, so feed_latency stays an empty-n dict — assert that graceful default.
    bars = _ramp([100, 101])
    runner = _run([HOLD, HOLD], bars)
    assert isinstance(runner.stats.feed_latency, dict)
    assert runner.stats.feed_latency.get("n") == 0


# ============================================================================
# equity property direct invariants (unit-level, no run loop)
# ============================================================================
def test_equity_property_long_and_short_signed():
    runner = LiveRunner(ScriptedStrategy([]), "T", _router())
    runner.cash = 50_000.0
    runner._last_price = 20.0
    runner._positions["T"] = Position("T", 100.0, 18.0)   # long
    assert runner.equity == pytest.approx(50_000.0 + 100 * 20.0)
    runner._positions["T"] = Position("T", -100.0, 18.0)  # short
    assert runner.equity == pytest.approx(50_000.0 - 100 * 20.0)


def test_position_qty_reflects_dict():
    runner = LiveRunner(ScriptedStrategy([]), "T", _router())
    assert runner.position_qty == 0.0
    runner._positions["T"] = Position("T", -42.0, 10.0)
    assert runner.position_qty == -42.0


def test_account_fill_direct_signed_math():
    runner = LiveRunner(ScriptedStrategy([]), "T", _router())
    runner._last_price = 50.0
    runner._account_fill(Side.BUY, 10, 50.0)   # open long 10 @50
    assert runner.position_qty == 10.0
    assert runner.cash == pytest.approx(100_000.0 - 500.0)
    runner._account_fill(Side.SELL, 10, 60.0)  # close at 60 -> +100 realized
    assert runner.position_qty == 0.0
    assert runner.stats.realized_pnl == pytest.approx(100.0)
    assert runner.cash == pytest.approx(100_000.0 - 500.0 + 600.0)


def test_account_fill_ignores_nonpositive_qty_or_price():
    runner = LiveRunner(ScriptedStrategy([]), "T", _router())
    runner._account_fill(Side.BUY, 0, 50.0)
    runner._account_fill(Side.BUY, 10, 0.0)
    runner._account_fill(Side.BUY, -5, 50.0)
    assert runner.position_qty == 0.0
    assert runner.cash == 100_000.0
    assert runner.stats.realized_pnl == 0.0


def test_account_fill_flip_via_oversized_opposite():
    runner = LiveRunner(ScriptedStrategy([]), "T", _router())
    runner._last_price = 60.0
    runner._account_fill(Side.BUY, 10, 50.0)    # long 10
    runner._account_fill(Side.SELL, 30, 60.0)   # sell 30 -> flip to short 20
    assert runner.position_qty == -20.0
    # realized from closing the long 10: (60-50)*10 = 100
    assert runner.stats.realized_pnl == pytest.approx(100.0)
    assert runner.stats.shorts_opened == 1  # flipped into a short


def test_account_fill_shorts_opened_only_on_open_not_cover():
    runner = LiveRunner(ScriptedStrategy([]), "T", _router())
    runner._account_fill(Side.SELL, 10, 50.0)   # open short
    assert runner.stats.shorts_opened == 1
    runner._account_fill(Side.BUY, 10, 45.0)    # cover -> not a new short
    assert runner.stats.shorts_opened == 1


# ============================================================================
# Extreme magnitudes / numerical robustness on the pure math
# ============================================================================
@pytest.mark.parametrize("qty,price", [
    (1e9, 1e6), (1e-6, 1e-3),
])
def test_apply_fill_extreme_magnitudes_consistent(qty, price):
    eff = apply_fill(0.0, 0.0, "BUY", qty, price)
    assert eff.new_qty == pytest.approx(qty)
    assert eff.cash_delta == pytest.approx(-qty * price)
    assert math.isfinite(eff.new_avg)


def test_apply_fill_partial_reduce_keeps_avg():
    # long 100 @50, sell 40 @70 -> remaining 60 keeps avg 50, realized (70-50)*40
    eff = apply_fill(100.0, 50.0, "SELL", 40, 70.0)
    assert eff.new_qty == 60.0
    assert eff.new_avg == 50.0
    assert eff.realized == pytest.approx(800.0)
    assert eff.closed_qty == 40.0
    assert eff.opened_qty == 0.0
