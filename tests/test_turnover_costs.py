"""
Characterization tests for the turnover / transaction-cost dynamics that drive the
"momentum brains overtrade on fine bars and bleed to costs; the vol-gate throttles
them" conclusions. These pin the *direction* of those effects so a future change
that, say, silently makes a strategy trade 10x more is caught.

Deterministic synthetic series (math.sin), no network, no randomness.
"""

import math

from alpca.backtest.engine import run_backtest
from alpca.execution.fills import FillModel
from alpca.strategies.base import BUY, EXIT, SELL
from alpca.strategies.momentum import EMACrossMomentum, VolRegimeGate
from alpca.strategies.registry import make


def _bar(c, i):
    return {"open": c, "high": c + 0.3, "low": c - 0.3, "close": c,
            "volume": 10_000, "timestamp": i * 60}


def _choppy(n, amp=8.0):
    # mean-reverting chop: lots of EMA crossings, very few sustained breakouts
    return [100.0 + amp * math.sin(i / 7.0) + 3.0 * math.sin(i / 3.0) for i in range(n)]


def _bars_from(closes):
    return [_bar(c, i) for i, c in enumerate(closes)]


def _actions(strat, closes):
    """Count actionable signals (entries + exits) a strategy emits over a series."""
    return sum(1 for i, c in enumerate(closes) if strat.on_bar(_bar(c, i)).is_actionable)


# ===================================================== turnover scales with cadence
def test_finer_bars_produce_more_trades():
    # the SAME price path sampled finely vs coarsely: finer cadence = more crossings
    # = more trades. This is why 1-min momentum overtrades vs a higher timeframe.
    closes = _choppy(600)
    fine = _actions(EMACrossMomentum(12, 26), closes)
    coarse = _actions(EMACrossMomentum(12, 26), closes[::12])  # ~12-min bars, same window
    assert fine > coarse
    assert coarse >= 0


def test_faster_ema_trades_more_than_slower():
    # controlled turnover: faster EMAs cross on smaller wiggles, so they trade more
    # than slower EMAs on the SAME tape (the parameter knob behind "overtrading").
    closes = _choppy(500)
    fast = _actions(EMACrossMomentum(3, 6), closes)
    slow = _actions(EMACrossMomentum(12, 40), closes)
    assert fast > slow


# ===================================================== cost sensitivity
def test_higher_slippage_lowers_return_same_trades():
    bars = _bars_from(_choppy(400))
    lo = run_backtest(make("ema-momentum"), bars, fill_model=FillModel.flat(1.0), commission_bps=0.0)
    hi = run_backtest(make("ema-momentum"), bars, fill_model=FillModel.flat(60.0), commission_bps=0.0)
    assert hi.total_return < lo.total_return     # more cost -> worse net
    assert lo.n_trades == hi.n_trades            # price-based signals: cost doesn't move them


def test_return_degrades_monotonically_with_cost():
    bars = _bars_from(_choppy(400))
    rets = [run_backtest(make("ema-momentum"), bars,
                         fill_model=FillModel.flat(bps), commission_bps=0.0).total_return
            for bps in (1.0, 20.0, 60.0)]
    assert rets[0] > rets[1] > rets[2]


def test_higher_turnover_bleeds_more_to_costs():
    # the headline claim, controlled: a cost increase hurts the HIGHER-turnover
    # parameterization (fast EMAs) more than the lower-turnover one (slow EMAs).
    bars = _bars_from(_choppy(500))

    def ret(fast, slow, bps):
        return run_backtest(EMACrossMomentum(fast, slow), bars,
                            fill_model=FillModel.flat(bps), commission_bps=0.0).total_return

    fast_drop = ret(3, 6, 1.0) - ret(3, 6, 60.0)
    slow_drop = ret(12, 40, 1.0) - ret(12, 40, 60.0)
    assert fast_drop > slow_drop


# ===================================================== vol-regime gate throttles
def test_vol_gate_only_throttles_never_adds():
    closes = _choppy(400)
    ungated = _actions(EMACrossMomentum(3, 6), closes)
    wide = _actions(VolRegimeGate(EMACrossMomentum(3, 6), lookback=10, vol_cap=float("inf")), closes)
    assert wide <= ungated   # a gate can only block entries, never create them


def test_vol_gate_blocks_entries_outside_band():
    closes = _choppy(400)
    ungated = _actions(EMACrossMomentum(3, 6), closes)
    # cap below any real vol -> every entry is out-of-band -> blocked (and with no
    # entry there is nothing to exit), so far fewer actions than ungated.
    blocked = _actions(VolRegimeGate(EMACrossMomentum(3, 6), lookback=10, vol_cap=1e-9), closes)
    assert blocked < ungated


def test_vol_gate_passthrough_matches_when_band_wide_after_warmup():
    # once both warmups are past, a wide band must not suppress a clean trend entry
    closes = [100.0 + i for i in range(40)]          # steady uptrend
    ungated = _actions(EMACrossMomentum(3, 6), closes)
    wide = _actions(VolRegimeGate(EMACrossMomentum(3, 6), lookback=5, vol_cap=float("inf")), closes)
    assert ungated >= 1 and wide >= 1
