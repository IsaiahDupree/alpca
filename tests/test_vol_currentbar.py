"""
Regression: the fill model must use the EXECUTION bar's own volume (the bar the
fill happens on), not the previous bar's. A prior bug defined `vol` after the
pending-execution block, so impact was computed from the lagged volume.

Construct two bars sequences identical except for the execution bar's volume; the
square-root impact term must differ accordingly.
"""

from alpca.backtest.engine import run_backtest
from alpca.execution.fills import FillModel
from alpca.strategies.breakout import DonchianBreakout


def _bar(o, h, l, c, ts, vol):
    return {"open": o, "high": h, "low": l, "close": c, "volume": vol,
            "timestamp": ts, "symbol": "T"}


def _seq(exec_bar_volume):
    # 6 flat channel bars, breakout signal on bar 6, entry fills on bar 7's open.
    # Bar 7 (the EXECUTION bar) carries `exec_bar_volume`; the signal bar (6) has
    # a deliberately DIFFERENT volume so a lagged-volume bug would be visible.
    bars = [_bar(100, 100.5, 99.5, 100, i, vol=1e9) for i in range(6)]
    bars.append(_bar(105, 112, 104, 112, 6, vol=42.0))          # signal bar
    bars.append(_bar(120, 122, 119, 121, 7, vol=exec_bar_volume))  # execution bar
    bars.append(_bar(121, 123, 120, 122, 8, vol=1e9))
    return bars


def test_impact_uses_execution_bar_volume():
    fm = FillModel(half_spread_bps=1.0, impact_coef_bps=20.0,
                   participation_cap=1.0, min_tick=0.0)
    # scarce liquidity on the execution bar -> high participation -> big impact
    scarce = run_backtest(DonchianBreakout(period=5, atr_period=3), _seq(1_000.0),
                          fill_model=fm)
    # deep liquidity on the execution bar -> tiny participation -> small impact
    deep = run_backtest(DonchianBreakout(period=5, atr_period=3), _seq(1e9),
                        fill_model=fm)

    se = scarce.trades[0]
    de = deep.trades[0]
    # both enter at the same open ref (120) ...
    assert abs(se.entry_ref - 120.0) < 1e-9
    assert abs(de.entry_ref - 120.0) < 1e-9
    # ... but scarce-liquidity execution pays a strictly higher fill price.
    assert se.entry_price > de.entry_price
    # and if the lagged-volume bug returned, both would use the SAME (signal-bar
    # or prior-bar) volume and the prices would match — guard against that.
    assert (se.entry_price - de.entry_price) > 1e-4
