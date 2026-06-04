"""
Regression guard for the keystone realism property: NO LOOK-AHEAD.

A signal derived from bar i is only known at bar i's close, so it must execute at
bar i+1's OPEN — never at bar i's own close. These tests construct bars where the
signal-bar close and the next-bar open are deliberately far apart, so a
look-ahead regression would be unmistakable.
"""

from alpca.backtest.engine import run_backtest
from alpca.strategies.breakout import DonchianBreakout


def _bar(o, h, l, c, ts=0):
    return {"open": o, "high": h, "low": l, "close": c, "volume": 1000,
            "timestamp": ts, "symbol": "T"}


def test_entry_fills_at_next_bar_open_not_signal_close():
    # bars 0-5 flat at 100 -> establishes the Donchian channel (prior high ~100.5)
    bars = [_bar(100, 100.5, 99.5, 100, ts=i) for i in range(6)]
    # bar 6: the breakout — close 110 > prior high. Signal is generated HERE.
    bars.append(_bar(105, 110, 104, 110, ts=6))
    # bar 7: a big gap up — open 120. A no-look-ahead fill must land at ~120,
    # NOT at the signal bar's close of 110.
    bars.append(_bar(120, 122, 119, 121, ts=7))
    # a couple more so the position can be marked/liquidated
    bars.append(_bar(121, 123, 120, 122, ts=8))

    res = run_backtest(DonchianBreakout(period=5, atr_period=3),
                       bars, slippage_bps=0.0, commission_bps=0.0)

    assert res.n_trades >= 1
    entry = res.trades[0]
    # With zero slippage, the entry reference AND fill must equal bar 7's open.
    assert abs(entry.entry_ref - 120.0) < 1e-9, entry.entry_ref
    assert abs(entry.entry_price - 120.0) < 1e-9, entry.entry_price
    # And must NOT be the signal bar's close (110) — that would be look-ahead.
    assert abs(entry.entry_price - 110.0) > 1.0


def test_signal_on_final_bar_never_executes():
    # Channel, then a breakout on the VERY LAST bar — there is no next bar to
    # fill against, so no trade may be opened (and certainly none look-ahead).
    bars = [_bar(100, 100.5, 99.5, 100, ts=i) for i in range(6)]
    bars.append(_bar(105, 111, 104, 111, ts=6))  # breakout on the last bar
    res = run_backtest(DonchianBreakout(period=5, atr_period=3),
                       bars, slippage_bps=0.0, commission_bps=0.0)
    # no executable next bar -> no closed trades, flat equity
    assert res.n_trades == 0
    assert abs(res.total_return) < 1e-9


def test_exit_also_fills_at_next_open():
    bars = [_bar(100, 100.5, 99.5, 100, ts=i) for i in range(6)]
    bars.append(_bar(105, 110, 104, 110, ts=6))   # breakout signal
    bars.append(_bar(120, 122, 119, 121, ts=7))   # entry fills here at open 120
    # now force a Donchian exit: close below the prior 5-bar low
    bars.append(_bar(118, 119, 90, 95, ts=8))     # exit signal generated here
    bars.append(_bar(92, 93, 88, 90, ts=9))       # exit fills here at open 92
    res = run_backtest(DonchianBreakout(period=5, atr_period=3),
                       bars, slippage_bps=0.0, commission_bps=0.0)
    assert res.n_trades == 1
    t = res.trades[0]
    assert abs(t.entry_ref - 120.0) < 1e-9
    # exit must fill at bar 9's open (92), not bar 8's close (95)
    assert abs(t.exit_ref - 92.0) < 1e-9, t.exit_ref
