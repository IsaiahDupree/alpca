"""
GapFade (#1) — opening-gap fade. Sessions are detected from REAL epoch-second
timestamps via the NYSE calendar, so these tests build true ET datetimes across
two trading days (2026-06-02 Tue, 2026-06-03 Wed — both regular sessions).
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from alpca.backtest.runner_backtest import backtest_resting
from alpca.strategies.base import BUY, EXIT, HOLD, SELL
from alpca.strategies.event_driven import GapFade
from alpca.strategies.registry import available, make

ET = ZoneInfo("America/New_York")


def _ts(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=ET).timestamp()


def _bar(close, open_=None, ts=0.0, sym="SPY"):
    o = close if open_ is None else open_
    return {"open": o, "high": max(o, close), "low": min(o, close),
            "close": close, "volume": 1e6, "timestamp": ts, "symbol": sym}


def _day1(close=100.0):
    """Three day-1 bars; last close is the reference for the day-2 gap."""
    return [
        _bar(close, ts=_ts(2026, 6, 2, 9, 30)),
        _bar(close, ts=_ts(2026, 6, 2, 9, 31)),
        _bar(close, ts=_ts(2026, 6, 2, 9, 32)),
    ]


def test_gap_down_fades_long_then_reverts():
    s = GapFade(entry_pct=0.01, exit_pct=0.002, stop_pct=0.05, hold_bars=30)
    for b in _day1(100.0):
        assert s.on_bar(b).side == HOLD          # warmup / same session
    # day-2 opens 2% BELOW prior close -> fade long
    enter = s.on_bar(_bar(98.0, open_=98.0, ts=_ts(2026, 6, 3, 9, 30)))
    assert enter.side == BUY
    assert enter.metadata["gap"] < -0.01
    assert s._in_position and s._side == "LONG"
    assert s.on_bar(_bar(99.0, ts=_ts(2026, 6, 3, 9, 31))).side == HOLD
    # reverted to within exit_pct of the 100.0 reference -> take profit
    ex = s.on_bar(_bar(99.95, ts=_ts(2026, 6, 3, 9, 32)))
    assert ex.side == EXIT and "revert" in ex.reason
    assert not s._in_position


def test_gap_up_fades_short_when_allow_short():
    s = GapFade(entry_pct=0.01, exit_pct=0.002, stop_pct=0.05, allow_short=True)
    for b in _day1(100.0):
        s.on_bar(b)
    enter = s.on_bar(_bar(102.0, open_=102.0, ts=_ts(2026, 6, 3, 9, 30)))
    assert enter.side == SELL                     # short the gap up
    assert enter.metadata["gap"] > 0.01
    assert s._side == "SHORT"
    ex = s.on_bar(_bar(100.05, ts=_ts(2026, 6, 3, 9, 31)))
    assert ex.side == EXIT and "revert" in ex.reason


def test_long_only_does_not_short_a_gap_up():
    s = GapFade(entry_pct=0.01, allow_short=False)   # default
    for b in _day1(100.0):
        s.on_bar(b)
    sig = s.on_bar(_bar(102.0, open_=102.0, ts=_ts(2026, 6, 3, 9, 30)))
    assert sig.side == HOLD                        # gap up + no shorting -> no trade
    assert not s._in_position


def test_gap_long_stop_out():
    s = GapFade(entry_pct=0.01, exit_pct=0.002, stop_pct=0.01, hold_bars=30)
    for b in _day1(100.0):
        s.on_bar(b)
    s.on_bar(_bar(98.0, open_=98.0, ts=_ts(2026, 6, 3, 9, 30)))   # long @ 98
    # falls another >1% from entry -> stop (98 * 0.99 = 97.02)
    ex = s.on_bar(_bar(96.5, ts=_ts(2026, 6, 3, 9, 31)))
    assert ex.side == EXIT and "stop" in ex.reason


def test_gap_long_time_stop():
    s = GapFade(entry_pct=0.01, exit_pct=0.002, stop_pct=0.20, hold_bars=2)
    for b in _day1(100.0):
        s.on_bar(b)
    s.on_bar(_bar(98.0, open_=98.0, ts=_ts(2026, 6, 3, 9, 30)))   # long @ 98
    assert s.on_bar(_bar(98.4, ts=_ts(2026, 6, 3, 9, 31))).side == HOLD   # held=1
    ex = s.on_bar(_bar(98.4, ts=_ts(2026, 6, 3, 9, 32)))                  # held=2 -> time
    assert ex.side == EXIT and "time" in ex.reason


def test_no_gap_no_trade():
    s = GapFade(entry_pct=0.01)
    for b in _day1(100.0):
        s.on_bar(b)
    sig = s.on_bar(_bar(100.2, open_=100.2, ts=_ts(2026, 6, 3, 9, 30)))  # 0.2% < 1%
    assert sig.side == HOLD and not s._in_position


def test_single_session_never_trades():
    s = GapFade(entry_pct=0.001)
    for b in _day1(100.0):                          # only one session present
        assert s.on_bar(b).side == HOLD


def test_overnight_position_flattened_at_next_open():
    # enter long on a day-2 gap-down that never reverts within the session, then a
    # day-3 bar arrives -> the carried position is flattened at the rollover.
    s = GapFade(entry_pct=0.01, exit_pct=0.0001, stop_pct=0.50, hold_bars=999)
    for b in _day1(100.0):
        s.on_bar(b)
    assert s.on_bar(_bar(98.0, open_=98.0, ts=_ts(2026, 6, 3, 9, 30))).side == BUY
    assert s._in_position
    flat = s.on_bar(_bar(98.5, open_=98.5, ts=_ts(2026, 6, 4, 9, 30)))  # next session
    assert flat.side == EXIT and "rollover" in flat.reason
    assert not s._in_position


def test_registry_has_gap_fade_variants():
    assert {"gap-fade", "gap-fade-ls"}.issubset(set(available()))
    assert isinstance(make("gap-fade"), GapFade)
    assert make("gap-fade").allow_short is False
    assert make("gap-fade-ls").allow_short is True


def test_gap_fade_trades_through_backtest_resting():
    bars = _day1(100.0)
    # day-2 gap down then revert toward 100 -> one long round trip
    bars += [
        _bar(98.0, open_=98.0, ts=_ts(2026, 6, 3, 9, 30)),
        _bar(99.0, ts=_ts(2026, 6, 3, 9, 31)),
        _bar(99.95, ts=_ts(2026, 6, 3, 9, 32)),
        _bar(100.0, ts=_ts(2026, 6, 3, 9, 33)),
    ]
    res = backtest_resting(GapFade(entry_pct=0.01, exit_pct=0.002, hold_bars=30), bars)
    assert res.n_trades >= 1
    assert res.ending_equity > 0
