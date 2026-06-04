"""Strategy behavior tests on synthetic series engineered to trigger entry/exit."""

from alpca.strategies.base import BUY, EXIT, HOLD
from alpca.strategies.breakout import ORB, DonchianBreakout
from alpca.strategies.mean_reversion import ZScoreMeanReversion
from alpca.strategies.registry import available, make


def _bar(c, h=None, l=None, v=1000):
    return {"open": c, "high": c if h is None else h, "low": c if l is None else l,
            "close": c, "volume": v, "timestamp": 0}


def test_registry_roundtrip():
    # the three core strategies must always be present (more may be registered)
    assert {"donchian", "orb", "zscore"}.issubset(set(available()))
    s = make("orb", range_bars=3)
    assert isinstance(s, ORB)
    assert s.range_bars == 3


def test_donchian_enters_on_breakout_and_exits():
    s = DonchianBreakout(period=5, atr_period=3, stop_atr_mult=2.0)
    for _ in range(7):
        s.on_bar(_bar(100.0, h=100.5, l=99.5))
    enter = s.on_bar(_bar(110.0, h=110.0, l=108.0))
    assert enter.side == BUY
    assert enter.price == 110.0
    assert s._in_position
    ex = s.on_bar(_bar(90.0, h=95.0, l=90.0))
    assert ex.side == EXIT
    assert not s._in_position


def test_orb_breakout_then_target_exit():
    s = ORB(range_bars=3, stop_pct=0.02, take_profit_pct=0.04)
    for c, h, l in [(100, 101, 99), (100, 100.5, 99.5), (100, 100.8, 99.2)]:
        assert s.on_bar(_bar(c, h, l)).side == HOLD
    enter = s.on_bar(_bar(102.0, 102.0, 101.0))
    assert enter.side == BUY
    entry = s._entry_price
    ex = s.on_bar(_bar(entry * 1.05, entry * 1.05, entry))
    assert ex.side == EXIT
    assert "target" in ex.reason


def test_orb_stop_exit():
    s = ORB(range_bars=2, stop_pct=0.02, take_profit_pct=0.10)
    for c, h, l in [(100, 101, 99), (100, 100.5, 99.5)]:
        s.on_bar(_bar(c, h, l))
    enter = s.on_bar(_bar(102.0, 102.0, 101.0))
    assert enter.side == BUY
    entry = s._entry_price
    ex = s.on_bar(_bar(entry * 0.97, entry * 0.99, entry * 0.97))
    assert ex.side == EXIT
    assert "stop" in ex.reason


def test_zscore_enters_oversold_and_reverts():
    s = ZScoreMeanReversion(lookback=20, entry_z=2.0, exit_z=0.5, stop_z=3.5)
    for i in range(20):
        s.on_bar(_bar(100.0 + (0.1 if i % 2 else -0.1)))
    enter = s.on_bar(_bar(98.5))
    assert enter.side == BUY
    assert enter.metadata["z"] < -2.0
    ex = s.on_bar(_bar(100.0))
    assert ex.side == EXIT


def test_no_signal_during_warmup():
    s = DonchianBreakout(period=10)
    for _ in range(3):
        assert s.on_bar(_bar(100.0)).side == HOLD
