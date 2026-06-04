"""Behavior tests for the added trend strategies (Keltner, Supertrend)."""

from alpca.strategies.base import BUY, EXIT, HOLD
from alpca.strategies.breakout import Supertrend, VolatilityBreakout
from alpca.strategies.registry import available, make


def _bar(c, h=None, l=None):
    return {"open": c, "high": c if h is None else h, "low": c if l is None else l,
            "close": c, "volume": 1000, "timestamp": 0}


def test_registry_has_core_strategies():
    # core set must be present; more may be registered (e.g. zscore-ls long/short)
    assert {"donchian", "orb", "keltner", "supertrend", "zscore"}.issubset(set(available()))
    assert isinstance(make("keltner", multiplier=2.0), VolatilityBreakout)
    assert isinstance(make("supertrend", atr_period=7), Supertrend)
    # the long/short variant is registered and shorting-enabled
    from alpca.strategies.mean_reversion import ZScoreMeanReversion
    ls = make("zscore-ls")
    assert isinstance(ls, ZScoreMeanReversion) and ls.allow_short is True


def test_keltner_enters_on_upper_break_and_exits_on_reenter():
    s = VolatilityBreakout(ema_period=10, atr_period=5, multiplier=1.5, stop_pct=0.05)
    # flat warmup with a little range so ATR > 0
    for _ in range(12):
        s.on_bar(_bar(100.0, h=100.6, l=99.4))
    # strong push above the upper band
    enter = None
    for px in (104, 106, 108):
        sig = s.on_bar(_bar(float(px), h=px + 1, l=px - 1))
        if sig.side == BUY:
            enter = sig
            break
    assert enter is not None and enter.side == BUY
    assert s._in_position
    # collapse back inside the band -> exit
    ex = s.on_bar(_bar(99.0, h=99.5, l=98.5))
    assert ex.side == EXIT
    assert not s._in_position


def test_supertrend_long_in_uptrend_then_flat_in_downtrend():
    s = Supertrend(atr_period=5, multiplier=2.0)
    # establish an uptrend
    entered = False
    px = 100.0
    for _ in range(20):
        px *= 1.01
        sig = s.on_bar(_bar(px, h=px * 1.005, l=px * 0.997))
        if sig.side == BUY:
            entered = True
    assert entered, "supertrend should go long during a sustained uptrend"
    assert s._in_position
    # sharp sustained downtrend should flip direction and EXIT
    exited = False
    for _ in range(20):
        px *= 0.97
        sig = s.on_bar(_bar(px, h=px * 1.003, l=px * 0.99))
        if sig.side == EXIT:
            exited = True
    assert exited, "supertrend should flatten when trend flips down"
    assert not s._in_position


def test_warmup_holds():
    s = Supertrend(atr_period=10)
    for _ in range(3):
        assert s.on_bar(_bar(100.0)).side == HOLD
