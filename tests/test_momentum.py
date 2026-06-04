"""
Momentum / trend family: EMA cross, MACD, RSI momentum, ATR breakout, and the
composable VolRegimeGate. Deterministic crafted series.
"""

import pytest

from alpca.strategies.base import BUY, EXIT, HOLD, SELL
from alpca.strategies.momentum import (
    ATRBreakout,
    EMACrossMomentum,
    MACDTrend,
    RSIMomentum,
    VolRegimeGate,
    ema,
)
from alpca.strategies.registry import available, make


def _bar(c, h=None, l=None, i=0):
    return {"open": c, "high": h if h is not None else c + 0.5,
            "low": l if l is not None else c - 0.5, "close": c, "volume": 1000, "timestamp": i}


def _run(strat, closes, hl=None):
    out = []
    for i, c in enumerate(closes):
        if hl:
            h, l = hl[i]
            out.append(strat.on_bar(_bar(c, h, l, i)))
        else:
            out.append(strat.on_bar(_bar(c, i=i)))
    return out


def _sides(sigs):
    return [s.side for s in sigs]


# ----------------------------------------------------------------- ema() helper
def test_ema_seeds_with_first_value():
    assert ema(None, 10.0, 3) == 10.0


def test_ema_step_uses_alpha():
    # period 3 -> alpha = 2/4 = 0.5
    assert ema(10.0, 20.0, 3) == pytest.approx(15.0)


@pytest.mark.parametrize("period", [2, 5, 12, 26])
def test_ema_stays_between_prev_and_value(period):
    e = ema(100.0, 110.0, period)
    assert 100.0 <= e <= 110.0


# ----------------------------------------------------------- EMA cross momentum
def test_ema_requires_fast_lt_slow():
    with pytest.raises(ValueError):
        EMACrossMomentum(fast=10, slow=5)


def test_ema_goes_long_on_uptrend():
    s = EMACrossMomentum(fast=3, slow=6)
    sigs = _run(s, [100 + i for i in range(15)])
    assert BUY in _sides(sigs)
    assert s._side == "LONG"


def test_ema_exits_long_when_trend_reverses():
    s = EMACrossMomentum(fast=3, slow=6)
    sigs = _run(s, [100 + i for i in range(12)] + [112 - 3 * i for i in range(12)])
    sides = _sides(sigs)
    assert BUY in sides and EXIT in sides
    assert sides.index(EXIT) > sides.index(BUY)


def test_ema_shorts_on_downtrend_when_enabled():
    s = EMACrossMomentum(fast=3, slow=6, allow_short=True)
    sigs = _run(s, [200 - 2 * i for i in range(15)])
    assert SELL in _sides(sigs)
    assert s._side == "SHORT"


def test_ema_long_only_never_shorts():
    s = EMACrossMomentum(fast=3, slow=6, allow_short=False)
    sigs = _run(s, [200 - 2 * i for i in range(15)])
    assert SELL not in _sides(sigs)


def test_ema_warmup_holds():
    s = EMACrossMomentum(fast=3, slow=6)
    sigs = _run(s, [100 + i for i in range(5)])  # < slow
    assert set(_sides(sigs)) == {HOLD}


# --------------------------------------------------------------------- MACD
def test_macd_goes_long_on_uptrend():
    s = MACDTrend(fast=3, slow=6, signal=3)
    sigs = _run(s, [100 + 2 * i for i in range(25)])
    assert BUY in _sides(sigs)


def test_macd_requires_fast_lt_slow():
    with pytest.raises(ValueError):
        MACDTrend(fast=9, slow=9)


def test_macd_shorts_on_downtrend_when_enabled():
    s = MACDTrend(fast=3, slow=6, signal=3, allow_short=True)
    sigs = _run(s, [300 - 3 * i for i in range(25)])
    assert SELL in _sides(sigs)


# -------------------------------------------------------------- RSI momentum
def test_rsi_momentum_longs_on_strength():
    s = RSIMomentum(rsi_period=3, entry_high=60, exit_level=50)
    sigs = _run(s, [100 + i for i in range(10)])   # monotonic up -> RSI ~100
    assert BUY in _sides(sigs)


def test_rsi_momentum_exits_on_fade():
    s = RSIMomentum(rsi_period=3, entry_high=60, exit_level=50)
    sigs = _run(s, [100 + i for i in range(8)] + [108 - 2 * i for i in range(8)])
    sides = _sides(sigs)
    assert BUY in sides and EXIT in sides


def test_rsi_momentum_is_opposite_of_meanrev():
    # momentum BUYS strength (high RSI); confirm it does NOT buy on weakness
    s = RSIMomentum(rsi_period=3, entry_high=60, exit_level=50, allow_short=False)
    sigs = _run(s, [100 - i for i in range(10)])   # monotonic down -> RSI ~0
    assert BUY not in _sides(sigs)


def test_rsi_momentum_shorts_weakness_when_enabled():
    s = RSIMomentum(rsi_period=3, entry_high=60, exit_level=50, entry_low=40, allow_short=True)
    sigs = _run(s, [100 - i for i in range(10)])
    assert SELL in _sides(sigs)


# -------------------------------------------------------------- ATR breakout
def test_atr_breakout_triggers_on_clean_break():
    s = ATRBreakout(lookback=4, atr_period=4, atr_mult=0.5)
    # flat channel around 100, then a decisive break to 106
    closes = [100, 100, 100, 100, 100, 100, 106]
    hl = [(c + 0.5, c - 0.5) for c in closes[:-1]] + [(107.0, 105.0)]
    sigs = _run(s, closes, hl)
    assert BUY in _sides(sigs)


def test_atr_breakout_ignores_tiny_break():
    s = ATRBreakout(lookback=4, atr_period=4, atr_mult=2.0)  # needs a big cushion
    closes = [100, 100, 100, 100, 100, 100, 100.3]          # barely above channel
    hl = [(c + 0.5, c - 0.5) for c in closes]
    sigs = _run(s, closes, hl)
    assert BUY not in _sides(sigs)


def test_atr_breakout_shorts_down_when_enabled():
    s = ATRBreakout(lookback=4, atr_period=4, atr_mult=0.5, allow_short=True)
    closes = [100, 100, 100, 100, 100, 100, 94]
    hl = [(c + 0.5, c - 0.5) for c in closes[:-1]] + [(95.0, 93.0)]
    sigs = _run(s, closes, hl)
    assert SELL in _sides(sigs)


# ------------------------------------------------------------- VolRegimeGate
def test_vol_gate_passes_entries_in_wide_band():
    base = EMACrossMomentum(fast=3, slow=6)
    g = VolRegimeGate(base, lookback=5, vol_floor=0.0, vol_cap=float("inf"))
    sigs = _run(g, [100 + i for i in range(15)])
    assert BUY in _sides(sigs)


def test_vol_gate_blocks_entries_above_cap():
    base = EMACrossMomentum(fast=3, slow=6)
    g = VolRegimeGate(base, lookback=5, vol_floor=0.0, vol_cap=1e-9)  # any real vol blocked
    sigs = _run(g, [100 + i for i in range(15)])
    assert BUY not in _sides(sigs)


def test_vol_gate_blocks_entries_below_floor():
    base = EMACrossMomentum(fast=3, slow=6)
    g = VolRegimeGate(base, lookback=5, vol_floor=1e9, vol_cap=float("inf"))
    sigs = _run(g, [100 + i for i in range(15)])
    assert BUY not in _sides(sigs)


def test_vol_gate_never_blocks_exit():
    # take a long in a wide band, then a reversal must still EXIT even if we narrow
    base = EMACrossMomentum(fast=3, slow=6)
    g = VolRegimeGate(base, lookback=5, vol_floor=0.0, vol_cap=float("inf"))
    sigs = _run(g, [100 + i for i in range(12)] + [112 - 3 * i for i in range(12)])
    assert EXIT in _sides(sigs)


# ------------------------------------------------------------------ registry
@pytest.mark.parametrize("name", [
    "ema-momentum", "ema-momentum-ls", "macd", "macd-ls",
    "rsi-momentum", "rsi-momentum-ls", "atr-breakout", "atr-breakout-ls",
])
def test_registry_resolves_momentum(name):
    assert name in available()
    assert make(name) is not None


def test_ls_variants_allow_short():
    assert make("ema-momentum-ls").allow_short is True
    assert make("rsi-momentum-ls").allow_short is True
    assert make("atr-breakout-ls").allow_short is True
    assert make("macd-ls").allow_short is True
