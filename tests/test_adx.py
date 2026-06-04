"""
ADX trend-strength indicator + ADXTrendGate (and the donchian-adx / donchian-vol
regime-gated combos). The gate is the direct fix for false-breakout whipsaw: it lets
a breakout entry through only when the market is actually trending (high ADX).
"""

import math

import pytest

from alpca.strategies.base import BUY, EXIT, HOLD, SELL
from alpca.strategies.breakout import DonchianBreakout
from alpca.strategies.momentum import ADXTrendGate, compute_adx
from alpca.strategies.registry import available, make


def _hlc(closes, spread=0.3):
    h = [c + spread for c in closes]
    l = [c - spread for c in closes]
    return h, l, list(closes)


def _adx(closes, period=14):
    h, l, c = _hlc(closes)
    return compute_adx(h, l, c, period)


def _bar(c, i, spread=0.3):
    return {"open": c, "high": c + spread, "low": c - spread, "close": c,
            "volume": 10_000, "timestamp": i * 60}


def _bars(closes):
    return [_bar(c, i) for i, c in enumerate(closes)]


def _entries(strat, bars):
    return sum(1 for b in bars if strat.on_bar(b).is_actionable)


# ----------------------------------------------------------------- compute_adx
def test_adx_none_until_warmup():
    assert _adx([100 + i for i in range(10)], period=14) is None  # < 2*period+1


@pytest.mark.parametrize("n", [40, 60, 100])
def test_adx_high_on_strong_uptrend(n):
    adx = _adx([100.0 + i for i in range(n)], period=14)
    assert adx is not None
    assert adx > 40.0          # a clean linear trend -> very strong ADX


def test_adx_high_on_strong_downtrend():
    adx = _adx([300.0 - i for i in range(60)], period=14)
    assert adx is not None and adx > 40.0


def test_adx_low_on_fast_chop():
    # tight fast oscillation -> direction flips constantly -> weak trend
    closes = [100.0 + 2.0 * math.sin(i / 2.0) for i in range(120)]
    adx = _adx(closes, period=14)
    assert adx is not None and adx < 25.0


@pytest.mark.parametrize("n", [50, 80])
def test_adx_in_unit_range(n):
    adx = _adx([100.0 + 3.0 * math.sin(i / 5.0) for i in range(n)], period=14)
    assert adx is not None and 0.0 <= adx <= 100.0


# ------------------------------------------------------------- ADXTrendGate
def test_gate_passes_breakout_in_strong_trend():
    base = DonchianBreakout(period=20, atr_period=14)
    g = ADXTrendGate(base, period=14, threshold=25.0)
    sigs = [g.on_bar(b) for b in _bars([100.0 + i for i in range(80)])]
    assert any(s.side == BUY and s.is_actionable for s in sigs)  # trend -> entry allowed


def test_gate_blocks_breakouts_in_chop():
    # tiny drift makes marginal new highs (Donchian whipsaws) but the fast oscillation
    # keeps ADX in the low-20s — the exact ranging-day setup that bled today.
    closes = [100.0 + 0.02 * i + 3.0 * math.sin(i / 3.0) for i in range(300)]
    bars = _bars(closes)
    ungated = _entries(DonchianBreakout(period=20, atr_period=14), bars)
    gated = _entries(ADXTrendGate(DonchianBreakout(period=20, atr_period=14),
                                  period=14, threshold=25.0), bars)
    assert ungated > 0          # donchian DOES whipsaw on this chop
    assert gated < ungated      # the ADX gate cuts the false-breakout entries


def test_gate_never_blocks_exit():
    # enter in a trend, then a reversal must still EXIT even though ADX may dip
    closes = [100.0 + i for i in range(60)] + [160.0 - 2.0 * i for i in range(40)]
    g = ADXTrendGate(DonchianBreakout(period=20, atr_period=14), period=14, threshold=25.0)
    sigs = [g.on_bar(b) for b in _bars(closes)]
    assert EXIT in [s.side for s in sigs]


def test_gate_warmup_holds():
    g = ADXTrendGate(DonchianBreakout(period=20, atr_period=14), period=14, threshold=25.0)
    sigs = [g.on_bar(b) for b in _bars([100.0 + i for i in range(10)])]
    assert all(not s.is_actionable for s in sigs)


def test_gate_threshold_monotone_in_entries():
    # a HIGHER ADX threshold can only allow FEWER (or equal) entries
    closes = [100.0 + 5.0 * math.sin(i / 4.0) + 0.05 * i for i in range(300)]
    bars = _bars(closes)
    lax = _entries(ADXTrendGate(DonchianBreakout(period=20), period=14, threshold=15.0), bars)
    strict = _entries(ADXTrendGate(DonchianBreakout(period=20), period=14, threshold=40.0), bars)
    assert strict <= lax


# ------------------------------------------------------------------ registry
def test_registry_has_gated_combos():
    assert "donchian-adx" in available()
    assert "donchian-vol" in available()


def test_donchian_adx_is_gated():
    s = make("donchian-adx")
    assert isinstance(s, ADXTrendGate)
    assert isinstance(s.base, DonchianBreakout)
