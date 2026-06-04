"""
EnsembleVote — trade only when >= min_agree of the voter brains agree on direction.
Combining EMA cross + MACD + Donchian filters the false signals any one fires.
"""

import math

import pytest

from alpca.strategies.base import BUY, EXIT, HOLD, SELL
from alpca.strategies.momentum import EMACrossMomentum, EnsembleVote, MACDTrend
from alpca.strategies.registry import available, make


def _bar(c, i, spread=0.3):
    return {"open": c, "high": c + spread, "low": c - spread, "close": c,
            "volume": 10_000, "timestamp": i * 60}


def _bars(closes):
    return [_bar(c, i) for i, c in enumerate(closes)]


def _sigs(strat, closes):
    return [strat.on_bar(b) for b in _bars(closes)]


def _entries(strat, closes):
    return sum(1 for b in _bars(closes) if strat.on_bar(b).is_actionable)


def test_ensemble_enters_long_on_strong_uptrend():
    # all three voters turn bullish on a clean trend -> consensus -> long
    sigs = _sigs(make("ensemble"), [100.0 + i for i in range(120)])
    assert any(s.side == BUY and s.is_actionable for s in sigs)


def test_unanimous_ensemble_is_most_selective_in_chop():
    # requiring ALL voters to agree (min_agree=3) is far more selective than any single
    # vote (min_agree=1) on a choppy tape where the voters frequently disagree.
    closes = [100.0 + 0.02 * i + 3.0 * math.sin(i / 3.0) for i in range(300)]
    loose = _entries(EnsembleVote(min_agree=1), closes)
    unanimous = _entries(EnsembleVote(min_agree=3), closes)
    assert unanimous < loose


def test_higher_min_agree_never_increases_entries():
    closes = [100.0 + 4.0 * math.sin(i / 9.0) + 0.03 * i for i in range(300)]
    loose = _entries(EnsembleVote(min_agree=1), closes)
    tight = _entries(EnsembleVote(min_agree=3), closes)
    assert tight <= loose


def test_ensemble_exits_when_consensus_breaks():
    closes = [100.0 + i for i in range(80)] + [180.0 - 3.0 * i for i in range(40)]
    sigs = _sigs(make("ensemble"), closes)
    sides = [s.side for s in sigs]
    assert BUY in sides and EXIT in sides
    assert sides.index(EXIT) > sides.index(BUY)


def test_ensemble_short_requires_consensus_and_flag():
    closes = [200.0 - i for i in range(120)]
    long_only = _sigs(make("ensemble"), [dict_c for dict_c in closes])
    assert all(s.side != SELL for s in long_only)          # no short leg by default
    ls = _sigs(make("ensemble-ls"), closes)
    assert any(s.side == SELL for s in ls)                 # downtrend consensus -> short


def test_custom_voters_respected():
    e = EnsembleVote(voters=[EMACrossMomentum(3, 6), MACDTrend(3, 6, 3)], min_agree=2)
    assert len(e.voters) == 2
    sigs = _sigs(e, [100.0 + i for i in range(60)])
    assert any(s.side == BUY for s in sigs)


def test_registry_has_ensemble():
    assert "ensemble" in available() and "ensemble-ls" in available()
    assert isinstance(make("ensemble"), EnsembleVote)


def test_reset_clears_voters():
    e = make("ensemble")
    _sigs(e, [100.0 + i for i in range(60)])
    e.reset()
    assert e._side == ""
    assert all(not getattr(v, "_in_position", False) for v in e.voters)
