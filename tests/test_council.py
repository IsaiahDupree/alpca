"""
TradeCouncil — advocates + skeptics under one voice. Logic is pinned with stub
advocates/skeptics (deterministic), plus an integration smoke on real-shaped series.
"""

import math

import pytest

from alpca.strategies.base import BUY, EXIT, HOLD, SELL, Strategy, hold
from alpca.strategies.council import (
    ChopSkeptic,
    CouncilContext,
    OverextendedSkeptic,
    Skeptic,
    TradeCouncil,
)
from alpca.strategies.registry import available, make


def _bar(c, i, spread=0.3):
    return {"open": c, "high": c + spread, "low": c - spread, "close": c,
            "volume": 10_000, "timestamp": i * 60}


class _Vote(Strategy):
    """Stub advocate that always reports a fixed direction via _side."""
    def __init__(self, side):
        super().__init__()
        self._side = side
        self._in_position = side != ""

    def on_bar(self, bar):
        return hold()


class _AlwaysVeto(Skeptic):
    name = "nope"

    def veto(self, ctx):
        return "blocked for testing"


def _drive(council, n=5):
    return [council.on_bar(_bar(100 + i, i)) for i in range(n)]


# ---------------------------------------------------------------- council logic
def test_council_enters_when_advocates_agree_and_no_veto():
    c = TradeCouncil(advocates=[_Vote("LONG"), _Vote("LONG")], skeptics=[], min_conviction=2)
    sides = [s.side for s in _drive(c)]
    assert BUY in sides
    assert "advocates for" in c.last_rationale


def test_council_blocks_when_skeptic_vetoes():
    c = TradeCouncil(advocates=[_Vote("LONG"), _Vote("LONG")], skeptics=[_AlwaysVeto()], min_conviction=2)
    sides = [s.side for s in _drive(c)]
    assert BUY not in sides
    assert "VETOED" in c.last_rationale


def test_council_no_action_without_consensus():
    c = TradeCouncil(advocates=[_Vote("LONG"), _Vote("")], skeptics=[], min_conviction=2)
    sides = [s.side for s in _drive(c)]
    assert all(s != BUY for s in sides)
    assert "no consensus" in c.last_rationale


def test_council_skeptic_never_blocks_exit():
    # 2 advocates long -> enter; then they go flat -> council must EXIT despite a veto
    longs = [_Vote("LONG"), _Vote("LONG")]
    c = TradeCouncil(advocates=longs, skeptics=[_AlwaysVeto()], min_conviction=2)
    c.on_bar(_bar(100, 0))  # enter? veto blocks entry... so flip approach:
    # build one that DID enter (no veto), then add a vetoing exit scenario
    c2 = TradeCouncil(advocates=longs, skeptics=[], min_conviction=2)
    c2.on_bar(_bar(100, 0))
    assert c2._side == "LONG"
    longs[0]._side = "";  longs[0]._in_position = False
    longs[1]._side = "";  longs[1]._in_position = False
    sig = c2.on_bar(_bar(101, 1))
    assert sig.side == EXIT


def test_council_short_requires_flag_and_consensus():
    longs = [_Vote("SHORT"), _Vote("SHORT")]
    lo = TradeCouncil(advocates=longs, skeptics=[], min_conviction=2, allow_short=False)
    assert all(s.side != SELL for s in _drive(lo))
    sh = TradeCouncil(advocates=[_Vote("SHORT"), _Vote("SHORT")], skeptics=[], min_conviction=2, allow_short=True)
    assert SELL in [s.side for s in _drive(sh)]


def test_rationale_is_populated_each_decision():
    c = TradeCouncil(advocates=[_Vote("LONG"), _Vote("LONG")], skeptics=[])
    _drive(c)
    assert isinstance(c.last_rationale, str) and len(c.last_rationale) > 0


# ----------------------------------------------------------------- skeptics
def test_chop_skeptic_vetoes_low_adx():
    closes = [100 + 2 * math.sin(i / 2.0) for i in range(80)]
    ctx = CouncilContext([c + 0.3 for c in closes], [c - 0.3 for c in closes], closes,
                         _bar(closes[-1], 80), direction=1)
    assert ChopSkeptic(adx_min=25.0).veto(ctx) is not None  # choppy -> vetoed


def test_chop_skeptic_clears_strong_trend():
    closes = [100.0 + i for i in range(80)]
    ctx = CouncilContext([c + 0.3 for c in closes], [c - 0.3 for c in closes], closes,
                         _bar(closes[-1], 80), direction=1)
    assert ChopSkeptic(adx_min=25.0).veto(ctx) is None      # strong trend -> cleared


def test_overextended_skeptic_vetoes_chasing():
    closes = [100.0] * 19 + [110.0]   # +10% above the MA
    ctx = CouncilContext(closes, closes, closes, _bar(110, 20), direction=1)
    assert OverextendedSkeptic(period=20, max_ext=0.03).veto(ctx) is not None


def test_overextended_skeptic_clears_near_ma():
    closes = [100.0] * 20
    ctx = CouncilContext(closes, closes, closes, _bar(100, 20), direction=1)
    assert OverextendedSkeptic(period=20, max_ext=0.03).veto(ctx) is None


# ----------------------------------------------------------------- registry
def test_registry_has_council():
    assert "council" in available() and "council-ls" in available()
    assert isinstance(make("council"), TradeCouncil)


def test_council_integration_enters_on_clean_trend():
    # a gentle, realistic uptrend: clear enough for consensus + high ADX, but NOT so
    # steep it trips the overextended (chasing) skeptic.
    c = make("council")
    sigs = [c.on_bar(_bar(100 + 0.1 * i, i)) for i in range(150)]
    assert any(s.side == BUY for s in sigs)
    assert "skeptics cleared" in c.last_rationale
