"""
Roadmap #6/#7/#10: VWAPReclaim, BollingerExpansion, SessionMomentum.
Deterministic crafted series; VWAP/session use real epoch ts so the session-reset hook
can be exercised.
"""

import math

import pytest

from alpca.strategies.base import BUY, EXIT, HOLD, SELL
from alpca.strategies.intraday import SessionMomentum, VWAPReclaim
from alpca.strategies.momentum import BollingerExpansion
from alpca.strategies.registry import available, make

DAY = 86400
T0 = 1_700_000_000


def _bar(c, i, vol=1000.0, spread=0.2, ts=None):
    return {"open": c, "high": c + spread, "low": c - spread, "close": c,
            "volume": vol, "timestamp": T0 + i * 60 if ts is None else ts}


def _sigs(strat, closes, vols=None):
    out = []
    for i, c in enumerate(closes):
        v = vols[i] if vols else 1000.0
        out.append(strat.on_bar(_bar(c, i, vol=v)))
    return out


def _sides(sigs):
    return [s.side for s in sigs]


# ===================================================================== VWAP
def test_vwap_longs_on_reclaim_from_below():
    # dip below the running VWAP, then rally back up through it -> reclaim -> long
    closes = [100, 99, 98, 97, 98, 99, 100, 101, 102, 103]
    sigs = _sigs(VWAPReclaim(), closes)
    assert BUY in _sides(sigs)


def test_vwap_exits_when_it_loses_vwap():
    closes = [100, 99, 98, 99, 100, 101, 102, 101, 99, 97, 95]
    sigs = _sigs(VWAPReclaim(), closes)
    sides = _sides(sigs)
    assert BUY in sides and EXIT in sides


def test_vwap_no_volume_holds():
    sigs = _sigs(VWAPReclaim(), [100, 101, 102, 103], vols=[0, 0, 0, 0])
    assert set(_sides(sigs)) == {HOLD}


def test_vwap_short_on_reject_when_enabled():
    # start above VWAP then break down through it -> reject -> short
    closes = [100, 101, 102, 103, 102, 101, 100, 99, 98, 97, 96]
    sigs = _sigs(VWAPReclaim(allow_short=True), closes)
    assert SELL in _sides(sigs)


def test_vwap_resets_each_session():
    v = VWAPReclaim()
    # day 1 builds a VWAP; session start must wipe the cumulative sums
    for i, c in enumerate([100, 101, 102, 103]):
        v.on_bar(_bar(c, i))
    assert v._cum_v > 0
    v.on_session_start()
    assert v._cum_v == 0.0 and v._prev_above is None


# =========================================================== SessionMomentum
def test_session_momentum_longs_on_early_up_move():
    s = SessionMomentum(entry_pct=0.005, min_bars=3)
    sigs = _sigs(s, [100 + i for i in range(15)])   # steadily up from the open
    assert BUY in _sides(sigs)


def test_session_momentum_holds_when_flat():
    s = SessionMomentum(entry_pct=0.02, min_bars=3)   # needs +2% to trigger
    sigs = _sigs(s, [100 + 0.1 * math.sin(i) for i in range(20)])  # barely moves
    assert BUY not in _sides(sigs)


def test_session_momentum_shorts_on_early_down_move():
    s = SessionMomentum(entry_pct=0.005, min_bars=3, allow_short=True)
    sigs = _sigs(s, [100 - i for i in range(15)])
    assert SELL in _sides(sigs)


def test_session_momentum_resets_open_each_session():
    s = SessionMomentum()
    for i, c in enumerate([100, 101, 102]):
        s.on_bar(_bar(c, i))
    assert s._open == 100
    s.on_session_start()
    assert s._open is None and s._bars == 0


# ========================================================= BollingerExpansion
def test_bollinger_expansion_breaks_up_on_volatility_spike():
    # quiet (tight band) then a sharp up-move that both expands the band AND closes
    # above the upper band -> momentum entry
    closes = [100 + 0.05 * (i % 2) for i in range(25)] + [101, 103, 106, 110]
    sigs = _sigs(BollingerExpansion(period=20, k=2.0), closes)
    assert BUY in _sides(sigs)


def test_bollinger_expansion_warmup_holds():
    sigs = _sigs(BollingerExpansion(period=20), [100 + i for i in range(10)])
    assert set(_sides(sigs)) == {HOLD}


def test_bollinger_expansion_exits_back_to_mean():
    closes = ([100 + 0.05 * (i % 2) for i in range(25)] + [101, 103, 106, 110]
              + [108, 104, 100, 98])   # pop then revert to the mean
    sigs = _sigs(BollingerExpansion(period=20, k=2.0), closes)
    sides = _sides(sigs)
    assert BUY in sides and EXIT in sides


def test_bollinger_short_on_downside_expansion():
    closes = [100 + 0.05 * (i % 2) for i in range(25)] + [99, 97, 94, 90]
    sigs = _sigs(BollingerExpansion(period=20, k=2.0, allow_short=True), closes)
    assert SELL in _sides(sigs)


# ================================================================== registry
@pytest.mark.parametrize("name", [
    "vwap", "vwap-ls", "session-momentum", "session-momentum-ls",
    "bollinger-expansion", "bollinger-expansion-ls",
])
def test_registry_resolves_new_intraday(name):
    assert name in available()
    assert make(name) is not None


def test_ls_variants_enable_short():
    assert make("vwap-ls").allow_short is True
    assert make("session-momentum-ls").allow_short is True
    assert make("bollinger-expansion-ls").allow_short is True
