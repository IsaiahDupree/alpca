"""
Deep, deterministic, offline tests for the three pure-logic microstructure /
event-driven strategies:

  - alpca/strategies/event_driven.py     (GapFade)
  - alpca/strategies/microstructure.py   (microprice*, MicropriceGate, MicropriceTilt)
  - alpca/strategies/order_flow.py       (ofi_event, L1OFI)

No network, no mocks, no live Alpaca. Every bar is hand-built with REAL epoch
timestamps (built from America/New_York wall-clock via zoneinfo, the exact same
path the source's calendar.session_date uses) so the NYSE session classifier
actually rolls over. RNG, where used, is explicitly seeded.
"""

from __future__ import annotations

import math
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from alpca.strategies.base import BUY, EXIT, HOLD, SELL
from alpca.strategies.event_driven import GapFade
from alpca.strategies.microstructure import (
    MicropriceGate,
    MicropriceTilt,
    gate,
    microprice,
    microprice_signal,
    microprice_tilt,
)
from alpca.strategies.order_flow import L1OFI, ofi_event

_ET = ZoneInfo("America/New_York")


# --------------------------------------------------------------------------- helpers
def et_epoch(y, mo, d, h=9, mi=30, s=0) -> float:
    """Epoch seconds for an America/New_York wall-clock instant."""
    return datetime(y, mo, d, h, mi, s, tzinfo=_ET).timestamp()


def gap_bar(open_, close, ts):
    return {"open": open_, "close": close, "timestamp": ts}


def qbar(close=100.0, bid=None, ask=None, bid_size=None, ask_size=None):
    b = {"close": close}
    if bid is not None:
        b["bid"] = bid
    if ask is not None:
        b["ask"] = ask
    if bid_size is not None:
        b["bid_size"] = bid_size
    if ask_size is not None:
        b["ask_size"] = ask_size
    return b


# ============================================================================
# microprice() pure function
# ============================================================================
class TestMicroprice:
    @pytest.mark.parametrize("bid,ask,bs,asz", [
        (None, 100.1, 5, 5),
        (100.0, None, 5, 5),
        (100.0, 100.1, None, 5),
        (100.0, 100.1, 5, None),
    ])
    def test_none_inputs_return_none(self, bid, ask, bs, asz):
        assert microprice(bid, ask, bs, asz) is None

    @pytest.mark.parametrize("bs,asz", [(0, 0), (0, -1), (-3, 3), (-1, 0)])
    def test_nonpositive_total_size_returns_none(self, bs, asz):
        assert microprice(100.0, 100.1, bs, asz) is None

    def test_equal_sizes_gives_mid(self):
        # equal queues -> microprice == midpoint
        assert microprice(100.0, 100.2, 7, 7) == pytest.approx(100.1)

    def test_tilts_toward_ask_when_bid_heavy(self):
        # bid_size >> ask_size -> microprice rides up toward the ask
        mp = microprice(100.0, 100.2, 100, 1)
        assert 100.1 < mp < 100.2

    def test_tilts_toward_bid_when_ask_heavy(self):
        mp = microprice(100.0, 100.2, 1, 100)
        assert 100.0 < mp < 100.1

    def test_exact_weighted_value(self):
        # (ask*bid_size + bid*ask_size)/(bid_size+ask_size)
        # (100.2*3 + 100.0*1)/4 = (300.6+100.0)/4 = 100.15
        assert microprice(100.0, 100.2, 3, 1) == pytest.approx(100.15)

    def test_crossed_quote_still_returns_value(self):
        # microprice itself does not require a positive spread (that's tilt's job)
        assert microprice(100.2, 100.0, 5, 5) == pytest.approx(100.1)


# ============================================================================
# microprice_tilt() pure function
# ============================================================================
class TestMicropriceTilt:
    @pytest.mark.parametrize("bid,ask", [
        (100.1, 100.1),   # locked (half_spread == 0)
        (100.2, 100.0),   # crossed (half_spread < 0)
    ])
    def test_locked_or_crossed_returns_none(self, bid, ask):
        assert microprice_tilt(bid, ask, 5, 5) is None

    def test_none_quote_returns_none(self):
        assert microprice_tilt(None, 100.1, 5, 5) is None

    def test_balanced_book_zero_tilt(self):
        assert microprice_tilt(100.0, 100.2, 9, 9) == pytest.approx(0.0)

    @pytest.mark.parametrize("bs,asz,sign", [
        (100, 1, +1),   # bid-heavy -> positive tilt
        (1, 100, -1),   # ask-heavy -> negative tilt
    ])
    def test_tilt_sign(self, bs, asz, sign):
        t = microprice_tilt(100.0, 100.2, bs, asz)
        assert math.copysign(1, t) == sign

    @pytest.mark.parametrize("bs,asz", [
        (1000, 1), (1, 1000), (5, 5), (3, 1), (1, 3), (50, 7),
    ])
    def test_tilt_bounded_pm1(self, bs, asz):
        t = microprice_tilt(100.0, 100.2, bs, asz)
        assert -1.0 <= t <= 1.0

    def test_extreme_bid_heavy_approaches_plus_one(self):
        t = microprice_tilt(100.0, 100.2, 10**9, 1)
        assert t == pytest.approx(1.0, abs=1e-6)


# ============================================================================
# microprice_signal() pure function — deadband k
# ============================================================================
class TestMicropriceSignal:
    def test_none_quote_returns_none(self):
        assert microprice_signal(None, 100.1, 5, 5) is None
        assert microprice_signal(100.1, 100.1, 5, 5) is None  # locked

    def test_balanced_is_flat(self):
        assert microprice_signal(100.0, 100.2, 5, 5, k=0.0) == "flat"

    def test_bull_when_tilt_above_k(self):
        assert microprice_signal(100.0, 100.2, 100, 1, k=0.0) == "bull"

    def test_bear_when_tilt_below_neg_k(self):
        assert microprice_signal(100.0, 100.2, 1, 100, k=0.0) == "bear"

    def test_deadband_suppresses_weak_imbalance(self):
        # mild bid-heavy tilt, but a high deadband -> flat
        t = microprice_tilt(100.0, 100.2, 3, 1)
        assert 0 < t < 0.9
        assert microprice_signal(100.0, 100.2, 3, 1, k=0.99) == "flat"
        assert microprice_signal(100.0, 100.2, 3, 1, k=0.0) == "bull"

    def test_exactly_at_k_is_flat_not_bull(self):
        # strict > k / < -k: a tilt exactly equal to k is NOT a signal
        t = microprice_tilt(100.0, 100.2, 3, 1)
        assert microprice_signal(100.0, 100.2, 3, 1, k=t) == "flat"


# ============================================================================
# MicropriceTilt standalone strategy
# ============================================================================
class TestMicropriceTiltStrategy:
    def test_no_quote_holds(self):
        s = MicropriceTilt()
        sig = s.on_bar(qbar(close=100.0))  # no bid/ask
        assert sig.side == HOLD
        assert s._side == ""
        assert s._in_position is False

    def test_enter_long_on_bull(self):
        s = MicropriceTilt(k=0.5)
        sig = s.on_bar(qbar(100.0, 100.0, 100.2, 100, 1))
        assert sig.side == BUY
        assert s._side == "LONG"
        assert s._in_position is True
        assert sig.price == 100.0

    def test_hold_long_while_still_bull(self):
        s = MicropriceTilt(k=0.5)
        s.on_bar(qbar(100.0, 100.0, 100.2, 100, 1))
        sig = s.on_bar(qbar(101.0, 100.9, 101.1, 100, 1))
        assert sig.side == HOLD
        assert s._side == "LONG"

    def test_exit_long_when_tilt_fades_to_flat(self):
        s = MicropriceTilt(k=0.5)
        s.on_bar(qbar(100.0, 100.0, 100.2, 100, 1))
        sig = s.on_bar(qbar(100.0, 100.0, 100.2, 5, 5))  # balanced -> flat
        assert sig.side == EXIT
        assert s._side == ""
        assert s._in_position is False

    def test_exit_long_when_tilt_flips_bear(self):
        s = MicropriceTilt(k=0.5)
        s.on_bar(qbar(100.0, 100.0, 100.2, 100, 1))
        sig = s.on_bar(qbar(100.0, 100.0, 100.2, 1, 100))  # bear
        assert sig.side == EXIT
        assert s._side == ""

    def test_no_short_entry_by_default(self):
        s = MicropriceTilt(k=0.5)  # allow_short False
        sig = s.on_bar(qbar(100.0, 100.0, 100.2, 1, 100))  # bear
        assert sig.side == HOLD
        assert s._side == ""
        assert s._in_position is False

    def test_short_entry_when_allowed(self):
        s = MicropriceTilt(k=0.5, allow_short=True)
        sig = s.on_bar(qbar(100.0, 100.0, 100.2, 1, 100))  # bear
        assert sig.side == SELL
        assert s._side == "SHORT"
        assert s._in_position is True

    def test_short_exits_when_not_bear(self):
        s = MicropriceTilt(k=0.5, allow_short=True)
        s.on_bar(qbar(100.0, 100.0, 100.2, 1, 100))  # bear -> short
        sig = s.on_bar(qbar(100.0, 100.0, 100.2, 5, 5))  # flat
        assert sig.side == EXIT
        assert s._side == ""

    def test_no_quote_while_long_holds_position(self):
        # microprice_signal None -> "no quote" hold, position untouched
        s = MicropriceTilt(k=0.5)
        s.on_bar(qbar(100.0, 100.0, 100.2, 100, 1))
        sig = s.on_bar(qbar(close=100.0))  # no quote
        assert sig.side == HOLD
        assert s._side == "LONG"
        assert s._in_position is True

    def test_reset_clears_state(self):
        s = MicropriceTilt(k=0.5)
        s.on_bar(qbar(100.0, 100.0, 100.2, 100, 1))
        s.reset()
        assert s._side == ""
        assert s._in_position is False
        assert s._entry_price == 0.0


# ============================================================================
# MicropriceGate confirmation gate
# ============================================================================
class _AlwaysBuy(MicropriceTilt):
    """A tiny deterministic base that emits a fresh BUY each bar from flat,
    flipping its own optimistic _enter state (so we can observe rollback)."""

    name = "alwaysbuy"

    def on_bar(self, bar):
        return self._enter(bar["close"], 1.0, "always buy")


class _AlwaysSell(MicropriceTilt):
    name = "alwayssell"

    def on_bar(self, bar):
        return self._short(bar["close"], 1.0, "always short")


class _AlwaysHold(MicropriceTilt):
    name = "alwayshold"

    def on_bar(self, bar):
        from alpca.strategies.base import hold
        return hold("base hold")


class TestMicropriceGate:
    def test_name_composition(self):
        g = gate(_AlwaysBuy(k=0.5))
        assert g.name == "alwaysbuy+mp"
        assert isinstance(g, MicropriceGate)

    def test_buy_passes_when_bull_confirms(self):
        g = MicropriceGate(_AlwaysBuy(k=0.5), k=0.5)
        sig = g.on_bar(qbar(100.0, 100.0, 100.2, 100, 1))  # bull
        assert sig.side == BUY
        assert g.base._in_position is True
        assert g.base._entry_price == 100.0

    def test_buy_blocked_when_no_confirm_rolls_back_state(self):
        base = _AlwaysBuy(k=0.5)
        g = MicropriceGate(base, k=0.5)
        sig = g.on_bar(qbar(100.0, 100.0, 100.2, 1, 100))  # bear, contradicts BUY
        assert sig.side == HOLD
        assert "blocked" in sig.reason
        # optimistic entry flip undone:
        assert base._in_position is False
        assert base._entry_price == 0.0
        assert base._side == ""

    def test_buy_blocked_flat_book_rolls_back(self):
        base = _AlwaysBuy(k=0.5)
        g = MicropriceGate(base, k=0.5)
        sig = g.on_bar(qbar(100.0, 100.0, 100.2, 5, 5))  # flat tilt
        assert sig.side == HOLD
        assert base._in_position is False

    def test_sell_passes_when_bear_confirms(self):
        base = _AlwaysSell(k=0.5, allow_short=True)
        g = MicropriceGate(base, k=0.5)
        sig = g.on_bar(qbar(100.0, 100.0, 100.2, 1, 100))  # bear
        assert sig.side == SELL
        assert base._in_position is True

    def test_sell_blocked_when_bull(self):
        base = _AlwaysSell(k=0.5, allow_short=True)
        g = MicropriceGate(base, k=0.5)
        sig = g.on_bar(qbar(100.0, 100.0, 100.2, 100, 1))  # bull contradicts SELL
        assert sig.side == HOLD
        assert base._in_position is False
        assert base._side == ""

    def test_require_quote_blocks_entry_when_no_quote(self):
        base = _AlwaysBuy(k=0.5)
        g = MicropriceGate(base, k=0.5, require_quote=True)
        sig = g.on_bar(qbar(close=100.0))  # no quote
        assert sig.side == HOLD
        assert "no quote" in sig.reason
        assert base._in_position is False  # rolled back

    def test_no_require_quote_passes_entry_when_no_quote(self):
        base = _AlwaysBuy(k=0.5)
        g = MicropriceGate(base, k=0.5, require_quote=False)
        sig = g.on_bar(qbar(close=100.0))  # no quote
        assert sig.side == BUY
        assert base._in_position is True

    def test_no_require_quote_still_blocks_on_contradicting_quote(self):
        # a usable quote that contradicts is still a block even when require_quote False
        base = _AlwaysBuy(k=0.5)
        g = MicropriceGate(base, k=0.5, require_quote=False)
        sig = g.on_bar(qbar(100.0, 100.0, 100.2, 1, 100))  # bear vs BUY
        assert sig.side == HOLD
        assert base._in_position is False

    def test_hold_always_passes_unblocked(self):
        base = _AlwaysHold(k=0.5)
        g = MicropriceGate(base, k=0.5)
        sig = g.on_bar(qbar(100.0, 100.0, 100.2, 1, 100))  # contradicting book
        assert sig.side == HOLD
        assert sig.reason == "base hold"  # passed through unchanged, not "blocked"

    def test_exit_always_passes(self):
        # build a base that is in position and exits
        base = MicropriceTilt(k=0.5)
        base.on_bar(qbar(100.0, 100.0, 100.2, 100, 1))  # long
        g = MicropriceGate(base, k=0.5)
        # next bar flat tilt -> base wants to EXIT; gate must never block exits
        sig = g.on_bar(qbar(100.0, 100.0, 100.2, 5, 5))
        assert sig.side == EXIT

    def test_reset_resets_base(self):
        base = _AlwaysBuy(k=0.5)
        g = MicropriceGate(base, k=0.5)
        g.on_bar(qbar(100.0, 100.0, 100.2, 100, 1))
        g.reset()
        assert base._in_position is False
        assert base._side == ""

    def test_on_fill_delegates_to_base(self):
        base = MicropriceTilt(k=0.5)
        g = MicropriceGate(base, k=0.5)
        g.on_fill("SELL", 1.0, 100.0)
        assert base._side == "SHORT"
        assert base._in_position is True

    def test_block_then_retry_next_bar_succeeds(self):
        # idempotency / retry: blocked entry leaves base flat, so a confirming
        # next bar lets the SAME base re-enter.
        base = _AlwaysBuy(k=0.5)
        g = MicropriceGate(base, k=0.5)
        s1 = g.on_bar(qbar(100.0, 100.0, 100.2, 1, 100))   # blocked
        assert s1.side == HOLD and base._in_position is False
        s2 = g.on_bar(qbar(100.0, 100.0, 100.2, 100, 1))   # confirms
        assert s2.side == BUY and base._in_position is True


# ============================================================================
# ofi_event() pure function
# ============================================================================
class TestOfiEvent:
    def test_bid_rose_adds_full_bid_size(self):
        # bid up -> dW = bid_size ; ask unchanged -> dV = ask_size - prev_ask_size
        e = ofi_event(101, 10, 102, 7, 100, 5, 102, 7)
        # dW = 10 ; dV = 7 - 7 = 0 -> e = 10
        assert e == 10

    def test_bid_unchanged_adds_delta(self):
        e = ofi_event(100, 12, 102, 7, 100, 5, 102, 7)
        # dW = 12 - 5 = 7 ; dV = 0 -> e = 7
        assert e == 7

    def test_bid_fell_subtracts_prev_bid_size(self):
        e = ofi_event(99, 12, 102, 7, 100, 5, 102, 7)
        # dW = -5 ; dV = 0 -> e = -5
        assert e == -5

    def test_ask_rose_supply_retreat_is_bullish(self):
        # ask up -> dV = -prev_ask_size (negative dV -> +e -> bullish)
        e = ofi_event(100, 5, 103, 9, 100, 5, 102, 8)
        # dW = 5 - 5 = 0 ; dV = -8 -> e = 0 - (-8) = 8
        assert e == 8

    def test_ask_fell_aggressive_sell_is_bearish(self):
        e = ofi_event(100, 5, 101, 9, 100, 5, 102, 8)
        # dW = 0 ; dV = ask_size = 9 -> e = -9
        assert e == -9

    def test_full_static_book_zero(self):
        e = ofi_event(100, 5, 102, 8, 100, 5, 102, 8)
        assert e == 0

    @pytest.mark.parametrize("scale", [1, 100, 10_000])
    def test_scales_linearly_in_size(self, scale):
        e = ofi_event(101, 10 * scale, 102, 7 * scale,
                      100, 5 * scale, 102, 7 * scale)
        assert e == 10 * scale


# ============================================================================
# L1OFI strategy
# ============================================================================
def _ofibar(bid, ask, bs, az, close=100.0):
    return {"bid": bid, "ask": ask, "bid_size": bs, "ask_size": az, "close": close}


class TestL1OFI:
    def test_missing_quote_holds(self):
        s = L1OFI(window=3)
        sig = s.on_bar({"close": 100.0})
        assert sig.side == HOLD
        assert sig.reason == "no quote"

    @pytest.mark.parametrize("missing", ["bid", "ask", "bid_size", "ask_size"])
    def test_partial_quote_holds(self, missing):
        s = L1OFI(window=3)
        bar = _ofibar(100.0, 100.2, 5, 5)
        bar[missing] = None
        sig = s.on_bar(bar)
        assert sig.side == HOLD and sig.reason == "no quote"

    def test_first_bar_is_warmup(self):
        s = L1OFI(window=3)
        sig = s.on_bar(_ofibar(100.0, 100.2, 5, 5))
        assert sig.side == HOLD and sig.reason == "warmup"
        assert s._prev == (100.0, 5, 100.2, 5)

    def test_warmup_until_window_full(self):
        s = L1OFI(window=3)
        s.on_bar(_ofibar(100.0, 100.2, 5, 5))   # prev set
        # after this there is 1 event; window 3 not full
        s.on_bar(_ofibar(100.1, 100.2, 5, 5))
        sig = s.on_bar(_ofibar(100.2, 100.2, 5, 5))
        # now 2 events, still < window 3 -> warmup
        assert sig.side == HOLD and sig.reason == "warmup"

    def test_sustained_buy_pressure_goes_long(self):
        # rising bid every step -> each e = bid_size, normOFI large positive
        s = L1OFI(window=3, entry=0.19, exit=0.08)
        s.on_bar(_ofibar(100.0, 100.5, 10, 10))   # warmup prev
        s.on_bar(_ofibar(100.1, 100.5, 10, 10))   # e=10
        s.on_bar(_ofibar(100.2, 100.5, 10, 10))   # e=10
        sig = s.on_bar(_ofibar(100.3, 100.5, 10, 10))  # window full, ofi>0.19
        assert sig.side == BUY
        assert s._side == "LONG"
        assert s._in_position is True
        assert sig.metadata["ofi"] > 0.19

    def test_strength_capped_at_one(self):
        s = L1OFI(window=2, entry=0.19, exit=0.08)
        s.on_bar(_ofibar(100.0, 100.5, 10, 10))
        s.on_bar(_ofibar(100.1, 100.5, 10, 10))
        sig = s.on_bar(_ofibar(100.2, 100.5, 10, 10))
        assert sig.side == BUY
        assert sig.strength == pytest.approx(1.0)

    def test_long_holds_then_exits_on_revert(self):
        s = L1OFI(window=2, entry=0.19, exit=0.08)
        s.on_bar(_ofibar(100.0, 100.5, 10, 10))
        s.on_bar(_ofibar(100.1, 100.5, 10, 10))
        e = s.on_bar(_ofibar(100.2, 100.5, 10, 10))
        assert e.side == BUY
        # now feed a static book: events become 0, normOFI decays below exit
        s.on_bar(_ofibar(100.2, 100.5, 10, 10))   # e=0
        sig = s.on_bar(_ofibar(100.2, 100.5, 10, 10))  # window now all-zero
        assert sig.side == EXIT
        assert s._side == ""
        assert s._in_position is False

    def test_no_short_by_default(self):
        # falling bid -> normOFI strongly negative, but allow_short False
        s = L1OFI(window=2, entry=0.19, exit=0.08)
        s.on_bar(_ofibar(100.3, 100.5, 10, 10))
        s.on_bar(_ofibar(100.2, 100.5, 10, 10))
        sig = s.on_bar(_ofibar(100.1, 100.5, 10, 10))
        assert sig.side == HOLD
        assert s._side == ""

    def test_short_when_allowed(self):
        s = L1OFI(window=2, entry=0.19, exit=0.08, allow_short=True)
        s.on_bar(_ofibar(100.3, 100.5, 10, 10))
        s.on_bar(_ofibar(100.2, 100.5, 10, 10))
        sig = s.on_bar(_ofibar(100.1, 100.5, 10, 10))
        assert sig.side == SELL
        assert s._side == "SHORT"
        assert sig.metadata["ofi"] < -0.19

    def test_on_session_start_clears_window_not_position(self):
        s = L1OFI(window=2, entry=0.19, exit=0.08)
        s.on_bar(_ofibar(100.0, 100.5, 10, 10))
        s.on_bar(_ofibar(100.1, 100.5, 10, 10))
        e = s.on_bar(_ofibar(100.2, 100.5, 10, 10))
        assert e.side == BUY and s._in_position is True
        s.on_session_start()
        # rolling window cleared, prev cleared; POSITION kept
        assert len(s._e) == 0
        assert len(s._sz) == 0
        assert s._prev is None
        assert s._side == "LONG"
        assert s._in_position is True
        # next bar after session start is a fresh warmup (prev was cleared)
        sig = s.on_bar(_ofibar(100.2, 100.5, 10, 10))
        assert sig.reason == "warmup"

    def test_reset_clears_everything(self):
        s = L1OFI(window=2)
        s.on_bar(_ofibar(100.0, 100.5, 10, 10))
        s.on_bar(_ofibar(100.1, 100.5, 10, 10))
        s.reset()
        assert s._prev is None
        assert len(s._e) == 0
        assert s._side == ""
        assert s._in_position is False

    def test_window_is_bounded_deque(self):
        s = L1OFI(window=3)
        for i in range(10):
            s.on_bar(_ofibar(100.0 + i * 0.1, 100.9, 10, 10))
        assert len(s._e) <= 3
        assert len(s._sz) <= 3

    def test_zero_entry_strength_is_one(self):
        # entry=0 path: strength becomes 1.0 (guard min(...) skipped)
        s = L1OFI(window=2, entry=0.0, exit=0.0)
        s.on_bar(_ofibar(100.0, 100.5, 10, 10))
        s.on_bar(_ofibar(100.1, 100.5, 10, 10))
        sig = s.on_bar(_ofibar(100.2, 100.5, 10, 10))
        assert sig.side == BUY
        assert sig.strength == pytest.approx(1.0)


# ============================================================================
# GapFade event-driven strategy (real epoch session rollover)
# ============================================================================
class TestGapFadeBasics:
    def test_first_bar_is_warmup_no_ref(self):
        g = GapFade()
        sig = g.on_bar(gap_bar(100, 100, et_epoch(2025, 6, 2, 9, 30)))
        assert sig.side == HOLD
        assert sig.reason == "warmup"
        assert g._ref_close is None
        assert g._running_close == 100

    def test_same_session_no_trade(self):
        g = GapFade()
        ts = et_epoch(2025, 6, 2, 9, 30)
        g.on_bar(gap_bar(100, 100, ts))
        sig = g.on_bar(gap_bar(100, 101, et_epoch(2025, 6, 2, 9, 31)))
        # still same date, not a new session -> flat
        assert sig.side == HOLD
        assert sig.reason == "flat"

    def test_gap_down_goes_long(self):
        g = GapFade(entry_pct=0.01)
        g.on_bar(gap_bar(100, 100, et_epoch(2025, 6, 2, 9, 30)))  # ref close=100
        sig = g.on_bar(gap_bar(98, 98, et_epoch(2025, 6, 3, 9, 30)))  # gap -2%
        assert sig.side == BUY
        assert g._side == "LONG"
        assert g._in_position is True
        assert g._ref_close == 100
        assert sig.metadata["gap"] == pytest.approx(-0.02)

    def test_gap_up_no_short_when_long_only(self):
        g = GapFade(entry_pct=0.01, allow_short=False)
        g.on_bar(gap_bar(100, 100, et_epoch(2025, 6, 2, 9, 30)))
        sig = g.on_bar(gap_bar(102, 102, et_epoch(2025, 6, 3, 9, 30)))  # gap +2%
        assert sig.side == HOLD
        assert g._side == ""
        assert g._in_position is False

    def test_gap_up_shorts_when_allowed(self):
        g = GapFade(entry_pct=0.01, allow_short=True)
        g.on_bar(gap_bar(100, 100, et_epoch(2025, 6, 2, 9, 30)))
        sig = g.on_bar(gap_bar(102, 102, et_epoch(2025, 6, 3, 9, 30)))  # gap +2%
        assert sig.side == SELL
        assert g._side == "SHORT"
        assert g._in_position is True
        assert sig.metadata["gap"] == pytest.approx(0.02)

    @pytest.mark.parametrize("open_,expect", [
        (99.5, "HOLD"),    # gap -0.5% inside band -> no trade
        (99.0, "BUY"),     # gap -1.0% exactly at band -> long (<=)
        (101.0, "HOLD"),   # gap +1.0% at band but long-only -> hold
    ])
    def test_entry_threshold_boundaries(self, open_, expect):
        g = GapFade(entry_pct=0.01, allow_short=False)
        g.on_bar(gap_bar(100, 100, et_epoch(2025, 6, 2, 9, 30)))
        sig = g.on_bar(gap_bar(open_, open_, et_epoch(2025, 6, 3, 9, 30)))
        assert sig.side == expect


class TestGapFadeExits:
    def _open_long(self, **kw):
        g = GapFade(entry_pct=0.01, **kw)
        g.on_bar(gap_bar(100, 100, et_epoch(2025, 6, 2, 9, 30)))  # ref=100
        # gap down to 95 (open & close 95) -> long, entry price 95
        sig = g.on_bar(gap_bar(95, 95, et_epoch(2025, 6, 3, 9, 30)))
        assert sig.side == BUY and g._entry_price == 95
        return g

    def test_long_exit_on_revert(self):
        g = self._open_long(exit_pct=0.002)
        # close near ref 100 (within 0.2%) -> revert exit
        sig = g.on_bar(gap_bar(100, 100.1, et_epoch(2025, 6, 3, 9, 40)))
        assert sig.side == EXIT
        assert "revert" in sig.reason
        assert g._side == ""
        assert g._in_position is False

    def test_long_exit_on_stop(self):
        g = self._open_long(exit_pct=0.0001, stop_pct=0.01)
        # entry 95; stop at 95*(1-0.01)=94.05; close 94 <= stop, and not reverted
        sig = g.on_bar(gap_bar(94, 94, et_epoch(2025, 6, 3, 9, 41)))
        assert sig.side == EXIT
        assert "stop" in sig.reason
        assert g._side == ""

    def test_long_exit_on_time(self):
        g = self._open_long(exit_pct=0.00001, stop_pct=0.99, hold_bars=3)
        # avoid revert (ref 100, stay far) and stop (stop_pct huge); bars_held hits 3
        t = et_epoch(2025, 6, 3, 9, 31)
        s1 = g.on_bar(gap_bar(96, 96, t))            # held=1, hold
        s2 = g.on_bar(gap_bar(96, 96, t + 60))       # held=2, hold
        s3 = g.on_bar(gap_bar(96, 96, t + 120))      # held=3, time exit
        assert s1.side == HOLD and s2.side == HOLD
        assert s3.side == EXIT and "time" in s3.reason
        assert g._side == ""

    def test_short_exit_on_stop(self):
        g = GapFade(entry_pct=0.01, exit_pct=0.0001, stop_pct=0.01, allow_short=True)
        g.on_bar(gap_bar(100, 100, et_epoch(2025, 6, 2, 9, 30)))  # ref=100
        sig = g.on_bar(gap_bar(105, 105, et_epoch(2025, 6, 3, 9, 30)))  # gap +5% short, entry 105
        assert sig.side == SELL
        # short stop = 105*(1+0.01)=106.05; close 107 >= stop -> stop exit
        ex = g.on_bar(gap_bar(107, 107, et_epoch(2025, 6, 3, 9, 31)))
        assert ex.side == EXIT
        assert "stop" in ex.reason
        assert g._side == ""

    def test_position_held_while_no_exit_condition(self):
        g = self._open_long(exit_pct=0.00001, stop_pct=0.99, hold_bars=100)
        sig = g.on_bar(gap_bar(96, 96, et_epoch(2025, 6, 3, 9, 31)))
        assert sig.side == HOLD
        assert "holding" in sig.reason
        assert g._in_position is True


class TestGapFadeRollover:
    def test_position_flattened_at_new_session(self):
        g = GapFade(entry_pct=0.01, exit_pct=0.00001, stop_pct=0.99, hold_bars=999)
        g.on_bar(gap_bar(100, 100, et_epoch(2025, 6, 2, 9, 30)))   # ref=100
        sig = g.on_bar(gap_bar(95, 95, et_epoch(2025, 6, 3, 9, 30)))  # long
        assert sig.side == BUY and g._in_position is True
        # next session opens while still in position -> flatten at rollover
        flat = g.on_bar(gap_bar(96, 96, et_epoch(2025, 6, 4, 9, 30)))
        assert flat.side == EXIT
        assert "rollover flatten" in flat.reason
        assert g._side == ""
        assert g._in_position is False
        # ref_close becomes the prior session's running close (95)
        assert g._ref_close == 95
        assert g._cur_date == "2025-06-04"

    def test_no_trade_carried_overnight_when_flat(self):
        # if flat at rollover, just re-arm; a same-direction gap can fire fresh
        g = GapFade(entry_pct=0.01)
        g.on_bar(gap_bar(100, 100, et_epoch(2025, 6, 2, 9, 30)))   # ref=100
        s2 = g.on_bar(gap_bar(99.5, 99.5, et_epoch(2025, 6, 3, 9, 30)))  # gap -0.5% no trade
        assert s2.side == HOLD
        # new session, ref now 99.5; gap to 98 is -1.5% -> long
        s3 = g.on_bar(gap_bar(98, 98, et_epoch(2025, 6, 4, 9, 30)))
        assert s3.side == BUY
        assert g._ref_close == pytest.approx(99.5)

    def test_reset_clears_all_state(self):
        g = GapFade(entry_pct=0.01)
        g.on_bar(gap_bar(100, 100, et_epoch(2025, 6, 2, 9, 30)))
        g.on_bar(gap_bar(95, 95, et_epoch(2025, 6, 3, 9, 30)))
        g.reset()
        assert g._side == ""
        assert g._cur_date is None
        assert g._ref_close is None
        assert g._running_close == 0.0
        assert g._bars_held == 0
        assert g._in_position is False


class TestGapFadeDegenerate:
    def test_missing_timestamp_collapses_to_one_date(self):
        # documented caveat: no/zero timestamp -> all bars share one calendar
        # date (the epoch-0 date) so no session ever rolls -> never trades.
        g = GapFade(entry_pct=0.01)
        s1 = g.on_bar({"open": 100, "close": 100})
        s2 = g.on_bar({"open": 90, "close": 90})
        s3 = g.on_bar({"open": 80, "close": 80})
        assert s1.side == HOLD
        assert s2.side == HOLD
        assert s3.side == HOLD
        assert g._in_position is False

    def test_on_fill_short_sync(self):
        # mirrors ZScore: a SELL fill from flat syncs _side/_in_position
        g = GapFade()
        g.on_fill("SELL", 1.0, 100.0)
        assert g._side == "SHORT"
        assert g._in_position is True

    def test_on_fill_buy_is_noop_for_state(self):
        g = GapFade()
        g.on_fill("BUY", 1.0, 100.0)
        assert g._side == ""
        assert g._in_position is False
