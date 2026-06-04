"""
Deep, deterministic tests for alpca/strategies/mean_reversion.py.

Covers the pure/offline logic only (no network, no mocks, no live Alpaca):
  - wilder_rsi: warmup gating, all-up=100, all-down=0, exact RS computations,
    Wilder smoothing past the seed window, and degenerate inputs.
  - rolling_return_vol: population stdev of bar returns, warmup gating, zero/NaN
    price handling, and the "<2 returns" guard.
  - ZScoreMeanReversion: z entry/exit/stop, long & short via allow_short,
    warmup, flat (std==0), _side bookkeeping, reset, on_fill sync.
  - RSIMeanReversion: entry_low/entry_high, exit_level, stop_pct, vol-regime
    gate band (floor/cap), allow_short, reset, on_fill side sync.

All expected values are computed by hand from the source formulas and confirmed
against the real functions. No reliance on wall-clock time or unseeded RNG.
"""

from __future__ import annotations

import math
import statistics

import pytest

from alpca.strategies.base import BUY, SELL, EXIT, HOLD
from alpca.strategies.mean_reversion import (
    wilder_rsi,
    rolling_return_vol,
    ZScoreMeanReversion,
    RSIMeanReversion,
)


# --------------------------------------------------------------------------- #
# tiny self-contained helpers (no imports from other tests/)                  #
# --------------------------------------------------------------------------- #
def bar(close: float) -> dict:
    """Minimal bar dict; the strategies only read ['close']."""
    return {"close": float(close)}


def feed(strategy, closes):
    """Push a list of closes through on_bar; return the list of Signals."""
    return [strategy.on_bar(bar(c)) for c in closes]


def manual_rsi(closes, period):
    """Independent reference implementation of Wilder RSI (mirrors source)."""
    if len(closes) < period + 1:
        return None
    gain = loss = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gain += d
        else:
            loss += -d
    avg_gain = gain / period
    avg_loss = loss / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# =========================================================================== #
# wilder_rsi                                                                  #
# =========================================================================== #
class TestWilderRSIWarmup:
    @pytest.mark.parametrize(
        "closes,period",
        [
            ([], 2),
            ([1.0], 2),
            ([1.0, 2.0], 2),       # need period+1 = 3
            ([1, 2, 3, 4], 4),     # need 5
            ([10] * 14, 14),       # exactly period, one short
        ],
    )
    def test_returns_none_until_period_plus_one(self, closes, period):
        assert wilder_rsi(list(map(float, closes)), period) is None

    @pytest.mark.parametrize(
        "closes,period",
        [
            ([1.0, 2.0, 3.0], 2),          # exactly period+1
            ([1, 2, 3, 4, 5], 4),
            ([10] * 15, 14),               # exactly period+1
        ],
    )
    def test_returns_value_at_exactly_period_plus_one(self, closes, period):
        out = wilder_rsi(list(map(float, closes)), period)
        assert out is not None
        assert 0.0 <= out <= 100.0


class TestWilderRSIExact:
    def test_all_up_is_100(self):
        # avg_loss == 0 -> 100.0 by definition
        assert wilder_rsi([1.0, 2.0, 3.0, 4.0, 5.0], 4) == 100.0

    def test_all_up_short_window_is_100(self):
        assert wilder_rsi([10.0, 11.0, 12.0], 2) == 100.0

    def test_flat_series_is_neutral_50(self):
        # a perfectly flat series has no up OR down moves -> NEUTRAL 50 (not the
        # all-up sentinel 100): avg_gain == 0 and avg_loss == 0.
        assert wilder_rsi([7.0, 7.0, 7.0, 7.0], 3) == 50.0

    def test_all_down_is_0(self):
        # avg_gain == 0, avg_loss > 0 -> rs == 0 -> 100 - 100/1 == 0
        assert wilder_rsi([5.0, 4.0, 3.0, 2.0, 1.0], 4) == 0.0

    def test_known_rs_seed_window(self):
        # closes [10, 8, 11], period 2:
        #   d1 = -2 -> loss 2 ; d2 = +3 -> gain 3
        #   avg_gain = 1.5, avg_loss = 1.0 -> rs = 1.5
        #   rsi = 100 - 100/2.5 = 60.0
        assert wilder_rsi([10.0, 8.0, 11.0], 2) == pytest.approx(60.0)

    def test_wilder_smoothing_past_seed(self):
        # closes [10, 11, 12, 11], period 2:
        #   seed (i=1,2): gains 1,1 -> avg_gain=1, avg_loss=0
        #   i=3: d=-1 -> g=0,l=1 ; avg_gain=(1*1+0)/2=0.5, avg_loss=(0*1+1)/2=0.5
        #   rs = 1 -> rsi = 50.0
        assert wilder_rsi([10.0, 11.0, 12.0, 11.0], 2) == pytest.approx(50.0)

    @pytest.mark.parametrize(
        "closes,period",
        [
            ([10, 8, 11, 9, 12, 7, 13], 2),
            ([100, 102, 101, 103, 99, 104, 98, 105], 3),
            ([50, 49, 51, 48, 52, 47, 53, 46, 54, 45], 4),
            ([1, 3, 2, 6, 4, 9, 5, 12, 6, 15, 7], 5),
        ],
    )
    def test_matches_independent_reference(self, closes, period):
        c = list(map(float, closes))
        assert wilder_rsi(c, period) == pytest.approx(manual_rsi(c, period))

    @pytest.mark.parametrize(
        "closes,period",
        [
            ([10, 8, 11, 9, 12], 2),
            ([5, 4, 3, 2, 1, 2, 3, 4], 3),
            ([100, 90, 110, 95, 105, 92, 108], 4),
        ],
    )
    def test_bounded_0_to_100(self, closes, period):
        out = wilder_rsi(list(map(float, closes)), period)
        assert 0.0 <= out <= 100.0


class TestWilderRSIEdge:
    def test_extreme_magnitudes(self):
        # huge spread up only -> still 100
        assert wilder_rsi([1e-6, 1e9, 2e9], 2) == 100.0

    def test_negative_prices_pure_gains(self):
        # the function operates on diffs, so negative price levels are fine;
        # -5 -> -3 -> -1 is strictly increasing -> all gains -> 100
        assert wilder_rsi([-5.0, -3.0, -1.0], 2) == 100.0

    def test_nan_propagates_not_crash(self):
        out = wilder_rsi([1.0, float("nan"), 3.0, 4.0], 2)
        # NaN in the diff stream poisons the average; must not raise.
        assert out is None or isinstance(out, float)

    def test_idempotent_no_mutation_of_input(self):
        closes = [10.0, 8.0, 11.0]
        snapshot = list(closes)
        wilder_rsi(closes, 2)
        wilder_rsi(closes, 2)
        assert closes == snapshot


# =========================================================================== #
# rolling_return_vol                                                          #
# =========================================================================== #
class TestRollingReturnVolWarmup:
    @pytest.mark.parametrize(
        "closes,lookback",
        [
            ([], 3),
            ([5.0], 3),
            ([1.0, 2.0, 3.0], 3),   # need lookback+1 = 4
            ([10] * 20, 20),
        ],
    )
    def test_none_until_lookback_plus_one(self, closes, lookback):
        assert rolling_return_vol(list(map(float, closes)), lookback) is None


class TestRollingReturnVolExact:
    def test_population_stdev_of_returns(self):
        closes = [10.0, 11.0, 12.0, 13.0]
        rets = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, 4)]
        expected = statistics.pstdev(rets)
        assert rolling_return_vol(closes, 3) == pytest.approx(expected)

    def test_uses_only_last_lookback_returns(self):
        # leading bars outside the window must not affect the result
        tail = [100.0, 101.0, 102.0, 103.0]
        with_lead = [1.0, 50.0] + tail
        v_lead = rolling_return_vol(with_lead, 3)
        v_tail = rolling_return_vol(tail, 3)
        assert v_lead == pytest.approx(v_tail)

    def test_constant_returns_zero_vol(self):
        # geometric-ish: equal multiplicative steps still give differing simple
        # returns, so use additive-equal that yields identical simple returns:
        # [0, 1, 2, 4] over lookback 3 -> rets [skip p0=0], [1/1], [2/2]=[1,1] pstdev 0
        assert rolling_return_vol([0.0, 1.0, 2.0, 4.0], 3) == pytest.approx(0.0)

    @pytest.mark.parametrize(
        "closes,lookback",
        [
            ([100, 102, 99, 101, 103], 4),
            ([50, 51, 49, 52, 48, 53], 5),
            ([10, 10.5, 9.5, 10.2, 9.8], 4),
        ],
    )
    def test_nonnegative(self, closes, lookback):
        v = rolling_return_vol(list(map(float, closes)), lookback)
        assert v is not None and v >= 0.0


class TestRollingReturnVolEdge:
    def test_zero_price_skipped_then_too_few(self):
        # [0,0,1,2] lookback2: window i=2 (p0=closes[1]=0 -> skip), i=3 (ret=1)
        # -> only 1 return -> < 2 -> None
        assert rolling_return_vol([0.0, 0.0, 1.0, 2.0], 2) is None

    def test_zero_price_partial_skip_keeps_enough(self):
        # [0,1,2,4] lookback3: i=1 p0=0 skip, i=2 ret 1, i=3 ret 1 -> 2 rets -> 0.0
        assert rolling_return_vol([0.0, 1.0, 2.0, 4.0], 3) == pytest.approx(0.0)

    def test_negative_prices(self):
        v = rolling_return_vol([-10.0, -11.0, -12.0, -13.0], 3)
        assert v is not None and v >= 0.0

    def test_nan_handled_gracefully_no_raise(self):
        # FIXED BEHAVIOR: rolling_return_vol now filters out non-finite returns
        # and closes before calling statistics.pstdev, so a NaN close no longer
        # raises ValueError. It returns None (if <2 usable returns remain) or a
        # finite float otherwise.
        out = rolling_return_vol([10.0, float("nan"), 12.0, 13.0], 3)
        assert out is None or math.isfinite(out)

    def test_idempotent_no_mutation(self):
        closes = [10.0, 11.0, 12.0, 13.0]
        snap = list(closes)
        rolling_return_vol(closes, 3)
        rolling_return_vol(closes, 3)
        assert closes == snap


# =========================================================================== #
# ZScoreMeanReversion                                                         #
# =========================================================================== #
class TestZScoreWarmupAndFlat:
    def test_warmup_holds_until_lookback(self):
        s = ZScoreMeanReversion(lookback=3, entry_z=1.0)
        outs = feed(s, [10.0, 10.0])
        assert all(o.side == HOLD and o.reason == "warmup" for o in outs)

    def test_flat_when_zero_std(self):
        s = ZScoreMeanReversion(lookback=3, entry_z=1.0)
        outs = feed(s, [5.0, 5.0, 5.0])
        assert outs[-1].side == HOLD and outs[-1].reason == "flat"
        assert s._side == ""

    def test_defaults(self):
        s = ZScoreMeanReversion()
        assert s.lookback == 60
        assert s.entry_z == 2.0
        assert s.exit_z == 0.5
        assert s.stop_z == 3.5
        assert s.allow_short is False
        assert s.name == "zscore"


class TestZScoreLong:
    def test_buy_on_oversold(self):
        # [10,10,7] -> mean 9, pstd sqrt(2), z=(7-9)/sqrt2 = -1.414 < -1.0
        s = ZScoreMeanReversion(lookback=3, entry_z=1.0, exit_z=0.6, stop_z=10.0)
        outs = feed(s, [10.0, 10.0, 7.0])
        buy = outs[-1]
        assert buy.side == BUY
        assert buy.strength == 1.0
        assert buy.price == 7.0
        assert buy.metadata["z"] == pytest.approx(-math.sqrt(2.0))
        assert s._side == "LONG"
        assert s._in_position is True
        assert s._entry_price == 7.0

    def test_no_buy_when_above_neg_entry(self):
        # mild dip not crossing entry threshold
        s = ZScoreMeanReversion(lookback=3, entry_z=5.0)
        outs = feed(s, [10.0, 10.0, 9.9])
        assert outs[-1].side == HOLD
        assert s._side == ""

    def test_exit_on_revert(self):
        # enter at [10,10,7]; next bar 10 -> window [10,10,7,10] z=+0.577 < exit 0.6
        s = ZScoreMeanReversion(lookback=3, entry_z=1.0, exit_z=0.6, stop_z=10.0)
        outs = feed(s, [10.0, 10.0, 7.0, 10.0])
        assert outs[2].side == BUY
        assert outs[3].side == EXIT
        assert "revert" in outs[3].reason
        assert s._side == ""
        assert s._in_position is False

    def test_holding_long_when_not_reverted(self):
        # tight exit_z keeps us holding
        s = ZScoreMeanReversion(lookback=3, entry_z=1.0, exit_z=0.1, stop_z=10.0)
        outs = feed(s, [10.0, 10.0, 7.0, 10.0])
        assert outs[2].side == BUY
        assert outs[3].side == HOLD
        assert outs[3].reason == "holding long"
        assert s._side == "LONG"

    def test_stop_blows_through_downside(self):
        # Long, then price collapses further -> z < -stop_z triggers EXIT.
        s = ZScoreMeanReversion(lookback=4, entry_z=1.0, exit_z=0.01, stop_z=1.2)
        # build a long then a deep negative z
        outs = feed(s, [10.0, 10.0, 10.0, 8.5, 3.0])
        # find the BUY then a subsequent EXIT
        sides = [o.side for o in outs]
        assert BUY in sides
        assert EXIT in sides
        assert s._side == ""


class TestZScoreShort:
    def test_short_on_overbought_when_allowed(self):
        s = ZScoreMeanReversion(lookback=3, entry_z=1.0, exit_z=0.6,
                                stop_z=10.0, allow_short=True)
        outs = feed(s, [10.0, 10.0, 13.0])
        sell = outs[-1]
        assert sell.side == SELL
        assert sell.metadata["z"] == pytest.approx(math.sqrt(2.0))
        assert sell.price == 13.0
        assert s._side == "SHORT"
        assert s._in_position is True

    def test_no_short_when_not_allowed(self):
        s = ZScoreMeanReversion(lookback=3, entry_z=1.0, allow_short=False)
        outs = feed(s, [10.0, 10.0, 13.0])
        assert outs[-1].side == HOLD
        assert s._side == ""

    def test_short_cover_on_revert(self):
        s = ZScoreMeanReversion(lookback=3, entry_z=1.0, exit_z=0.6,
                                stop_z=10.0, allow_short=True)
        outs = feed(s, [10.0, 10.0, 13.0, 10.0])
        assert outs[2].side == SELL
        assert outs[3].side == EXIT
        assert "cover" in outs[3].reason
        assert s._side == ""

    def test_holding_short_when_not_reverted(self):
        s = ZScoreMeanReversion(lookback=3, entry_z=1.0, exit_z=0.1,
                                stop_z=10.0, allow_short=True)
        outs = feed(s, [10.0, 10.0, 13.0, 10.0])
        assert outs[2].side == SELL
        assert outs[3].side == HOLD
        assert outs[3].reason == "holding short"
        assert s._side == "SHORT"


class TestZScoreStateMgmt:
    def test_reset_clears_state(self):
        s = ZScoreMeanReversion(lookback=3, entry_z=1.0, exit_z=0.6, stop_z=10.0)
        feed(s, [10.0, 10.0, 7.0])
        assert s._side == "LONG" and len(s._closes) == 3
        s.reset()
        assert s._side == ""
        assert len(s._closes) == 0
        assert s._in_position is False
        assert s._entry_price == 0.0

    def test_on_fill_sell_from_flat_sets_short(self):
        s = ZScoreMeanReversion()
        s.on_fill("SELL", 1.0, 100.0)
        assert s._side == "SHORT"
        assert s._in_position is True

    def test_on_fill_sell_noop_when_already_in_position(self):
        s = ZScoreMeanReversion()
        s._in_position = True
        s._side = "LONG"
        s.on_fill("SELL", 1.0, 100.0)  # an exit fill of a long
        # must NOT flip a LONG into a SHORT
        assert s._side == "LONG"
        assert s._in_position is True

    def test_on_fill_buy_is_noop(self):
        s = ZScoreMeanReversion()
        s.on_fill("BUY", 1.0, 100.0)
        assert s._side == ""
        assert s._in_position is False

    def test_deque_bounded_to_lookback_plus_one(self):
        s = ZScoreMeanReversion(lookback=3, entry_z=99.0)  # never enters
        feed(s, [float(i) for i in range(20)])
        assert len(s._closes) == 4  # maxlen = lookback + 1


# =========================================================================== #
# RSIMeanReversion                                                            #
# =========================================================================== #
class TestRSIDefaults:
    def test_defaults(self):
        s = RSIMeanReversion()
        assert s.rsi_period == 2
        assert s.entry_low == 10.0
        assert s.entry_high == 90.0
        assert s.exit_level == 50.0
        assert s.vol_lookback == 20
        assert s.vol_floor == 0.0
        assert s.vol_cap == float("inf")
        assert s.stop_pct == 0.05
        assert s.allow_short is False
        assert s.name == "rsi-mr"

    def test_warmup_holds(self):
        s = RSIMeanReversion(rsi_period=2, vol_lookback=2)
        outs = feed(s, [100.0, 99.0])  # not enough for rsi(2)
        assert all(o.side == HOLD and o.reason == "warmup" for o in outs)


class TestRSILong:
    def test_buy_on_oversold_in_band(self):
        # steep drop -> rsi(2)=0 < entry_low=30; wide-open vol band by default
        s = RSIMeanReversion(rsi_period=2, entry_low=30.0, exit_level=50.0,
                             vol_lookback=2)
        outs = feed(s, [100.0, 99.0, 98.0])
        buy = outs[-1]
        assert buy.side == BUY
        assert buy.strength == 1.0
        assert buy.price == 98.0
        assert buy.metadata["rsi"] == pytest.approx(0.0)
        assert "rsi" in buy.metadata and "vol" in buy.metadata
        assert s._side == "LONG"
        assert s._entry_price == 98.0

    def test_exit_on_rsi_recovery(self):
        # enter oversold, then a strong up bar that lifts rsi >= exit_level
        s = RSIMeanReversion(rsi_period=2, entry_low=30.0, exit_level=50.0,
                             vol_lookback=2, stop_pct=0.99)  # disable stop
        outs = feed(s, [100.0, 99.0, 98.0, 200.0])
        assert outs[2].side == BUY
        assert outs[3].side == EXIT
        assert s._side == ""

    def test_stop_pct_exit_long(self):
        # entry at 98, drop to <= 98*(1-0.05)=93.1 -> stopped
        s = RSIMeanReversion(rsi_period=2, entry_low=30.0, exit_level=99.0,
                             vol_lookback=2, stop_pct=0.05)
        outs = feed(s, [100.0, 99.0, 98.0, 90.0])
        assert outs[2].side == BUY
        assert outs[3].side == EXIT
        assert "stop" in outs[3].reason
        assert s._side == ""

    def test_holding_long_no_exit(self):
        # entered, rsi still below exit and no stop hit
        s = RSIMeanReversion(rsi_period=2, entry_low=30.0, exit_level=99.0,
                             vol_lookback=2, stop_pct=0.99)
        outs = feed(s, [100.0, 99.0, 98.0, 98.5])
        assert outs[2].side == BUY
        assert outs[3].side == HOLD
        assert outs[3].reason == "holding long"
        assert s._side == "LONG"


class TestRSIVolGate:
    def test_vol_floor_blocks_entry(self):
        # impossibly high floor -> never in band -> no entry even when oversold
        s = RSIMeanReversion(rsi_period=2, entry_low=30.0, vol_lookback=2,
                             vol_floor=1.0)
        outs = feed(s, [100.0, 99.0, 98.0])
        assert outs[-1].side == HOLD
        assert outs[-1].reason == "flat"
        assert s._side == ""

    def test_vol_cap_blocks_entry(self):
        # near-zero cap -> realized vol exceeds it -> blocked
        s = RSIMeanReversion(rsi_period=2, entry_low=30.0, vol_lookback=2,
                             vol_cap=1e-12)
        outs = feed(s, [100.0, 99.0, 98.0])
        assert outs[-1].side == HOLD
        assert s._side == ""

    def test_in_band_allows_entry(self):
        # wide band [0, inf] -> oversold entry proceeds
        s = RSIMeanReversion(rsi_period=2, entry_low=30.0, vol_lookback=2,
                             vol_floor=0.0, vol_cap=float("inf"))
        outs = feed(s, [100.0, 99.0, 98.0])
        assert outs[-1].side == BUY

    def test_vol_gate_never_blocks_exit(self):
        # enter in a wide band, then tighten conceptually is not possible at
        # runtime; instead verify exit fires regardless of vol by using a
        # recovery bar. The gate only guards entries by construction.
        s = RSIMeanReversion(rsi_period=2, entry_low=30.0, exit_level=50.0,
                             vol_lookback=2, vol_floor=0.0, stop_pct=0.99)
        outs = feed(s, [100.0, 99.0, 98.0, 300.0])
        assert outs[2].side == BUY
        assert outs[3].side == EXIT


class TestRSIShort:
    def test_short_on_overbought_when_allowed(self):
        s = RSIMeanReversion(rsi_period=2, entry_low=10.0, entry_high=70.0,
                             exit_level=50.0, vol_lookback=2, allow_short=True)
        outs = feed(s, [100.0, 101.0, 102.0])
        sell = outs[-1]
        assert sell.side == SELL
        assert sell.metadata["rsi"] == pytest.approx(100.0)
        assert sell.price == 102.0
        assert s._side == "SHORT"

    def test_no_short_when_not_allowed(self):
        s = RSIMeanReversion(rsi_period=2, entry_high=70.0, vol_lookback=2,
                             allow_short=False)
        outs = feed(s, [100.0, 101.0, 102.0])
        assert outs[-1].side == HOLD
        assert s._side == ""

    def test_short_cover_on_rsi_drop(self):
        s = RSIMeanReversion(rsi_period=2, entry_low=10.0, entry_high=70.0,
                             exit_level=50.0, vol_lookback=2, allow_short=True,
                             stop_pct=0.99)
        # overbought entry, then a down bar pulls rsi <= exit_level -> cover
        outs = feed(s, [100.0, 101.0, 102.0, 50.0])
        assert outs[2].side == SELL
        assert outs[3].side == EXIT
        assert s._side == ""

    def test_short_stop_on_upside(self):
        # entry at 102, rise to >= 102*(1+0.05)=107.1 -> stopped cover
        s = RSIMeanReversion(rsi_period=2, entry_low=10.0, entry_high=70.0,
                             exit_level=1.0, vol_lookback=2, allow_short=True,
                             stop_pct=0.05)
        outs = feed(s, [100.0, 101.0, 102.0, 110.0])
        assert outs[2].side == SELL
        assert outs[3].side == EXIT
        assert "stop" in outs[3].reason
        assert s._side == ""

    def test_holding_short_no_exit(self):
        s = RSIMeanReversion(rsi_period=2, entry_low=10.0, entry_high=70.0,
                             exit_level=1.0, vol_lookback=2, allow_short=True,
                             stop_pct=0.99)
        outs = feed(s, [100.0, 101.0, 102.0, 102.5])
        assert outs[2].side == SELL
        assert outs[3].side == HOLD
        assert outs[3].reason == "holding short"
        assert s._side == "SHORT"


class TestRSIStateMgmt:
    def test_reset_clears(self):
        s = RSIMeanReversion(rsi_period=2, entry_low=30.0, vol_lookback=2)
        feed(s, [100.0, 99.0, 98.0])
        assert s._side == "LONG"
        s.reset()
        assert s._side == ""
        assert len(s._closes) == 0
        assert s._in_position is False
        assert s._entry_price == 0.0

    def test_on_fill_sell_from_flat_sets_short(self):
        s = RSIMeanReversion()
        s.on_fill("SELL", 1.0, 100.0)
        assert s._side == "SHORT"
        assert s._in_position is True

    def test_on_fill_sell_noop_when_in_position(self):
        s = RSIMeanReversion()
        s._in_position = True
        s._side = "LONG"
        s.on_fill("SELL", 1.0, 100.0)
        assert s._side == "LONG"
        assert s._in_position is True

    def test_on_fill_buy_is_noop(self):
        s = RSIMeanReversion()
        s.on_fill("BUY", 1.0, 100.0)
        assert s._side == ""
        assert s._in_position is False

    def test_buffer_sizing(self):
        s = RSIMeanReversion(rsi_period=2, vol_lookback=20)
        assert s._closes.maxlen == max(2, 20) * 3 + 5
