"""
Deep, deterministic tests for alpca/calibration/volatility.py.

Covers the pure/offline volatility math only (no network, no Alpaca):
  - compute_rolling_volatility: prior fallbacks, non-negativity, zero on flat,
    monotonicity vs. wildness, annualize on/off, bars_per_day override.
  - build_vol_series: keys == bar timestamps, all >= 0, prior on warm-up bars,
    and the sliding window AGREES with a direct compute_rolling_volatility
    recompute of the final window for several synthetic series.
  - _infer_bars_per_day: median-bars-per-ET-session-date logic.

All inputs are deterministic synthetic bars with real epoch-second timestamps.
RNG, where used, is seeded with a fixed value.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from alpca.calibration.volatility import (
    _DEFAULT_PRIOR,
    _TRADING_DAYS,
    _infer_bars_per_day,
    _window_bars,
    build_vol_series,
    compute_rolling_volatility,
)
from alpca.data.calendar import session_date

# --------------------------------------------------------------------------
# Tiny self-contained helpers (no imports from other tests/ files).
# --------------------------------------------------------------------------

# A weekday inside the hand-maintained calendar range, away from any holiday.
_DAILY_ANCHOR = datetime(2025, 3, 3, 14, 30, 0, tzinfo=timezone.utc)  # Mon ~09:30 ET
# An intraday anchor: a regular-session minute (14:30 UTC == 09:30 ET).
_INTRA_ANCHOR = datetime(2025, 3, 3, 14, 30, 0, tzinfo=timezone.utc)


def _epoch(dt: datetime) -> float:
    return dt.timestamp()


def daily_bars(closes):
    """One bar per consecutive ET session date (skips weekends/holidays so each
    bar lands on a distinct, tradeable-looking calendar date). 1 bar/day."""
    bars = []
    cur = _DAILY_ANCHOR
    i = 0
    while len(bars) < len(closes):
        # advance to a weekday (Mon-Fri); good enough — distinct ET dates is all
        # the inference needs.
        while cur.weekday() >= 5:
            cur = cur + timedelta(days=1)
        bars.append({"timestamp": _epoch(cur), "close": float(closes[len(bars)])})
        cur = cur + timedelta(days=1)
        i += 1
    return bars


def intraday_bars(closes, per_day, minutes=1):
    """`per_day` bars on each ET session date, spaced `minutes` apart starting at
    the regular open, rolling over to the next weekday once the day is full."""
    bars = []
    day = _INTRA_ANCHOR
    while day.weekday() >= 5:
        day = day + timedelta(days=1)
    k = 0
    while k < len(closes):
        for j in range(per_day):
            if k >= len(closes):
                break
            ts = _epoch(day + timedelta(minutes=minutes * j))
            bars.append({"timestamp": ts, "close": float(closes[k])})
            k += 1
        day = day + timedelta(days=1)
        while day.weekday() >= 5:
            day = day + timedelta(days=1)
    return bars


def geometric_walk(n, step, start=100.0):
    """Deterministic alternating up/down geometric path: close[i]/close[i-1] is
    exp(+step) then exp(-step) alternating, giving constant |log return| == step."""
    closes = [start]
    for i in range(1, n):
        sign = 1.0 if (i % 2 == 1) else -1.0
        closes.append(closes[-1] * math.exp(sign * step))
    return closes


def seeded_walk(n, vol, start=100.0, seed=12345):
    import random

    rng = random.Random(seed)
    closes = [start]
    for _ in range(1, n):
        r = rng.gauss(0.0, vol)
        closes.append(closes[-1] * math.exp(r))
    return closes


def population_stdev(xs):
    n = len(xs)
    mean = sum(xs) / n
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / n)


def direct_final_window_vol(bars, lookback_days, *, annualize, bars_per_day, prior):
    """Reference recompute that MUST equal build_vol_series' last value: replicate
    compute_rolling_volatility on the trailing window."""
    return compute_rolling_volatility(
        bars,
        lookback_days,
        annualize=annualize,
        bars_per_day=bars_per_day,
        prior=prior,
    )


# ==========================================================================
# compute_rolling_volatility — prior / degenerate paths
# ==========================================================================


@pytest.mark.parametrize("bars", [[], None])
def test_compute_empty_or_none_returns_prior(bars):
    # Source: `if not bars: return prior` — both [] and None are falsy.
    assert compute_rolling_volatility(bars) == _DEFAULT_PRIOR


def test_compute_custom_prior_on_empty():
    assert compute_rolling_volatility([], prior=0.42) == 0.42


def test_compute_single_bar_returns_prior():
    bars = daily_bars([100.0])
    assert compute_rolling_volatility(bars, bars_per_day=1.0) == _DEFAULT_PRIOR


def test_compute_two_bars_one_return_returns_prior():
    # Two bars => exactly 1 log return => len(rets) < 2 => prior.
    bars = daily_bars([100.0, 101.0])
    assert compute_rolling_volatility(bars, bars_per_day=1.0, prior=0.21) == 0.21


def test_compute_three_bars_two_returns_not_prior():
    # 3 bars => 2 returns => crosses the len(rets) >= 2 threshold.
    bars = daily_bars([100.0, 101.0, 100.5])
    v = compute_rolling_volatility(bars, bars_per_day=1.0, annualize=False)
    assert v != _DEFAULT_PRIOR
    assert v > 0.0


# ==========================================================================
# compute_rolling_volatility — non-negativity & flatness
# ==========================================================================


@pytest.mark.parametrize("price", [1.0, 50.0, 100.0, 9999.0])
def test_compute_flat_series_is_zero(price):
    bars = daily_bars([price] * 12)
    assert compute_rolling_volatility(bars, bars_per_day=1.0, annualize=False) == 0.0
    assert compute_rolling_volatility(bars, bars_per_day=1.0, annualize=True) == 0.0


@pytest.mark.parametrize("step", [0.001, 0.01, 0.05, 0.2])
def test_compute_nonnegative_geometric(step):
    bars = daily_bars(geometric_walk(30, step))
    v = compute_rolling_volatility(bars, bars_per_day=1.0)
    assert v >= 0.0


@pytest.mark.parametrize("seed", [1, 7, 12345, 99999])
def test_compute_nonnegative_random(seed):
    bars = daily_bars(seeded_walk(40, 0.02, seed=seed))
    v = compute_rolling_volatility(bars, bars_per_day=1.0)
    assert v >= 0.0
    assert math.isfinite(v)


# ==========================================================================
# compute_rolling_volatility — exact computed value (constant |return| path)
# ==========================================================================


@pytest.mark.parametrize("step", [0.01, 0.03, 0.07])
def test_compute_constant_abs_return_exact_unannualized(step):
    # geometric_walk gives log returns alternating +step,-step,+step...
    # population stdev of a zero-mean symmetric +/-step sequence == step
    # (when the count of +step equals the count of -step it is exactly step;
    # for an odd count the mean is tiny but nonzero — we use an even # of returns).
    # 11 closes => 10 returns => 5 up, 5 down => mean 0 => pstdev == step.
    bars = daily_bars(geometric_walk(11, step))
    v = compute_rolling_volatility(
        bars, lookback_days=999, bars_per_day=1.0, annualize=False
    )
    assert v == pytest.approx(step, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("step,bpd", [(0.01, 1.0), (0.02, 390.0), (0.05, 6.5)])
def test_compute_annualize_factor_exact(step, bpd):
    bars = daily_bars(geometric_walk(11, step))
    raw = compute_rolling_volatility(
        bars, lookback_days=999, bars_per_day=bpd, annualize=False
    )
    ann = compute_rolling_volatility(
        bars, lookback_days=999, bars_per_day=bpd, annualize=True
    )
    assert ann == pytest.approx(raw * math.sqrt(bpd * _TRADING_DAYS), rel=1e-12)


def test_compute_annualize_matches_manual_pstdev():
    closes = seeded_walk(15, 0.02, seed=2026)
    bars = daily_bars(closes)
    rets = [math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))]
    expected_raw = population_stdev(rets)
    got_raw = compute_rolling_volatility(
        bars, lookback_days=999, bars_per_day=1.0, annualize=False
    )
    assert got_raw == pytest.approx(expected_raw, rel=1e-12)


# ==========================================================================
# compute_rolling_volatility — monotonicity (wilder => higher)
# ==========================================================================


def test_compute_wilder_series_has_higher_vol():
    calm = daily_bars(geometric_walk(30, 0.005))
    wild = daily_bars(geometric_walk(30, 0.05))
    vc = compute_rolling_volatility(calm, bars_per_day=1.0, annualize=False)
    vw = compute_rolling_volatility(wild, bars_per_day=1.0, annualize=False)
    assert vw > vc
    assert vc > 0.0


@pytest.mark.parametrize(
    "lo,hi", [(0.002, 0.02), (0.01, 0.04), (0.005, 0.1)]
)
def test_compute_monotone_in_step(lo, hi):
    a = compute_rolling_volatility(
        daily_bars(geometric_walk(21, lo)), bars_per_day=1.0, annualize=False
    )
    b = compute_rolling_volatility(
        daily_bars(geometric_walk(21, hi)), bars_per_day=1.0, annualize=False
    )
    assert b > a


# ==========================================================================
# compute_rolling_volatility — annualize toggle & bars_per_day override
# ==========================================================================


def test_compute_annualize_true_ge_false_when_factor_ge_one():
    bars = daily_bars(geometric_walk(20, 0.02))
    raw = compute_rolling_volatility(bars, bars_per_day=1.0, annualize=False)
    ann = compute_rolling_volatility(bars, bars_per_day=1.0, annualize=True)
    # bpd*252 = 252 > 1 so annualized magnitude is larger for nonzero vol.
    assert ann > raw > 0.0


@pytest.mark.parametrize("bpd", [1.0, 6.5, 78.0, 390.0])
def test_compute_bars_per_day_scales_annualization(bpd):
    bars = daily_bars(geometric_walk(20, 0.02))
    raw = compute_rolling_volatility(bars, bars_per_day=bpd, annualize=False)
    ann = compute_rolling_volatility(bars, bars_per_day=bpd, annualize=True)
    assert ann == pytest.approx(raw * math.sqrt(bpd * _TRADING_DAYS), rel=1e-12)


def test_compute_unannualized_is_bpd_independent():
    # Raw (population) stdev of the SAME returns must not depend on bars_per_day.
    # bars_per_day also sizes the window (lookback_days*bpd), so use a lookback
    # large enough that both bpd values include the entire series of returns.
    bars = daily_bars(geometric_walk(20, 0.02))
    v1 = compute_rolling_volatility(
        bars, lookback_days=999, bars_per_day=1.0, annualize=False
    )
    v2 = compute_rolling_volatility(
        bars, lookback_days=999, bars_per_day=390.0, annualize=False
    )
    assert v1 == pytest.approx(v2, rel=1e-12)


def test_compute_window_truncates_to_lookback():
    # A long quiet tail then a wild prefix: a short lookback should only see the
    # quiet tail and report low vol; a long lookback sees the wild prefix too.
    wild = geometric_walk(6, 0.10)          # 6 wild closes
    quiet = [wild[-1]] * 8                   # 8 flat closes (zero returns)
    bars = daily_bars(wild + quiet)
    short = compute_rolling_volatility(
        bars, lookback_days=3, bars_per_day=1.0, annualize=False
    )
    long = compute_rolling_volatility(
        bars, lookback_days=999, bars_per_day=1.0, annualize=False
    )
    assert short == pytest.approx(0.0, abs=1e-12)
    assert long > short


# ==========================================================================
# compute_rolling_volatility — robustness to degenerate prices
# ==========================================================================


def test_compute_skips_nonpositive_close_pairs():
    # Source only forms a return when both closes > 0; a zero close breaks the
    # chain around it but the rest still yields returns.
    closes = [100.0, 0.0, 101.0, 102.0, 101.5, 103.0]
    bars = daily_bars(closes)
    v = compute_rolling_volatility(
        bars, lookback_days=999, bars_per_day=1.0, annualize=False
    )
    assert math.isfinite(v)
    assert v >= 0.0


def test_compute_extreme_magnitudes_finite():
    bars = daily_bars([1e-6, 2e-6, 1.5e-6, 3e6, 1e6, 2e6])
    v = compute_rolling_volatility(bars, bars_per_day=1.0, annualize=False)
    assert math.isfinite(v)
    assert v > 0.0


def test_compute_all_zero_closes_returns_prior():
    # Every close is 0 => no valid returns => prior.
    bars = daily_bars([0.0] * 6)
    assert compute_rolling_volatility(bars, bars_per_day=1.0) == _DEFAULT_PRIOR


def test_compute_idempotent_repeated_call():
    bars = daily_bars(seeded_walk(25, 0.02, seed=555))
    a = compute_rolling_volatility(bars, bars_per_day=1.0)
    b = compute_rolling_volatility(bars, bars_per_day=1.0)
    assert a == b


# ==========================================================================
# _infer_bars_per_day — median over distinct ET session dates
# ==========================================================================


def test_infer_single_date_returns_none():
    # All bars share one ET date => fewer than 2 distinct counts => None.
    bars = intraday_bars([100.0] * 5, per_day=5)
    assert _infer_bars_per_day(bars) is None


def test_infer_daily_one_per_date_is_one():
    bars = daily_bars([100.0] * 6)  # 6 distinct dates, 1 each
    assert _infer_bars_per_day(bars) == 1.0


@pytest.mark.parametrize("per_day", [3, 5, 10])
def test_infer_constant_intraday_cadence(per_day):
    # 4 full days at `per_day` each => median == per_day.
    bars = intraday_bars([100.0] * (per_day * 4), per_day=per_day)
    assert _infer_bars_per_day(bars) == float(per_day)


def test_infer_median_of_uneven_days():
    # Build dates with counts [2, 5, 5, 9] => median == (5+5)/2 == 5.0.
    days = []
    cur = _DAILY_ANCHOR
    for _ in range(4):
        while cur.weekday() >= 5:
            cur = cur + timedelta(days=1)
        days.append(cur)
        cur = cur + timedelta(days=1)
    counts = [2, 5, 5, 9]
    bars = []
    for day, n in zip(days, counts):
        for j in range(n):
            bars.append({"timestamp": _epoch(day + timedelta(minutes=j)), "close": 100.0})
    # sanity: all four days are distinct ET session dates
    assert len({session_date(b["timestamp"]) for b in bars}) == 4
    assert _infer_bars_per_day(bars) == 5.0


def test_infer_ignores_nonpositive_timestamps():
    good = daily_bars([100.0] * 3)  # 3 distinct dates
    bars = [{"timestamp": 0, "close": 100.0}, {"timestamp": -5, "close": 100.0}] + good
    # Only the 3 good bars (1 each) count => median 1.0.
    assert _infer_bars_per_day(bars) == 1.0


def test_infer_empty_returns_none():
    assert _infer_bars_per_day([]) is None


def test_infer_missing_timestamp_key_treated_as_zero():
    # b.get("timestamp", 0) => no key means skipped. Mix one keyless bar in.
    good = daily_bars([100.0] * 2)
    bars = [{"close": 100.0}] + good
    # 2 good distinct dates 1-each => median 1.0.
    assert _infer_bars_per_day(bars) == 1.0


# ==========================================================================
# _window_bars — window/bpd resolution (helper used by both public fns)
# ==========================================================================


def test_window_bars_override_takes_precedence():
    bars = daily_bars([100.0] * 10)
    window, bpd = _window_bars(bars, lookback_days=2.0, bars_per_day=5.0)
    assert bpd == 5.0
    assert window == max(2, int(round(2.0 * 5.0)))  # == 10


def test_window_bars_min_two():
    bars = daily_bars([100.0] * 10)
    window, bpd = _window_bars(bars, lookback_days=0.0, bars_per_day=1.0)
    assert window == 2  # floored at 2


def test_window_bars_inferred_when_no_override():
    bars = intraday_bars([100.0] * 12, per_day=3)  # 4 days, 3 each => bpd 3
    window, bpd = _window_bars(bars, lookback_days=2.0, bars_per_day=None)
    assert bpd == 3.0
    assert window == 6


def test_window_bars_falls_back_to_len_when_uninferrable():
    # Single session date => infer None => fallback float(len(bars)).
    bars = intraday_bars([100.0] * 4, per_day=4)
    window, bpd = _window_bars(bars, lookback_days=1.0, bars_per_day=None)
    assert bpd == float(len(bars))  # 4.0
    assert window == 4


# ==========================================================================
# build_vol_series — keys, non-negativity, prior warm-up
# ==========================================================================


def test_build_keys_equal_bar_timestamps():
    bars = daily_bars(seeded_walk(20, 0.02, seed=1))
    series = build_vol_series(bars, bars_per_day=1.0)
    assert list(series.keys()) == [float(b["timestamp"]) for b in bars]


def test_build_all_values_nonnegative_and_finite():
    bars = daily_bars(seeded_walk(50, 0.03, seed=2))
    series = build_vol_series(bars, bars_per_day=1.0)
    assert all(v >= 0.0 and math.isfinite(v) for v in series.values())


def test_build_warmup_bars_get_prior():
    # First bar: 0 returns -> prior. Second bar: 1 return (window has 1) -> prior.
    # Third bar onward: >= 2 returns in window -> computed.
    closes = seeded_walk(8, 0.02, seed=3)
    bars = daily_bars(closes)
    series = build_vol_series(bars, bars_per_day=1.0, prior=0.123, annualize=False)
    vals = list(series.values())
    assert vals[0] == 0.123
    assert vals[1] == 0.123
    assert vals[2] != 0.123  # has 2 returns now


def test_build_flat_series_all_prior_then_zero():
    bars = daily_bars([100.0] * 6)
    series = build_vol_series(bars, bars_per_day=1.0, prior=0.5, annualize=False)
    vals = list(series.values())
    # bar0: no return -> prior; bar1: r=0 (1 return) -> prior;
    # bar2+: >=2 zero returns -> variance 0.
    assert vals[0] == 0.5
    assert vals[1] == 0.5
    assert all(v == 0.0 for v in vals[2:])


def test_build_empty_returns_empty_dict():
    assert build_vol_series([], bars_per_day=1.0) == {}


def test_build_custom_prior_used_on_short_input():
    bars = daily_bars([100.0, 101.0])  # at most 1 return -> all prior
    series = build_vol_series(bars, bars_per_day=1.0, prior=0.77)
    assert set(series.values()) == {0.77}


# ==========================================================================
# build_vol_series — sliding window AGREES with direct recompute
# (the core invariant of the O(n) implementation)
# ==========================================================================


@pytest.mark.parametrize("seed", [1, 42, 2026])
@pytest.mark.parametrize("lookback_days,bpd", [(5.0, 1.0), (3.0, 1.0)])
def test_build_last_value_matches_direct_recompute_daily(seed, lookback_days, bpd):
    closes = seeded_walk(40, 0.025, seed=seed)
    bars = daily_bars(closes)
    series = build_vol_series(
        bars, lookback_days=lookback_days, bars_per_day=bpd, annualize=True
    )
    last_ts = float(bars[-1]["timestamp"])
    direct = direct_final_window_vol(
        bars, lookback_days, annualize=True, bars_per_day=bpd, prior=_DEFAULT_PRIOR
    )
    assert series[last_ts] == pytest.approx(direct, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("per_day,lookback_days", [(3, 2.0), (5, 1.0), (10, 1.0)])
def test_build_last_value_matches_direct_recompute_intraday(per_day, lookback_days):
    closes = seeded_walk(per_day * 6, 0.01, seed=777)
    bars = intraday_bars(closes, per_day=per_day)
    bpd = float(per_day)
    series = build_vol_series(
        bars, lookback_days=lookback_days, bars_per_day=bpd, annualize=True
    )
    last_ts = float(bars[-1]["timestamp"])
    direct = direct_final_window_vol(
        bars, lookback_days, annualize=True, bars_per_day=bpd, prior=_DEFAULT_PRIOR
    )
    assert series[last_ts] == pytest.approx(direct, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("annualize", [True, False])
def test_build_last_value_matches_direct_geometric(annualize):
    bars = daily_bars(geometric_walk(30, 0.03))
    series = build_vol_series(
        bars, lookback_days=4.0, bars_per_day=1.0, annualize=annualize
    )
    last_ts = float(bars[-1]["timestamp"])
    direct = direct_final_window_vol(
        bars, 4.0, annualize=annualize, bars_per_day=1.0, prior=_DEFAULT_PRIOR
    )
    assert series[last_ts] == pytest.approx(direct, rel=1e-9, abs=1e-12)


def test_build_every_window_position_matches_compute_on_prefix():
    # Stronger invariant: for each bar index i (>=2), build's value equals a
    # direct compute over the same trailing window of the prefix bars[:i+1].
    closes = seeded_walk(18, 0.02, seed=909)
    bars = daily_bars(closes)
    lookback, bpd = 3.0, 1.0
    series = build_vol_series(
        bars, lookback_days=lookback, bars_per_day=bpd, annualize=True
    )
    vals = list(series.values())
    for i in range(len(bars)):
        prefix = bars[: i + 1]
        expected = compute_rolling_volatility(
            prefix, lookback_days=lookback, bars_per_day=bpd, annualize=True
        )
        # compute returns prior when <2 returns; build also stores prior there.
        assert vals[i] == pytest.approx(expected, rel=1e-9, abs=1e-12)


# ==========================================================================
# build_vol_series — annualize toggle consistency & determinism
# ==========================================================================


def test_build_annualize_factor_consistent():
    bars = daily_bars(geometric_walk(25, 0.02))
    raw = build_vol_series(bars, lookback_days=999, bars_per_day=4.0, annualize=False)
    ann = build_vol_series(bars, lookback_days=999, bars_per_day=4.0, annualize=True)
    factor = math.sqrt(4.0 * _TRADING_DAYS)
    for ts in raw:
        if raw[ts] == _DEFAULT_PRIOR:
            # warm-up bars store prior unscaled in both series
            assert ann[ts] == _DEFAULT_PRIOR
        else:
            assert ann[ts] == pytest.approx(raw[ts] * factor, rel=1e-12)


def test_build_deterministic_repeat():
    bars = daily_bars(seeded_walk(30, 0.02, seed=4321))
    a = build_vol_series(bars, bars_per_day=1.0)
    b = build_vol_series(bars, bars_per_day=1.0)
    assert a == b


def test_build_handles_nonpositive_close_without_crash():
    closes = [100.0, 0.0, 101.0, 102.0, 100.5, 101.0, 103.0]
    bars = daily_bars(closes)
    series = build_vol_series(bars, bars_per_day=1.0, annualize=False)
    assert len(series) == len(bars)
    assert all(math.isfinite(v) and v >= 0.0 for v in series.values())


def test_build_missing_timestamp_uses_zero_key():
    # A bar without 'timestamp' => key 0.0 via b.get("timestamp", 0).
    bars = [{"close": 100.0}] + daily_bars([101.0, 102.0, 101.5])
    series = build_vol_series(bars, bars_per_day=1.0)
    assert 0.0 in series
