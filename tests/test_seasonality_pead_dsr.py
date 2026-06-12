"""Invariants for seasonality sleeves, PEAD, the earnings layer, and the Deflated Sharpe Ratio."""

import math

from alpca.backtest.evaluation import (
    deflated_sharpe_ratio, expected_max_sharpe, probabilistic_sharpe_ratio, _norm_ppf)
from alpca.backtest.pead import backtest_pead
from alpca.backtest.seasonality import (
    FOMC_ANNOUNCEMENTS, backtest_seasonal, pre_fomc_position, turn_of_month_position)
from alpca.data.earnings import _epoch


# ---- DSR / PSR ----
def _strong_equity(seed=0, n=800):
    import random
    rng = random.Random(seed)
    eq = [100_000.0]
    for _ in range(n):
        eq.append(eq[-1] * (1 + rng.gauss(0.0008, 0.01)))
    return eq


def test_norm_ppf_inverts_cdf():
    assert abs(_norm_ppf(0.975) - 1.959964) < 1e-3
    assert abs(_norm_ppf(0.5)) < 1e-6


def test_psr_high_for_strong_low_for_noise():
    import random
    rng = random.Random(1)
    noise = [100_000.0]
    for _ in range(800):
        noise.append(noise[-1] * (1 + rng.gauss(0.0, 0.01)))
    assert probabilistic_sharpe_ratio(_strong_equity()) > 0.9
    assert probabilistic_sharpe_ratio(noise) < 0.8


def test_dsr_deflates_with_more_trials():
    eq = _strong_equity()
    few = deflated_sharpe_ratio(eq, n_trials=2, sharpe_variance=0.001)
    many = deflated_sharpe_ratio(eq, n_trials=200, sharpe_variance=0.001)
    assert many <= few                      # more trials -> harder to be significant
    assert 0.0 <= many <= 1.0 and 0.0 <= few <= 1.0


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(100, 0.001) > expected_max_sharpe(5, 0.001) > 0


# ---- seasonality ----
def _spy_like(n=600):
    import random
    rng = random.Random(3)
    base = 1_600_000_000  # epoch; ~daily steps
    bars, p = [], 100.0
    for i in range(n):
        p *= (1 + rng.gauss(0.0004, 0.01))
        bars.append({"timestamp": base + i * 86400, "close": p})
    return bars


def test_turn_of_month_is_binary_and_partial_exposure():
    bars = _spy_like()
    pos = turn_of_month_position(bars, days_before=4, days_after=3)
    assert set(pos) <= {0.0, 1.0}
    frac = sum(pos) / len(pos)
    assert 0.1 < frac < 0.7                 # in-market only around month turns


def test_pre_fomc_position_sparse_and_binary():
    bars = _spy_like()
    pos = pre_fomc_position(bars)
    assert set(pos) <= {0.0, 1.0}
    assert sum(pos) <= len(FOMC_ANNOUNCEMENTS) + 1   # at most ~one day per announcement


def test_backtest_seasonal_runs_finite():
    bars = _spy_like()
    r = backtest_seasonal(bars, turn_of_month_position(bars), name="tom")
    assert all(math.isfinite(e) and e > 0 for e in r.equity_curve)
    assert 0.0 <= r.exposure <= 1.0


# ---- PEAD ----
def test_pead_long_short_legs_run():
    import random
    rng = random.Random(4)
    base = 1_600_000_000
    bars_by, events_by = {}, {}
    for j in range(6):
        p, bars = 100.0, []
        for i in range(400):
            p *= (1 + rng.gauss(0.0003, 0.012))
            bars.append({"timestamp": base + i * 86400, "close": p})
        bars_by[f"S{j}"] = bars
        # one surprise event mid-series
        events_by[f"S{j}"] = [{"date": base + 150 * 86400, "surprise_pct": 5.0 if j % 2 else -5.0}]
    for leg in ("long", "short", "both"):
        r = backtest_pead(bars_by, events_by, hold=20, entry_thr=2.0, leg=leg)
        assert all(math.isfinite(e) and e > 0 for e in r.equity_curve)
        assert r.n_events_used == 6


def test_pead_skips_events_outside_price_window():
    # an event BEFORE the bars start must be ignored (else it piles in at bar 0)
    base = 1_600_000_000
    bars = [{"timestamp": base + i * 86400, "close": 100.0 + i} for i in range(300)]
    bars_by = {"A": bars, "B": bars, "C": bars}
    pre = base - 500 * 86400          # well before ts[0]
    post = base + 5000 * 86400        # well after ts[-1]
    events = {"A": [{"date": pre, "surprise_pct": 9.0}],     # outside -> skipped
              "B": [{"date": post, "surprise_pct": 9.0}],    # outside -> skipped
              "C": [{"date": base + 100 * 86400, "surprise_pct": 9.0}]}  # inside -> used
    r = backtest_pead(bars_by, events, hold=20, entry_thr=2.0, leg="both")
    assert r.n_events_used == 1


def test_pead_threshold_filters_events():
    base = 1_600_000_000
    bars = [{"timestamp": base + i * 86400, "close": 100.0 + i} for i in range(300)]
    bars_by = {"A": bars, "B": bars, "C": bars}            # backtest needs >=3 symbols
    events = {"A": [{"date": base + 100 * 86400, "surprise_pct": 1.0}],   # below thr 2.0 -> skipped
              "B": [{"date": base + 100 * 86400, "surprise_pct": 9.0}],   # above -> used
              "C": [{"date": base + 120 * 86400, "surprise_pct": -8.0}]}  # above (abs) -> used
    r = backtest_pead(bars_by, events, hold=20, entry_thr=2.0, leg="both")
    assert r.n_events_used == 2


# ---- earnings parsing ----
def test_earnings_epoch_parsing():
    assert _epoch("01/15/2024", "%m/%d/%Y") is not None
    assert _epoch("2024-01-15", "%Y-%m-%d") is not None
    assert _epoch("garbage", "%Y-%m-%d") is None
