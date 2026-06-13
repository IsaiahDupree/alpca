"""Invariants for EAR-PEAD (earnings-announcement-return drift)."""

import math
import random

from alpca.backtest.ear_pead import backtest_ear_pead, _ear_signal_events


def _drift_book(n_sym=8, n_days=400, seed=0):
    """Synthetic universe: each symbol has one earnings event mid-series with a sharp 3-day
    reaction (EAR) followed by a continuing drift in the SAME direction (so high-EAR drifts up).
    Plus a market factor so a beta hedge has something to remove."""
    rng = random.Random(seed)
    base = 1_600_000_000
    mkt = [0.0]
    for _ in range(n_days):
        mkt.append(rng.gauss(0.0005, 0.008))     # common market factor
    bars_by, events_by, bench = {}, {}, []
    pb = 100.0
    for i in range(n_days):
        pb *= (1 + mkt[i + 1])
        bench.append({"timestamp": base + i * 86400, "open": pb, "high": pb, "low": pb,
                      "close": pb, "volume": 1e6})
    for j in range(n_sym):
        p, bars = 100.0, []
        ev_day = 150 + j        # stagger events
        up = (j % 2 == 0)       # half beat (EAR up + drift up), half miss
        for i in range(n_days):
            r = mkt[i + 1] * 1.0
            if ev_day <= i < ev_day + 3:          # 3-day reaction window
                r += (0.03 if up else -0.03)
            elif ev_day + 3 <= i < ev_day + 30:   # continuing drift
                r += (0.002 if up else -0.002)
            p *= (1 + r)
            o = p / (1 + r)
            bars.append({"timestamp": base + i * 86400, "open": o, "high": max(o, p),
                         "low": min(o, p), "close": p, "volume": 1e6})
        bars_by[f"S{j}"] = bars
        events_by[f"S{j}"] = [{"date": base + ev_day * 86400, "surprise_pct": 0.0}]
    return bars_by, events_by, bench


def test_ear_signal_measures_announcement_return_and_skips_outside_window():
    bars_by, events_by, _ = _drift_book()
    tagged = _ear_signal_events(bars_by, events_by, ear_window=3, skip_after_ear=1)
    # every symbol has exactly one in-window event with a finite EAR
    assert all(len(v) == 1 and math.isfinite(v[0]["ear"]) for v in tagged.values())
    # beats (even j) have positive EAR, misses (odd j) negative
    assert tagged["S0"][0]["ear"] > 0 and tagged["S1"][0]["ear"] < 0


def test_long_mode_skips_low_ear_events():
    bars_by, events_by, bench = _drift_book()
    lo = backtest_ear_pead(bars_by, events_by, hold=25, entry_thr=2.0, mode="long", bench_bars=bench)
    ne = backtest_ear_pead(bars_by, events_by, hold=25, entry_thr=2.0, mode="neutral", bench_bars=bench)
    assert lo.n_events_used < ne.n_events_used     # long takes only the high-EAR (beats); neutral takes both
    assert lo.total_return > 0                      # the built-in long drift is harvestable


def test_beta_hedge_removes_market_beta():
    bars_by, events_by, bench = _drift_book()
    hed = backtest_ear_pead(bars_by, events_by, hold=25, entry_thr=2.0, mode="beta_hedged", bench_bars=bench)
    assert hed.beta != 0.0                          # a beta was estimated and removed
    assert all(math.isfinite(e) and e > 0 for e in hed.equity_curve)
    assert len(hed.dates) == len(hed.daily_returns)  # dates aligned with returns (regime breakdown)


def test_trailing_hedge_no_lookahead_runs():
    # the trailing-beta hedge (audit's no-lookahead variant) runs and stays finite
    bars_by, events_by, bench = _drift_book()
    full = backtest_ear_pead(bars_by, events_by, hold=25, entry_thr=2.0, mode="beta_hedged",
                             bench_bars=bench, hedge_window=0)
    trail = backtest_ear_pead(bars_by, events_by, hold=25, entry_thr=2.0, mode="beta_hedged",
                              bench_bars=bench, hedge_window=60)
    assert all(math.isfinite(e) and e > 0 for e in trail.equity_curve)
    assert trail.beta != full.beta                  # trailing avg-beta differs from full-sample beta


def test_entry_threshold_filters_events():
    bars_by, events_by, bench = _drift_book()
    lowthr = backtest_ear_pead(bars_by, events_by, hold=25, entry_thr=1.0, mode="long", bench_bars=bench)
    highthr = backtest_ear_pead(bars_by, events_by, hold=25, entry_thr=10.0, mode="long", bench_bars=bench)
    assert highthr.n_events_used <= lowthr.n_events_used


def test_too_few_symbols_returns_empty():
    r = backtest_ear_pead({"A": []}, {"A": []}, mode="long")
    assert r.n_events_used == 0 and r.equity_curve == [100_000.0]
