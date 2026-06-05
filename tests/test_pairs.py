"""
Market-neutral pairs / spread mean-reversion backtester.
"""

import math

import pytest

from alpca.backtest.pairs import (
    adf_stat,
    align,
    backtest_pairs,
    kalman_spread,
    mean_reversion_stats,
    screen_pairs,
    walkforward_pairs,
    _hedge_ratio,
)

DAY = 86400
T0 = 1_700_000_000


def _bars(closes, start=T0, step=DAY):
    return [{"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1000,
             "timestamp": start + i * step} for i, c in enumerate(closes)]


def test_align_inner_joins_on_timestamp():
    a = _bars([10, 11, 12])
    b = _bars([20, 21, 22])
    rows = align(a, b)
    assert len(rows) == 3
    assert rows[0] == (T0, 10.0, 20.0)


def test_align_drops_unmatched_timestamps():
    a = _bars([10, 11, 12])
    b = [dict(bb, timestamp=bb["timestamp"] + 1) for bb in _bars([20, 21, 22])]  # shifted
    assert align(a, b) == []


def test_hedge_ratio_recovers_slope():
    # log(a) = 2*log(b) + c  -> hedge ~ 2
    lb = [math.log(b) for b in range(10, 60)]
    la = [2 * x + 0.5 for x in lb]
    assert _hedge_ratio(la, lb) == pytest.approx(2.0, rel=1e-6)


def _cointegrated(n=600):
    # a DOMINANT common trend (so the legs are positively correlated, hedge ~1) plus a
    # SMALLER mean-reverting spread deviation -> a real pair.
    common = [100.0 + 25.0 * math.sin(i / 60.0) for i in range(n)]
    dev = [1.5 * math.sin(i / 8.0) for i in range(n)]
    return [common[i] + dev[i] for i in range(n)], [common[i] - dev[i] for i in range(n)]


def test_cointegrated_pair_is_profitable():
    a, b = _cointegrated()
    res = backtest_pairs(_bars(a), _bars(b), lookback=40, entry_z=1.2, exit_z=0.3, cost_bps=0.0)
    assert res.hedge > 0                 # positively-correlated legs
    assert res.n_trades > 0
    assert res.total_return > 0          # the spread edge pays


def test_no_trades_when_series_too_short():
    res = backtest_pairs(_bars([1, 2, 3]), _bars([1, 2, 3]), lookback=60)
    assert res.n_trades == 0 and res.total_return == 0.0


def test_costs_reduce_return():
    a, b = _cointegrated()
    free = backtest_pairs(_bars(a), _bars(b), lookback=40, entry_z=1.2, exit_z=0.3, cost_bps=0.0)
    costly = backtest_pairs(_bars(a), _bars(b), lookback=40, entry_z=1.2, exit_z=0.3, cost_bps=20.0)
    assert costly.total_return < free.total_return
    assert free.n_trades == costly.n_trades   # cost doesn't change the signals


def test_equity_curve_length_matches_aligned_bars():
    a = _bars([100 + math.sin(i) for i in range(200)])
    b = _bars([100 + math.cos(i) for i in range(200)])
    res = backtest_pairs(a, b, lookback=30)
    assert len(res.equity_curve) == 200


# ------------------------------------------------- cointegration / half-life
def test_mean_reversion_stats_detects_reverting_spread():
    spread = [math.sin(i / 5.0) for i in range(200)]   # oscillates around 0 -> reverts
    lam, hl = mean_reversion_stats(spread)
    assert lam < 0 and math.isfinite(hl) and hl > 0


def test_mean_reversion_stats_rejects_trend():
    spread = [float(i) for i in range(200)]             # monotonic -> not mean-reverting
    _, hl = mean_reversion_stats(spread)
    assert hl == float("inf")


def test_screen_pairs_finds_the_cointegrated_one():
    n = 400
    common = [100.0 + 25.0 * math.sin(i / 60.0) for i in range(n)]
    dev = [2.0 * math.sin(i / 4.0) for i in range(n)]    # FAST-reverting spread (period ~25)
    A = _bars([common[i] + dev[i] for i in range(n)])    # A,B cointegrated
    B = _bars([common[i] - dev[i] for i in range(n)])
    C = _bars([100.0 + 0.1 * i for i in range(n)])        # C independent trend
    found = screen_pairs(["A", "B", "C"], {"A": A, "B": B, "C": C},
                         min_overlap=100, max_half_life=120, min_half_life=2)
    assert any({r["a"], r["b"]} == {"A", "B"} for r in found)


# ----------------------------------------------------------- walk-forward
def test_walkforward_trades_oos_and_is_profitable_on_cointegrated_universe():
    n = 500
    common = [100.0 + 25.0 * math.sin(i / 70.0) for i in range(n)]

    def leg(amp, ph, sign):
        return _bars([common[i] + sign * amp * math.sin(i / 5.0 + ph) for i in range(n)])
    bars = {"A1": leg(2.0, 0, 1), "B1": leg(2.0, 0, -1),
            "A2": leg(1.5, 1, 1), "B2": leg(1.5, 1, -1),
            "A3": leg(2.5, 2, 1), "B3": leg(2.5, 2, -1)}
    res = walkforward_pairs(bars, train=80, test=30, top_n=3, max_half_life=80,
                            min_half_life=2, cost_bps=0.0)
    assert res.n_windows > 1          # rolled forward several times
    assert res.n_oos_bars > 0
    assert res.total_return > 0       # OOS-traded cointegrated pairs pay


def test_walkforward_safe_when_too_short():
    bars = {f"S{k}": _bars([100 + k + i for i in range(50)]) for k in range(4)}
    res = walkforward_pairs(bars, train=80, test=30)
    assert res.n_windows == 0 and res.total_return == 0.0


# ----------------------------------------------- ADF cointegration test
def test_adf_strongly_negative_for_stationary():
    # a strongly mean-reverting AR(1) (phi=0.2) -> DF stat very negative
    y = [0.0]
    for i in range(1, 300):
        y.append(0.2 * y[-1] + math.sin(i * 1.7))
    assert adf_stat(y) < -2.86          # rejects unit root -> mean-reverting


def test_adf_near_zero_for_trend():
    trend = [0.5 * i for i in range(200)]         # non-stationary
    assert adf_stat(trend) > -2.86


def test_adf_screen_is_stricter_than_half_life():
    # build a small universe; the ADF filter must pass <= the half-life-only screen
    n = 400
    common = [100.0 + 25.0 * math.sin(i / 60.0) for i in range(n)]
    syms, bars = [], {}
    for k in range(6):
        dev = [(1.5 + 0.3 * k) * math.sin(i / (4.0 + k) + k) for i in range(n)]
        bars[f"A{k}"] = _bars([common[i] + dev[i] for i in range(n)])
        bars[f"B{k}"] = _bars([common[i] - dev[i] for i in range(n)])
        syms += [f"A{k}", f"B{k}"]
    loose = screen_pairs(syms, bars, min_overlap=200, max_half_life=60)
    strict = screen_pairs(syms, bars, min_overlap=200, max_half_life=60, max_adf=-3.0)
    assert len(strict) <= len(loose)
    assert all("adf" in r and r["adf"] < -3.0 for r in strict)


# ----------------------------------------------- Kalman dynamic hedge
def test_kalman_converges_to_true_hedge():
    lb = [math.log(10 + i) for i in range(300)]
    la = [2.0 * x + 0.1 for x in lb]              # a = 2*b + const in logs
    betas, innov, sd = kalman_spread(la, lb)
    assert len(betas) == len(innov) == len(sd) == 300
    assert betas[-1] == pytest.approx(2.0, abs=0.25)   # tracks the true hedge


def test_backtest_pairs_kalman_runs_on_cointegrated():
    a, b = _cointegrated()
    res = backtest_pairs(_bars(a), _bars(b), lookback=30, entry_z=1.5, exit_z=0.3,
                         cost_bps=0.0, use_kalman=True)
    assert len(res.equity_curve) == len(a)
    assert math.isfinite(res.total_return)
