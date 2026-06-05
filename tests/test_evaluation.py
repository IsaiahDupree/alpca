"""
Honest evaluation harness: Sharpe / maxDD / buy-and-hold benchmark / OOS split / verdict.
"""

import math

import pytest

from alpca.backtest.evaluation import (
    beta_alpha,
    buy_and_hold,
    evaluate,
    infer_periods_per_year,
    information_ratio,
    max_drawdown_of,
    segment_sharpes,
    sharpe_of,
    sharpe_pvalue,
    sharpe_tstat,
    sortino_of,
    vol_of,
)
from alpca.data.bars import synthetic_bars

DAY = 86400
T0 = 1_700_000_000


def _daily(closes, start=T0):
    return [{"open": c, "high": c + 0.5, "low": c - 0.5, "close": c, "volume": 1000,
             "timestamp": start + i * DAY} for i, c in enumerate(closes)]


# ------------------------------------------------------------------ primitives
def test_maxdd_of_monotonic_up_is_zero():
    assert max_drawdown_of([100, 101, 102, 103]) == 0.0


def test_maxdd_of_known_drop():
    # 100 -> 120 -> 90 : peak 120, trough 90 -> -25%
    assert max_drawdown_of([100, 120, 90, 100]) == pytest.approx(-0.25)


def test_sharpe_zero_on_flat_equity():
    assert sharpe_of([100, 100, 100, 100], 252) == 0.0


def test_sharpe_positive_on_steady_gains():
    eq = [100 * (1.01 ** i) for i in range(50)]   # steady +1%/period
    assert sharpe_of(eq, 252) > 0


@pytest.mark.parametrize("closes,exp_ret", [
    ([100, 110], 0.10),
    ([100, 200], 1.0),
    ([100, 90], -0.10),
])
def test_buy_and_hold_return(closes, exp_ret):
    bh = buy_and_hold(_daily(closes), 252)
    assert bh.total_return == pytest.approx(exp_ret, rel=1e-6)


def test_buy_and_hold_degenerate():
    assert buy_and_hold([], 252).total_return == 0.0
    assert buy_and_hold(_daily([100]), 252).total_return == 0.0


def test_infer_ppy_daily():
    bars = _daily([100 + i for i in range(40)])
    ppy = infer_periods_per_year(bars)
    assert ppy == pytest.approx(252, rel=0.01)   # 1 bar / session-date -> 252


# ------------------------------------------------------------------ evaluate()
def test_evaluate_returns_report_with_benchmark():
    bars = synthetic_bars("X", n=400, seed=3)
    rep = evaluate("donchian", bars, periods_per_year=252)
    assert rep.name == "donchian"
    assert isinstance(rep.beats_return, bool)
    assert isinstance(rep.verdict, str) and rep.verdict
    # the benchmark fields are populated and self-consistent
    assert rep.excess_return == pytest.approx(rep.strat_return - rep.bh_return, abs=1e-9)


def test_evaluate_flags_underperformance_as_beta():
    # a strong steady uptrend: a sometimes-flat strategy can't beat always-long B&H
    bars = _daily([100 + i for i in range(300)])
    rep = evaluate("donchian", bars, periods_per_year=252)
    assert rep.bh_return > 0
    if not rep.beats_return:
        assert "BETA" in rep.verdict or "RISK-REDUCED" in rep.verdict


def test_evaluate_exposure_in_unit_interval():
    bars = synthetic_bars("X", n=300, seed=1)
    rep = evaluate("supertrend", bars, periods_per_year=252)
    assert 0.0 <= rep.exposure <= 1.0


def test_evaluate_oos_split_runs():
    bars = synthetic_bars("X", n=400, seed=7)
    rep = evaluate("ema-momentum", bars, periods_per_year=252, oos_frac=0.3)
    # both halves produced a number; verdict references OOS consistency logic
    assert isinstance(rep.is_return, float) and isinstance(rep.oos_return, float)
    assert isinstance(rep.oos_beats_bh, bool)


# -------------------------------------------------- new rigour: risk metrics
def _eq(rets, start=100.0):
    e = [start]
    for r in rets:
        e.append(e[-1] * (1 + r))
    return e


def test_sortino_zero_on_flat():
    assert sortino_of([100, 100, 100, 100], 252) == 0.0


def test_sortino_positive_on_gains():
    assert sortino_of(_eq([0.01] * 30 + [-0.005] * 5), 252) > 0


def test_vol_of_scales_with_volatility():
    calm = vol_of(_eq([0.001 * math.sin(i) for i in range(100)]), 252)
    wild = vol_of(_eq([0.02 * math.sin(i) for i in range(100)]), 252)
    assert wild > calm > 0


# -------------------------------------------------- significance
def test_sharpe_tstat_grows_with_sample_size():
    short = _eq([0.01, -0.005] * 10)
    long = _eq([0.01, -0.005] * 100)
    assert abs(sharpe_tstat(long)) > abs(sharpe_tstat(short))


def test_pvalue_low_for_strong_steady_signal():
    eq = _eq([0.01] * 200)            # relentless gains -> clearly not noise
    assert sharpe_pvalue(eq) < 0.05


def test_pvalue_high_for_noise():
    eq = _eq([0.02 * math.sin(i / 1.7) for i in range(200)])  # zero-mean oscillation
    assert sharpe_pvalue(eq) > 0.05


# -------------------------------------------------- benchmark-relative
def test_beta_alpha_recovers_2x():
    bench_r = [0.01 * math.sin(i / 3.0) for i in range(120)]
    strat_r = [2.0 * x for x in bench_r]
    beta, alpha = beta_alpha(_eq(strat_r), _eq(bench_r), 252)
    assert beta == pytest.approx(2.0, abs=0.05)
    assert alpha == pytest.approx(0.0, abs=0.02)


def test_information_ratio_positive_when_beating_benchmark():
    bench_r = [0.005 * math.sin(i / 4.0) for i in range(150)]
    strat_r = [x + 0.001 for x in bench_r]   # steady excess over benchmark
    assert information_ratio(_eq(strat_r), _eq(bench_r), 252) > 0


# -------------------------------------------------- stability + report wiring
def test_segment_sharpes_returns_k_values():
    eq = _eq([0.01 * math.sin(i / 5.0) for i in range(200)])
    assert len(segment_sharpes(eq, 252, k=4)) == 4


def test_evaluate_populates_rigour_fields():
    bars = synthetic_bars("X", n=400, seed=2)
    rep = evaluate("supertrend", bars, periods_per_year=252)
    for attr in ("beta", "alpha", "info_ratio", "sharpe_tstat", "sharpe_pvalue",
                 "strat_sortino", "strat_vol", "calmar"):
        assert isinstance(getattr(rep, attr), float)
    assert isinstance(rep.significant, bool) and isinstance(rep.stable, bool)
    assert isinstance(rep.segment_sharpes, list)
    assert 0.0 <= rep.sharpe_pvalue <= 1.0
