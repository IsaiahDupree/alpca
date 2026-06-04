"""
Honest evaluation harness: Sharpe / maxDD / buy-and-hold benchmark / OOS split / verdict.
"""

import math

import pytest

from alpca.backtest.evaluation import (
    buy_and_hold,
    evaluate,
    infer_periods_per_year,
    max_drawdown_of,
    sharpe_of,
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
