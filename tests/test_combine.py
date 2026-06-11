"""Invariants for the portfolio combiner (alpca/backtest/combine.py)."""

import math

from alpca.backtest.combine import (
    combined_sharpe_formula, correlation, evaluate_combo, inverse_vol_weights,
    half_kelly_leverage, return_translation)


def test_combined_sharpe_formula_matches_known_cases():
    # 4 uncorrelated 0.5-Sharpe legs -> 1.0
    assert abs(combined_sharpe_formula(0.5, 4, 0.0) - 1.0) < 1e-9
    # rho=1 (same edge) -> no gain
    assert abs(combined_sharpe_formula(0.5, 4, 1.0) - 0.5) < 1e-9
    # higher correlation -> lower combined Sharpe
    assert combined_sharpe_formula(0.5, 4, 0.3) < combined_sharpe_formula(0.5, 4, 0.0)


def test_correlation_bounds_and_signs():
    a = [0.01, -0.02, 0.03, -0.01, 0.02]
    assert abs(correlation(a, a) - 1.0) < 1e-9           # self-corr = 1
    assert abs(correlation(a, [-x for x in a]) + 1.0) < 1e-9  # anti = -1


def test_inverse_vol_weights_favor_low_vol():
    streams = {"calm": [0.001, -0.001, 0.001, -0.001] * 10,
               "wild": [0.05, -0.05, 0.05, -0.05] * 10}
    w = inverse_vol_weights(streams)
    assert w["calm"] > w["wild"]
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_half_kelly_capped_and_nonneg():
    assert abs(half_kelly_leverage(0.2, 0.10, fraction=0.5, cap=10.0) - 1.0) < 1e-9  # 0.5*0.2/0.1
    assert half_kelly_leverage(1.0, 0.10, cap=2.0) == 2.0   # 5.0 -> capped at 2.0
    assert half_kelly_leverage(-1.0, 0.10) == 0.0           # negative Sharpe -> no leverage


def test_return_translation_daily_edge_below_noise():
    t = return_translation(0.9, 0.08, ppy=252.0)
    # daily expected excess is tiny vs daily vol (the honest point)
    assert t["expected_daily_excess"] < t["daily_vol"]
    assert t["noise_to_edge_ratio"] > 5
    assert abs(t["expected_excess_annual"] - 0.9 * 0.08) < 1e-9


def test_evaluate_combo_uncorrelated_beats_or_ties_and_reports():
    import random
    rng = random.Random(0)
    streams = {f"leg{i}": [rng.gauss(0.0003, 0.01) for _ in range(400)] for i in range(4)}
    rep = evaluate_combo(streams, ppy=252.0)
    assert rep.avg_abs_corr < 0.3                       # independent legs -> low corr
    assert len(rep.corr_matrix) == 4
    assert math.isfinite(rep.invvol_sharpe) and math.isfinite(rep.equalweight_sharpe)
    assert abs(sum(rep.invvol_weights.values()) - 1.0) < 1e-9
