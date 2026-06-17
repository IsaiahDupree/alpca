"""Invariants for the deployed-portfolio layer (weights policy + combine_tracks)."""

import math

from alpca.live.portfolio import (
    DEPLOYED, deployed_weights, combine_tracks, CombinedBook)


def test_deployed_weights_enforce_caps_and_zero_probation():
    w = deployed_weights()
    # short_vol is hard-capped at 0.08
    assert w["short_vol"] == 0.08
    # momentum is PROBATION -> 0 trading capital
    assert w["momentum"] == 0.0
    # pairs is the funded core
    assert w["pairs"] > 0.5
    # every cap actually binds (weight never exceeds cap)
    for s in DEPLOYED:
        if s.cap is not None:
            assert w[s.name] <= s.cap + 1e-12


def test_combine_blends_funded_sleeves_at_weights():
    # two funded sleeves on the same two days; momentum present but unfunded -> ignored
    tr = {
        "pairs":     {100: 0.01, 200: -0.02},
        "short_vol": {100: 0.05, 200: 0.05},
        "momentum":  {100: 0.99, 200: 0.99},     # PROBATION (weight 0) -> must NOT affect the book
    }
    book = combine_tracks(tr)
    assert isinstance(book, CombinedBook) and book.n_days == 2
    # day 1: pairs (core) 0.92 + short_vol (capped) 0.08 — both present, fully invested
    assert math.isclose(book.daily_returns[0], 0.92 * 0.01 + 0.08 * 0.05, rel_tol=1e-9)
    assert "momentum" not in book.weights        # unfunded excluded


def test_capped_sleeve_is_NEVER_amplified_when_core_missing():
    """The bug the canonical backtest caught: when the pairs CORE has no data, the capped short-vol
    sleeve must stay at its 8% cap (the rest is cash), NOT be renormalized to ~100% (a −46% tail)."""
    tr = {"pairs": {100: 0.01}, "short_vol": {100: 0.05, 200: 0.05}}
    book = combine_tracks(tr)
    assert book.n_days == 2
    # day 100: both present -> 0.92*pairs + 0.08*shortvol
    assert math.isclose(book.daily_returns[0], 0.92 * 0.01 + 0.08 * 0.05, rel_tol=1e-9)
    # day 200: ONLY short-vol present -> it stays pinned at 0.08 (92% cash), NOT 1.0
    assert math.isclose(book.daily_returns[1], 0.08 * 0.05, rel_tol=1e-9)


def test_missing_capped_leg_leaves_core_at_its_weight_plus_cash():
    # short_vol missing on day 200 -> that day is 0.92*pairs + 8% cash (NOT renormalized to 100% pairs)
    tr = {"pairs": {100: 0.01, 200: 0.03}, "short_vol": {100: 0.05}}
    book = combine_tracks(tr)
    assert book.n_days == 2
    assert math.isclose(book.daily_returns[1], 0.92 * 0.03, rel_tol=1e-9)   # core at its weight + cash


def test_multiple_cores_renormalize_among_present():
    # two equal cores share residual 1.0; when one is missing the other absorbs the full core weight
    tr = {"a": {1: 0.02, 2: 0.04}, "b": {1: 0.02}}
    book = combine_tracks(tr, weights={"a": 0.5, "b": 0.5}, capped=set())
    assert math.isclose(book.daily_returns[0], 0.5 * 0.02 + 0.5 * 0.02, rel_tol=1e-9)  # both present
    assert math.isclose(book.daily_returns[1], 1.0 * 0.04, rel_tol=1e-9)               # only a -> a absorbs


def test_combine_empty_is_safe():
    book = combine_tracks({"pairs": {}, "short_vol": {}})
    assert book.n_days == 0 and book.equity_curve == [1.0]


def test_combine_custom_weights_override():
    tr = {"pairs": {1: 0.0}, "short_vol": {1: 0.10}}
    book = combine_tracks(tr, weights={"pairs": 0.5, "short_vol": 0.5})
    assert math.isclose(book.daily_returns[0], 0.05, rel_tol=1e-9)     # 50/50 blend
