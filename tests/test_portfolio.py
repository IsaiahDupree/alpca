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
    # day 1: weights pairs 0.92 / short_vol 0.08 renormalized over the two present (sum 1.0)
    wsum = 0.92 + 0.08
    expect0 = (0.92 / wsum) * 0.01 + (0.08 / wsum) * 0.05
    assert math.isclose(book.daily_returns[0], expect0, rel_tol=1e-9)
    assert "momentum" not in book.weights        # unfunded excluded


def test_combine_renormalizes_when_a_sleeve_is_missing_that_day():
    # short_vol missing on day 200 -> that day is 100% pairs (renormalized), not shrunk
    tr = {"pairs": {100: 0.01, 200: 0.03}, "short_vol": {100: 0.05}}
    book = combine_tracks(tr)
    assert book.n_days == 2
    assert math.isclose(book.daily_returns[1], 0.03, rel_tol=1e-9)     # pure pairs that day


def test_combine_empty_is_safe():
    book = combine_tracks({"pairs": {}, "short_vol": {}})
    assert book.n_days == 0 and book.equity_curve == [1.0]


def test_combine_custom_weights_override():
    tr = {"pairs": {1: 0.0}, "short_vol": {1: 0.10}}
    book = combine_tracks(tr, weights={"pairs": 0.5, "short_vol": 0.5})
    assert math.isclose(book.daily_returns[0], 0.05, rel_tol=1e-9)     # 50/50 blend
