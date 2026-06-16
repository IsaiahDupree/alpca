"""The second-leg gate must reproduce the Cases 47-51 verdicts on synthetic stand-ins."""

import math
import random

from alpca.backtest.leg_gate import evaluate_leg_candidate

DAY = 86400
BASE = 1_600_000_000      # 2020-09-13; ~4 years of daily epochs spans 2021-2024


def _book(n=900, seed=0):
    """A positive, low-vol 'pairs-like' book."""
    rng = random.Random(seed)
    return {BASE + i * DAY: 0.0006 + rng.gauss(0, 0.004) for i in range(n)}


def test_uncorrelated_positive_leg_PASSES():
    book = _book()
    rng = random.Random(1)
    cand = {t: 0.0005 + rng.gauss(0, 0.006) for t in book}     # positive, independent
    v = evaluate_leg_candidate(cand, book)
    assert v.checks["forward_positive"] and v.checks["uncorrelated"]
    assert v.passed, v.reasons


def test_negative_leg_FAILS_forward_positive():
    book = _book()
    rng = random.Random(2)
    cand = {t: -0.0008 + rng.gauss(0, 0.006) for t in book}    # negative drift (momentum-over-2022 case)
    v = evaluate_leg_candidate(cand, book)
    assert not v.checks["forward_positive"] and not v.passed


def test_correlated_leg_FAILS_uncorrelated():
    book = _book()
    cand = {t: 0.5 * r + 0.0003 for t, r in book.items()}      # basically the book again
    v = evaluate_leg_candidate(cand, book)
    assert not v.checks["uncorrelated"] and not v.passed


def test_lift_carried_by_one_year_FAILS_robustness():
    """A leg that's only positive in the most-recent year (a partial-year-artifact stand-in) must fail
    robust_loo and/or partial_year_safe."""
    book = _book(n=1000)
    rng = random.Random(3)
    cand = {}
    for t in book:
        import time as _t
        yr = _t.gmtime(t).tm_year
        cand[t] = (0.004 if yr == max(_t.gmtime(x).tm_year for x in book) else -0.0006) + rng.gauss(0, 0.003)
    v = evaluate_leg_candidate(cand, book)
    assert not (v.checks["robust_loo"] and v.checks["partial_year_safe"])
    assert not v.passed


def test_insufficient_overlap_is_safe():
    book = {BASE + i * DAY: 0.001 for i in range(30)}
    cand = {BASE + i * DAY: 0.001 for i in range(30)}
    v = evaluate_leg_candidate(cand, book)
    assert not v.passed and v.n_common == 30
