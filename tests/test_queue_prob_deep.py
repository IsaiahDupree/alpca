"""
Deep, deterministic tests for alpca/execution/queue_prob.py.

Covers:
  - All 5 PROB_FUNCS exact numeric values + edges (front,0)->1, (0,back)->0,
    (0,0)->0.5, monotonicity (up in front, down in back), power n=1/2/3 ordering.
  - QueuePosition.advance: cancellation (depth_reduction) with back, trade FIFO
    eat-then-fill, at_front property, front0 immutability, filled accumulation,
    negative/None/NaN/inf robustness.

Fully offline: queue_prob only imports `math`. No network, mocks, or fixtures.
"""

from __future__ import annotations

import math

import pytest

from alpca.execution.queue_prob import (
    PROB_FUNCS,
    LogProbQueueFunc,
    PowerProbQueueFunc,
    QueuePosition,
    SqrtProbQueueFunc,
    power_prob,
)


# --------------------------------------------------------------------------- #
# tiny self-contained helpers (no imports from other tests)
# --------------------------------------------------------------------------- #
def _expected_power(front: float, back: float, n: float) -> float:
    a = max(0.0, front) ** n
    b = max(0.0, back) ** n
    tot = a + b
    return a / tot if tot > 0 else 0.5


def _expected_log(front: float, back: float) -> float:
    a = math.log1p(max(0.0, front))
    b = math.log1p(max(0.0, back))
    tot = a + b
    return a / tot if tot > 0 else 0.5


def _expected_sqrt(front: float, back: float) -> float:
    a = math.sqrt(max(0.0, front))
    b = math.sqrt(max(0.0, back))
    tot = a + b
    return a / tot if tot > 0 else 0.5


_REL = 1e-12


# --------------------------------------------------------------------------- #
# PROB_FUNCS registry shape
# --------------------------------------------------------------------------- #
def test_prob_funcs_keys_exact():
    assert set(PROB_FUNCS) == {"power1", "power2", "power3", "log", "sqrt"}


def test_prob_funcs_types_and_n():
    assert isinstance(PROB_FUNCS["power1"], PowerProbQueueFunc)
    assert PROB_FUNCS["power1"].n == 1.0
    assert PROB_FUNCS["power2"].n == 2.0
    assert PROB_FUNCS["power3"].n == 3.0
    assert isinstance(PROB_FUNCS["log"], LogProbQueueFunc)
    assert isinstance(PROB_FUNCS["sqrt"], SqrtProbQueueFunc)


def test_power_prob_factory():
    f = power_prob(2.5)
    assert isinstance(f, PowerProbQueueFunc)
    assert f.n == 2.5
    assert f(3.0, 1.0) == pytest.approx(_expected_power(3.0, 1.0, 2.5), rel=_REL)


def test_power_prob_factory_default_n_is_two():
    assert power_prob().n == 2.0


# --------------------------------------------------------------------------- #
# Exact numeric values across all five funcs at several (front, back) points
# --------------------------------------------------------------------------- #
_POINTS = [
    (2.0, 1.0),
    (1.0, 2.0),
    (5.0, 5.0),
    (10.0, 1.0),
    (1.0, 10.0),
    (3.0, 7.0),
    (100.0, 1.0),
    (0.5, 0.25),
]


@pytest.mark.parametrize("front,back", _POINTS)
def test_power1_exact(front, back):
    assert PROB_FUNCS["power1"](front, back) == pytest.approx(
        _expected_power(front, back, 1.0), rel=_REL
    )


@pytest.mark.parametrize("front,back", _POINTS)
def test_power2_exact(front, back):
    assert PROB_FUNCS["power2"](front, back) == pytest.approx(
        _expected_power(front, back, 2.0), rel=_REL
    )


@pytest.mark.parametrize("front,back", _POINTS)
def test_power3_exact(front, back):
    assert PROB_FUNCS["power3"](front, back) == pytest.approx(
        _expected_power(front, back, 3.0), rel=_REL
    )


@pytest.mark.parametrize("front,back", _POINTS)
def test_log_exact(front, back):
    assert PROB_FUNCS["log"](front, back) == pytest.approx(
        _expected_log(front, back), rel=_REL
    )


@pytest.mark.parametrize("front,back", _POINTS)
def test_sqrt_exact(front, back):
    assert PROB_FUNCS["sqrt"](front, back) == pytest.approx(
        _expected_sqrt(front, back), rel=_REL
    )


# --------------------------------------------------------------------------- #
# Edges: (front,0)->1, (0,back)->0, (0,0)->0.5 for every func
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_edge_back_zero_is_one(name):
    # everything ahead -> prob_ahead = 1
    assert PROB_FUNCS[name](5.0, 0.0) == pytest.approx(1.0, rel=_REL)


@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_edge_front_zero_is_zero(name):
    # nothing ahead -> prob_ahead = 0
    assert PROB_FUNCS[name](0.0, 5.0) == 0.0


@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_edge_both_zero_is_half(name):
    # both empty -> symmetric 0.5
    assert PROB_FUNCS[name](0.0, 0.0) == 0.5


@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_symmetric_equal_inputs_is_half(name):
    for v in (1.0, 4.0, 9.0, 50.0):
        assert PROB_FUNCS[name](v, v) == pytest.approx(0.5, rel=_REL)


@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_output_in_unit_interval(name):
    f = PROB_FUNCS[name]
    for front in (0.0, 0.3, 1.0, 7.0, 1000.0):
        for back in (0.0, 0.3, 1.0, 7.0, 1000.0):
            p = f(front, back)
            assert 0.0 <= p <= 1.0


# --------------------------------------------------------------------------- #
# Monotonicity: increasing in front (back fixed), decreasing in back (front fixed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_monotone_increasing_in_front(name):
    f = PROB_FUNCS[name]
    back = 5.0
    fronts = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    vals = [f(x, back) for x in fronts]
    assert all(b > a for a, b in zip(vals, vals[1:])), vals


@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_monotone_decreasing_in_back(name):
    f = PROB_FUNCS[name]
    front = 5.0
    backs = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    vals = [f(front, x) for x in backs]
    assert all(b < a for a, b in zip(vals, vals[1:])), vals


# --------------------------------------------------------------------------- #
# Power ordering: at a point with front > back, higher n => higher prob_ahead.
# When front < back, higher n => lower prob_ahead. At front==back all equal 0.5.
# --------------------------------------------------------------------------- #
def test_power_ordering_front_gt_back():
    p1 = PROB_FUNCS["power1"](6.0, 2.0)
    p2 = PROB_FUNCS["power2"](6.0, 2.0)
    p3 = PROB_FUNCS["power3"](6.0, 2.0)
    assert p1 < p2 < p3


def test_power_ordering_front_lt_back():
    p1 = PROB_FUNCS["power1"](2.0, 6.0)
    p2 = PROB_FUNCS["power2"](2.0, 6.0)
    p3 = PROB_FUNCS["power3"](2.0, 6.0)
    assert p1 > p2 > p3


def test_power_ordering_equal_all_half():
    assert (
        PROB_FUNCS["power1"](4.0, 4.0)
        == PROB_FUNCS["power2"](4.0, 4.0)
        == PROB_FUNCS["power3"](4.0, 4.0)
        == 0.5
    )


def test_nonpower_funcs_below_linear_and_above_half():
    # At front>back, power1 (linear) is the steepest -> highest prob_ahead, and
    # both sqrt and log sit between 0.5 and power1. (Empirically at this ratio
    # log > sqrt; both are gentler than the linear power1.)
    front, back = 6.0, 2.0
    log = PROB_FUNCS["log"](front, back)
    sqrt = PROB_FUNCS["sqrt"](front, back)
    pw1 = PROB_FUNCS["power1"](front, back)
    assert 0.5 < sqrt < pw1
    assert 0.5 < log < pw1
    assert sqrt < log  # actual current ordering at (6, 2)


# --------------------------------------------------------------------------- #
# Robustness of prob funcs: negative clamps to 0, NaN/inf handling
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_negative_front_clamps_to_zero(name):
    # negative front clamps -> treated as 0 ahead -> prob 0 (with positive back)
    assert PROB_FUNCS[name](-10.0, 5.0) == 0.0


@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_negative_back_clamps_to_zero(name):
    # negative back clamps -> treated as 0 behind -> prob 1 (with positive front)
    assert PROB_FUNCS[name](5.0, -10.0) == pytest.approx(1.0, rel=_REL)


@pytest.mark.parametrize("name", list(PROB_FUNCS))
def test_both_negative_clamp_to_half(name):
    assert PROB_FUNCS[name](-3.0, -7.0) == 0.5


def test_power_inf_front_is_nan_documented():
    # inf**n = inf, inf/inf = nan: power funcs do not special-case infinity.
    p = PROB_FUNCS["power2"](math.inf, 5.0)
    assert math.isnan(p)


def test_power_huge_front_saturates_to_one():
    # very large but finite front dominates -> prob ~ 1
    assert PROB_FUNCS["power2"](1e6, 1.0) == pytest.approx(1.0, rel=1e-9)


def test_power_huge_back_saturates_to_zero():
    assert PROB_FUNCS["power2"](1.0, 1e6) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# QueuePosition construction & immutable front0
# --------------------------------------------------------------------------- #
def test_init_defaults():
    q = QueuePosition(10.0)
    assert q.front0 == 10.0
    assert q.front == 10.0
    assert q.filled == 0.0
    assert isinstance(q.prob, PowerProbQueueFunc)
    assert q.prob.n == 2.0


def test_init_negative_front_clamped():
    q = QueuePosition(-5.0)
    assert q.front0 == 0.0
    assert q.front == 0.0


def test_front0_immutable_after_advance():
    q = QueuePosition(10.0)
    q.advance(traded_qty=4.0)
    q.advance(depth_reduction=3.0, back=2.0)
    assert q.front0 == 10.0  # never changes
    assert q.front < 10.0    # current position did move


def test_at_front_true_when_zero():
    assert QueuePosition(0.0).at_front is True


def test_at_front_false_when_positive():
    assert QueuePosition(1.0).at_front is False


def test_at_front_tolerance():
    q = QueuePosition(1.0)
    q.advance(traded_qty=1.0 - 1e-10)  # leaves tiny residual below 1e-9
    assert q.at_front is True


# --------------------------------------------------------------------------- #
# advance: cancellation only (depth_reduction with back) — no fill
# --------------------------------------------------------------------------- #
def test_cancel_only_no_fill_power2():
    # front=10, back=10, power2 prob=0.5, reduction=4 -> front=10-0.5*4=8
    q = QueuePosition(10.0)
    filled = q.advance(depth_reduction=4.0, back=10.0)
    assert filled == 0.0
    assert q.front == pytest.approx(8.0, rel=_REL)
    assert q.filled == 0.0


def test_cancel_back_zero_prob_one():
    # back=0 -> prob_ahead=1 -> full reduction applied ahead
    q = QueuePosition(10.0)
    q.advance(depth_reduction=3.0, back=0.0)
    assert q.front == pytest.approx(7.0, rel=_REL)


def test_cancel_cannot_drive_front_negative():
    # reduction larger than front, prob=1 -> clamps at 0
    q = QueuePosition(2.0)
    q.advance(depth_reduction=100.0, back=0.0)
    assert q.front == 0.0
    assert q.filled == 0.0
    assert q.at_front is True


def test_cancel_noop_when_front_already_zero():
    q = QueuePosition(0.0)
    filled = q.advance(depth_reduction=50.0, back=5.0)
    assert filled == 0.0
    assert q.front == 0.0


def test_cancel_negative_reduction_ignored():
    # depth_reduction <= 0 guard: no movement
    q = QueuePosition(10.0)
    q.advance(depth_reduction=-5.0, back=1.0)
    assert q.front == 10.0


def test_cancel_uses_dynamic_prob():
    # custom prob func with n=1: front=8, back=8 -> prob 0.5, reduction=2 -> front 7
    q = QueuePosition(8.0, prob_func=PowerProbQueueFunc(1.0))
    q.advance(depth_reduction=2.0, back=8.0)
    assert q.front == pytest.approx(7.0, rel=_REL)


# --------------------------------------------------------------------------- #
# advance: trade FIFO eat-then-fill
# --------------------------------------------------------------------------- #
def test_trade_eats_partial_no_fill():
    # front=10, trade=4 -> eat 4, front 6, no fill
    q = QueuePosition(10.0)
    filled = q.advance(traded_qty=4.0)
    assert filled == 0.0
    assert q.front == pytest.approx(6.0, rel=_REL)
    assert q.filled == 0.0


def test_trade_eats_exact_no_fill():
    # front=5, trade=5 -> eat 5, front 0, no leftover -> no fill
    q = QueuePosition(5.0)
    filled = q.advance(traded_qty=5.0)
    assert filled == 0.0
    assert q.front == 0.0


def test_trade_eat_then_fill():
    # front=5, trade=8 -> eat 5, fill 3
    q = QueuePosition(5.0)
    filled = q.advance(traded_qty=8.0)
    assert filled == pytest.approx(3.0, rel=_REL)
    assert q.front == 0.0
    assert q.filled == pytest.approx(3.0, rel=_REL)


def test_trade_fills_fully_when_at_front():
    # already at front -> entire trade fills
    q = QueuePosition(0.0)
    filled = q.advance(traded_qty=7.0)
    assert filled == pytest.approx(7.0, rel=_REL)
    assert q.front == 0.0
    assert q.filled == pytest.approx(7.0, rel=_REL)


def test_trade_negative_qty_clamped():
    q = QueuePosition(5.0)
    filled = q.advance(traded_qty=-9.0)
    assert filled == 0.0
    assert q.front == 5.0


def test_trade_zero_qty_noop():
    q = QueuePosition(5.0)
    filled = q.advance(traded_qty=0.0)
    assert filled == 0.0
    assert q.front == 5.0


# --------------------------------------------------------------------------- #
# advance: combined cancellation + trade in one bar
# --------------------------------------------------------------------------- #
def test_combined_cancel_then_trade_no_fill():
    # front=10, back=0 -> cancel prob 1: front 10-2=8; trade 3 eats -> front 5, no fill
    q = QueuePosition(10.0)
    filled = q.advance(traded_qty=3.0, depth_reduction=2.0, back=0.0)
    assert filled == 0.0
    assert q.front == pytest.approx(5.0, rel=_REL)


def test_combined_cancel_then_trade_with_fill():
    # front=4, back=0 -> cancel reduction 2 -> front 2; trade 5 eats 2 -> fill 3
    q = QueuePosition(4.0)
    filled = q.advance(traded_qty=5.0, depth_reduction=2.0, back=0.0)
    assert filled == pytest.approx(3.0, rel=_REL)
    assert q.front == 0.0
    assert q.filled == pytest.approx(3.0, rel=_REL)


# --------------------------------------------------------------------------- #
# advance: accumulation across multiple bars
# --------------------------------------------------------------------------- #
def test_filled_accumulates_across_bars():
    q = QueuePosition(3.0)
    q.advance(traded_qty=1.0)              # eat 1, front 2
    assert q.front == pytest.approx(2.0, rel=_REL)
    f2 = q.advance(traded_qty=5.0)         # eat 2, fill 3
    assert f2 == pytest.approx(3.0, rel=_REL)
    f3 = q.advance(traded_qty=2.0)         # already front -> fill 2
    assert f3 == pytest.approx(2.0, rel=_REL)
    assert q.filled == pytest.approx(5.0, rel=_REL)


def test_full_sequence_eat_cancel_fill():
    # front=20: cancel back=0 reduction 5 -> 15; trade 10 -> 5; cancel back=0 5 -> 0; trade 4 -> fill 4
    q = QueuePosition(20.0)
    assert q.advance(depth_reduction=5.0, back=0.0) == 0.0
    assert q.front == pytest.approx(15.0, rel=_REL)
    assert q.advance(traded_qty=10.0) == 0.0
    assert q.front == pytest.approx(5.0, rel=_REL)
    assert q.advance(depth_reduction=5.0, back=0.0) == 0.0
    assert q.front == pytest.approx(0.0, abs=1e-12)
    f = q.advance(traded_qty=4.0)
    assert f == pytest.approx(4.0, rel=_REL)
    assert q.filled == pytest.approx(4.0, rel=_REL)


def test_advance_all_defaults_is_noop():
    q = QueuePosition(5.0)
    filled = q.advance()
    assert filled == 0.0
    assert q.front == 5.0
    assert q.filled == 0.0


def test_idempotent_noop_advances():
    q = QueuePosition(5.0)
    for _ in range(10):
        assert q.advance() == 0.0
    assert q.front == 5.0
    assert q.filled == 0.0


# --------------------------------------------------------------------------- #
# advance: degenerate numeric inputs
# --------------------------------------------------------------------------- #
def test_trade_nan_qty_behavior():
    # max(0.0, nan) -> nan; min(front, nan) -> nan propagates. Document actual behavior.
    q = QueuePosition(5.0)
    filled = q.advance(traded_qty=float("nan"))
    assert math.isnan(filled) or filled == 0.0


def test_trade_inf_qty_fills_inf():
    # huge trade fully consumes finite front then fills the remainder (inf)
    q = QueuePosition(5.0)
    filled = q.advance(traded_qty=math.inf)
    assert filled == math.inf
    assert q.front == pytest.approx(0.0, abs=1e-9) or q.front <= 0
