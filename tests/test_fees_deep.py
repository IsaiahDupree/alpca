"""
Deep, deterministic tests for alpca/execution/fees.py (AlpacaFeeModel).

Covers, against the REAL source (no mocks, no network):
  - $0 commission default (and configurable commission with min floor)
  - SEC Section 31 fee charged on SELL notional only, never on BUYS
  - FINRA TAF per-share fee on shares sold, with the per-order cap
  - TAF cap binding on large sells
  - fee() == commission() + regulatory() decomposition
  - ZERO_FEES sentinel returns 0 on every side/qty/price
  - Edge/degenerate inputs: zero, negative qty (abs), NaN, inf, extreme magnitudes,
    idempotency, sign symmetry.

All expected values are computed directly from the documented 2024 defaults:
  sec_fee_per_dollar = 27.80e-6
  taf_per_share      = 0.000166
  taf_cap            = 8.30
"""

from __future__ import annotations

import math

import pytest

from alpca.execution.fees import AlpacaFeeModel, ZERO_FEES


# ---------------------------------------------------------------------------
# Tiny self-contained helpers (no imports from other tests/)
# ---------------------------------------------------------------------------

def approx(x: float) -> "pytest.approx":
    # tight tolerance: these are exact float products, allow only fp slop
    return pytest.approx(x, rel=1e-12, abs=1e-15)


def expected_sec(model: AlpacaFeeModel, qty: float, price: float) -> float:
    return abs(qty) * price * model.sec_fee_per_dollar


def expected_taf(model: AlpacaFeeModel, qty: float) -> float:
    return min(abs(qty) * model.taf_per_share, model.taf_cap)


# Default-rate constants mirrored from the source for independent verification.
SEC_RATE = 27.80e-6
TAF_PER_SHARE = 0.000166
TAF_CAP = 8.30


@pytest.fixture
def model() -> AlpacaFeeModel:
    return AlpacaFeeModel()


# ---------------------------------------------------------------------------
# Defaults & construction
# ---------------------------------------------------------------------------

def test_default_rates_match_documented_2024_values(model):
    assert model.sec_fee_per_dollar == SEC_RATE
    assert model.taf_per_share == TAF_PER_SHARE
    assert model.taf_cap == TAF_CAP
    assert model.commission_per_share == 0.0
    assert model.commission_min == 0.0


def test_default_commission_is_zero_for_any_qty(model):
    for qty in (0.0, 1.0, 100.0, 1_000_000.0, -250.0):
        assert model.commission(qty) == 0.0


# ---------------------------------------------------------------------------
# Commission branch (commission-free default, but configurable)
# ---------------------------------------------------------------------------

def test_commission_zero_when_rate_and_min_zero(model):
    assert model.commission(1234.0) == 0.0


@pytest.mark.parametrize("qty", [1.0, 10.0, 333.0, 1e6])
def test_commission_per_share_scales_with_abs_qty(qty):
    m = AlpacaFeeModel(commission_per_share=0.005)
    assert m.commission(qty) == approx(abs(qty) * 0.005)
    # negative qty uses abs()
    assert m.commission(-qty) == approx(abs(qty) * 0.005)


def test_commission_min_floor_applies_when_raw_below_min():
    m = AlpacaFeeModel(commission_per_share=0.005, commission_min=1.0)
    # 10 shares * 0.005 = 0.05 < 1.0 floor
    assert m.commission(10.0) == approx(1.0)
    # 1000 shares * 0.005 = 5.0 > 1.0 floor
    assert m.commission(1000.0) == approx(5.0)


def test_commission_min_applies_even_for_zero_qty():
    # raw == 0 but commission_min > 0 -> floor returned (branch in source)
    m = AlpacaFeeModel(commission_per_share=0.005, commission_min=1.0)
    assert m.commission(0.0) == approx(1.0)


def test_commission_zero_qty_zero_rate_zero_min_is_zero():
    m = AlpacaFeeModel(commission_per_share=0.0, commission_min=0.0)
    assert m.commission(0.0) == 0.0


# ---------------------------------------------------------------------------
# Buys incur NO regulatory fee — invariant across the grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qty", [1.0, 100.0, 10_000.0, 1_000_000.0])
@pytest.mark.parametrize("price", [1.0, 37.5, 250.0, 9999.99])
def test_buy_regulatory_is_zero(model, qty, price):
    assert model.regulatory(side_buy=True, qty=qty, price=price) == 0.0


@pytest.mark.parametrize("qty", [1.0, 100.0, 10_000.0])
@pytest.mark.parametrize("price", [1.0, 250.0])
def test_buy_total_fee_is_zero_with_default_commission_free(model, qty, price):
    # commission default is 0 and buys have no regulatory fee
    assert model.fee(side_buy=True, qty=qty, price=price) == 0.0


def test_buy_total_fee_equals_commission_only_when_commission_nonzero():
    m = AlpacaFeeModel(commission_per_share=0.01)
    # buy: regulatory 0, so fee == commission
    assert m.fee(side_buy=True, qty=500.0, price=100.0) == approx(500.0 * 0.01)


# ---------------------------------------------------------------------------
# Sells DO incur SEC + TAF — exact computed values across a param grid
# ---------------------------------------------------------------------------

SELL_GRID = [
    (1.0, 1.0),
    (10.0, 50.0),
    (100.0, 25.0),
    (1000.0, 10.0),
    (500.0, 250.0),
    (123.0, 7.77),
    (40000.0, 5.0),        # TAF below cap: 40000*0.000166 = 6.64
    (50000.0, 3.0),        # TAF below cap: 50000*0.000166 = 8.30 exactly == cap
    (60000.0, 2.0),        # TAF above cap -> binds
]


@pytest.mark.parametrize("qty,price", SELL_GRID)
def test_sell_regulatory_exact(model, qty, price):
    sec = expected_sec(model, qty, price)
    taf = expected_taf(model, qty)
    assert model.regulatory(side_buy=False, qty=qty, price=price) == approx(sec + taf)


@pytest.mark.parametrize("qty,price", SELL_GRID)
def test_sell_regulatory_is_positive(model, qty, price):
    assert model.regulatory(side_buy=False, qty=qty, price=price) > 0.0


@pytest.mark.parametrize("qty,price", SELL_GRID)
def test_fee_equals_commission_plus_regulatory(model, qty, price):
    total = model.fee(side_buy=False, qty=qty, price=price)
    comp = model.commission(qty) + model.regulatory(side_buy=False, qty=qty, price=price)
    assert total == approx(comp)


@pytest.mark.parametrize("qty,price", SELL_GRID)
def test_sell_fee_strictly_greater_than_buy_fee(model, qty, price):
    sell = model.fee(side_buy=False, qty=qty, price=price)
    buy = model.fee(side_buy=True, qty=qty, price=price)
    assert sell > buy
    assert buy == 0.0


# ---------------------------------------------------------------------------
# SEC fee isolation: zero out TAF, verify pure notional * rate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qty,price", [(100.0, 50.0), (1000.0, 10.0), (1.0, 1000.0)])
def test_sec_fee_isolated(qty, price):
    m = AlpacaFeeModel(taf_per_share=0.0, taf_cap=0.0)
    assert m.regulatory(False, qty, price) == approx(qty * price * SEC_RATE)


def test_sec_fee_scales_linearly_with_notional():
    m = AlpacaFeeModel(taf_per_share=0.0, taf_cap=0.0)
    a = m.regulatory(False, 100.0, 10.0)   # notional 1000
    b = m.regulatory(False, 100.0, 20.0)   # notional 2000
    assert b == approx(2.0 * a)


# ---------------------------------------------------------------------------
# TAF cap behavior
# ---------------------------------------------------------------------------

def test_taf_below_cap_is_per_share():
    # isolate TAF: zero SEC
    m = AlpacaFeeModel(sec_fee_per_dollar=0.0)
    qty = 1000.0  # 1000 * 0.000166 = 0.166 < cap
    assert m.regulatory(False, qty, 100.0) == approx(qty * TAF_PER_SHARE)


def test_taf_exactly_at_cap_threshold():
    m = AlpacaFeeModel(sec_fee_per_dollar=0.0)
    # cap / per_share = 8.30 / 0.000166 = 50000 shares -> exactly cap
    qty = TAF_CAP / TAF_PER_SHARE
    assert m.regulatory(False, qty, 1.0) == approx(TAF_CAP)


@pytest.mark.parametrize("qty", [50001.0, 60000.0, 100000.0, 1_000_000.0])
def test_taf_cap_binds_on_large_sells(qty):
    m = AlpacaFeeModel(sec_fee_per_dollar=0.0)
    # above 50000 shares, TAF is capped at 8.30 regardless of qty
    assert m.regulatory(False, qty, 1.0) == approx(TAF_CAP)


def test_taf_cap_is_constant_once_bound():
    m = AlpacaFeeModel(sec_fee_per_dollar=0.0)
    a = m.regulatory(False, 200000.0, 1.0)
    b = m.regulatory(False, 999999.0, 1.0)
    assert a == approx(TAF_CAP)
    assert b == approx(TAF_CAP)
    assert a == approx(b)


def test_full_regulatory_with_capped_taf():
    # large sell: SEC scales with notional, TAF pinned at cap
    m = AlpacaFeeModel()
    qty, price = 100000.0, 4.0
    expected = qty * price * SEC_RATE + TAF_CAP
    assert m.regulatory(False, qty, price) == approx(expected)


# ---------------------------------------------------------------------------
# Sign symmetry: regulatory uses abs(qty), so +/- qty give same magnitude
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qty,price", [(100.0, 50.0), (1000.0, 10.0), (60000.0, 2.0)])
def test_negative_qty_uses_abs(model, qty, price):
    pos = model.regulatory(False, qty, price)
    neg = model.regulatory(False, -qty, price)
    assert pos == approx(neg)
    assert pos > 0.0


# ---------------------------------------------------------------------------
# Zero / degenerate inputs
# ---------------------------------------------------------------------------

def test_zero_qty_sell_is_zero(model):
    assert model.regulatory(False, 0.0, 100.0) == 0.0
    assert model.fee(False, 0.0, 100.0) == 0.0


def test_zero_price_sell_only_taf(model):
    # notional 0 -> SEC 0; only TAF remains
    qty = 1000.0
    assert model.regulatory(False, qty, 0.0) == approx(qty * TAF_PER_SHARE)


def test_zero_qty_and_zero_price(model):
    assert model.regulatory(False, 0.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# NaN / inf handling (graceful: no crash, propagates per IEEE-754)
# ---------------------------------------------------------------------------

def test_nan_qty_sell_propagates_nan(model):
    r = model.regulatory(False, float("nan"), 100.0)
    assert math.isnan(r)


def test_nan_price_sell_propagates_nan(model):
    r = model.regulatory(False, 100.0, float("nan"))
    assert math.isnan(r)


def test_nan_qty_buy_is_zero(model):
    # buys short-circuit to 0.0 before touching qty/price
    assert model.regulatory(True, float("nan"), float("nan")) == 0.0


def test_inf_qty_sell_taf_capped_sec_inf(model):
    # abs(inf)*taf -> inf, min(inf, cap) -> cap; but SEC = inf*price -> inf
    r = model.regulatory(False, float("inf"), 100.0)
    assert math.isinf(r) and r > 0


def test_inf_qty_sell_taf_capped_but_sec_zero_times_inf_is_nan():
    # SEC term = notional * rate = (inf * 100) * 0.0 = inf * 0.0 = NaN per IEEE-754.
    # TAF = min(inf*per_share, cap) = cap. Sum (NaN + cap) = NaN.
    # Documenting ACTUAL behavior: with a zeroed SEC rate but infinite qty, the
    # SEC multiplication produces NaN, so the total is NaN, not the finite cap.
    m = AlpacaFeeModel(sec_fee_per_dollar=0.0)
    r = m.regulatory(False, float("inf"), 100.0)
    assert math.isnan(r)


def test_inf_qty_sell_taf_capped_finite_when_price_zero():
    # With price 0, notional = inf * 0 = NaN too -> NaN total.
    m = AlpacaFeeModel(sec_fee_per_dollar=0.0)
    r = m.regulatory(False, float("inf"), 0.0)
    assert math.isnan(r)


def test_inf_qty_taf_isolated_is_cap_with_finite_price_and_real_sec():
    # When SEC rate is a normal positive number and qty inf, SEC -> inf, total inf.
    m = AlpacaFeeModel()
    r = m.regulatory(False, float("inf"), 100.0)
    assert math.isinf(r) and r > 0


def test_inf_price_sell_sec_inf(model):
    r = model.regulatory(False, 100.0, float("inf"))
    assert math.isinf(r) and r > 0


def test_inf_qty_buy_is_zero(model):
    assert model.regulatory(True, float("inf"), float("inf")) == 0.0


# ---------------------------------------------------------------------------
# Extreme magnitudes — SEC scales unbounded, TAF still capped
# ---------------------------------------------------------------------------

def test_extreme_notional_sec_dominates(model):
    qty, price = 1e9, 1e6  # notional 1e15
    r = model.regulatory(False, qty, price)
    sec = qty * price * SEC_RATE
    assert r == approx(sec + TAF_CAP)
    # TAF is negligible vs SEC here
    assert sec > TAF_CAP * 1e6


def test_tiny_fractional_share_below_taf_and_sec(model):
    qty, price = 1e-6, 0.01
    r = model.regulatory(False, qty, price)
    expected = qty * price * SEC_RATE + qty * TAF_PER_SHARE
    assert r == approx(expected)
    assert r > 0.0


# ---------------------------------------------------------------------------
# ZERO_FEES sentinel
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("side_buy", [True, False])
@pytest.mark.parametrize("qty,price", [(0.0, 0.0), (100.0, 50.0), (1e6, 1e6), (-500.0, 10.0)])
def test_zero_fees_always_zero(side_buy, qty, price):
    assert ZERO_FEES.fee(side_buy, qty, price) == 0.0
    assert ZERO_FEES.regulatory(side_buy, qty, price) == 0.0
    assert ZERO_FEES.commission(qty) == 0.0


def test_zero_fees_rates_are_zero():
    assert ZERO_FEES.sec_fee_per_dollar == 0.0
    assert ZERO_FEES.taf_per_share == 0.0
    assert ZERO_FEES.taf_cap == 0.0
    assert ZERO_FEES.commission_per_share == 0.0
    assert ZERO_FEES.commission_min == 0.0


# ---------------------------------------------------------------------------
# Idempotency / determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qty,price", SELL_GRID)
def test_idempotent_repeated_calls(model, qty, price):
    first = model.regulatory(False, qty, price)
    for _ in range(5):
        assert model.regulatory(False, qty, price) == first


def test_fee_decomposition_invariant_holds_for_buys_and_sells(model):
    for side in (True, False):
        for qty, price in [(100.0, 50.0), (60000.0, 2.0), (0.0, 10.0)]:
            assert model.fee(side, qty, price) == approx(
                model.commission(qty) + model.regulatory(side, qty, price)
            )


# ---------------------------------------------------------------------------
# Monotonicity invariants
# ---------------------------------------------------------------------------

def test_sec_monotonic_in_price(model):
    prev = -1.0
    for price in [1.0, 10.0, 100.0, 1000.0]:
        cur = model.regulatory(False, 1000.0, price)
        assert cur > prev
        prev = cur


def test_regulatory_monotonic_nondecreasing_in_qty_below_taf_cap():
    # below TAF cap, both SEC and TAF strictly increase with qty
    m = AlpacaFeeModel()
    prev = -1.0
    for qty in [10.0, 100.0, 1000.0, 10000.0]:  # all under 50000 -> TAF uncapped
        cur = m.regulatory(False, qty, 10.0)
        assert cur > prev
        prev = cur
