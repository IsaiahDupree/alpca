"""
Deep, deterministic, real (no-mock) tests for alpca/runtime/account.py.

Covers the three pure/offline cash-account realism components:
  * SettlementLedger  — T+1 settlement of sale proceeds
  * PdtGuard          — rolling 5-session pattern-day-trade cap under $25k
  * BorrowFeeLedger   — daily short-borrow fee = annual_rate / trading_days

All inputs are fixed constants; no network, RNG, or wall-clock dependence.
"""

from __future__ import annotations

import math

import pytest

from alpca.runtime.account import (
    BorrowFeeLedger,
    PdtGuard,
    SettlementLedger,
)


# --------------------------------------------------------------------------- #
# tiny self-contained helpers
# --------------------------------------------------------------------------- #
def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol + tol * abs(b)


# =========================================================================== #
# SettlementLedger
# =========================================================================== #
class TestSettlementLedgerBasics:
    def test_initial_state(self):
        led = SettlementLedger(10_000.0)
        assert led.settled == 10_000.0
        assert led.pending_total == 0.0
        assert led.total == 10_000.0
        assert led.available() == 10_000.0
        assert led.settle_lag == 1

    def test_starting_cash_coerced_to_float(self):
        led = SettlementLedger(5_000)  # int in
        assert isinstance(led.settled, float)
        assert led.settled == 5_000.0

    def test_buy_draws_settled_only(self):
        led = SettlementLedger(1_000.0)
        led.record_buy(300.0)
        assert led.settled == 700.0
        assert led.available() == 700.0
        assert led.total == 700.0

    def test_buy_can_go_negative(self):
        # No guard in source; record_buy simply subtracts.
        led = SettlementLedger(100.0)
        led.record_buy(250.0)
        assert led.settled == -150.0
        assert led.available() == -150.0

    def test_sell_goes_to_pending_not_available(self):
        led = SettlementLedger(0.0)
        led.record_sell(500.0, current_session=0)
        # proceeds pending, NOT available
        assert led.available() == 0.0
        assert led.pending_total == 500.0
        assert led.total == 500.0

    def test_tplus1_settlement_timing(self):
        led = SettlementLedger(0.0, settle_lag=1)
        led.record_sell(500.0, current_session=3)  # settles at session 4
        # advancing to the SAME session does nothing
        assert led.advance_to(3) == 0.0
        assert led.available() == 0.0
        # advancing to session 4 settles it
        newly = led.advance_to(4)
        assert newly == 500.0
        assert led.available() == 500.0
        assert led.pending_total == 0.0

    def test_advance_returns_newly_settled_amount(self):
        led = SettlementLedger(0.0)
        led.record_sell(100.0, current_session=0)  # -> 1
        led.record_sell(200.0, current_session=1)  # -> 2
        assert led.advance_to(1) == 100.0
        assert led.available() == 100.0
        assert led.advance_to(2) == 200.0
        assert led.available() == 300.0

    def test_advance_settles_all_matured_at_or_before_index(self):
        led = SettlementLedger(0.0)
        led.record_sell(10.0, current_session=0)  # settle 1
        led.record_sell(20.0, current_session=1)  # settle 2
        led.record_sell(40.0, current_session=2)  # settle 3
        # jump straight to 3 -> all three mature
        assert led.advance_to(3) == 70.0
        assert led.pending_total == 0.0
        assert led.available() == 70.0

    def test_advance_idempotent_after_settle(self):
        led = SettlementLedger(0.0)
        led.record_sell(100.0, current_session=0)
        led.advance_to(5)
        # second advance settles nothing more
        assert led.advance_to(5) == 0.0
        assert led.advance_to(10) == 0.0
        assert led.available() == 100.0

    def test_multiple_sells_same_settle_session_accumulate(self):
        led = SettlementLedger(0.0)
        led.record_sell(100.0, current_session=2)  # settle 3
        led.record_sell(50.0, current_session=2)   # settle 3 (same key)
        assert led.pending_total == 150.0
        assert led.advance_to(3) == 150.0

    def test_total_invariant_holds_through_lifecycle(self):
        led = SettlementLedger(1_000.0)
        # buy 400 settled -> 600, total 600
        led.record_buy(400.0)
        # sell 700 pending
        led.record_sell(700.0, current_session=0)
        assert approx(led.total, led.settled + led.pending_total)
        assert approx(led.total, 600.0 + 700.0)
        led.advance_to(1)
        assert approx(led.total, led.settled + led.pending_total)
        assert approx(led.available(), 1_300.0)


@pytest.mark.parametrize("lag", [0, 1, 2, 3, 5])
def test_settle_lag_param(lag):
    led = SettlementLedger(0.0, settle_lag=lag)
    led.record_sell(100.0, current_session=10)
    settle_at = 10 + lag
    # one session before settle_at -> not yet settled (regardless of lag)
    assert led.advance_to(settle_at - 1) == 0.0
    assert led.available() == 0.0
    # reaching settle_at settles it
    assert led.advance_to(settle_at) == 100.0
    assert led.available() == 100.0


@pytest.mark.parametrize(
    "proceeds",
    [0.0, 0.01, 1_000.0, 1_000_000.0, 1e12],
)
def test_sell_proceeds_magnitudes(proceeds):
    led = SettlementLedger(0.0)
    led.record_sell(proceeds, current_session=0)
    assert approx(led.pending_total, proceeds)
    assert led.advance_to(1) == proceeds
    assert approx(led.available(), proceeds)


@pytest.mark.parametrize(
    "sessions",
    [
        [0, 1, 2, 3],
        [5, 5, 6, 10],
        [2, 0, 1],          # out-of-order sell sessions
        [0, 0, 0, 0],       # all same session
    ],
)
def test_settlement_session_grids(sessions):
    led = SettlementLedger(0.0)
    each = 100.0
    for s in sessions:
        led.record_sell(each, current_session=s)
    expected_total = each * len(sessions)
    assert approx(led.pending_total, expected_total)
    # advance past everything: max settle session = max(s)+1
    final = max(sessions) + 1
    led.advance_to(final)
    assert led.pending_total == 0.0
    assert approx(led.available(), expected_total)


def test_negative_proceeds_edge():
    # Degenerate: a negative "proceeds" simply nets pending down. Source does
    # no validation, so assert the actual arithmetic behavior.
    led = SettlementLedger(0.0)
    led.record_sell(100.0, current_session=0)
    led.record_sell(-30.0, current_session=0)
    assert approx(led.pending_total, 70.0)
    assert led.advance_to(1) == 70.0


def test_advance_to_far_past_index():
    led = SettlementLedger(0.0)
    led.record_sell(100.0, current_session=0)
    # advancing way past settle session still settles exactly once
    assert led.advance_to(10_000) == 100.0
    assert led.pending_total == 0.0


def test_advance_to_earlier_index_does_not_settle():
    led = SettlementLedger(0.0)
    led.record_sell(100.0, current_session=10)  # settle 11
    assert led.advance_to(5) == 0.0
    assert led.pending_total == 100.0


# =========================================================================== #
# PdtGuard
# =========================================================================== #
class TestPdtGuardBasics:
    def test_defaults(self):
        g = PdtGuard()
        assert g.min_equity == 25_000.0
        assert g.max_day_trades == 3
        assert g.window_sessions == 5
        assert g.day_trade_count(0) == 0

    def test_would_be_day_trade_requires_opposite_side(self):
        g = PdtGuard()
        assert g.would_be_day_trade(0, "AAPL", "BUY") is False
        g.record_fill(0, "AAPL", "BUY")
        # selling now closes the round trip
        assert g.would_be_day_trade(0, "AAPL", "SELL") is True
        # buying again does not (same side already there)
        assert g.would_be_day_trade(0, "AAPL", "BUY") is False

    def test_symbol_case_insensitive(self):
        g = PdtGuard()
        g.record_fill(0, "aapl", "BUY")
        assert g.would_be_day_trade(0, "AAPL", "SELL") is True
        assert g.would_be_day_trade(0, "AaPl", "SELL") is True

    def test_day_trade_recorded_on_round_trip(self):
        g = PdtGuard()
        g.record_fill(0, "AAPL", "BUY")
        assert g.day_trade_count(0) == 0  # open only
        g.record_fill(0, "AAPL", "SELL")
        assert g.day_trade_count(0) == 1  # closed -> one day trade

    def test_different_sessions_not_a_day_trade(self):
        g = PdtGuard()
        g.record_fill(0, "AAPL", "BUY")
        g.record_fill(1, "AAPL", "SELL")  # different session -> overnight, not DT
        assert g.day_trade_count(1) == 0

    def test_high_equity_never_blocks(self):
        g = PdtGuard()
        # rack up 3 day trades this session
        for sym in ("A", "B", "C"):
            g.record_fill(0, sym, "BUY")
            g.record_fill(0, sym, "SELL")
        assert g.day_trade_count(0) == 3
        # set up a 4th round-trip
        g.record_fill(0, "D", "BUY")
        ok, msg = g.check(0, "D", "SELL", equity=25_000.0)
        assert ok is True and msg == ""
        ok, msg = g.check(0, "D", "SELL", equity=100_000.0)
        assert ok is True and msg == ""

    def test_fourth_day_trade_blocked_under_threshold(self):
        g = PdtGuard()
        for sym in ("A", "B", "C"):
            g.record_fill(0, sym, "BUY")
            g.record_fill(0, sym, "SELL")
        assert g.day_trade_count(0) == 3
        g.record_fill(0, "D", "BUY")
        ok, msg = g.check(0, "D", "SELL", equity=24_999.0)
        assert ok is False
        assert "PDT" in msg
        assert "#4" in msg

    def test_opening_trade_not_blocked_even_at_cap(self):
        g = PdtGuard()
        for sym in ("A", "B", "C"):
            g.record_fill(0, sym, "BUY")
            g.record_fill(0, sym, "SELL")
        # An opening BUY on a fresh symbol is not a day-trade -> allowed
        ok, msg = g.check(0, "Z", "BUY", equity=1_000.0)
        assert ok is True and msg == ""

    def test_third_round_trip_still_allowed(self):
        g = PdtGuard()
        # two prior day trades
        for sym in ("A", "B"):
            g.record_fill(0, sym, "BUY")
            g.record_fill(0, sym, "SELL")
        assert g.day_trade_count(0) == 2
        g.record_fill(0, "C", "BUY")
        # this would be the 3rd -> count (2) < max (3) -> allowed
        ok, msg = g.check(0, "C", "SELL", equity=10_000.0)
        assert ok is True and msg == ""


class TestPdtGuardWindow:
    def test_rolling_window_prunes_old_day_trades(self):
        # NOTE: _prune permanently popleft()s expired entries, so day_trade_count
        # is order-dependent. Query each session boundary with a fresh guard.
        def fresh():
            g = PdtGuard()  # window 5
            g.record_fill(0, "A", "BUY")
            g.record_fill(0, "A", "SELL")
            return g

        assert fresh().day_trade_count(0) == 1
        # at session 4, cutoff = 0, still counted
        assert fresh().day_trade_count(4) == 1
        # at session 5, cutoff = 5 - 5 + 1 = 1, so session-0 DT is pruned
        assert fresh().day_trade_count(5) == 0

    def test_window_boundary_inclusive(self):
        g = PdtGuard()
        g.record_fill(2, "A", "BUY")
        g.record_fill(2, "A", "SELL")  # DT at session 2
        # session 6: cutoff = 6-5+1 = 2 -> 2 >= 2 kept
        assert g.day_trade_count(6) == 1
        # session 7: cutoff = 3 -> 2 < 3 pruned
        assert g.day_trade_count(7) == 0

    def test_block_then_unblock_after_window_rolls(self):
        g = PdtGuard()
        # 3 day trades across sessions 0,1,2
        g.record_fill(0, "A", "BUY"); g.record_fill(0, "A", "SELL")
        g.record_fill(1, "B", "BUY"); g.record_fill(1, "B", "SELL")
        g.record_fill(2, "C", "BUY"); g.record_fill(2, "C", "SELL")
        assert g.day_trade_count(2) == 3
        # at session 2 a 4th would be blocked
        g.record_fill(2, "D", "BUY")
        ok, _ = g.check(2, "D", "SELL", equity=5_000.0)
        assert ok is False
        # by session 5, the session-0 DT is pruned (cutoff=1) -> count 2 -> allowed
        g2 = PdtGuard()
        g2.record_fill(0, "A", "BUY"); g2.record_fill(0, "A", "SELL")
        g2.record_fill(1, "B", "BUY"); g2.record_fill(1, "B", "SELL")
        g2.record_fill(2, "C", "BUY"); g2.record_fill(2, "C", "SELL")
        assert g2.day_trade_count(5) == 2
        g2.record_fill(5, "E", "BUY")
        ok, _ = g2.check(5, "E", "SELL", equity=5_000.0)
        assert ok is True


@pytest.mark.parametrize("equity", [0.0, 24_999.99, 25_000.0, 25_000.01, 1e9])
def test_pdt_threshold_boundary(equity):
    g = PdtGuard()
    for sym in ("A", "B", "C"):
        g.record_fill(0, sym, "BUY")
        g.record_fill(0, sym, "SELL")
    g.record_fill(0, "D", "BUY")
    ok, _ = g.check(0, "D", "SELL", equity=equity)
    # blocked only strictly below 25k
    assert ok == (equity >= 25_000.0)


@pytest.mark.parametrize("n_prior", [0, 1, 2, 3, 4, 5])
def test_block_depends_on_prior_count(n_prior):
    g = PdtGuard()
    syms = [f"S{i}" for i in range(n_prior)]
    for sym in syms:
        g.record_fill(0, sym, "BUY")
        g.record_fill(0, sym, "SELL")
    assert g.day_trade_count(0) == n_prior
    g.record_fill(0, "X", "BUY")
    ok, _ = g.check(0, "X", "SELL", equity=10_000.0)
    # blocked when prior count >= max_day_trades (3)
    assert ok == (n_prior < 3)


@pytest.mark.parametrize("side", ["BUY", "SELL"])
def test_round_trip_either_direction(side):
    g = PdtGuard()
    opp = "SELL" if side == "BUY" else "BUY"
    g.record_fill(0, "AAPL", side)
    assert g.would_be_day_trade(0, "AAPL", opp) is True
    g.record_fill(0, "AAPL", opp)
    assert g.day_trade_count(0) == 1


def test_multiple_round_trips_same_symbol_same_session():
    g = PdtGuard()
    g.record_fill(0, "AAPL", "BUY")
    g.record_fill(0, "AAPL", "SELL")  # DT #1
    assert g.day_trade_count(0) == 1
    # both sides now present in set; recording BUY again sees opp SELL -> another DT
    g.record_fill(0, "AAPL", "BUY")
    assert g.day_trade_count(0) == 2


def test_prune_with_empty_deque_is_safe():
    g = PdtGuard()
    assert g.day_trade_count(0) == 0
    assert g.day_trade_count(1_000) == 0


def test_check_unknown_symbol_no_history():
    g = PdtGuard()
    ok, msg = g.check(0, "NEW", "SELL", equity=1_000.0)
    assert ok is True and msg == ""


def test_independent_symbols_each_make_day_trades():
    g = PdtGuard()
    for i, sym in enumerate(["A", "B", "C", "D"]):
        g.record_fill(0, sym, "BUY")
        g.record_fill(0, sym, "SELL")
    # four independent round trips -> 4 day trades counted
    assert g.day_trade_count(0) == 4


# =========================================================================== #
# BorrowFeeLedger
# =========================================================================== #
class TestBorrowFeeLedger:
    def test_defaults(self):
        b = BorrowFeeLedger()
        assert b.annual_rate == 0.03
        assert b.trading_days == 252
        assert b.total_accrued == 0.0
        assert approx(b.daily_rate, 0.03 / 252)

    def test_daily_rate_formula(self):
        b = BorrowFeeLedger(annual_rate=0.05, trading_days=252)
        assert approx(b.daily_rate, 0.05 / 252)

    def test_accrue_on_short(self):
        b = BorrowFeeLedger(annual_rate=0.03, trading_days=252)
        fee = b.accrue_for_session(10_000.0)
        assert approx(fee, 10_000.0 * 0.03 / 252)
        assert approx(b.total_accrued, fee)

    def test_zero_when_flat(self):
        b = BorrowFeeLedger()
        assert b.accrue_for_session(0.0) == 0.0
        assert b.total_accrued == 0.0

    def test_zero_when_long_negative_smv(self):
        # short_market_value <= 0 means not short -> 0 fee.
        b = BorrowFeeLedger()
        assert b.accrue_for_session(-5_000.0) == 0.0
        assert b.total_accrued == 0.0

    def test_total_accrued_accumulates(self):
        b = BorrowFeeLedger(annual_rate=0.03, trading_days=252)
        rate = 0.03 / 252
        b.accrue_for_session(10_000.0)
        b.accrue_for_session(20_000.0)
        b.accrue_for_session(0.0)  # flat day adds nothing
        b.accrue_for_session(5_000.0)
        expected = (10_000.0 + 20_000.0 + 5_000.0) * rate
        assert approx(b.total_accrued, expected)

    def test_held_n_sessions_constant_notional(self):
        b = BorrowFeeLedger(annual_rate=0.10, trading_days=252)
        smv = 50_000.0
        for _ in range(10):
            b.accrue_for_session(smv)
        assert approx(b.total_accrued, 10 * smv * 0.10 / 252)


@pytest.mark.parametrize(
    "apr,days",
    [
        (0.03, 252),
        (0.10, 252),
        (0.0, 252),
        (0.5, 360),
        (0.25, 100),
    ],
)
def test_daily_rate_grid(apr, days):
    b = BorrowFeeLedger(annual_rate=apr, trading_days=days)
    assert approx(b.daily_rate, apr / days)
    fee = b.accrue_for_session(100_000.0)
    assert approx(fee, 100_000.0 * apr / days)


@pytest.mark.parametrize(
    "smv,apr",
    [
        (1_000.0, 0.03),
        (1e6, 0.05),
        (0.01, 0.03),
        (1e9, 0.02),
    ],
)
def test_fee_magnitudes(smv, apr):
    b = BorrowFeeLedger(annual_rate=apr)
    fee = b.accrue_for_session(smv)
    assert approx(fee, smv * apr / 252)
    assert fee >= 0.0


@pytest.mark.parametrize("zero_apr", [0.0])
def test_zero_apr_no_fee(zero_apr):
    b = BorrowFeeLedger(annual_rate=zero_apr)
    assert b.accrue_for_session(1_000_000.0) == 0.0
    assert b.total_accrued == 0.0


def test_nan_smv_propagates_but_no_crash():
    # NaN <= 0 is False in Python, so the multiply path runs and yields NaN.
    b = BorrowFeeLedger()
    fee = b.accrue_for_session(float("nan"))
    assert math.isnan(fee)
    assert math.isnan(b.total_accrued)


def test_inf_smv_yields_inf_fee():
    b = BorrowFeeLedger()
    fee = b.accrue_for_session(float("inf"))
    assert math.isinf(fee)
    assert math.isinf(b.total_accrued)


@pytest.mark.parametrize(
    "smv_sequence,expected_count",
    [
        ([10_000.0, 0.0, 10_000.0], 2),       # short, flat, short again
        ([0.0, 0.0, 0.0], 0),                 # always flat
        ([5_000.0, 5_000.0, 5_000.0, 5_000.0], 4),
        ([-1.0, 100.0, -1.0], 1),             # only the positive day accrues
    ],
)
def test_accrue_only_while_short_grid(smv_sequence, expected_count):
    b = BorrowFeeLedger(annual_rate=0.03, trading_days=252)
    rate = 0.03 / 252
    accrued_days = 0
    for smv in smv_sequence:
        fee = b.accrue_for_session(smv)
        if smv > 0:
            accrued_days += 1
            assert approx(fee, smv * rate)
        else:
            assert fee == 0.0
    assert accrued_days == expected_count
    expected_total = sum(s for s in smv_sequence if s > 0) * rate
    assert approx(b.total_accrued, expected_total)


def test_borrow_total_accrued_field_can_be_seeded():
    # total_accrued is a plain dataclass field; constructing with a prior value
    # should carry through and keep accumulating.
    b = BorrowFeeLedger(annual_rate=0.03, trading_days=252, total_accrued=100.0)
    fee = b.accrue_for_session(10_000.0)
    assert approx(b.total_accrued, 100.0 + fee)
