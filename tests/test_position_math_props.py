"""
Property/invariant tests for the signed-position fill math (apply_fill).

These complement the hand-computed cases in test_position_math.py by asserting the
INVARIANTS that must hold for every fill, swept across a grid of inputs.
"""

import itertools

import pytest

from alpca.runtime.position_math import apply_fill

# a deterministic grid of (pos_qty, side, qty, price); pos_avg fixed at 50 when held
_POS = [0.0, 100.0, -100.0]
_SIDE = ["BUY", "SELL"]
_QTY = [10.0, 60.0, 120.0]
_PX = [55.0]
_GRID = list(itertools.product(_POS, _SIDE, _QTY, _PX))


@pytest.mark.parametrize("pos_qty,side,qty,price", _GRID)
def test_equity_conserved_at_fill_price(pos_qty, side, qty, price):
    # Marked at the fill price, total equity is unchanged by a fill: the cash move
    # exactly offsets the change in position value. cash_delta + signed*price == 0.
    avg = 50.0 if pos_qty else 0.0
    eff = apply_fill(pos_qty, avg, side, qty, price)
    signed = qty if side == "BUY" else -qty
    assert eff.cash_delta + signed * price == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("pos_qty,side,qty,price", _GRID)
def test_new_qty_is_pos_plus_signed(pos_qty, side, qty, price):
    avg = 50.0 if pos_qty else 0.0
    eff = apply_fill(pos_qty, avg, side, qty, price)
    signed = qty if side == "BUY" else -qty
    assert eff.new_qty == pytest.approx(pos_qty + signed, abs=1e-9)


@pytest.mark.parametrize("pos_qty,side,qty,price", _GRID)
def test_cash_delta_sign_and_magnitude(pos_qty, side, qty, price):
    avg = 50.0 if pos_qty else 0.0
    eff = apply_fill(pos_qty, avg, side, qty, price)
    assert eff.cash_delta == pytest.approx((-1 if side == "BUY" else 1) * qty * price, abs=1e-9)


@pytest.mark.parametrize("pos_qty,side,qty,price", _GRID)
def test_closed_and_opened_nonneg_and_bounded(pos_qty, side, qty, price):
    avg = 50.0 if pos_qty else 0.0
    eff = apply_fill(pos_qty, avg, side, qty, price)
    assert eff.closed_qty >= -1e-12
    assert eff.opened_qty >= -1e-12
    assert eff.closed_qty <= min(qty, abs(pos_qty)) + 1e-9


@pytest.mark.parametrize("pos_qty,side,qty,price", _GRID)
def test_avg_is_nonneg(pos_qty, side, qty, price):
    avg = 50.0 if pos_qty else 0.0
    eff = apply_fill(pos_qty, avg, side, qty, price)
    assert eff.new_avg >= -1e-12
    if abs(eff.new_qty) < 1e-12:
        assert eff.new_avg == 0.0  # flat -> avg resets


@pytest.mark.parametrize("side", ["BUY", "SELL"])
@pytest.mark.parametrize("bad_qty", [0.0, -5.0])
def test_noop_on_nonpositive_qty(side, bad_qty):
    eff = apply_fill(100.0, 50.0, side, bad_qty, 55.0)
    assert (eff.new_qty, eff.new_avg, eff.realized, eff.cash_delta) == (100.0, 50.0, 0.0, 0.0)


@pytest.mark.parametrize("side", ["BUY", "SELL"])
@pytest.mark.parametrize("bad_px", [0.0, -1.0])
def test_noop_on_nonpositive_price(side, bad_px):
    eff = apply_fill(-100.0, 50.0, side, 10.0, bad_px)
    assert eff.new_qty == -100.0 and eff.cash_delta == 0.0


# ---- explicit realized-PnL sign cases ----
def test_long_close_profit_is_positive():
    eff = apply_fill(100.0, 50.0, "SELL", 100.0, 55.0)
    assert eff.realized == pytest.approx(500.0)
    assert eff.new_qty == 0.0


def test_long_close_loss_is_negative():
    eff = apply_fill(100.0, 50.0, "SELL", 100.0, 45.0)
    assert eff.realized == pytest.approx(-500.0)


def test_short_cover_profit_is_positive():
    # shorted at 50, cover at 45 -> +5/share
    eff = apply_fill(-100.0, 50.0, "BUY", 100.0, 45.0)
    assert eff.realized == pytest.approx(500.0)
    assert eff.new_qty == 0.0


def test_short_cover_loss_is_negative():
    eff = apply_fill(-100.0, 50.0, "BUY", 100.0, 55.0)
    assert eff.realized == pytest.approx(-500.0)


def test_add_to_long_blends_average_no_realized():
    eff = apply_fill(100.0, 50.0, "BUY", 100.0, 60.0)
    assert eff.new_qty == 200.0
    assert eff.new_avg == pytest.approx(55.0)
    assert eff.realized == 0.0
    assert eff.opened_qty == 100.0


def test_partial_reduce_keeps_average():
    eff = apply_fill(100.0, 50.0, "SELL", 40.0, 55.0)
    assert eff.new_qty == 60.0
    assert eff.new_avg == pytest.approx(50.0)        # remaining keeps cost
    assert eff.realized == pytest.approx(40.0 * 5.0)
    assert eff.closed_qty == 40.0


def test_flip_long_to_short_books_close_then_opens_remainder():
    # long 100 @50, SELL 150 @60 -> close 100 (+1000), open short 50 @60
    eff = apply_fill(100.0, 50.0, "SELL", 150.0, 60.0)
    assert eff.new_qty == -50.0
    assert eff.new_avg == pytest.approx(60.0)
    assert eff.realized == pytest.approx(1000.0)
    assert eff.closed_qty == 100.0
    assert eff.opened_qty == 50.0


def test_flip_short_to_long():
    eff = apply_fill(-100.0, 50.0, "BUY", 150.0, 45.0)
    assert eff.new_qty == 50.0
    assert eff.new_avg == pytest.approx(45.0)
    assert eff.realized == pytest.approx(500.0)      # covered 100 at +5
    assert eff.opened_qty == 50.0


@pytest.mark.parametrize("seq_seed", range(12))
def test_cash_plus_cost_basis_tracks_realized_over_sequence(seq_seed):
    # Run a deterministic sequence of fills; the accounting identity that holds at
    # every step is: cash + (cost-basis value of the open position) == starting
    # cash + cumulative realized PnL. (Unrealized PnL is NOT in either side here.)
    sides = ["BUY", "SELL"]
    cash = 100_000.0
    qty = 0.0
    avg = 0.0
    realized_cum = 0.0
    for i in range(8):
        side = sides[(seq_seed + i) % 2]
        n = 10 + ((seq_seed * 7 + i * 13) % 90)
        price = 40.0 + ((seq_seed * 3 + i * 5) % 40)
        eff = apply_fill(qty, avg, side, float(n), float(price))
        cash += eff.cash_delta
        realized_cum += eff.realized
        qty, avg = eff.new_qty, eff.new_avg
        assert cash + qty * avg == pytest.approx(100_000.0 + realized_cum, abs=1e-6)
