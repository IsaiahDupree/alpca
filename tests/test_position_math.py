"""
Exhaustive tests for signed-position fill math. Every scenario is hand-computed
in the comment so a reviewer can verify the expected numbers independently.
"""

import math

from alpca.runtime.position_math import apply_fill


def _close(a, b, tol=1e-9):
    return abs(a - b) < tol


# ----------------------------------------------------------------- opening
def test_open_long():
    # flat -> BUY 100 @ 50: qty +100, avg 50, no realized, cash -5000
    e = apply_fill(0.0, 0.0, "BUY", 100, 50.0)
    assert e.new_qty == 100 and _close(e.new_avg, 50.0)
    assert e.realized == 0.0 and _close(e.cash_delta, -5000.0)
    assert e.opened_qty == 100 and e.closed_qty == 0


def test_open_short():
    # flat -> SELL 100 @ 50: qty -100, avg 50, no realized, cash +5000 (proceeds)
    e = apply_fill(0.0, 0.0, "SELL", 100, 50.0)
    assert e.new_qty == -100 and _close(e.new_avg, 50.0)
    assert e.realized == 0.0 and _close(e.cash_delta, 5000.0)
    assert e.opened_qty == 100


# ----------------------------------------------------------------- adding
def test_add_to_long_blends_avg():
    # long 100 @ 50, BUY 100 @ 60 -> 200 @ 55, cash -6000
    e = apply_fill(100.0, 50.0, "BUY", 100, 60.0)
    assert e.new_qty == 200 and _close(e.new_avg, 55.0)
    assert e.realized == 0.0 and _close(e.cash_delta, -6000.0)


def test_add_to_short_blends_avg():
    # short -100 @ 50, SELL 100 @ 60 -> -200 @ 55, cash +6000
    e = apply_fill(-100.0, 50.0, "SELL", 100, 60.0)
    assert e.new_qty == -200 and _close(e.new_avg, 55.0)
    assert _close(e.cash_delta, 6000.0)


# ----------------------------------------------------------------- reducing
def test_partial_close_long_profit():
    # long 100 @ 50, SELL 40 @ 60: realized (60-50)*40=+400, remain 60 @ 50
    e = apply_fill(100.0, 50.0, "SELL", 40, 60.0)
    assert e.new_qty == 60 and _close(e.new_avg, 50.0)
    assert _close(e.realized, 400.0) and _close(e.cash_delta, 2400.0)
    assert e.closed_qty == 40 and e.opened_qty == 0


def test_partial_cover_short_profit():
    # short -100 @ 50, BUY 40 @ 45: realized (50-45)*40=+200, remain -60 @ 50
    e = apply_fill(-100.0, 50.0, "BUY", 40, 45.0)
    assert e.new_qty == -60 and _close(e.new_avg, 50.0)
    assert _close(e.realized, 200.0) and _close(e.cash_delta, -1800.0)


def test_partial_cover_short_loss():
    # short -100 @ 50, BUY 40 @ 55 (price rose): realized (50-55)*40 = -200
    e = apply_fill(-100.0, 50.0, "BUY", 40, 55.0)
    assert _close(e.realized, -200.0)


# ----------------------------------------------------------------- closing
def test_exact_close_long():
    e = apply_fill(100.0, 50.0, "SELL", 100, 55.0)
    assert e.new_qty == 0 and e.new_avg == 0.0
    assert _close(e.realized, 500.0)  # (55-50)*100


def test_exact_cover_short():
    e = apply_fill(-100.0, 50.0, "BUY", 100, 45.0)
    assert e.new_qty == 0 and e.new_avg == 0.0
    assert _close(e.realized, 500.0)  # (50-45)*100 short profit


# ----------------------------------------------------------------- flipping
def test_flip_long_to_short():
    # long 100 @ 50, SELL 150 @ 60: close 100 (realized +1000), open -50 @ 60
    e = apply_fill(100.0, 50.0, "SELL", 150, 60.0)
    assert e.new_qty == -50 and _close(e.new_avg, 60.0)
    assert _close(e.realized, 1000.0)       # (60-50)*100
    assert e.closed_qty == 100 and e.opened_qty == 50
    assert _close(e.cash_delta, 9000.0)     # +150*60


def test_flip_short_to_long():
    # short -100 @ 50, BUY 150 @ 45: cover 100 (realized +500), open +50 @ 45
    e = apply_fill(-100.0, 50.0, "BUY", 150, 45.0)
    assert e.new_qty == 50 and _close(e.new_avg, 45.0)
    assert _close(e.realized, 500.0)        # (50-45)*100
    assert e.closed_qty == 100 and e.opened_qty == 50
    assert _close(e.cash_delta, -6750.0)    # -150*45


# ----------------------------------------------------------------- invariants
def test_short_equity_falls_when_price_rises():
    # open short, then mark to market: equity = cash + qty*price
    e = apply_fill(0.0, 0.0, "SELL", 100, 50.0)
    cash = 100_000 + e.cash_delta            # 105_000
    eq_at_50 = cash + e.new_qty * 50.0       # 105000 - 5000 = 100000 (no PnL yet)
    eq_at_60 = cash + e.new_qty * 60.0       # 105000 - 6000 = 99000 (lost 1000)
    assert _close(eq_at_50, 100_000.0)
    assert _close(eq_at_60, 99_000.0)


def test_realized_plus_unrealized_consistency_round_trip():
    # full short round trip: short 100 @ 50, cover 100 @ 47 -> +300; cash nets +300
    o = apply_fill(0.0, 0.0, "SELL", 100, 50.0)
    c = apply_fill(o.new_qty, o.new_avg, "BUY", 100, 47.0)
    assert _close(c.realized, 300.0)
    net_cash = o.cash_delta + c.cash_delta   # +5000 -4700 = +300
    assert _close(net_cash, 300.0)
    assert c.new_qty == 0


def test_noop_on_zero_qty_or_price():
    assert apply_fill(10, 50, "BUY", 0, 50).new_qty == 10
    assert apply_fill(10, 50, "BUY", 5, 0).new_qty == 10
