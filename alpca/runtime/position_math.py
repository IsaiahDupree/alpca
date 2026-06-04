"""
Signed-position fill math — the single source of truth for how a fill changes a
position, its average cost, realized PnL, and cash. Pure + side-effect-free so it
can be exhaustively unit-tested and adversarially verified in isolation.

Convention: position qty is SIGNED (positive = long, negative = short). avg is the
(positive) average entry price of the open position; it is 0.0 when flat.

Cash: a BUY pays out (cash decreases by qty*price); a SELL takes in (cash
increases by qty*price). This is correct for opening a short (you receive the sale
proceeds) and for covering (you pay to buy back).

Realized PnL on a reduce/close:
  long  closed: (exit - avg) * closed
  short closed: (avg - exit) * closed
  unified:      (exit - avg) * closed * sign(old_qty)

Mark-to-market equity at price P for a signed qty Q at any avg:
  cash + Q * P     (Q<0 for shorts -> a price rise reduces equity, as it should)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FillEffect:
    new_qty: float       # signed position after the fill
    new_avg: float       # average entry of the open position (0.0 if flat)
    realized: float      # realized PnL booked by this fill (0 if purely opening/adding)
    cash_delta: float    # change in cash (negative on a buy, positive on a sell)
    closed_qty: float    # shares of the prior position that were closed (>=0)
    opened_qty: float    # shares of NEW exposure opened (>=0; nonzero on open/flip)


def apply_fill(pos_qty: float, pos_avg: float, side: str, qty: float, price: float) -> FillEffect:
    """
    Apply a `qty`-share fill on `side` ("BUY"/"SELL") at `price` to a signed
    position (pos_qty @ pos_avg). Returns the resulting FillEffect.

    Handles all cases: open long/short, add to same side, partial reduce, exact
    close to flat, and FLIP (a fill larger than the opposite-side position closes
    it and opens a new position with the remainder at `price`).
    """
    if qty <= 0 or price <= 0:
        return FillEffect(pos_qty, pos_avg, 0.0, 0.0, 0.0, 0.0)

    signed = qty if side == "BUY" else -qty
    new_qty = pos_qty + signed
    cash_delta = -qty * price if side == "BUY" else qty * price

    # flat -> opening a fresh position
    if abs(pos_qty) < 1e-12:
        return FillEffect(new_qty, price, 0.0, cash_delta, 0.0, qty)

    same_direction = (pos_qty > 0) == (signed > 0)
    if same_direction:
        # add to the existing side: blend the average over total magnitude
        new_avg = (abs(pos_qty) * pos_avg + qty * price) / abs(new_qty)
        return FillEffect(new_qty, new_avg, 0.0, cash_delta, 0.0, qty)

    # opposite direction: reduce / close / flip
    closed = min(qty, abs(pos_qty))
    realized = (price - pos_avg) * closed * (1.0 if pos_qty > 0 else -1.0)

    if qty < abs(pos_qty) - 1e-12:
        # partial reduce — remaining position keeps its original average
        return FillEffect(new_qty, pos_avg, realized, cash_delta, closed, 0.0)
    if abs(new_qty) < 1e-12:
        # exact close to flat
        return FillEffect(0.0, 0.0, realized, cash_delta, closed, 0.0)
    # flip: closed the old side, opened a new position with the remainder
    opened = qty - closed
    return FillEffect(new_qty, price, realized, cash_delta, closed, opened)
