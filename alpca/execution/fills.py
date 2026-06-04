"""
Fill model — turns an intended (mid) price + size into a realistic fill.

Models the three things a naive "fill 100% at one price" backtest ignores:
  1. bid/ask SPREAD   — you buy at the ask, sell at the bid (half-spread each side)
  2. market IMPACT    — bigger orders push the price (square-root law vs volume)
  3. volume CAP       — you can't take more than a fraction of the bar's volume;
                        excess size is a PARTIAL fill

Used by both the backtester and the SimAdapter so offline research and the
live-sim path share one honest cost model.

Backward-compat: a FillModel(half_spread_bps=s, impact_coef_bps=0,
participation_cap=1.0, min_tick=0.0) reproduces the old flat-`s`-bps behavior
exactly, which is the default the backtester builds from its `slippage_bps` arg.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class FillResult:
    price: float            # realized fill price (per share)
    filled_qty: float       # how much actually filled (< requested if volume-capped)
    slippage_bps: float     # adverse slippage applied vs ref, in bps
    capped: bool            # True if volume cap reduced the fill size


@dataclass
class FillModel:
    """
    half_spread_bps : half the bid/ask spread, paid on every fill (adverse).
    impact_coef_bps : coefficient on the square-root impact term;
                      impact_bps = impact_coef_bps * sqrt(participation),
                      where participation = qty / bar_volume.
    participation_cap : max fraction of bar volume one order may take; size above
                      this fills partially. 1.0 = no cap.
    min_tick        : round fills to this tick ($0.01 equities). 0.0 = no rounding.
    """
    half_spread_bps: float = 1.0
    impact_coef_bps: float = 8.0
    participation_cap: float = 1.0
    min_tick: float = 0.01

    @classmethod
    def flat(cls, slippage_bps: float) -> "FillModel":
        """A pure flat-bps model (no impact, no cap, no tick rounding)."""
        return cls(half_spread_bps=slippage_bps, impact_coef_bps=0.0,
                   participation_cap=1.0, min_tick=0.0)

    def _round_tick(self, price: float) -> float:
        if self.min_tick and self.min_tick > 0:
            return round(price / self.min_tick) * self.min_tick
        return price

    def fill(self, side_buy: bool, ref_price: float, qty: float,
             bar_volume: Optional[float] = None) -> FillResult:
        if (not math.isfinite(ref_price) or not math.isfinite(qty)
                or ref_price <= 0 or qty <= 0):
            safe = ref_price if math.isfinite(ref_price) else 0.0
            return FillResult(price=safe, filled_qty=0.0, slippage_bps=0.0, capped=False)

        # 1) volume cap -> partial fill
        filled_qty = qty
        capped = False
        if bar_volume is not None and bar_volume > 0 and self.participation_cap < 1.0:
            max_qty = self.participation_cap * bar_volume
            if qty > max_qty:
                filled_qty = max_qty
                capped = True

        # 2) impact from participation (use the REQUESTED size — you move the
        #    market by trying to take that much, even if you only get part of it)
        impact_bps = 0.0
        if self.impact_coef_bps > 0 and bar_volume is not None and bar_volume > 0:
            participation = qty / bar_volume
            impact_bps = self.impact_coef_bps * math.sqrt(participation)

        # 3) total adverse slippage = half-spread + impact
        slippage_bps = self.half_spread_bps + impact_bps
        adj = slippage_bps / 10_000.0
        price = ref_price * (1 + adj) if side_buy else ref_price * (1 - adj)
        price = self._round_tick(price)

        return FillResult(price=price, filled_qty=filled_qty,
                          slippage_bps=slippage_bps, capped=capped)

    def fill_limit(self, side_buy: bool, limit_price: float,
                   bar_open: float, bar_high: float, bar_low: float, qty: float,
                   bar_volume: Optional[float] = None, *, queue_pos=None) -> FillResult:
        """
        Realistic LIMIT fill against one bar (audit gap: limits previously
        "filled" by clamping with no through-trade test).

        A limit order fills ONLY if the bar's price actually traded through it:
          - BUY  fills iff bar_low  <= limit (price came down to your bid)
          - SELL fills iff bar_high >= limit (price came up to your offer)
        On a gap THROUGH the limit (e.g. a buy whose bar opens below the limit),
        you fill at the better price (the open), modeling price improvement.
        You never pay worse than your limit, so slippage_bps <= 0 (favorable).

        Queue position: by default the volume cap is a coarse proxy — you can be
        filled for `participation_cap` of the bar's volume immediately. Pass a
        `queue_pos` (execution.queue_prob.QueuePosition) for a FIFO model instead:
        the order only fills once the displayed size AHEAD of it has been consumed
        by trades, so an order joining behind a deep queue waits its turn across
        bars. A no-fill returns filled_qty=0 (the caller leaves the order resting).
        """
        if (not math.isfinite(limit_price) or not math.isfinite(qty)
                or limit_price <= 0 or qty <= 0):
            safe = limit_price if math.isfinite(limit_price) else 0.0
            return FillResult(price=safe, filled_qty=0.0, slippage_bps=0.0, capped=False)

        reached = (bar_low <= limit_price) if side_buy else (bar_high >= limit_price)
        if not reached:
            # price never touched the limit this bar -> no fill (order rests)
            return FillResult(price=limit_price, filled_qty=0.0, slippage_bps=0.0, capped=False)

        # price improvement on a gap-through; otherwise fill exactly at the limit
        if side_buy:
            fill_price = min(bar_open, limit_price)
        else:
            fill_price = max(bar_open, limit_price)
        fill_price = self._round_tick(fill_price)

        # how much volume could trade at our level this bar (cap = the share of bar
        # volume that prints at/through our price); 1.0 cap => all of it. With no
        # volume context, assume enough traded to fill the request.
        if bar_volume is not None and bar_volume > 0:
            traded_at_level = self.participation_cap * bar_volume
        else:
            traded_at_level = qty

        filled_qty = qty
        capped = False
        if queue_pos is not None:
            # FIFO: trades first consume the queue ahead, then fill us
            fillable = queue_pos.advance(traded_qty=traded_at_level)
            filled_qty = min(qty, fillable)
            capped = filled_qty < qty
        elif bar_volume is not None and bar_volume > 0 and self.participation_cap < 1.0:
            # legacy volume-cap proxy (unchanged)
            if qty > traded_at_level:
                filled_qty = traded_at_level
                capped = True

        # slippage vs the limit: 0 when filled at limit, negative (favorable) on
        # a price-improving gap. Signed so positive=worse, matching .fill().
        raw = (fill_price - limit_price) / limit_price * 10_000.0
        slippage_bps = raw if side_buy else -raw
        return FillResult(price=fill_price, filled_qty=filled_qty,
                          slippage_bps=slippage_bps, capped=capped)
