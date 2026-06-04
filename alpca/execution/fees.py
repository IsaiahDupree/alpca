"""
Alpaca US-equity fee model.

Alpaca charges **$0 commission** on US equities, but the regulatory pass-through
fees still apply on the SELL side and a naive backtest that ignores them
overstates net PnL (they're small but real, and they scale with turnover):

  - SEC Section 31 fee  : on sell NOTIONAL. Rate changes ~annually; expressed
                          here as dollars per dollar of principal.
  - FINRA TAF (Trading Activity Fee) : on shares SOLD, per share, capped per trade.

Buys incur no regulatory fee (and $0 commission). Rates are configurable because
the SEC/FINRA publish new values periodically — defaults are 2024 figures.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AlpacaFeeModel:
    # SEC Section 31: 2024 rate = $27.80 per $1,000,000 of sell principal.
    sec_fee_per_dollar: float = 27.80e-6
    # FINRA TAF: 2024 = $0.000166 per share sold, capped at $8.30 per trade.
    taf_per_share: float = 0.000166
    taf_cap: float = 8.30
    # Alpaca US equities are commission-free; kept configurable for other venues.
    commission_per_share: float = 0.0
    commission_min: float = 0.0

    def commission(self, qty: float) -> float:
        raw = abs(qty) * self.commission_per_share
        return max(raw, self.commission_min) if (raw > 0 or self.commission_min > 0) else 0.0

    def regulatory(self, side_buy: bool, qty: float, price: float) -> float:
        """SEC + TAF — charged on SELLS only."""
        if side_buy:
            return 0.0
        notional = abs(qty) * price
        sec = notional * self.sec_fee_per_dollar
        taf = min(abs(qty) * self.taf_per_share, self.taf_cap)
        return sec + taf

    def fee(self, side_buy: bool, qty: float, price: float) -> float:
        """Total fee for a fill: commission (both sides) + regulatory (sells)."""
        return self.commission(qty) + self.regulatory(side_buy, qty, price)


# A zero-fee model (e.g. to reproduce a fee-free backtest or for unit isolation).
ZERO_FEES = AlpacaFeeModel(sec_fee_per_dollar=0.0, taf_per_share=0.0,
                           taf_cap=0.0, commission_per_share=0.0, commission_min=0.0)
