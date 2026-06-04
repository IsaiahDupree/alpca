"""
Microstructure (NBBO microprice) — strategy #3.

Full L2 order-flow-imbalance (OFI) is NOT available on Alpaca's IEX feed (no
depth), so we use the portable top-of-book kernel: the MICROPRICE, a size-weighted
fair value that needs only the best bid/ask and their sizes (which Alpaca quotes
carry). It tilts toward the side with the larger opposing queue:

    microprice = (ask * bid_size + bid * ask_size) / (bid_size + ask_size)

When bid_size >> ask_size (heavy buying pressure) the microprice rides up toward
the ask; the normalized tilt (microprice - mid) / half_spread lives in [-1, +1].

This is the most latency-sensitive idea in the codebase — the tilt decays in
milliseconds — so it most directly exercises the signal->fill latency metrics that
are this bot's reason to exist. It is best used as a CONFIRMATION GATE on another
strategy's entries (MicropriceGate), not standalone; a standalone MicropriceTilt
is provided for instrumentation/demos.

Quote plumbing: strategies read bid/ask/bid_size/ask_size off the BAR dict. The
runner already passes the whole bar to on_bar(), so QuoteEnrichedBarSource
(data/feed.py) just merges the latest NBBO onto each bar — no runner/ABC change.
NBBO-only: top-of-book, no depth, no spoofing resistance.

CALIBRATION (k default): the deadband k = 0.5 is the 75th percentile of |tilt| fit
from 972k real IEX NBBO quotes (SPY/QQQ/AAPL, 9 trading days × 3 intraday regimes,
regular session only — see scripts/analyze_microstructure.py + data/microstructure_
deadbands.json). p75|tilt| landed at 0.50 / 0.556 / 0.50 across the three symbols,
so the gate only confirms on a top-quartile book imbalance rather than any flicker.
Unlike the OFI bar-vs-tick mismatch, the microprice tilt is the SAME instantaneous
quantity on a tick or on a bar's attached NBBO, so this tick fit applies directly.
"""

from __future__ import annotations

from typing import Dict, Optional

from alpca.strategies.base import BUY, EXIT, HOLD, SELL, Signal, Strategy, hold


# --------------------------------------------------------------- pure functions
def microprice(bid, ask, bid_size, ask_size) -> Optional[float]:
    """Size-weighted fair value, or None if the quote is incomplete/degenerate."""
    if bid is None or ask is None or bid_size is None or ask_size is None:
        return None
    total = bid_size + ask_size
    if total <= 0:
        return None
    return (ask * bid_size + bid * ask_size) / total


def microprice_tilt(bid, ask, bid_size, ask_size) -> Optional[float]:
    """(microprice - mid) / half_spread in [-1, +1]; None for locked/crossed/empty
    quotes (half_spread <= 0) where the tilt can't be normalized."""
    mp = microprice(bid, ask, bid_size, ask_size)
    if mp is None:
        return None
    half = (ask - bid) / 2.0
    if half <= 0:
        return None
    mid = (bid + ask) / 2.0
    return (mp - mid) / half


def microprice_signal(bid, ask, bid_size, ask_size, k: float = 0.0) -> Optional[str]:
    """'bull' / 'bear' / 'flat' from the tilt with a deadband k in [0, 1); None if
    no usable quote. k filters out weak imbalances (k=0 => any nonzero tilt)."""
    t = microprice_tilt(bid, ask, bid_size, ask_size)
    if t is None:
        return None
    if t > k:
        return "bull"
    if t < -k:
        return "bear"
    return "flat"


def _quote_of(bar: Dict[str, float]):
    return (bar.get("bid"), bar.get("ask"), bar.get("bid_size"), bar.get("ask_size"))


# --------------------------------------------------------------- confirmation gate
class MicropriceGate(Strategy):
    """
    Wrap any strategy; let its ENTRY signals through only when the microprice tilt
    confirms the direction (BUY needs 'bull', SELL needs 'bear'). HOLD and EXIT
    always pass — never block an exit. This is the recommended use of microprice.

    On a block we ROLL BACK the base strategy's optimistic entry state
    (_in_position / _entry_price / _side, set by _enter/_short) so it re-evaluates
    the entry on the next bar instead of getting stuck "in position" — the same
    desync the risk engine would otherwise cause on a rejected entry.

    `require_quote`: if True (default) an entry is blocked when no usable quote is
    on the bar; if False, missing quotes pass entries through (gate is a no-op
    offline). Reads bid/ask/bid_size/ask_size off the bar dict.
    """

    def __init__(self, base: Strategy, *, k: float = 0.5, require_quote: bool = True) -> None:
        super().__init__()
        self.base = base
        self.k = k
        self.require_quote = require_quote
        self.name = f"{base.name}+mp"

    def reset(self) -> None:
        super().reset()
        self.base.reset()

    def on_fill(self, side: str, qty: float, price: float) -> None:
        self.base.on_fill(side, qty, price)

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        snap = (self.base._in_position, getattr(self.base, "_side", None),
                self.base._entry_price)
        sig = self.base.on_bar(bar)
        if sig.side in (HOLD, EXIT):
            return sig

        tilt = microprice_signal(*_quote_of(bar), k=self.k)
        confirmed = ((sig.side == BUY and tilt == "bull") or
                     (sig.side == SELL and tilt == "bear"))
        if confirmed or (tilt is None and not self.require_quote):
            return sig

        # blocked: undo the base's optimistic entry flip so it retries next bar
        self.base._in_position, self.base._entry_price = snap[0], snap[2]
        if hasattr(self.base, "_side"):
            self.base._side = snap[1]
        why = "no quote" if tilt is None else f"tilt={tilt}"
        return hold(f"mp gate blocked {sig.side} ({why})")


def gate(base: Strategy, *, k: float = 0.5, require_quote: bool = True) -> MicropriceGate:
    """Convenience: wrap `base` in a MicropriceGate. k=0.5 = calibrated p75|tilt|."""
    return MicropriceGate(base, k=k, require_quote=require_quote)


# --------------------------------------------------------------- standalone
class MicropriceTilt(Strategy):
    """
    Standalone microprice tilt follower (mainly for instrumentation). Go LONG while
    the tilt is 'bull', flatten when it stops being bull; with allow_short, go SHORT
    while 'bear'. Edge decays in milliseconds — this is the latency stress test.
    """

    name = "microprice"

    def __init__(self, k: float = 0.5, allow_short: bool = False) -> None:
        super().__init__()
        self.k = k          # calibrated p75|tilt| (see module CALIBRATION note)
        self.allow_short = allow_short
        self._side = ""

    def reset(self) -> None:
        super().reset()
        self._side = ""

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        bid, ask, bs, as_ = _quote_of(bar)
        direction = microprice_signal(bid, ask, bs, as_, k=self.k)
        if direction is None:
            return hold("no quote")
        close = bar["close"]
        t = microprice_tilt(bid, ask, bs, as_)

        if self._side == "":
            if direction == "bull":
                self._side = "LONG"
                return self._enter(close, 1.0, f"microprice bull tilt={t:.2f}", tilt=t)
            if self.allow_short and direction == "bear":
                self._side = "SHORT"
                return self._short(close, 1.0, f"microprice bear tilt={t:.2f}", tilt=t)
            return hold("flat")

        if self._side == "LONG":
            if direction != "bull":
                self._side = ""
                return self._exit(close, f"microprice tilt faded ({direction})")
            return hold("holding long")

        if direction != "bear":
            self._side = ""
            return self._exit(close, f"microprice tilt recovered ({direction})")
        return hold("holding short")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True
