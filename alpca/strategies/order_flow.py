"""
Order-flow-imbalance (OFI) strategy #N — a new microstructure SIGNAL computable
from top-of-book NBBO only (no L2 depth), so it runs on Alpaca/IEX quotes.

L1 OFI (Cont–Kukanov–Stoikov) measures net order flow pushing price up between two
consecutive quote snapshots:

    bid leg  ΔW = +bid_size            if bid price rose
                  (bid_size - prev)    if bid price unchanged
                  -prev_bid_size       if bid price fell
    ask leg  ΔV = -prev_ask_size       if ask price rose      (supply retreated → bullish)
                  (ask_size - prev)    if ask price unchanged
                  +ask_size            if ask price fell       (aggressive selling → bearish)
    OFI       e = ΔW - ΔV

We accumulate e over a rolling window and NORMALIZE by the window's total top-of-
book size so the threshold is scale-free across symbols (raw OFI is in shares).
Reimplemented from nicolezattarin/LOB-feature-analysis (Apache-2.0) + the Cont et
al. paper. Reads bid/ask/bid_size/ask_size off the (quote-enriched) bar dict.

CALIBRATION NOTE: entry=0.19 / exit=0.05→0.08 are the CALIBRATED bar-level deadbands
(default window=20 bars). Fit by scripts/analyze_ofi_bars.py on full-session contiguous
qbars (5 sessions × SPY/QQQ/AAPL, ~1,840 genuine 20-bar-window |normOFI| values each,
100% per-bar NBBO coverage from scripts/sample_quotes_fullsession.py) → entry = mean
p90 (0.204/0.213/0.157), exit = mean p50 (0.090/0.093/0.062); see data/ofi_deadbands
.json. NOTE the bar-level p90 (~0.19) is ~2x the TICK-level p90 (~0.10 from analyze_
microstructure.py): the tick window rolls over seconds, this over 20 minutes, so the
tick fit was correctly NOT transferred. Re-fit with the same script + window if you
change the bar interval (a 5-min-bar L1OFI would have a different distribution).
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional, Tuple

from alpca.strategies.base import Signal, Strategy, hold


def ofi_event(bid, bid_size, ask, ask_size,
              prev_bid, prev_bid_size, prev_ask, prev_ask_size) -> float:
    """One L1 OFI increment e = ΔW - ΔV between two NBBO snapshots (shares)."""
    if bid > prev_bid:
        dW = bid_size
    elif bid == prev_bid:
        dW = bid_size - prev_bid_size
    else:
        dW = -prev_bid_size

    if ask > prev_ask:
        dV = -prev_ask_size
    elif ask == prev_ask:
        dV = ask_size - prev_ask_size
    else:
        dV = ask_size
    return dW - dV


class L1OFI(Strategy):
    """
    Long (and optionally short) on persistent top-of-book order-flow imbalance.

    normalized OFI = Σ e over `window` / Σ (bid_size+ask_size) over `window`  ∈ ~[-1,1]
      BUY   when normOFI > +entry  (sustained buy pressure)
      SHORT when normOFI < -entry  (allow_short; needs RiskConfig.allow_short)
      EXIT  when the imbalance fades back inside ±exit
    """

    name = "ofi"

    def __init__(self, window: int = 20, entry: float = 0.19, exit: float = 0.08,
                 allow_short: bool = False) -> None:
        super().__init__()
        self.window = window
        self.entry = entry
        self.exit = exit
        self.allow_short = allow_short
        self._e: Deque[float] = deque(maxlen=window)
        self._sz: Deque[float] = deque(maxlen=window)
        self._prev: Optional[Tuple[float, float, float, float]] = None
        self._side = ""  # "LONG" | "SHORT" | ""

    def reset(self) -> None:
        super().reset()
        self._e.clear()
        self._sz.clear()
        self._prev = None
        self._side = ""

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        bid, ask = bar.get("bid"), bar.get("ask")
        bs, az = bar.get("bid_size"), bar.get("ask_size")
        if bid is None or ask is None or bs is None or az is None:
            return hold("no quote")

        if self._prev is None:
            self._prev = (bid, bs, ask, az)
            return hold("warmup")

        pb, pbs, pa, paz = self._prev
        e = ofi_event(bid, bs, ask, az, pb, pbs, pa, paz)
        self._prev = (bid, bs, ask, az)
        self._e.append(e)
        self._sz.append(bs + az)
        if len(self._e) < self.window:
            return hold("warmup")

        denom = sum(self._sz) or 1.0
        ofi = sum(self._e) / denom
        close = bar["close"]
        strength = min(1.0, abs(ofi) / self.entry) if self.entry > 0 else 1.0

        if self._side == "":
            if ofi > self.entry:
                self._side = "LONG"
                return self._enter(close, strength, f"OFI+ {ofi:.3f}", ofi=ofi)
            if self.allow_short and ofi < -self.entry:
                self._side = "SHORT"
                return self._short(close, strength, f"OFI- {ofi:.3f}", ofi=ofi)
            return hold("flat")

        if self._side == "LONG":
            if ofi < self.exit:
                self._side = ""
                return self._exit(close, f"OFI revert {ofi:.3f}", ofi=ofi)
            return hold("holding long")

        if ofi > -self.exit:
            self._side = ""
            return self._exit(close, f"OFI revert {ofi:.3f}", ofi=ofi)
        return hold("holding short")

    def on_session_start(self) -> None:
        # Reset the intraday OFI accumulation so the 20-bar window never straddles
        # an overnight gap (a quote-event computed against yesterday's last NBBO is
        # meaningless). The POSITION (_side / _in_position) is intentionally NOT
        # reset — if a position is carried overnight it persists; only the rolling
        # imbalance estimate restarts, re-warming over the new session's first bars.
        self._e.clear()
        self._sz.clear()
        self._prev = None

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True
