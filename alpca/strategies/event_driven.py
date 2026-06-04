"""
Event-driven (session-open) strategies.

  - GapFade: fade the opening gap (open vs prior-session close) on the first bar
    of a new trading session, expecting reversion back toward the prior close.

The open is the single most latency-sensitive, highest-edge-decay window for US
equities, which is exactly the regime this bot instruments: the seconds between
the gap forming and the order filling decide the realized-vs-backtest slippage.

Session boundaries are detected from the bar TIMESTAMP via the NYSE calendar
(`session_date`), because the runner does NOT call strategy.reset() between
sessions. The strategy therefore needs REAL epoch-second timestamps to fire;
synthetic integer-timestamp bars all collapse to one calendar date and produce
no gap (documented, same caveat as the calendar-gated backtest path).
"""

from __future__ import annotations

from typing import Dict, Optional

from alpca.data.calendar import session_date
from alpca.strategies.base import Signal, Strategy, hold


class GapFade(Strategy):
    """
    Opening-gap mean-reversion.

    On the FIRST bar of a new session, measure the gap of this session's open vs
    the prior session's close:
        gap = (open - prev_close) / prev_close
    Then fade it (bet on reversion toward prev_close):
        gap <= -entry_pct  -> BUY  (fade a gap DOWN; long-only path, no short needed)
        gap >= +entry_pct  -> SHORT (fade a gap UP; needs allow_short + RiskConfig.allow_short)

    Exit (checked every bar while in a position):
        revert : |close - prev_close| / prev_close <= exit_pct   (reached target)
        stop   : close moved stop_pct against the entry
        time   : held for hold_bars bars

    A position still open when a NEW session starts is flattened at that session's
    first bar (the gap trade is intraday; we never carry it overnight).

    `allow_short` defaults False (long-only gap-down fade). The two-sided variant
    is registered as 'gap-fade-ls' and also requires RiskConfig.allow_short on the
    runner for the short leg to actually execute.
    """

    name = "gap-fade"

    def __init__(self, entry_pct: float = 0.01, exit_pct: float = 0.002,
                 stop_pct: float = 0.01, hold_bars: int = 30,
                 allow_short: bool = False) -> None:
        super().__init__()
        self.entry_pct = entry_pct
        self.exit_pct = exit_pct
        self.stop_pct = stop_pct
        self.hold_bars = hold_bars
        self.allow_short = allow_short
        self._side = ""                       # "LONG" | "SHORT" | ""
        self._cur_date: Optional[str] = None  # ET trading date of the last bar
        self._ref_close: Optional[float] = None  # prior session's close (frozen)
        self._running_close: float = 0.0      # last close seen (becomes ref at rollover)
        self._bars_held: int = 0

    def reset(self) -> None:
        super().reset()
        self._side = ""
        self._cur_date = None
        self._ref_close = None
        self._running_close = 0.0
        self._bars_held = 0

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        open_, close = bar["open"], bar["close"]
        ts = float(bar.get("timestamp", 0) or 0)
        date = session_date(ts)
        first_ever = self._cur_date is None
        new_session = (not first_ever) and date != self._cur_date

        # --- session rollover -------------------------------------------------
        if new_session:
            prior_close = self._running_close  # last close of the prior session
            if self._in_position:
                # flatten an intraday gap trade carried into the new session
                self._side = ""
                self._bars_held = 0
                self._ref_close = prior_close
                self._cur_date = date
                self._running_close = close
                return self._exit(close, "gap-fade rollover flatten")
            self._ref_close = prior_close
            self._bars_held = 0
        self._cur_date = date

        # --- manage an open position -----------------------------------------
        if self._in_position:
            self._bars_held += 1
            sig = self._manage_exit(close)
            self._running_close = close
            return sig

        # --- flat: only fade the gap on the FIRST bar of a new session -------
        if new_session and self._ref_close:
            gap = (open_ - self._ref_close) / self._ref_close
            self._running_close = close
            if gap <= -self.entry_pct:
                self._side = "LONG"
                self._bars_held = 0
                return self._enter(close, 1.0, f"gap-fade long gap={gap:.3%}",
                                   gap=gap, ref=self._ref_close)
            if self.allow_short and gap >= self.entry_pct:
                self._side = "SHORT"
                self._bars_held = 0
                return self._short(close, 1.0, f"gap-fade short gap={gap:.3%}",
                                   gap=gap, ref=self._ref_close)
            return hold(f"no gap ({gap:.3%})")

        self._running_close = close
        return hold("warmup" if first_ever else "flat")

    def _manage_exit(self, close: float) -> Signal:
        ref = self._ref_close
        entry = self._entry_price
        reverted = bool(ref) and abs(close - ref) / ref <= self.exit_pct
        timed = self._bars_held >= self.hold_bars
        if self._side == "LONG":
            stopped = entry > 0 and close <= entry * (1 - self.stop_pct)
            if reverted or stopped or timed:
                self._side = ""
                r = "revert" if reverted else ("stop" if stopped else "time")
                return self._exit(close, f"gap-fade long exit ({r})")
            return hold("holding gap-long")
        if self._side == "SHORT":
            stopped = entry > 0 and close >= entry * (1 + self.stop_pct)
            if reverted or stopped or timed:
                self._side = ""
                r = "revert" if reverted else ("stop" if stopped else "time")
                return self._exit(close, f"gap-fade short exit ({r})")
            return hold("holding gap-short")
        return hold("flat")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        # keep _side/_in_position consistent if a short fills via a path that
        # doesn't go through _short() (mirrors ZScoreMeanReversion.on_fill).
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True
