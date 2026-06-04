"""
Session-anchored intraday strategies — they reset their reference each trading day
via the runner's on_session_start() hook (so VWAP / the session-open are fresh daily).

  VWAPReclaim     - long when price reclaims the rolling session VWAP from below,
                    exit/short when it rejects (loses VWAP from above)
  SessionMomentum - ride the early-session move: long if the session is up >= entry_pct
                    after a warmup, flat/short when it fades back toward the open

Both NEED real epoch-second bar timestamps for the daily reset to fire (synthetic
integer-ts bars stay on one session — they still work, just never re-anchor).
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from alpca.strategies.base import Signal, Strategy, hold


class VWAPReclaim(Strategy):
    """
    Rolling session VWAP = sum(typical_price * volume) / sum(volume), typical =
    (high+low+close)/3, reset each session. Long when the close crosses UP through
    VWAP (reclaim from below); exit when it falls back below; with allow_short, short
    on a reject (close crosses DOWN through VWAP) and cover when it reclaims.
    """

    name = "vwap"

    def __init__(self, allow_short: bool = False) -> None:
        super().__init__()
        self.allow_short = allow_short
        self._cum_pv = 0.0
        self._cum_v = 0.0
        self._prev_above: Optional[bool] = None
        self._side = ""

    def reset(self) -> None:
        super().reset()
        self.on_session_start()
        self._side = ""

    def on_session_start(self) -> None:
        # fresh VWAP each trading day (position is NOT force-flattened)
        self._cum_pv = 0.0
        self._cum_v = 0.0
        self._prev_above = None

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close = bar["close"]
        if not math.isfinite(close):
            return hold("non-finite close")
        vol = bar.get("volume", 0.0) or 0.0
        tp = (bar.get("high", close) + bar.get("low", close) + close) / 3.0
        self._cum_pv += tp * vol
        self._cum_v += vol
        if self._cum_v <= 0:
            return hold("no volume for VWAP")
        vwap = self._cum_pv / self._cum_v
        above = close > vwap
        prev, self._prev_above = self._prev_above, above
        if prev is None:
            return hold("session warmup")

        if self._side == "":
            # above VWAP = reclaimed/bullish -> long; below = rejected/bearish -> short.
            # The flat gap after an exit lets the opposite side open on the next bar.
            if above:                          # strictly above VWAP
                self._side = "LONG"
                return self._enter(close, 1.0, f"reclaimed VWAP {vwap:.2f}", vwap=vwap)
            if self.allow_short and close < vwap:   # strictly below; AT VWAP = no signal
                self._side = "SHORT"
                return self._short(close, 1.0, f"rejected VWAP {vwap:.2f}", vwap=vwap)
            return hold("at/below VWAP (flat)")
        if self._side == "LONG":
            if not above:
                self._side = ""
                return self._exit(close, f"lost VWAP {vwap:.2f}")
            return hold("holding long")
        if above:
            self._side = ""
            return self._exit(close, f"reclaimed VWAP {vwap:.2f} (cover)")
        return hold("holding short")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True


class SessionMomentum(Strategy):
    """
    Opening/session momentum: ride the early-session move. After `min_bars` of the
    session, go long while the session return (close vs session open) is >= entry_pct;
    exit when it fades back to <= exit_pct. With allow_short, mirror on the downside.
    Liquidity/trends cluster intraday, so the direction of the early move often persists.
    """

    name = "session-momentum"

    def __init__(self, entry_pct: float = 0.005, exit_pct: float = 0.0,
                 min_bars: int = 5, allow_short: bool = False) -> None:
        super().__init__()
        self.entry_pct = entry_pct
        self.exit_pct = exit_pct
        self.min_bars = min_bars
        self.allow_short = allow_short
        self._open: Optional[float] = None
        self._bars = 0
        self._side = ""

    def reset(self) -> None:
        super().reset()
        self.on_session_start()
        self._side = ""

    def on_session_start(self) -> None:
        self._open = None
        self._bars = 0

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close = bar["close"]
        if not math.isfinite(close):
            return hold("non-finite close")
        if self._open is None:
            self._open = close
        self._bars += 1
        ret = (close - self._open) / self._open if self._open else 0.0
        if self._bars < self.min_bars:
            return hold("session warmup")

        if self._side == "":
            if ret >= self.entry_pct:
                self._side = "LONG"
                return self._enter(close, 1.0, f"session up {ret:+.2%}", ret=ret)
            if self.allow_short and ret <= -self.entry_pct:
                self._side = "SHORT"
                return self._short(close, 1.0, f"session down {ret:+.2%}", ret=ret)
            return hold("flat")
        if self._side == "LONG":
            if ret <= self.exit_pct:
                self._side = ""
                return self._exit(close, f"session momentum faded {ret:+.2%}")
            return hold("holding long")
        if ret >= -self.exit_pct:
            self._side = ""
            return self._exit(close, f"session momentum faded {ret:+.2%}")
        return hold("holding short")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True
