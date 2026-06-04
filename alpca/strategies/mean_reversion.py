"""
Latency-sensitive mean-reversion, ported from TradingBot:
  - ZScoreMeanReversion (fast-bar reversion on a rolling z-score)
  - RSIMeanReversion    (Wilder-RSI reversion with a volatility-regime gate)

On fast bars, the reversion edge is short-lived — measuring how long it takes a
signal to become a fill (and the slippage incurred) is exactly the point.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Deque, Dict, List, Optional

from alpca.strategies.base import Signal, Strategy, hold


def wilder_rsi(closes: List[float], period: int) -> Optional[float]:
    """
    Wilder's RSI over `period` on a close series. Returns None until there are at
    least `period + 1` closes. Recomputed from the buffer each call (O(n)),
    matching the _atr recompute style in breakout.py; the strategy's deque is
    bounded so cost stays flat. avg_loss==0 -> RSI 100 (all-up window).
    """
    if len(closes) < period + 1:
        return None
    gain = 0.0
    loss = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        if d >= 0:
            gain += d
        else:
            loss += -d
    avg_gain = gain / period
    avg_loss = loss / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period
    if avg_loss == 0:
        # no DOWN moves: an all-UP series is maximally strong (100), but a perfectly
        # FLAT series (no up moves either) is NEUTRAL (50), not strong — otherwise a
        # zero-information tape reads as 100 and momentum strategies wrongly enter.
        return 50.0 if avg_gain == 0 else 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def rolling_return_vol(closes: List[float], lookback: int) -> Optional[float]:
    """
    Population stdev of the last `lookback` simple bar-to-bar returns — a cheap
    realized-vol estimate for regime gating. Returns None until enough closes.
    Reusable filter for any strategy that wants to trade only in a vol band.
    """
    if len(closes) < lookback + 1:
        return None
    rets = []
    for i in range(len(closes) - lookback, len(closes)):
        p0 = closes[i - 1]
        if p0 and math.isfinite(p0) and math.isfinite(closes[i]):
            r = (closes[i] - p0) / p0
            if math.isfinite(r):
                rets.append(r)
    if len(rets) < 2:
        return None
    return statistics.pstdev(rets)


class ZScoreMeanReversion(Strategy):
    """
    z = (close - rolling_mean) / rolling_pstd over `lookback`.

    Long-only mode (default):
      BUY  when z < -entry_z (oversold).
      EXIT when |z| < exit_z (reverted) or z < -stop_z (blown through).

    Long/short mode (`allow_short=True`, needs RiskConfig.allow_short on the runner):
      additionally SHORT when z > +entry_z (overbought), covering when |z| < exit_z
      or z > +stop_z. This is the natural symmetric mean-reversion trade and the
      reason to support shorts at all.

    Tracks `_side` ("LONG"/"SHORT"/"") so on_fill keeps it in sync when entries
    are placed as resting orders (or rejected).
    """

    name = "zscore"

    def __init__(self, lookback: int = 60, entry_z: float = 2.0,
                 exit_z: float = 0.5, stop_z: float = 3.5,
                 allow_short: bool = False) -> None:
        super().__init__()
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z
        self.allow_short = allow_short
        self._closes: Deque[float] = deque(maxlen=lookback + 1)
        self._side = ""  # "LONG" | "SHORT" | ""

    def reset(self) -> None:
        super().reset()
        self._closes.clear()
        self._side = ""

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close = bar["close"]
        if not math.isfinite(close):
            return hold("non-finite close")
        self._closes.append(close)
        if len(self._closes) < self.lookback:
            return hold("warmup")

        vals = list(self._closes)
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals)
        if std == 0:
            return hold("flat")
        z = (close - mean) / std

        if self._side == "":
            if z < -self.entry_z:
                self._side = "LONG"
                return self._enter(close, 1.0, f"z-score oversold z={z:.2f}", z=z)
            if self.allow_short and z > self.entry_z:
                self._side = "SHORT"
                return self._short(close, 1.0, f"z-score overbought z={z:.2f}", z=z)
            return hold()

        if self._side == "LONG":
            if abs(z) < self.exit_z or z < -self.stop_z:
                self._side = ""
                return self._exit(close, f"z-score revert/stop z={z:.2f}", z=z)
            return hold("holding long")

        # SHORT: cover when reverted or blown through to the upside
        if abs(z) < self.exit_z or z > self.stop_z:
            self._side = ""
            return self._exit(close, f"z-score cover z={z:.2f}", z=z)
        return hold("holding short")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        # keep _side / _in_position consistent with reality (e.g. a short that was
        # rejected when allow_short is off should not leave us thinking we're in).
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True
        # a flatten (cover/exit) is reflected by the strategy's own _exit() call;
        # nothing extra needed here for the market path.


class RSIMeanReversion(Strategy):
    """
    Wilder-RSI mean-reversion with a volatility-regime gate — a second MR signal
    distinct from the z-score (different trigger) that adds a reusable vol filter.

    Long-only mode (default):
      BUY  when rsi < entry_low (oversold) AND realized vol is in [vol_floor, vol_cap].
      EXIT when rsi >= exit_level (reverted to neutral) or a stop_pct loss.

    Long/short mode (`allow_short=True`, needs RiskConfig.allow_short on the runner):
      additionally SHORT when rsi > entry_high (overbought), covering when
      rsi <= exit_level or a stop_pct loss.

    The vol gate constrains ENTRIES only (never blocks an exit). It defaults wide
    open (vol_floor=0, vol_cap=inf) so the bare RSI behaves predictably; set the
    band to trade only in a chosen volatility regime. Classic config is RSI(2)
    with entry_low=10/entry_high=90 (Connors); RSI(14) also works.
    """

    name = "rsi-mr"

    def __init__(self, rsi_period: int = 2, entry_low: float = 10.0,
                 entry_high: float = 90.0, exit_level: float = 50.0,
                 vol_lookback: int = 20, vol_floor: float = 0.0,
                 vol_cap: float = float("inf"), stop_pct: float = 0.05,
                 allow_short: bool = False) -> None:
        super().__init__()
        self.rsi_period = rsi_period
        self.entry_low = entry_low
        self.entry_high = entry_high
        self.exit_level = exit_level
        self.vol_lookback = vol_lookback
        self.vol_floor = vol_floor
        self.vol_cap = vol_cap
        self.stop_pct = stop_pct
        self.allow_short = allow_short
        buf = max(rsi_period, vol_lookback) * 3 + 5
        self._closes: Deque[float] = deque(maxlen=buf)
        self._side = ""  # "LONG" | "SHORT" | ""

    def reset(self) -> None:
        super().reset()
        self._closes.clear()
        self._side = ""

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close = bar["close"]
        if not math.isfinite(close):
            return hold("non-finite close")
        self._closes.append(close)
        vals = list(self._closes)
        rsi = wilder_rsi(vals, self.rsi_period)
        if rsi is None:
            return hold("warmup")

        if self._side == "":
            vol = rolling_return_vol(vals, self.vol_lookback)
            in_band = vol is not None and self.vol_floor <= vol <= self.vol_cap
            if in_band and rsi < self.entry_low:
                self._side = "LONG"
                return self._enter(close, 1.0, f"RSI {rsi:.1f} oversold", rsi=rsi, vol=vol)
            if in_band and self.allow_short and rsi > self.entry_high:
                self._side = "SHORT"
                return self._short(close, 1.0, f"RSI {rsi:.1f} overbought", rsi=rsi, vol=vol)
            return hold("flat")

        if self._side == "LONG":
            stopped = self._entry_price > 0 and close <= self._entry_price * (1 - self.stop_pct)
            if rsi >= self.exit_level or stopped:
                self._side = ""
                return self._exit(close, f"RSI {rsi:.1f} exit" + (" stop" if stopped else ""), rsi=rsi)
            return hold("holding long")

        # SHORT: cover when reverted to neutral or stopped out to the upside
        stopped = self._entry_price > 0 and close >= self._entry_price * (1 + self.stop_pct)
        if rsi <= self.exit_level or stopped:
            self._side = ""
            return self._exit(close, f"RSI {rsi:.1f} cover" + (" stop" if stopped else ""), rsi=rsi)
        return hold("holding short")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True
