"""
Latency-sensitive breakout / trend strategies, ported from TradingBot:
  - DonchianBreakout   (channel breakout, ATR stop)
  - ORB                (Opening Range Breakout)
  - VolatilityBreakout (Keltner: EMA +/- mult*ATR bands)
  - Supertrend         (ATR bands around HL2, classic carry-forward)

Breakouts/trend flips are latency-sensitive: the edge decays in the seconds after
the level breaks, so the signal->order->fill latency this bot measures directly
bears on realized vs. backtested fills.

All strategies are long-only (BUY to enter, EXIT to flatten), matching the
Signal contract the router/backtester consume.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Optional

from alpca.strategies.base import Signal, Strategy, hold


def _atr(highs: List[float], lows: List[float], closes: List[float], period: int) -> Optional[float]:
    """True-range ATR over the last `period` bars (uses prior close for gaps)."""
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(-period, 0):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    return sum(trs) / len(trs)


class DonchianBreakout(Strategy):
    """
    BUY when close breaks above the highest high of the prior `period` bars;
    EXIT on a break below the prior `period`-bar low or an ATR-based stop.
    """

    name = "donchian"

    def __init__(self, period: int = 20, atr_period: int = 14, stop_atr_mult: float = 2.0,
                 entry: str = "market") -> None:
        super().__init__()
        self.period = period
        self.atr_period = atr_period
        self.stop_atr_mult = stop_atr_mult
        # entry="market": act at next open when close>prior_high (legacy).
        # entry="stop":   rest a buy-STOP at the channel high so a future bar that
        #                 trades up THROUGH the level fills it (classic turtle
        #                 execution; captures the intrabar break, no look-ahead).
        self.entry = entry
        self._highs: Deque[float] = deque(maxlen=period + 1)
        self._lows: Deque[float] = deque(maxlen=period + 1)
        self._closes: Deque[float] = deque(maxlen=max(period, atr_period) + 2)
        self._stop_price: Optional[float] = None
        self._last_atr: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._stop_price = None
        self._last_atr = None

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        high, low, close = bar["high"], bar["low"], bar["close"]

        prior_high = max(self._highs) if len(self._highs) >= self.period else None
        prior_low = min(self._lows) if len(self._lows) >= self.period else None

        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        atr = _atr(list(self._highs), list(self._lows), list(self._closes), self.atr_period)
        self._last_atr = atr

        if prior_high is None or prior_low is None or atr is None:
            return hold("warmup")

        if not self._in_position:
            if self.entry == "stop":
                # Maintain a resting buy-stop at the current channel high. The
                # runner dedups (cancel-replaces) as the level moves; on_fill()
                # flips us in-position when it triggers.
                return self._rest_buy_stop(prior_high, reason=f"buy-stop @ {prior_high:.2f}",
                                           tif="GTC", atr=atr)
            if close > prior_high:
                self._stop_price = close - self.stop_atr_mult * atr
                return self._enter(close, 1.0, f"Donchian breakout > {prior_high:.2f}",
                                   stop=self._stop_price, atr=atr)
            return hold()
        if close < prior_low or (self._stop_price is not None and close <= self._stop_price):
            self._stop_price = None
            return self._exit(close, "Donchian exit (low break/stop)")
        return hold("holding")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "BUY":
            self._in_position = True
            if self._last_atr is not None:
                self._stop_price = price - self.stop_atr_mult * self._last_atr
        else:
            self._in_position = False
            self._stop_price = None


class ORB(Strategy):
    """
    Opening Range Breakout: build the opening range (high/low) over the first
    `range_bars` bars, BUY on a break above the range high, EXIT on stop /
    take-profit / break below the range low. Call reset() at each new session.
    """

    name = "orb"

    def __init__(self, range_bars: int = 5, stop_pct: float = 0.02,
                 take_profit_pct: float = 0.04) -> None:
        super().__init__()
        self.range_bars = range_bars
        self.stop_pct = stop_pct
        self.take_profit_pct = take_profit_pct
        self._bar_count = 0
        self._range_high: Optional[float] = None
        self._range_low: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self._bar_count = 0
        self._range_high = None
        self._range_low = None

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        high, low, close = bar["high"], bar["low"], bar["close"]
        self._bar_count += 1

        if self._bar_count <= self.range_bars:
            self._range_high = max(self._range_high if self._range_high is not None else high, high)
            self._range_low = min(self._range_low if self._range_low is not None else low, low)
            return hold("building range")

        if self._range_high is None or self._range_low is None:
            return hold("no range")

        if not self._in_position:
            if close > self._range_high:
                return self._enter(close, 1.0, f"ORB breakout > {self._range_high:.2f}",
                                   range_high=self._range_high, range_low=self._range_low)
            return hold()
        stop = self._entry_price * (1 - self.stop_pct)
        target = self._entry_price * (1 + self.take_profit_pct)
        if close <= stop or close >= target or close < self._range_low:
            reason = ("stop" if close <= stop else
                      "target" if close >= target else "range low break")
            return self._exit(close, f"ORB exit ({reason})")
        return hold("holding")


class VolatilityBreakout(Strategy):
    """
    Keltner-style volatility breakout (ported from TradingBot VolatilityBreakout).
    Long-only: BUY when close breaks above EMA + multiplier*ATR; EXIT when close
    re-enters below the upper band or hits a percentage stop.
    """

    name = "keltner"

    def __init__(self, ema_period: int = 20, atr_period: int = 10,
                 multiplier: float = 1.5, stop_pct: float = 0.02) -> None:
        super().__init__()
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.stop_pct = stop_pct
        self._closes: Deque[float] = deque(maxlen=max(ema_period, atr_period) + 2)
        self._highs: Deque[float] = deque(maxlen=atr_period + 2)
        self._lows: Deque[float] = deque(maxlen=atr_period + 2)
        self._ema: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self._closes.clear()
        self._highs.clear()
        self._lows.clear()
        self._ema = None

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close, high, low = bar["close"], bar["high"], bar["low"]
        self._closes.append(close)
        self._highs.append(high)
        self._lows.append(low)

        if len(self._closes) < self.ema_period:
            return hold("warmup")

        if self._ema is None:
            self._ema = sum(self._closes) / len(self._closes)
        else:
            k = 2.0 / (self.ema_period + 1)
            self._ema = close * k + self._ema * (1 - k)

        atr = _atr(list(self._highs), list(self._lows), list(self._closes), self.atr_period)
        if atr is None or atr == 0:
            return hold("atr-warmup")

        upper = self._ema + self.multiplier * atr

        if not self._in_position:
            if close > upper:
                return self._enter(close, min(1.0, (close - upper) / atr),
                                   "vol breakout up", upper=upper, atr=atr)
            return hold()
        if close <= self._entry_price * (1 - self.stop_pct) or close < upper:
            return self._exit(close, "vol exit (re-enter/stop)")
        return hold("holding")


class Supertrend(Strategy):
    """
    Supertrend (ported from TradingBot): ATR bands around HL2 with the classic
    carry-forward rules; long-only — long while direction is up, flat otherwise.
    """

    name = "supertrend"

    def __init__(self, atr_period: int = 10, multiplier: float = 3.0) -> None:
        super().__init__()
        self.atr_period = atr_period
        self.multiplier = multiplier
        self._highs: Deque[float] = deque(maxlen=atr_period + 2)
        self._lows: Deque[float] = deque(maxlen=atr_period + 2)
        self._closes: Deque[float] = deque(maxlen=atr_period + 2)
        self._direction: int = 0
        self._final_upper: Optional[float] = None
        self._final_lower: Optional[float] = None
        self._prev_close: Optional[float] = None

    def reset(self) -> None:
        super().reset()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._direction = 0
        self._final_upper = None
        self._final_lower = None
        self._prev_close = None

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close, high, low = bar["close"], bar["high"], bar["low"]
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)

        atr = _atr(list(self._highs), list(self._lows), list(self._closes), self.atr_period)
        if atr is None or atr == 0:
            self._prev_close = close
            return hold("warmup")

        hl2 = (high + low) / 2.0
        basic_upper = hl2 + self.multiplier * atr
        basic_lower = hl2 - self.multiplier * atr

        if self._final_upper is None or self._prev_close is None:
            final_upper, final_lower = basic_upper, basic_lower
        else:
            pc = self._prev_close
            final_upper = (basic_upper if (basic_upper < self._final_upper or pc > self._final_upper)
                           else self._final_upper)
            final_lower = (basic_lower if (basic_lower > self._final_lower or pc < self._final_lower)
                           else self._final_lower)

        if self._direction == 0:
            self._direction = 1 if close >= hl2 else -1
        elif self._direction == 1 and close < final_lower:
            self._direction = -1
        elif self._direction == -1 and close > final_upper:
            self._direction = 1

        self._final_upper, self._final_lower, self._prev_close = final_upper, final_lower, close

        if self._direction == 1 and not self._in_position:
            return self._enter(close, 1.0, "supertrend up")
        if self._direction == -1 and self._in_position:
            return self._exit(close, "supertrend down")
        return hold("holding" if self._in_position else "flat")
