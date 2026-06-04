"""
Momentum / trend-following strategies — "buy strength" brains, distinct from the
mean-reversion family. All single-asset, bar-only (work on the equities runner);
each is long-only by default and gains a symmetric short leg with allow_short=True.

Contents:
  ema()                  - incremental EMA update helper
  EMACrossMomentum       - fast/slow EMA crossover (reacts faster than SMA)
  MACDTrend              - MACD line vs signal + histogram confirmation
  RSIMomentum            - long while RSI is STRONG (momentum), not fading it
  ATRBreakout            - breakout only counts if it clears the range by k*ATR
  VolRegimeGate          - composable filter: pass a base strategy's entries only
                           when realized volatility is inside a chosen band

These intentionally overlap in spirit with Supertrend (already in breakout.py) but
trigger on different signals, giving an ensemble genuinely different voters.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from typing import Deque, Dict, Optional

from alpca.strategies.base import BUY, EXIT, HOLD, SELL, Signal, Strategy, hold
from alpca.strategies.mean_reversion import rolling_return_vol, wilder_rsi


def ema(prev: Optional[float], value: float, period: int) -> float:
    """One incremental EMA step. Seeds with `value` when prev is None."""
    if prev is None:
        return value
    alpha = 2.0 / (period + 1.0)
    return prev + alpha * (value - prev)


class EMACrossMomentum(Strategy):
    """
    Long while the fast EMA is above the slow EMA; flat (or short) while below.
    Faster to react than an SMA cross — the canonical intraday momentum brain.
    """

    name = "ema-momentum"

    def __init__(self, fast: int = 12, slow: int = 26, allow_short: bool = False) -> None:
        super().__init__()
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        self.fast = fast
        self.slow = slow
        self.allow_short = allow_short
        self._fast: Optional[float] = None
        self._slow: Optional[float] = None
        self._n = 0
        self._side = ""

    def reset(self) -> None:
        super().reset()
        self._fast = self._slow = None
        self._n = 0
        self._side = ""

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close = bar["close"]
        if not math.isfinite(close):
            return hold("non-finite close")
        self._fast = ema(self._fast, close, self.fast)
        self._slow = ema(self._slow, close, self.slow)
        self._n += 1
        if self._n < self.slow:
            return hold("warmup")

        bull = self._fast > self._slow
        if self._side == "":
            if bull:
                self._side = "LONG"
                return self._enter(close, 1.0, "EMA fast>slow (momentum up)")
            if self.allow_short and self._fast < self._slow:   # strict bear; flat (fast==slow) is no-signal
                self._side = "SHORT"
                return self._short(close, 1.0, "EMA fast<slow (momentum down)")
            return hold("flat")
        if self._side == "LONG":
            if not bull:
                self._side = ""
                return self._exit(close, "EMA cross down")
            return hold("holding long")
        # SHORT
        if bull:
            self._side = ""
            return self._exit(close, "EMA cross up")
        return hold("holding short")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True


class MACDTrend(Strategy):
    """
    MACD = EMA(fast) - EMA(slow); signal = EMA(MACD, signal_period); hist = MACD-signal.
    Long when MACD is above its signal line (hist > 0); flat/short when below. A second
    momentum voter that reacts to the *rate of change* of the trend, not just its level.
    """

    name = "macd"

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9,
                 allow_short: bool = False) -> None:
        super().__init__()
        if fast >= slow:
            raise ValueError("fast period must be < slow period")
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.allow_short = allow_short
        self._fast: Optional[float] = None
        self._slow: Optional[float] = None
        self._sig: Optional[float] = None
        self._n = 0
        self._side = ""

    def reset(self) -> None:
        super().reset()
        self._fast = self._slow = self._sig = None
        self._n = 0
        self._side = ""

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close = bar["close"]
        if not math.isfinite(close):
            return hold("non-finite close")
        self._fast = ema(self._fast, close, self.fast)
        self._slow = ema(self._slow, close, self.slow)
        macd = self._fast - self._slow
        self._sig = ema(self._sig, macd, self.signal)
        self._n += 1
        if self._n < self.slow + self.signal:
            return hold("warmup")

        hist = macd - self._sig
        bull = hist > 0
        if self._side == "":
            if bull:
                self._side = "LONG"
                return self._enter(close, 1.0, f"MACD hist+ {hist:.4f}", hist=hist)
            if self.allow_short and hist < 0:   # strict; hist==0 (flat) is no-signal
                self._side = "SHORT"
                return self._short(close, 1.0, f"MACD hist- {hist:.4f}", hist=hist)
            return hold("flat")
        if self._side == "LONG":
            if not bull:
                self._side = ""
                return self._exit(close, "MACD hist turned negative")
            return hold("holding long")
        if bull:
            self._side = ""
            return self._exit(close, "MACD hist turned positive")
        return hold("holding short")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True


class RSIMomentum(Strategy):
    """
    MOMENTUM RSI (the opposite of RSIMeanReversion): go long while RSI is STRONG
    (>= entry_high), exit when it fades back through exit_level. "Overbought stays
    overbought" — buying strength rather than fading it. With allow_short, go short
    while RSI is WEAK (<= entry_low), cover when it recovers through exit_level.
    """

    name = "rsi-momentum"

    def __init__(self, rsi_period: int = 14, entry_high: float = 60.0,
                 exit_level: float = 50.0, entry_low: float = 40.0,
                 allow_short: bool = False) -> None:
        super().__init__()
        self.rsi_period = rsi_period
        self.entry_high = entry_high
        self.exit_level = exit_level
        self.entry_low = entry_low
        self.allow_short = allow_short
        self._closes: Deque[float] = deque(maxlen=rsi_period * 4 + 5)
        self._side = ""

    def reset(self) -> None:
        super().reset()
        self._closes.clear()
        self._side = ""

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close = bar["close"]
        if not math.isfinite(close):
            return hold("non-finite close")
        self._closes.append(close)
        rsi = wilder_rsi(list(self._closes), self.rsi_period)
        if rsi is None:
            return hold("warmup")

        if self._side == "":
            if rsi >= self.entry_high:
                self._side = "LONG"
                return self._enter(close, 1.0, f"RSI strong {rsi:.1f}", rsi=rsi)
            if self.allow_short and rsi <= self.entry_low:
                self._side = "SHORT"
                return self._short(close, 1.0, f"RSI weak {rsi:.1f}", rsi=rsi)
            return hold("flat")
        if self._side == "LONG":
            if rsi < self.exit_level:
                self._side = ""
                return self._exit(close, f"RSI faded {rsi:.1f}", rsi=rsi)
            return hold("holding long")
        if rsi > self.exit_level:
            self._side = ""
            return self._exit(close, f"RSI recovered {rsi:.1f}", rsi=rsi)
        return hold("holding short")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True


class ATRBreakout(Strategy):
    """
    Range breakout with an ATR filter: only go long when the close clears the prior
    `lookback`-bar high by `atr_mult` * ATR (and symmetrically short below the low).
    The ATR cushion rejects tiny fake breakouts that a raw Donchian channel takes.
    Exit a long when price falls back below the prior channel low (and vice-versa).
    """

    name = "atr-breakout"

    def __init__(self, lookback: int = 20, atr_period: int = 14, atr_mult: float = 0.5,
                 allow_short: bool = False) -> None:
        super().__init__()
        self.lookback = lookback
        self.atr_period = atr_period
        self.atr_mult = atr_mult
        self.allow_short = allow_short
        n = max(lookback, atr_period) + 1
        self._highs: Deque[float] = deque(maxlen=n)
        self._lows: Deque[float] = deque(maxlen=n)
        self._closes: Deque[float] = deque(maxlen=n)
        self._side = ""

    def reset(self) -> None:
        super().reset()
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        self._side = ""

    def _atr(self) -> Optional[float]:
        if len(self._closes) < self.atr_period + 1:
            return None
        trs = []
        h, l, c = list(self._highs), list(self._lows), list(self._closes)
        for i in range(len(c) - self.atr_period, len(c)):
            tr = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
            trs.append(tr)
        return statistics.fmean(trs)

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        high, low, close = bar.get("high", bar["close"]), bar.get("low", bar["close"]), bar["close"]
        if not (math.isfinite(high) and math.isfinite(low) and math.isfinite(close)):
            return hold("non-finite bar")
        # channel from bars BEFORE this one (no look-ahead)
        prev_high = max(self._highs) if self._highs else None
        prev_low = min(self._lows) if self._lows else None
        atr = self._atr()
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(close)
        if prev_high is None or prev_low is None or atr is None or len(self._closes) <= self.lookback:
            return hold("warmup")

        up_trigger = prev_high + self.atr_mult * atr
        dn_trigger = prev_low - self.atr_mult * atr
        if self._side == "":
            if close > up_trigger:
                self._side = "LONG"
                return self._enter(close, 1.0, f"ATR breakout up (>{up_trigger:.2f})", atr=atr)
            if self.allow_short and close < dn_trigger:
                self._side = "SHORT"
                return self._short(close, 1.0, f"ATR breakout down (<{dn_trigger:.2f})", atr=atr)
            return hold("flat")
        if self._side == "LONG":
            if close < prev_low:
                self._side = ""
                return self._exit(close, "fell below channel low")
            return hold("holding long")
        if close > prev_high:
            self._side = ""
            return self._exit(close, "rose above channel high")
        return hold("holding short")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True


class VolRegimeGate(Strategy):
    """
    Composable volatility-regime FILTER (not a standalone strategy): wrap any base
    strategy and let its ENTRY signals through only when realized volatility is
    inside [vol_floor, vol_cap]. HOLD and EXIT always pass (never block an exit).
    On a block, roll back the base's optimistic entry state so it re-evaluates next
    bar — the same pattern as MicropriceGate. Trend brains often only pay in a
    chosen vol regime; this gates them without touching their logic.
    """

    def __init__(self, base: Strategy, *, lookback: int = 20,
                 vol_floor: float = 0.0, vol_cap: float = float("inf"),
                 annualize: bool = False) -> None:
        super().__init__()
        self.base = base
        self.lookback = lookback
        self.vol_floor = vol_floor
        self.vol_cap = vol_cap
        self.annualize = annualize
        self._closes: Deque[float] = deque(maxlen=lookback * 3 + 5)
        self.name = f"{base.name}+vol"

    def reset(self) -> None:
        super().reset()
        self.base.reset()
        self._closes.clear()

    def on_fill(self, side: str, qty: float, price: float) -> None:
        self.base.on_fill(side, qty, price)

    def _in_band(self) -> Optional[bool]:
        from alpca.calibration.volatility import compute_rolling_volatility
        if len(self._closes) < self.lookback + 1:
            return None
        bars = [{"close": c, "timestamp": 0} for c in self._closes]
        vol = compute_rolling_volatility(bars, lookback_days=len(self._closes),
                                         annualize=self.annualize, bars_per_day=1.0)
        return self.vol_floor <= vol <= self.vol_cap

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close = bar["close"]
        if math.isfinite(close):
            self._closes.append(close)
        snap = (self.base._in_position, getattr(self.base, "_side", None), self.base._entry_price)
        sig = self.base.on_bar(bar)
        if sig.side in (HOLD, EXIT):
            return sig

        in_band = self._in_band()
        if in_band:
            return sig
        # blocked: undo the base's optimistic entry flip so it retries next bar
        self.base._in_position, self.base._entry_price = snap[0], snap[2]
        if hasattr(self.base, "_side"):
            self.base._side = snap[1]
        why = "warmup" if in_band is None else "out-of-band vol"
        return hold(f"vol gate blocked {sig.side} ({why})")


def compute_adx(highs, lows, closes, period: int = 14) -> Optional[float]:
    """
    Wilder's ADX (Average Directional Index, 0–100) — trend STRENGTH (not direction).
    >~25 = trending, <~20 = ranging/choppy. Returns None until warmed (needs about
    2*period+1 bars). Pure, recomputed from the (bounded) buffers each call.

    +DM/-DM directional movement, smoothed TR -> +DI/-DI -> DX = |+DI - -DI| /
    (+DI + -DI); ADX = Wilder moving average of DX. The +DI/-DI use the same Wilder
    smoothing on numerator and denominator, so summed (not averaged) smoothing is
    fine — the ratio is scale-invariant.
    """
    n = len(closes)
    if n < 2 * period + 1:
        return None
    trs, pdm, mdm = [], [], []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        dn = lows[i - 1] - lows[i]
        trs.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
        pdm.append(up if (up > dn and up > 0) else 0.0)
        mdm.append(dn if (dn > up and dn > 0) else 0.0)

    def _wilder(vals, p):
        s = sum(vals[:p])
        out = [s]
        for v in vals[p:]:
            s = s - s / p + v
            out.append(s)
        return out

    atr, spdm, smdm = _wilder(trs, period), _wilder(pdm, period), _wilder(mdm, period)
    dxs = []
    for a, pd_, md_ in zip(atr, spdm, smdm):
        if a <= 0:
            dxs.append(0.0)
            continue
        pdi, mdi = 100.0 * pd_ / a, 100.0 * md_ / a
        denom = pdi + mdi
        dxs.append(100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0)
    if len(dxs) < period:
        return None
    adx = sum(dxs[:period]) / period
    for dx in dxs[period:]:
        adx = (adx * (period - 1) + dx) / period
    return adx


class ADXTrendGate(Strategy):
    """
    Composable TREND-STRENGTH filter: pass a base strategy's ENTRY signals only when
    ADX >= threshold (the market is actually trending). HOLD and EXIT always pass.

    Directly targets false-breakout whipsaw — a breakout strategy SITS OUT ranging
    days (low ADX) instead of buying every fakeout. On a block, roll back the base's
    optimistic entry state so it re-evaluates next bar (same pattern as VolRegimeGate
    / MicropriceGate). Reads high/low/close off the bar (falls back to close).
    """

    def __init__(self, base: Strategy, *, period: int = 14, threshold: float = 25.0) -> None:
        super().__init__()
        self.base = base
        self.period = period
        self.threshold = threshold
        m = period * 5 + 5
        self._h: Deque[float] = deque(maxlen=m)
        self._l: Deque[float] = deque(maxlen=m)
        self._c: Deque[float] = deque(maxlen=m)
        self.name = f"{base.name}+adx"

    def reset(self) -> None:
        super().reset()
        self.base.reset()
        self._h.clear()
        self._l.clear()
        self._c.clear()

    def on_fill(self, side: str, qty: float, price: float) -> None:
        self.base.on_fill(side, qty, price)

    def _adx(self) -> Optional[float]:
        if len(self._c) < 2 * self.period + 1:
            return None
        return compute_adx(list(self._h), list(self._l), list(self._c), self.period)

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        h = bar.get("high", bar["close"])
        l = bar.get("low", bar["close"])
        c = bar["close"]
        if math.isfinite(h) and math.isfinite(l) and math.isfinite(c):
            self._h.append(h)
            self._l.append(l)
            self._c.append(c)
        snap = (self.base._in_position, getattr(self.base, "_side", None), self.base._entry_price)
        sig = self.base.on_bar(bar)
        if sig.side in (HOLD, EXIT):
            return sig

        adx = self._adx()
        if adx is not None and adx >= self.threshold:
            return sig
        # blocked: market isn't trending — undo the base's optimistic entry flip
        self.base._in_position, self.base._entry_price = snap[0], snap[2]
        if hasattr(self.base, "_side"):
            self.base._side = snap[1]
        why = "warmup" if adx is None else f"ADX {adx:.1f}<{self.threshold:g}"
        return hold(f"adx gate blocked {sig.side} ({why})")


class BollingerExpansion(Strategy):
    """
    Bollinger bands = SMA(period) ± k*std(period). The usual play FADES the bands; this
    one rides momentum the other way: enter LONG only when the bands are EXPANDING
    (width rising = volatility breaking out) AND the close prints ABOVE the upper band.
    Exit when price falls back to the mean. Symmetric short below the lower band with
    allow_short. Expansion-only avoids buying a quiet drift through a tight band.
    """

    name = "bollinger-expansion"

    def __init__(self, period: int = 20, k: float = 2.0, allow_short: bool = False) -> None:
        super().__init__()
        self.period = period
        self.k = k
        self.allow_short = allow_short
        self._closes: Deque[float] = deque(maxlen=period + 2)
        self._prev_width: Optional[float] = None
        self._side = ""

    def reset(self) -> None:
        super().reset()
        self._closes.clear()
        self._prev_width = None
        self._side = ""

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close = bar["close"]
        if not math.isfinite(close):
            return hold("non-finite close")
        self._closes.append(close)
        if len(self._closes) < self.period:
            return hold("warmup")
        vals = list(self._closes)[-self.period:]
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals)
        upper, lower = mean + self.k * std, mean - self.k * std
        width = upper - lower
        expanding = self._prev_width is not None and width > self._prev_width
        self._prev_width = width

        if self._side == "":
            if expanding and close > upper:
                self._side = "LONG"
                return self._enter(close, 1.0, f"band expansion break up ({width:.2f})")
            if self.allow_short and expanding and close < lower:
                self._side = "SHORT"
                return self._short(close, 1.0, f"band expansion break down ({width:.2f})")
            return hold("flat")
        if self._side == "LONG":
            if close < mean:
                self._side = ""
                return self._exit(close, "back to band mean")
            return hold("holding long")
        if close > mean:
            self._side = ""
            return self._exit(close, "back to band mean")
        return hold("holding short")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True


class EnsembleVote(Strategy):
    """
    Vote across several momentum/trend brains and act only when >= min_agree of them
    point the SAME way. Combining genuinely different signals (EMA cross + MACD +
    Donchian breakout) filters the false signals any single one fires — go long when
    a majority are long, flat when consensus breaks, short on a short majority
    (allow_short). The voters are driven for their internal direction only; they
    place no orders — just this ensemble does.
    """

    name = "ensemble"

    def __init__(self, voters=None, min_agree: int = 2, allow_short: bool = False) -> None:
        super().__init__()
        if voters is None:
            from alpca.strategies.breakout import DonchianBreakout
            # voters are direction INDICATORS — EMA/MACD vote both ways so a bearish
            # consensus is visible; the ensemble's own allow_short decides whether to
            # act on it. (Donchian is long-only, so it only contributes long votes.)
            voters = [EMACrossMomentum(12, 26, allow_short=True),
                      MACDTrend(12, 26, 9, allow_short=True),
                      DonchianBreakout(period=20)]
        self.voters = list(voters)
        self.min_agree = min_agree
        self.allow_short = allow_short
        self._side = ""

    def reset(self) -> None:
        super().reset()
        for v in self.voters:
            v.reset()
        self._side = ""

    @staticmethod
    def _dir(v) -> int:
        side = getattr(v, "_side", None)
        if side == "SHORT":
            return -1
        if side == "LONG":
            return 1
        if side in ("", None):
            return 1 if getattr(v, "_in_position", False) else 0
        return 0

    def on_bar(self, bar: Dict[str, float]) -> Signal:
        close = bar["close"]
        for v in self.voters:
            v.on_bar(bar)
        dirs = [self._dir(v) for v in self.voters]
        longs = sum(1 for d in dirs if d > 0)
        shorts = sum(1 for d in dirs if d < 0)

        if self._side == "":
            if longs >= self.min_agree and longs > shorts:
                self._side = "LONG"
                return self._enter(close, 1.0, f"ensemble long {longs}/{len(self.voters)}")
            if self.allow_short and shorts >= self.min_agree and shorts > longs:
                self._side = "SHORT"
                return self._short(close, 1.0, f"ensemble short {shorts}/{len(self.voters)}")
            return hold(f"no consensus (L{longs}/S{shorts})")
        if self._side == "LONG":
            if longs < self.min_agree:
                self._side = ""
                return self._exit(close, f"long consensus lost ({longs})")
            return hold("holding long")
        if shorts < self.min_agree:
            self._side = ""
            return self._exit(close, f"short consensus lost ({shorts})")
        return hold("holding short")

    def on_fill(self, side: str, qty: float, price: float) -> None:
        if side == "SELL" and self._side != "SHORT" and not self._in_position:
            self._side = "SHORT"
            self._in_position = True
