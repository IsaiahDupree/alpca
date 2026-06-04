"""
TradeCouncil — advocates and skeptics deliberating under ONE voice.

  * ADVOCATES argue FOR a trade: directional strategies (session-momentum, EMA, MACD,
    Donchian...). Each casts a long / short / flat vote.
  * SKEPTICS argue AGAINST: veto conditions that say "not now" even when the advocates
    agree — no trend (ADX), price overextended (chasing), vol out of band, etc.
  * ONE VOICE: a single synthesizer weighs the bull case against the vetoes and emits
    exactly ONE decision, carrying a human-readable rationale on Signal.reason (and
    .last_rationale) — e.g. "BUY: 3 advocates for (session-momentum, ema, macd);
    skeptics cleared" or "BUY proposed by 3 advocates — VETOED: ADX 17<20 (chop)".

A skeptic can VETO a trade the advocates wanted (risk-first); skeptics never block an
EXIT. This is a strictly more cautious EnsembleVote: consensus is necessary but not
sufficient — the trade must also survive the objections.
"""

from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from alpca.strategies.base import Signal, Strategy, hold


@dataclass
class CouncilContext:
    highs: List[float]
    lows: List[float]
    closes: List[float]
    bar: Dict[str, float]
    direction: int          # +1 = a long is proposed, -1 = a short is proposed


# --------------------------------------------------------------------- skeptics
class Skeptic:
    """Base skeptic: return a veto REASON string to block the proposed trade, else None."""
    name = "skeptic"

    def veto(self, ctx: CouncilContext) -> Optional[str]:
        return None


class ChopSkeptic(Skeptic):
    """Veto when the market isn't trending (ADX below threshold) — the direct fix for
    false-breakout whipsaw on ranging days."""
    name = "chop"

    def __init__(self, period: int = 14, adx_min: float = 20.0) -> None:
        self.period = period
        self.adx_min = adx_min

    def veto(self, ctx: CouncilContext) -> Optional[str]:
        from alpca.strategies.momentum import compute_adx
        adx = compute_adx(ctx.highs, ctx.lows, ctx.closes, self.period)
        if adx is None:
            return None  # not enough data -> abstain (don't block)
        if adx < self.adx_min:
            return f"ADX {adx:.0f}<{self.adx_min:g} (no trend / chop)"
        return None


class OverextendedSkeptic(Skeptic):
    """Veto CHASING: price already too far from its SMA in the trade's direction."""
    name = "overextended"

    def __init__(self, period: int = 20, max_ext: float = 0.03) -> None:
        self.period = period
        self.max_ext = max_ext

    def veto(self, ctx: CouncilContext) -> Optional[str]:
        if len(ctx.closes) < self.period:
            return None
        sma = statistics.fmean(ctx.closes[-self.period:])
        if sma <= 0:
            return None
        ext = (ctx.closes[-1] - sma) / sma
        if ctx.direction > 0 and ext > self.max_ext:
            return f"price {ext:+.1%} above MA{self.period} (chasing a top)"
        if ctx.direction < 0 and -ext > self.max_ext:
            return f"price {ext:+.1%} below MA{self.period} (chasing a bottom)"
        return None


class VolRegimeSkeptic(Skeptic):
    """Veto when realized volatility is outside a sane band (too quiet = no edge)."""
    name = "vol"

    def __init__(self, lookback: int = 20, floor: float = 0.0, cap: float = float("inf")) -> None:
        self.lookback = lookback
        self.floor = floor
        self.cap = cap

    def veto(self, ctx: CouncilContext) -> Optional[str]:
        from alpca.calibration.volatility import compute_rolling_volatility
        if len(ctx.closes) < self.lookback + 1:
            return None
        bars = [{"close": c, "timestamp": 0} for c in ctx.closes]
        vol = compute_rolling_volatility(bars, lookback_days=len(ctx.closes),
                                         annualize=False, bars_per_day=1.0)
        if vol < self.floor:
            return f"vol {vol:.4f}<floor (too quiet)"
        if vol > self.cap:
            return f"vol {vol:.4f}>cap (too wild)"
        return None


# ---------------------------------------------------------------------- council
class TradeCouncil(Strategy):
    name = "council"

    def __init__(self, advocates=None, skeptics=None, min_conviction: int = 2,
                 allow_short: bool = False) -> None:
        super().__init__()
        if advocates is None:
            from alpca.strategies.breakout import DonchianBreakout
            from alpca.strategies.intraday import SessionMomentum
            from alpca.strategies.momentum import EMACrossMomentum, MACDTrend
            advocates = [SessionMomentum(allow_short=True),
                         EMACrossMomentum(12, 26, allow_short=True),
                         MACDTrend(12, 26, 9, allow_short=True),
                         DonchianBreakout(period=20)]
        if skeptics is None:
            skeptics = [ChopSkeptic(adx_min=20.0), OverextendedSkeptic(period=20, max_ext=0.03)]
        self.advocates = list(advocates)
        self.skeptics = list(skeptics)
        self.min_conviction = min_conviction
        self.allow_short = allow_short
        n = 5 * 20 + 5
        self._h: Deque[float] = deque(maxlen=n)
        self._l: Deque[float] = deque(maxlen=n)
        self._c: Deque[float] = deque(maxlen=n)
        self._side = ""
        self.last_rationale = ""

    def reset(self) -> None:
        super().reset()
        for a in self.advocates:
            a.reset()
        self._h.clear()
        self._l.clear()
        self._c.clear()
        self._side = ""
        self.last_rationale = ""

    def on_session_start(self) -> None:
        for a in self.advocates:
            a.on_session_start()

    def on_fill(self, side: str, qty: float, price: float) -> None:
        for a in self.advocates:
            a.on_fill(side, qty, price)

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
        if not math.isfinite(close):
            return hold("non-finite close")
        self._h.append(bar.get("high", close))
        self._l.append(bar.get("low", close))
        self._c.append(close)
        for a in self.advocates:
            a.on_bar(bar)
        dirs = [(a, self._dir(a)) for a in self.advocates]
        longs = [a for a, d in dirs if d > 0]
        shorts = [a for a, d in dirs if d < 0]

        # hold-management: exit when the held side's advocate conviction collapses.
        # Skeptics never block an exit.
        if self._side == "LONG":
            if len(longs) < self.min_conviction:
                self._side = ""
                self.last_rationale = f"EXIT long: only {len(longs)} advocates still long (<{self.min_conviction})"
                return self._exit(close, self.last_rationale)
            return hold("council holds long")
        if self._side == "SHORT":
            if len(shorts) < self.min_conviction:
                self._side = ""
                self.last_rationale = f"EXIT short: only {len(shorts)} advocates still short"
                return self._exit(close, self.last_rationale)
            return hold("council holds short")

        # flat: is there a consensus to act on?
        direction = 0
        if len(longs) >= self.min_conviction and len(longs) > len(shorts):
            direction = 1
        elif self.allow_short and len(shorts) >= self.min_conviction and len(shorts) > len(longs):
            direction = -1
        if direction == 0:
            self.last_rationale = f"no consensus (L{len(longs)}/S{len(shorts)})"
            return hold(self.last_rationale)

        # advocates favor a trade — now hear the skeptics (one voice synthesizes)
        ctx = CouncilContext(list(self._h), list(self._l), list(self._c), bar, direction)
        vetoes = [v for v in (s.veto(ctx) for s in self.skeptics) if v]
        favor = longs if direction > 0 else shorts
        names = ", ".join(a.name for a in favor)
        verb = "BUY" if direction > 0 else "SHORT"
        if vetoes:
            self.last_rationale = f"{verb} proposed by {len(favor)} advocates ({names}) — VETOED: {'; '.join(vetoes)}"
            return hold(self.last_rationale)
        self.last_rationale = f"{verb}: {len(favor)} advocates for ({names}); skeptics cleared"
        if direction > 0:
            self._side = "LONG"
            return self._enter(close, 1.0, self.last_rationale)
        self._side = "SHORT"
        return self._short(close, 1.0, self.last_rationale)
