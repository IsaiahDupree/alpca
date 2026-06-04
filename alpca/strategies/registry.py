"""
Strategy registry — name -> factory, so the CLI/backtester/runner can build a
strategy from a string.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from alpca.strategies.base import Strategy
from alpca.strategies.breakout import ORB, DonchianBreakout, Supertrend, VolatilityBreakout
from alpca.strategies.event_driven import GapFade
from alpca.strategies.mean_reversion import RSIMeanReversion, ZScoreMeanReversion
from alpca.strategies.microstructure import MicropriceTilt
from alpca.strategies.council import TradeCouncil
from alpca.strategies.intraday import SessionMomentum, VWAPReclaim
from alpca.strategies.momentum import (
    ADXTrendGate,
    ATRBreakout,
    BollingerExpansion,
    EMACrossMomentum,
    EnsembleVote,
    MACDTrend,
    RSIMomentum,
    VolRegimeGate,
)
from alpca.strategies.order_flow import L1OFI


_REGISTRY: Dict[str, Callable[..., Strategy]] = {
    "donchian": DonchianBreakout,
    "orb": ORB,
    "keltner": VolatilityBreakout,
    "supertrend": Supertrend,
    "zscore": ZScoreMeanReversion,
    # long/short symmetric mean-reversion (needs RiskConfig.allow_short on the runner)
    "zscore-ls": lambda **kw: ZScoreMeanReversion(allow_short=True, **kw),
    # RSI mean-reversion with a vol-regime gate (#2)
    "rsi-mr": RSIMeanReversion,
    "rsi-mr-ls": lambda **kw: RSIMeanReversion(allow_short=True, **kw),
    # opening-gap fade (#1) — needs real epoch-second bar timestamps to fire
    "gap-fade": GapFade,
    "gap-fade-ls": lambda **kw: GapFade(allow_short=True, **kw),
    # NBBO microprice tilt (#3) — needs quote-enriched bars (bid/ask/sizes).
    # Best used as a confirmation GATE (microstructure.gate / MicropriceGate).
    "microprice": MicropriceTilt,
    "microprice-ls": lambda **kw: MicropriceTilt(allow_short=True, **kw),
    # L1 order-flow imbalance — needs quote-enriched bars (top-of-book NBBO).
    "ofi": L1OFI,
    "ofi-ls": lambda **kw: L1OFI(allow_short=True, **kw),
    # --- momentum / trend family (buy strength) ---
    "ema-momentum": EMACrossMomentum,
    "ema-momentum-ls": lambda **kw: EMACrossMomentum(allow_short=True, **kw),
    "macd": MACDTrend,
    "macd-ls": lambda **kw: MACDTrend(allow_short=True, **kw),
    "rsi-momentum": RSIMomentum,
    "rsi-momentum-ls": lambda **kw: RSIMomentum(allow_short=True, **kw),
    "atr-breakout": ATRBreakout,
    "atr-breakout-ls": lambda **kw: ATRBreakout(allow_short=True, **kw),
    # --- regime-gated combos: take breakouts only when the market is trending/volatile,
    #     so they sit out ranging "chop" days that produce false-breakout whipsaw ---
    "donchian-adx": lambda **kw: ADXTrendGate(DonchianBreakout(**kw)),
    "donchian-vol": lambda **kw: VolRegimeGate(DonchianBreakout(**kw), lookback=20),
    # ensemble: trade only when >=2 of EMA/MACD/Donchian agree on direction
    "ensemble": EnsembleVote,
    "ensemble-ls": lambda **kw: EnsembleVote(allow_short=True, **kw),
    # --- intraday / band strategies (roadmap #6/#7/#10) ---
    "vwap": VWAPReclaim,                         # reclaim/reject session VWAP
    "vwap-ls": lambda **kw: VWAPReclaim(allow_short=True, **kw),
    "session-momentum": SessionMomentum,         # ride the early-session move
    "session-momentum-ls": lambda **kw: SessionMomentum(allow_short=True, **kw),
    "bollinger-expansion": BollingerExpansion,   # band-expansion momentum break
    "bollinger-expansion-ls": lambda **kw: BollingerExpansion(allow_short=True, **kw),
    # advocates + skeptics under one voice (risk-aware deliberation)
    "council": TradeCouncil,
    "council-ls": lambda **kw: TradeCouncil(allow_short=True, **kw),
}


def available() -> List[str]:
    return sorted(_REGISTRY)


def make(name: str, **params) -> Strategy:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"unknown strategy {name!r}; available: {available()}")
    return _REGISTRY[key](**params)
