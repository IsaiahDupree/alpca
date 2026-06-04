"""
Microstructure (#3) — NBBO microprice: pure functions, the MicropriceGate
confirmation wrapper, the standalone MicropriceTilt, and the QuoteEnrichedBarSource
that merges NBBO onto bars (no runner change).
"""

import asyncio

from alpca.backtest.runner_backtest import backtest_resting
from alpca.data.feed import QuoteEnrichedBarSource, ReplayBarSource
from alpca.strategies.base import BUY, EXIT, HOLD, SELL, Strategy, hold
from alpca.strategies.microstructure import (
    MicropriceGate,
    MicropriceTilt,
    gate,
    microprice,
    microprice_signal,
    microprice_tilt,
)
from alpca.strategies.registry import available, make


def _qbar(close, bid, ask, bid_size, ask_size, ts=0.0, sym="SPY"):
    return {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close,
            "volume": 1e6, "timestamp": ts, "symbol": sym,
            "bid": bid, "ask": ask, "bid_size": bid_size, "ask_size": ask_size}


# ---- pure functions --------------------------------------------------------

def test_microprice_balanced_is_mid():
    assert microprice(99, 101, 5, 5) == 100.0
    assert microprice_tilt(99, 101, 5, 5) == 0.0
    assert microprice_signal(99, 101, 5, 5) == "flat"


def test_microprice_tilts_toward_heavy_queue():
    # heavy bid_size -> price pushed toward the ask -> bull
    assert microprice(99, 101, 9, 1) == 100.8
    assert abs(microprice_tilt(99, 101, 9, 1) - 0.8) < 1e-12
    assert microprice_signal(99, 101, 9, 1) == "bull"
    # heavy ask_size -> bear
    assert microprice(99, 101, 1, 9) == 99.2
    assert abs(microprice_tilt(99, 101, 1, 9) + 0.8) < 1e-12
    assert microprice_signal(99, 101, 1, 9) == "bear"


def test_microprice_deadband():
    assert microprice_signal(99, 101, 9, 1, k=0.9) == "flat"   # 0.8 tilt < 0.9 band
    assert microprice_signal(99, 101, 9, 1, k=0.5) == "bull"


def test_microprice_degenerate_quotes_return_none():
    assert microprice(None, 101, 5, 5) is None
    assert microprice_tilt(100, 100, 5, 5) is None        # locked (half_spread 0)
    assert microprice_tilt(101, 99, 5, 5) is None         # crossed
    assert microprice(99, 101, 0, 0) is None
    assert microprice_signal(99, None, 5, 5) is None


# ---- confirmation gate -----------------------------------------------------

class _AlwaysBuy(Strategy):
    name = "alwaysbuy"

    def on_bar(self, bar):
        if not self._in_position:
            return self._enter(bar["close"], 1.0, "buy")
        return hold("holding")


class _AlwaysExit(Strategy):
    name = "alwaysexit"

    def on_bar(self, bar):
        return self._exit(bar["close"], "x")


def test_gate_passes_confirmed_entry_blocks_and_retries():
    base = _AlwaysBuy()
    g = gate(base, k=0.1)
    # bearish quote contradicts the BUY -> blocked AND base rolled back to flat
    assert g.on_bar(_qbar(100, 99, 101, 1, 9)).side == HOLD
    assert base._in_position is False
    # bullish quote confirms the BUY -> passes, base now in position
    assert g.on_bar(_qbar(100, 99, 101, 9, 1)).side == BUY
    assert base._in_position is True


def test_gate_always_passes_exit():
    g = MicropriceGate(_AlwaysExit(), k=0.1)
    assert g.on_bar(_qbar(100, 99, 101, 1, 9)).side == EXIT   # exit not gated
    # even with no quote on the bar
    assert g.on_bar({"open": 100, "high": 100, "low": 100, "close": 100,
                     "volume": 1e6, "timestamp": 0}).side == EXIT


def test_gate_require_quote_toggle():
    nobar = {"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1e6, "timestamp": 0}
    assert MicropriceGate(_AlwaysBuy(), require_quote=True).on_bar(nobar).side == HOLD
    assert MicropriceGate(_AlwaysBuy(), require_quote=False).on_bar(nobar).side == BUY


def test_gate_reports_combined_name():
    assert gate(_AlwaysBuy()).name == "alwaysbuy+mp"


# ---- standalone tilt -------------------------------------------------------

def test_tilt_strategy_long_cycle():
    s = MicropriceTilt(k=0.3)
    assert s.on_bar(_qbar(100, 99, 101, 9, 1)).side == BUY      # bull -> long
    assert s.on_bar(_qbar(100, 99, 101, 5, 5)).side == EXIT     # flat -> out


def test_tilt_strategy_short_cycle():
    s = MicropriceTilt(k=0.3, allow_short=True)
    assert s.on_bar(_qbar(100, 99, 101, 1, 9)).side == SELL     # bear -> short
    assert s.on_bar(_qbar(100, 99, 101, 9, 1)).side == EXIT     # bull -> cover


def test_tilt_holds_without_quote():
    assert MicropriceTilt().on_bar({"open": 100, "high": 100, "low": 100,
                                    "close": 100, "volume": 1, "timestamp": 0}).side == HOLD


# ---- quote-enriched feed ---------------------------------------------------

class _StubQuote:
    async def latest(self):
        return {"bid": 99.0, "ask": 101.0, "bid_size": 9.0, "ask_size": 1.0}


class _FailQuote:
    async def latest(self):
        raise RuntimeError("quote feed down")


def _collect(src):
    async def go():
        out = []
        async for b in src:
            out.append(b)
        return out
    return asyncio.run(go())


def test_quote_enriched_bar_source_merges_nbbo():
    plain = [{"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1e6,
              "timestamp": 0, "symbol": "X"}]
    bars = _collect(QuoteEnrichedBarSource(ReplayBarSource(plain), _StubQuote()))
    b = bars[0]
    assert b["bid"] == 99.0 and b["ask"] == 101.0
    assert microprice_signal(b["bid"], b["ask"], b["bid_size"], b["ask_size"]) == "bull"


def test_quote_enriched_bar_source_tolerates_failure():
    plain = [{"open": 100, "high": 100, "low": 100, "close": 100, "volume": 1e6,
              "timestamp": 0, "symbol": "X"}]
    bars = _collect(QuoteEnrichedBarSource(ReplayBarSource(plain), _FailQuote()))
    assert "bid" not in bars[0]            # bar passes through unchanged


# ---- registry + integration ------------------------------------------------

def test_registry_has_microprice():
    assert {"microprice", "microprice-ls"}.issubset(set(available()))
    assert isinstance(make("microprice"), MicropriceTilt)
    assert make("microprice-ls").allow_short is True


def test_microprice_trades_through_runner():
    # alternate bullish / flat quotes so it enters long then flattens, twice
    bars = []
    for i in range(8):
        if i % 4 in (0, 1):
            bars.append(_qbar(100 + i * 0.1, 99, 101, 9, 1, ts=float(i)))   # bull
        else:
            bars.append(_qbar(100 + i * 0.1, 99, 101, 5, 5, ts=float(i)))   # flat
    res = backtest_resting(MicropriceTilt(k=0.3), bars)
    assert res.n_trades >= 1
    assert res.ending_equity > 0
