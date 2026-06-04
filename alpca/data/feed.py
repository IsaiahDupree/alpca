"""
Bar sources for the live runner — all expose the same async-iterator contract:

    async for bar in source:   # bar is a dict {open,high,low,close,volume,timestamp,symbol}
        ...

Three implementations:
  - ReplayBarSource     : replays a list of bars (offline; tests, parity).
  - AlpacaBarPoller     : polls Alpaca REST for the latest bar on a cadence
                          (real; simple, reliable, higher latency).
  - AlpacaWebSocketFeed : streams minute bars over alpaca-py's StockDataStream
                          (real; lowest data-arrival latency).

Live bars carry an extra `recv_ts` (local arrival time) alongside `timestamp`
(the bar's venue time), so the runner can measure FEED latency = recv_ts - timestamp.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncIterator, Deque, Dict, List, Optional

from alpca.config import AlpacaConfig


# --------------------------------------------------------------------------- ts
def parse_ts(value) -> float:
    """Best-effort -> epoch seconds. Handles datetime, ns/us/ms/s ints, RFC3339 str."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        v = float(value)
        # heuristics: ns ~1e18, us ~1e15, ms ~1e12, s ~1e9
        if v > 1e17:
            return v / 1e9
        if v > 1e14:
            return v / 1e6
        if v > 1e11:
            return v / 1e3
        return v
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        s = value.strip().replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(s).timestamp()
        except ValueError:
            return 0.0
    if hasattr(value, "timestamp"):
        try:
            return value.timestamp()
        except Exception:
            return 0.0
    return 0.0


@dataclass
class Tick:
    """A normalized market event (trade or quote), with feed-latency built in."""
    symbol: str
    source_ts: float            # venue timestamp (epoch s)
    recv_ts: float              # local arrival (epoch s)
    price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    kind: str = "trade"         # "trade" | "quote" | "bar"

    @property
    def feed_latency_ms(self) -> float:
        return (self.recv_ts - self.source_ts) * 1000.0


def _bar_obj_to_dict(symbol: str, b) -> Dict[str, float]:
    """Convert an alpaca-py Bar (or bar-like) to our bar dict, stamping recv_ts."""
    ts = parse_ts(getattr(b, "timestamp", None))
    return {
        "open": float(getattr(b, "open", 0.0)),
        "high": float(getattr(b, "high", 0.0)),
        "low": float(getattr(b, "low", 0.0)),
        "close": float(getattr(b, "close", 0.0)),
        "volume": float(getattr(b, "volume", 0.0) or 0.0),
        "timestamp": ts,
        "recv_ts": time.time(),
        "symbol": symbol,
    }


class _FeedLatencyTracker:
    def __init__(self, window: int = 500) -> None:
        self._samples: Deque[float] = deque(maxlen=window)

    def record(self, bar: Dict[str, float]) -> None:
        if "recv_ts" in bar and bar.get("timestamp"):
            self._samples.append((bar["recv_ts"] - bar["timestamp"]) * 1000.0)

    def stats(self) -> Dict[str, float]:
        vals = sorted(self._samples)
        if not vals:
            return {"n": 0}
        n = len(vals)
        return {
            "n": n,
            "mean_ms": sum(vals) / n,
            "p50_ms": vals[n // 2],
            "p95_ms": vals[min(n - 1, int(0.95 * (n - 1)))],
            "max_ms": vals[-1],
        }


# ----------------------------------------------------------------- offline replay
class ReplayBarSource:
    """Replays a fixed list of bars as an async iterator (offline)."""

    def __init__(self, bars: List[Dict[str, float]], interval_s: float = 0.0) -> None:
        self._bars = list(bars)
        self._interval_s = interval_s
        self.latency = _FeedLatencyTracker()

    async def __aiter__(self) -> AsyncIterator[Dict[str, float]]:
        for bar in self._bars:
            if self._interval_s > 0:
                await asyncio.sleep(self._interval_s)
            self.latency.record(bar)
            yield bar


# --------------------------------------------------------------- REST poller
class AlpacaBarPoller:
    """
    Polls Alpaca REST for the latest bar of one symbol on a cadence. Yields a bar
    only when a new (later-timestamped) bar appears. Reliable; higher latency
    than the websocket feed.
    """

    def __init__(self, config: AlpacaConfig, symbol: str, *,
                 timeframe: str = "1min", poll_interval_s: float = 5.0,
                 max_bars: Optional[int] = None) -> None:
        config.require_credentials()
        self.cfg = config
        self.symbol = symbol
        self.timeframe = timeframe
        self.poll_interval_s = poll_interval_s
        self.max_bars = max_bars
        self.latency = _FeedLatencyTracker()
        self._last_ts: float = 0.0
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(self.cfg.api_key, self.cfg.secret_key)
        return self._client

    async def _latest_bar(self) -> Optional[Dict[str, float]]:
        from alpaca.data.requests import StockLatestBarRequest
        req = StockLatestBarRequest(symbol_or_symbols=self.symbol, feed=self.cfg.data_feed)
        resp = await asyncio.to_thread(self.client.get_stock_latest_bar, req)
        bar = resp.get(self.symbol) if isinstance(resp, dict) else None
        if bar is None:
            return None
        return _bar_obj_to_dict(self.symbol, bar)

    async def __aiter__(self) -> AsyncIterator[Dict[str, float]]:
        emitted = 0
        while True:
            try:
                bar = await self._latest_bar()
            except Exception:
                bar = None
            if bar and bar["timestamp"] > self._last_ts:
                self._last_ts = bar["timestamp"]
                self.latency.record(bar)
                emitted += 1
                yield bar
                if self.max_bars and emitted >= self.max_bars:
                    return
            await asyncio.sleep(self.poll_interval_s)


# --------------------------------------------------------------- websocket feed
class AlpacaWebSocketFeed:
    """
    Streams minute bars via alpaca-py's StockDataStream. The SDK runs its own
    event loop, so we drive it in a daemon thread and bridge bars back to our
    asyncio loop through a thread-safe handoff onto an asyncio.Queue.

    Lowest data-arrival latency; per-bar feed latency is tracked.
    """

    def __init__(self, config: AlpacaConfig, symbols: List[str], *,
                 max_bars: Optional[int] = None) -> None:
        config.require_credentials()
        self.cfg = config
        self.symbols = symbols
        self.max_bars = max_bars
        self.latency = _FeedLatencyTracker()
        self._stream = None
        self._thread: Optional[threading.Thread] = None
        self._stopped = False

    def _build_stream(self):
        from alpaca.data.live import StockDataStream
        return StockDataStream(self.cfg.api_key, self.cfg.secret_key, feed=self.cfg.data_feed)

    async def __aiter__(self) -> AsyncIterator[Dict[str, float]]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        self._stream = self._build_stream()

        async def _bar_handler(b):
            bar = _bar_obj_to_dict(getattr(b, "symbol", self.symbols[0]), b)
            loop.call_soon_threadsafe(queue.put_nowait, bar)

        self._stream.subscribe_bars(_bar_handler, *self.symbols)

        def _runner():
            try:
                self._stream.run()  # blocking; manages its own loop
            except Exception:
                loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel on failure

        self._thread = threading.Thread(target=_runner, daemon=True)
        self._thread.start()

        emitted = 0
        try:
            while not self._stopped:
                bar = await queue.get()
                if bar is None:  # stream died
                    return
                self.latency.record(bar)
                emitted += 1
                yield bar
                if self.max_bars and emitted >= self.max_bars:
                    return
        finally:
            self.stop()

    def stop(self) -> None:
        self._stopped = True
        if self._stream is not None:
            try:
                self._stream.stop()
            except Exception:
                pass


# --------------------------------------------------------------- NBBO quotes (#3)
class AlpacaQuoteCache:
    """
    Fetches the latest NBBO quote for one symbol via Alpaca REST. IEX feed =
    top-of-book only (best bid/ask + sizes), which is exactly what the microprice
    needs — there is no L2 depth. Returns a dict {bid, ask, bid_size, ask_size,
    quote_ts, quote_recv_ts} or None.
    """

    def __init__(self, config: AlpacaConfig, symbol: str) -> None:
        config.require_credentials()
        self.cfg = config
        self.symbol = symbol
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient
            self._client = StockHistoricalDataClient(self.cfg.api_key, self.cfg.secret_key)
        return self._client

    async def latest(self) -> Optional[Dict[str, float]]:
        from alpaca.data.requests import StockLatestQuoteRequest
        req = StockLatestQuoteRequest(symbol_or_symbols=self.symbol, feed=self.cfg.data_feed)
        resp = await asyncio.to_thread(self.client.get_stock_latest_quote, req)
        q = resp.get(self.symbol) if isinstance(resp, dict) else None
        if q is None:
            return None
        bid = float(getattr(q, "bid_price", 0) or 0) or None
        ask = float(getattr(q, "ask_price", 0) or 0) or None
        return {
            "bid": bid,
            "ask": ask,
            "bid_size": float(getattr(q, "bid_size", 0) or 0) or None,
            "ask_size": float(getattr(q, "ask_size", 0) or 0) or None,
            "quote_ts": parse_ts(getattr(q, "timestamp", None)),
            "quote_recv_ts": time.time(),
        }


class QuoteEnrichedBarSource:
    """
    Wrap any bar source and merge the latest NBBO quote onto each bar, so
    microstructure strategies can read bid/ask/bid_size/ask_size off the bar dict
    (the runner already passes the whole bar to on_bar — no runner change needed).

    `quote_provider` is anything with `async latest() -> quote dict | None`
    (e.g. AlpacaQuoteCache, or a stub for tests). Best-effort: a failed/empty
    quote fetch leaves the bar unchanged so the bar stream never stalls.
    """

    def __init__(self, bar_source, quote_provider) -> None:
        self._src = bar_source
        self._q = quote_provider
        self.latency = getattr(bar_source, "latency", _FeedLatencyTracker())

    async def __aiter__(self) -> AsyncIterator[Dict[str, float]]:
        async for bar in self._src:
            try:
                q = await self._q.latest()
            except Exception:
                q = None
            yield {**bar, **q} if q else bar
