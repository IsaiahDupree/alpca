"""
Historical OHLCV bars.

Provides bars as a list of plain dicts ({open,high,low,close,volume,timestamp,
symbol}) — the schema strategies and the backtester consume. Real bars come from
Alpaca; a deterministic synthetic generator is available for offline use.

Adjustment policy (audit gap "no split/dividend adjustment"): raw Alpaca bars
contain price discontinuities on split/dividend ex-dates — a 2:1 split looks like
a fake -50% bar and generates phantom signals. fetch_alpaca_bars therefore
defaults to `adjustment="all"` (split + dividend adjusted) for honest signal
generation and return continuity. For live-fill PARITY you instead want raw
quotes (what you actually trade at) — pass adjustment="raw" for that path.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from alpca.config import AlpacaConfig


_TIMEFRAMES = {"1min", "5min", "15min", "1hour", "1day"}
_ADJUSTMENTS = {"raw", "split", "dividend", "all"}


def _to_alpaca_timeframe(tf: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    tf = tf.lower()
    if tf in ("1min", "min", "minute"):
        return TimeFrame(1, TimeFrameUnit.Minute)
    if tf == "5min":
        return TimeFrame(5, TimeFrameUnit.Minute)
    if tf == "15min":
        return TimeFrame(15, TimeFrameUnit.Minute)
    if tf in ("1hour", "hour", "1h"):
        return TimeFrame(1, TimeFrameUnit.Hour)
    if tf in ("1day", "day", "1d"):
        return TimeFrame(1, TimeFrameUnit.Day)
    raise ValueError(f"unsupported timeframe {tf!r}; use one of {sorted(_TIMEFRAMES)}")


def _to_alpaca_adjustment(adjustment: str):
    from alpaca.data.enums import Adjustment

    a = (adjustment or "all").lower()
    if a not in _ADJUSTMENTS:
        raise ValueError(f"unsupported adjustment {adjustment!r}; use one of {sorted(_ADJUSTMENTS)}")
    return Adjustment(a)


def fetch_alpaca_bars(
    config: AlpacaConfig,
    symbol: str,
    *,
    timeframe: str = "1hour",
    days: int = 30,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    adjustment: str = "all",
) -> List[Dict[str, float]]:
    """
    Fetch real OHLCV bars from Alpaca and return them as bar dicts.

    `start`/`end`: explicit UTC datetimes; when `start` is omitted it defaults to
    `end - days`. Use explicit windows to page a multi-year pull in chunks (the
    SDK also paginates internally within a single request).

    `adjustment`: "all" (split+dividend, default — for signals/returns), "split",
    "dividend", or "raw" (unadjusted — use this for live-fill parity). The chosen
    adjustment is recorded on every bar dict under "adjustment" so downstream code
    (and the parity check) can assert backtest and live used the same convention.
    """
    config.require_credentials()
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    client = StockHistoricalDataClient(config.api_key, config.secret_key)
    end = end or (datetime.now(timezone.utc) - timedelta(minutes=20))  # IEX 15min delay safety
    start = start or (end - timedelta(days=days))
    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_to_alpaca_timeframe(timeframe),
        start=start,
        end=end,
        feed=config.data_feed,
        adjustment=_to_alpaca_adjustment(adjustment),
    )
    barset = client.get_stock_bars(req)
    raw = barset.data.get(symbol, [])
    adj = (adjustment or "all").lower()
    out: List[Dict[str, float]] = []
    for b in raw:
        out.append({
            "open": float(b.open),
            "high": float(b.high),
            "low": float(b.low),
            "close": float(b.close),
            "volume": float(b.volume),
            "timestamp": b.timestamp.timestamp() if hasattr(b.timestamp, "timestamp") else 0.0,
            "symbol": symbol,
            "adjustment": adj,
        })
    return out


def fetch_alpaca_quotes(
    config: AlpacaConfig,
    symbol: str,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    days: int = 1,
    limit: Optional[int] = None,
) -> List[Dict[str, float]]:
    """
    Fetch historical NBBO quotes (top-of-book bid/ask + sizes) from Alpaca, the
    data the microprice strategy needs. IEX feed = top-of-book only (no L2 depth),
    and free IEX quotes are sparse/thin vs the paid SIP tape — treat sizes as
    indicative. Returns quote dicts sorted by venue timestamp (epoch seconds):
      {bid, ask, bid_size, ask_size, timestamp, symbol}

    `limit` caps the number of quotes returned (tick quotes are voluminous; a day
    of one symbol can be hundreds of thousands of rows).
    """
    config.require_credentials()
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockQuotesRequest

    client = StockHistoricalDataClient(config.api_key, config.secret_key)
    end = end or (datetime.now(timezone.utc) - timedelta(minutes=20))  # IEX 15min delay safety
    start = start or (end - timedelta(days=days))
    req = StockQuotesRequest(
        symbol_or_symbols=symbol,
        start=start,
        end=end,
        feed=config.data_feed,
        limit=limit,
    )
    qs = client.get_stock_quotes(req)
    raw = qs.data.get(symbol, []) if hasattr(qs, "data") else []
    out: List[Dict[str, float]] = []
    for q in raw:
        bid = getattr(q, "bid_price", None)
        ask = getattr(q, "ask_price", None)
        out.append({
            "bid": float(bid) if bid else None,
            "ask": float(ask) if ask else None,
            "bid_size": float(getattr(q, "bid_size", 0) or 0) or None,
            "ask_size": float(getattr(q, "ask_size", 0) or 0) or None,
            "timestamp": q.timestamp.timestamp() if hasattr(getattr(q, "timestamp", None), "timestamp") else 0.0,
            "symbol": symbol,
        })
    out.sort(key=lambda r: r["timestamp"])
    return out


def attach_quotes_to_bars(
    bars: List[Dict[str, float]],
    quotes: List[Dict[str, float]],
    *,
    max_staleness_s: Optional[float] = None,
) -> List[Dict[str, float]]:
    """
    Offline analog of feed.QuoteEnrichedBarSource: merge the prevailing NBBO onto
    each bar so microstructure strategies can read bid/ask/bid_size/ask_size off
    the bar dict in a BACKTEST. Each bar gets the LAST quote whose timestamp is
    <= the bar's timestamp (the quote in effect at the bar's open instant — known
    at decision time, so NO look-ahead). Bars before the first quote are left
    unenriched. Both inputs are sorted here; merge is linear (two-pointer).

    `max_staleness_s`: when set, only attach a quote if it is no older than this
    many seconds vs the bar (bar_ts - quote_ts <= max_staleness_s). This prevents
    a CROSS-DAY stale quote (e.g. only one day of NBBO merged onto several days of
    bars) from contaminating microprice/TCA with a quote from a different price
    regime. None = attach the most recent prior quote regardless of age.
    """
    bars = sorted(bars, key=lambda b: b.get("timestamp", 0) or 0)
    quotes = sorted(quotes, key=lambda q: q.get("timestamp", 0) or 0)
    out: List[Dict[str, float]] = []
    j = 0
    last: Optional[Dict[str, float]] = None
    for bar in bars:
        bt = bar.get("timestamp", 0) or 0
        while j < len(quotes) and (quotes[j].get("timestamp", 0) or 0) <= bt:
            last = quotes[j]
            j += 1
        fresh = last is not None and (
            max_staleness_s is None or (bt - (last.get("timestamp", 0) or 0)) <= max_staleness_s)
        if fresh:
            bar = {**bar, "bid": last["bid"], "ask": last["ask"],
                   "bid_size": last["bid_size"], "ask_size": last["ask_size"],
                   "quote_ts": last["timestamp"]}
        out.append(bar)
    return out


def synthetic_bars(
    symbol: str = "TEST",
    n: int = 300,
    *,
    start_price: float = 100.0,
    drift: float = 0.0005,
    vol: float = 0.01,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Deterministic geometric-random-walk bars for offline backtests/demos."""
    rng = random.Random(seed)
    price = start_price
    bars: List[Dict[str, float]] = []
    for i in range(n):
        ret = drift + rng.gauss(0.0, vol)
        new_price = max(0.01, price * math.exp(ret))
        hi = max(price, new_price) * (1 + abs(rng.gauss(0, vol / 2)))
        lo = min(price, new_price) * (1 - abs(rng.gauss(0, vol / 2)))
        bars.append({
            "open": round(price, 4),
            "high": round(hi, 4),
            "low": round(lo, 4),
            "close": round(new_price, 4),
            "volume": float(rng.randint(1000, 10000)),
            "timestamp": float(i),
            "symbol": symbol,
        })
        price = new_price
    return bars
