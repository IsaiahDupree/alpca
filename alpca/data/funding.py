"""
Perpetual-futures FUNDING RATE history as a free, non-price crowding/sentiment signal.

We cannot run the cash-and-carry arb (Alpaca is spot-only, no perps, no crypto shorting),
so funding is used only as a SIGNAL: persistently extreme-positive funding = over-leveraged
longs (crowded) -> fade/reduce; persistently negative = short-squeeze setup -> favor long.
The literature is explicit that funding has edge only at EXTREMES, so it must be a
low-turnover gate/tilt on a spot allocation, not a high-frequency signal.

Source: Kraken Futures public historical-funding endpoint (US-accessible; Binance's
fapi is geo-blocked 451 from US IPs). `relativeFundingRate` is the price-normalized
(percentage) funding per period. Returns plain dicts so the harness can consume a daily
funding series. NOTE: Kraken history is ~1 year — a short window; treat results as
exploratory, not multi-regime.
"""

from __future__ import annotations

import json
import urllib.request
from typing import Dict, List

KRAKEN_FUNDING = "https://futures.kraken.com/derivatives/api/v4/historicalfundingrates?symbol={sym}"


def fetch_kraken_funding(symbol: str = "PF_XBTUSD", *, timeout: int = 30) -> List[dict]:
    """Fetch historical funding for a Kraken perp (PF_XBTUSD, PF_ETHUSD, ...). Returns
    [{ts(epoch seconds), rate(relativeFundingRate, fraction per period)}] oldest-first."""
    url = KRAKEN_FUNDING.format(sym=symbol)
    with urllib.request.urlopen(url, timeout=timeout) as r:
        data = json.load(r)
    out: List[dict] = []
    for row in data.get("rates", []):
        t = row.get("timestamp")
        rel = row.get("relativeFundingRate")
        if t is None or rel is None:
            continue
        # ISO 8601 'YYYY-MM-DDTHH:MM:SSZ' -> epoch
        from datetime import datetime, timezone
        ep = datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
        out.append({"ts": ep, "rate": float(rel)})
    out.sort(key=lambda x: x["ts"])
    return out


def daily_funding(rows: List[dict]) -> Dict[str, float]:
    """Aggregate intraperiod funding to a per-UTC-date SUM (the day's total funding cost).
    Positive = longs paid shorts that day (bullish/crowded-long pressure)."""
    from datetime import datetime, timezone
    out: Dict[str, float] = {}
    for r in rows:
        d = datetime.fromtimestamp(r["ts"], timezone.utc).strftime("%Y-%m-%d")
        out[d] = out.get(d, 0.0) + r["rate"]
    return out
