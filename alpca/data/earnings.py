"""
Earnings-surprise data for the Post-Earnings-Announcement-Drift (PEAD) edge.

PEAD trades the SUE (standardized unexpected earnings) drift: a stock that beats its
consensus tends to keep drifting up for weeks (and miss -> down). It is one of the most
replicated anomalies in finance and is event-driven cross-sectional -> genuinely
diversifying from both long-beta and our price-mean-reversion pairs basket.

Data sources (no paid data):
  - Nasdaq earnings-surprise table: FREE, no key, US-accessible, but only the LAST ~4
    quarters per ticker -> a ~1-year window. Enough for an honest recent-period test, NOT
    a multi-regime one. (api.nasdaq.com is an unofficial JSON endpoint; needs a browser UA.)
  - Finnhub /calendar/earnings: deeper history (years) but needs a FREE key in
    FINNHUB_API_KEY. Used automatically when the key is present.

Both return a uniform list[{date(epoch), surprise_pct, eps, consensus}] oldest-first so the
PEAD backtest can consume either. No fabricated data: if a source is unavailable the
fetcher returns [] and the caller reports reduced coverage honestly.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from typing import List, Optional

_NASDAQ = "https://api.nasdaq.com/api/company/{sym}/earnings-surprise"
_FINNHUB = "https://finnhub.io/api/v1/calendar/earnings?from={start}&to={end}&symbol={sym}&token={tok}"
_UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def _epoch(date_str: str, fmt: str) -> Optional[float]:
    try:
        return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError):
        return None


def fetch_nasdaq_earnings_surprise(symbol: str, *, timeout: int = 15) -> List[dict]:
    """Last ~4 quarters of reported EPS vs consensus for `symbol` (free, no key)."""
    req = urllib.request.Request(_NASDAQ.format(sym=symbol.upper()), headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.load(r)
    except Exception:
        return []
    rows = (((d or {}).get("data") or {}).get("earningsSurpriseTable") or {}).get("rows") or []
    out = []
    for row in rows:
        ep = _epoch(row.get("dateReported", ""), "%m/%d/%Y")
        if ep is None:
            continue
        try:
            eps = float(row.get("eps"))
            cons = float(row.get("consensusForecast"))
            surp = float(str(row.get("percentageSurprise")).replace(",", ""))
        except (TypeError, ValueError):
            continue
        out.append({"date": ep, "surprise_pct": surp, "eps": eps, "consensus": cons})
    out.sort(key=lambda x: x["date"])
    return out


def fetch_finnhub_earnings(symbol: str, *, start: str, end: str, token: Optional[str] = None,
                           timeout: int = 15) -> List[dict]:
    """Deeper earnings history via Finnhub (needs FINNHUB_API_KEY). start/end are 'YYYY-MM-DD'."""
    tok = token or os.environ.get("FINNHUB_API_KEY", "")
    if not tok:
        return []
    url = _FINNHUB.format(start=start, end=end, sym=symbol.upper(), tok=tok)
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=timeout) as r:
            d = json.load(r)
    except Exception:
        return []
    out = []
    for e in (d or {}).get("earningsCalendar", []) or []:
        ep = _epoch(e.get("date", ""), "%Y-%m-%d")
        act, est = e.get("epsActual"), e.get("epsEstimate")
        if ep is None or act is None or est in (None, 0):
            continue
        try:
            surp = (float(act) - float(est)) / abs(float(est)) * 100.0
        except (TypeError, ValueError, ZeroDivisionError):
            continue
        out.append({"date": ep, "surprise_pct": surp, "eps": float(act), "consensus": float(est)})
    out.sort(key=lambda x: x["date"])
    return out


def fetch_earnings_surprise(symbol: str, *, prefer_finnhub: bool = True, **kw) -> List[dict]:
    """Prefer Finnhub (deep history) when a key exists, else fall back to Nasdaq (~1yr)."""
    if prefer_finnhub and os.environ.get("FINNHUB_API_KEY"):
        start = kw.get("start", "2019-01-01")
        end = kw.get("end", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
        rows = fetch_finnhub_earnings(symbol, start=start, end=end)
        if rows:
            return rows
    return fetch_nasdaq_earnings_surprise(symbol)


def download_universe_earnings(symbols, out_dir, *, delay_s: float = 0.3) -> dict:
    """Fetch + cache earnings surprise for a universe to <out_dir>/<sym>_earnings.json.
    Returns {sym: n_events}. Polite delay between requests; skips already-cached files."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for sym in symbols:
        fp = out / f"{sym}_earnings.json"
        if fp.exists():
            counts[sym] = len(json.loads(fp.read_text()))
            continue
        rows = fetch_earnings_surprise(sym)
        fp.write_text(json.dumps(rows))
        counts[sym] = len(rows)
        if rows:
            time.sleep(delay_s)
    return counts
