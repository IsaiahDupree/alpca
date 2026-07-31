"""
Download DEEP daily history (back to ~1993 for SPY, ~1999 for QQQ, etc.) from the
Yahoo Finance chart API -- a free, no-key, curl-able source -- to unblock research
that needs more than the 5yr Alpaca cache (e.g. seasonality/turn-of-month decay
tests that need pre-2011 data to test the literature's "effect has decayed" claim).

Source: https://query1.finance.yahoo.com/v8/finance/chart/<SYM>?period1=..&period2=..&interval=1d
Shape:  data["chart"]["result"][0]["timestamp"]                       -> unix seconds, one per bar
        data["chart"]["result"][0]["indicators"]["quote"][0]          -> open/high/low/close/volume arrays
        data["chart"]["result"][0]["indicators"]["adjclose"][0]["adjclose"] -> split+dividend-adjusted close

Output schema (one JSON object per line, matches alpca/data/bars.py convention,
adjustment="all" == split+dividend-adjusted):
  {"open": float, "high": float, "low": float, "close": float, "volume": float,
   "timestamp": float, "symbol": str, "adjustment": "all"}

CRITICAL: "close" is the Yahoo adjclose value (not raw quote close) to match the
rest of this codebase's adjusted-close convention. open/high/low/volume are scaled
by the same (adjclose / raw_close) ratio for that bar so OHLC stays internally
consistent (standard adjusted-bars practice) -- volume is scaled by the INVERSE
ratio (adjusted volume = raw_volume / ratio) since adjusted price * adjusted volume
should preserve dollar volume, consistent with how adjusted-close data providers
(including Alpaca's "all" adjustment) present volume.

Bars with null/zero volume are skipped (Yahoo sometimes returns holiday gaps) --
no fabricated values. Output is sorted by timestamp ascending and deduped.

Usage:
  python scripts/download_yahoo_deep.py --symbols SPY,QQQ --start 1993-01-01 \
      --out "/Volumes/My Passport/AlpcaData/cache_seasonality_deep"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"


def _fetch_yahoo_chart(symbol: str, period1: int, period2: int, retries: int = 5) -> dict:
    # NOTE: shells out to curl rather than using urllib. Empirically, Yahoo's edge
    # (Akamai/bot-mitigation) 429s Python's urllib requests (TLS/HTTP fingerprint)
    # even with an identical User-Agent and URL, while an equivalent curl request
    # from the same IP at the same time succeeds -- so curl is used as the actual
    # transport here to get a fingerprint that isn't blocked.
    params = f"?period1={period1}&period2={period2}&interval=1d&events=div,splits"
    url = YAHOO_CHART_URL.format(sym=symbol) + params
    last_err = None
    for attempt in range(retries):
        try:
            # -f: fail (nonzero exit) on HTTP >=400 instead of printing the error
            # body (e.g. "Too Many Requests") to stdout, which would otherwise look
            # like a JSON-parse failure and obscure that it was a 429/403.
            proc = subprocess.run(
                ["curl", "-sL", "-f", "-A", UA, "--max-time", "30", url],
                capture_output=True, timeout=35,
            )
            if proc.returncode != 0:
                last_err = RuntimeError(f"curl exit {proc.returncode}: {proc.stderr.decode(errors='replace')[:200]}")
            else:
                payload = json.loads(proc.stdout)
                err = payload.get("chart", {}).get("error")
                if err:
                    last_err = RuntimeError(f"Yahoo API error: {err}")
                else:
                    return payload
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            last_err = e
        # generous backoff -- Yahoo's edge rate-limits aggressively after a
        # handful of requests in quick succession (observed: several requests
        # within ~30s trips a 429 that takes 1-2 min to clear)
        time.sleep(20.0 * (attempt + 1))
    raise RuntimeError(f"Yahoo chart fetch failed for {symbol} after {retries} tries: {last_err}")


def _parse_bars(symbol: str, payload: dict) -> list[dict]:
    result_list = payload.get("chart", {}).get("result") or []
    if not result_list:
        err = payload.get("chart", {}).get("error")
        raise RuntimeError(f"Yahoo chart returned no result for {symbol}: {err}")
    result = result_list[0]

    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators", {})
    quote_list = indicators.get("quote") or []
    quote = quote_list[0] if quote_list else {}
    adjclose_list = indicators.get("adjclose") or []
    adjclose = (adjclose_list[0] or {}).get("adjclose") if adjclose_list else None

    o = quote.get("open") or []
    h = quote.get("high") or []
    l = quote.get("low") or []
    c = quote.get("close") or []
    v = quote.get("volume") or []
    ac = adjclose or c  # fall back to raw close if adjclose missing (shouldn't happen)

    n = len(timestamps)
    bars = []
    seen_ts = set()
    for i in range(n):
        ts = timestamps[i]
        raw_close = c[i] if i < len(c) else None
        raw_open = o[i] if i < len(o) else None
        raw_high = h[i] if i < len(h) else None
        raw_low = l[i] if i < len(l) else None
        raw_vol = v[i] if i < len(v) else None
        adj_close = ac[i] if i < len(ac) else None

        # skip incomplete / holiday-gap bars -- do not fabricate values
        if ts is None or raw_close is None or raw_open is None or raw_high is None \
                or raw_low is None or raw_vol is None or adj_close is None:
            continue
        if not raw_vol or raw_vol <= 0:
            continue
        if not raw_close or raw_close <= 0:
            continue
        if ts in seen_ts:
            continue
        seen_ts.add(ts)

        # scale OHLC by adjclose/close ratio so the whole bar is internally
        # consistent on the adjusted-close basis; scale volume by the inverse
        # ratio to preserve approximate dollar volume (adjusted price * adjusted
        # volume ~= raw price * raw volume)
        ratio = adj_close / raw_close
        bars.append({
            "open": raw_open * ratio,
            "high": raw_high * ratio,
            "low": raw_low * ratio,
            "close": adj_close,
            "volume": raw_vol / ratio,
            "timestamp": float(ts),
            "symbol": symbol,
            "adjustment": "all",
        })

    bars.sort(key=lambda b: b["timestamp"])
    return bars


def _write_jsonl(path: Path, rows: list[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="SPY")
    ap.add_argument("--start", default="1993-01-01", help="YYYY-MM-DD, earliest date to request")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD, defaults to today")
    ap.add_argument("--out", default="data/cache_seasonality_deep")
    args = ap.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    start_dt = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = (datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
              if args.end else datetime.now(timezone.utc))
    period1 = int(start_dt.timestamp())
    period2 = int(end_dt.timestamp())

    out = Path(args.out)
    print(f"[ok] symbols={symbols}  start={args.start}  end={end_dt.date()}  out={out}")

    grand = 0
    for i, sym in enumerate(symbols):
        if i > 0:
            time.sleep(15.0)  # avoid tripping Yahoo's edge rate-limit between symbols
        print(f"\n{sym}:")
        try:
            payload = _fetch_yahoo_chart(sym, period1, period2)
            bars = _parse_bars(sym, payload)
        except Exception as e:
            print(f"  FAIL: {e}")
            continue
        out_path = out / f"{sym}_1day_bars.jsonl"
        n = _write_jsonl(out_path, bars)
        grand += n
        if bars:
            first_d = datetime.fromtimestamp(bars[0]["timestamp"], tz=timezone.utc).date()
            last_d = datetime.fromtimestamp(bars[-1]["timestamp"], tz=timezone.utc).date()
            print(f"  {sym}: {n} bars  {first_d} .. {last_d}  -> {out_path}")
        else:
            print(f"  {sym}: 0 bars written (empty result) -> {out_path}")

    print(f"\n[done] {out}/   total bars={grand}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
