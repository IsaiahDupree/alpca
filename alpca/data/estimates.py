"""
AlphaVantage EARNINGS_ESTIMATES — analyst consensus EPS estimates + revision trends (free tier).

Returns, for the nearest fiscal-year horizon, the current consensus EPS, the consensus 7/30/60/90
days ago, and the count of up/down analyst revisions over the trailing 7/30 days. This is the raw
material for the analyst-revision-drift signal (estimate momentum + revision breadth).

IMPORTANT (honesty): this endpoint is a CURRENT SNAPSHOT — it does NOT provide a point-in-time
history of past estimates at arbitrary dates. So a revision signal can only be FORWARD-tracked from
when we start snapshotting it; there is no way to backtest it on free data. Quota: AV free = ~25
calls/day, SHARED with the earnings job — fetch a small slice per day.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Optional

AV_URL = "https://www.alphavantage.co/query"


def av_key() -> str:
    for line in (Path(__file__).resolve().parents[2] / ".env").read_text().splitlines():
        if line.startswith("ALPHAVANTAGE_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("no ALPHAVANTAGE_API_KEY in .env")


def _f(v) -> Optional[float]:
    try:
        return float(v) if v not in (None, "None", "") else None
    except (TypeError, ValueError):
        return None


def fetch_estimates(symbol: str, key: Optional[str] = None, timeout: int = 30) -> Optional[dict]:
    """Return the nearest fiscal-year estimate snapshot for `symbol`, or None on miss/rate-limit."""
    key = key or av_key()
    url = f"{AV_URL}?function=EARNINGS_ESTIMATES&symbol={symbol}&apikey={key}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            d = json.load(r)
    except Exception:
        return None
    est = d.get("estimates")
    if not est:
        return None  # rate-limited / no coverage
    fy = next((e for e in est if e.get("horizon") == "fiscal year"
               and _f(e.get("eps_estimate_average")) is not None), None)
    if not fy:
        return None
    avg = _f(fy.get("eps_estimate_average"))
    a90 = _f(fy.get("eps_estimate_average_90_days_ago"))
    a30 = _f(fy.get("eps_estimate_average_30_days_ago"))
    up30 = _f(fy.get("eps_estimate_revision_up_trailing_30_days")) or 0.0
    dn30 = _f(fy.get("eps_estimate_revision_down_trailing_30_days")) or 0.0
    n = _f(fy.get("eps_estimate_analyst_count")) or 0.0
    # estimate momentum (90d consensus drift, scaled) + revision breadth (net up/down / analysts)
    est_mom = ((avg - a90) / abs(a90)) if (avg is not None and a90 not in (None, 0)) else None
    breadth = ((up30 - dn30) / n) if n > 0 else None
    return {"symbol": symbol, "horizon_date": fy.get("date"),
            "eps_avg": avg, "eps_avg_30d": a30, "eps_avg_90d": a90,
            "rev_up_30d": up30, "rev_down_30d": dn30, "analyst_count": n,
            "est_mom_90d": (round(est_mom, 5) if est_mom is not None else None),
            "rev_breadth_30d": (round(breadth, 4) if breadth is not None else None)}
