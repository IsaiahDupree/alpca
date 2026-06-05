"""
Alpaca news (alt-data) — paginated fetch + daily per-symbol article counts.

A single get_news call returns only the newest ~50 articles; this paginates via the
page_token to pull a full historical window. News COUNT/flow is a non-price feature
testable WITHOUT an NLP sentiment model (does an attention spike predict drift or
reversal?); proper sentiment would need a model on top. Returns plain dicts so the
evaluation harness / cross-sectional tools can consume daily news-flow series.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Dict, List

from alpca.config import AlpacaConfig


def fetch_alpaca_news(config: AlpacaConfig, symbols, *, start: datetime, end: datetime,
                      max_pages: int = 100) -> List[dict]:
    """Paginate Alpaca news for `symbols` over [start, end]. Returns
    [{symbols, created_at(epoch), headline}] oldest-first. max_pages caps API calls."""
    config.require_credentials()
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest

    client = NewsClient(config.api_key, config.secret_key)
    sym = ",".join(symbols) if not isinstance(symbols, str) else symbols
    out: List[dict] = []
    token = None
    for _ in range(max_pages):
        req = NewsRequest(symbols=sym, start=start, end=end, limit=50, page_token=token)
        resp = client.get_news(req)
        arts = resp.data.get("news", []) if hasattr(resp, "data") else []
        for a in arts:
            ca = getattr(a, "created_at", None)
            out.append({
                "symbols": list(getattr(a, "symbols", []) or []),
                "created_at": ca.timestamp() if hasattr(ca, "timestamp") else 0.0,
                "headline": getattr(a, "headline", "") or "",
            })
        token = getattr(resp, "next_page_token", None)
        if not token:
            break
    out.sort(key=lambda r: r["created_at"])
    return out


def daily_news_counts(articles: List[dict], symbol: str) -> Dict[str, int]:
    """Per-ET-date article counts for `symbol` (attention proxy / news-flow feature)."""
    from alpca.data.calendar import session_date
    counts: Dict[str, int] = {}
    for a in articles:
        if symbol in a.get("symbols", []) and a.get("created_at"):
            d = session_date(a["created_at"])
            counts[d] = counts.get(d, 0) + 1
    return counts


def news_history_depth(config: AlpacaConfig, symbol: str = "AAPL", max_pages: int = 40) -> dict:
    """Diagnostic: how far back does Alpaca news go for one symbol, and how dense?"""
    end = datetime.now(timezone.utc)
    arts = fetch_alpaca_news(config, symbol, start=end - timedelta(days=3650), end=end, max_pages=max_pages)
    if not arts:
        return {"symbol": symbol, "n": 0}
    span_days = (arts[-1]["created_at"] - arts[0]["created_at"]) / 86400.0
    return {"symbol": symbol, "n": len(arts), "span_days": round(span_days, 1),
            "oldest": arts[0]["created_at"], "newest": arts[-1]["created_at"],
            "per_day": round(len(arts) / span_days, 2) if span_days > 0 else 0}
