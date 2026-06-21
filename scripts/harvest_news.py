"""
Harvest the API-based article sources into the research corpus (data/research/articles.jsonl):
  - AlphaVantage NEWS_SENTIMENT : title/url/summary + sentiment + per-ticker relevance, history via
    time_from/time_to (monthly windows). Free tier ~25 req/day — SHARED with the earnings job —
    so --max-calls is enforced and defaults low.
  - Alpaca news (Benzinga)       : paginated newest-first (shallow history on free, but real-time).

The web / Reddit / Perplexity sources are harvested by the agent's tools and written to the same
corpus via alpca.research.corpus — this script only covers the pure-HTTP providers.

Run:
  .venv/bin/python scripts/harvest_news.py --av-topics financial_markets,earnings --av-months 2 --max-calls 6
  .venv/bin/python scripts/harvest_news.py --av-tickers AAPL,MSFT --av-months 3 --alpaca-tickers AAPL --max-calls 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpca.research.corpus import Article, Corpus  # noqa: E402

AV_URL = "https://www.alphavantage.co/query"


def _av_key() -> str:
    for line in (Path(__file__).resolve().parents[1] / ".env").read_text().splitlines():
        if line.startswith("ALPHAVANTAGE_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no ALPHAVANTAGE_API_KEY in .env")


def _parse_av_time(s: str):
    # AV time_published format: 20260621T181658
    try:
        return datetime.strptime(s, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc).timestamp()
    except Exception:
        return None


def av_news(key, *, tickers=None, topics=None, months=2, max_calls=6, sleep=1.2):
    """Page NEWS_SENTIMENT over monthly windows (newest->older). Returns Article list."""
    arts, calls = [], 0
    now = datetime.now(timezone.utc)
    for m in range(months):
        if calls >= max_calls:
            break
        hi = now - timedelta(days=30 * m)
        lo = hi - timedelta(days=30)
        params = {"function": "NEWS_SENTIMENT", "apikey": key, "sort": "LATEST", "limit": 1000,
                  "time_from": lo.strftime("%Y%m%dT%H%M"), "time_to": hi.strftime("%Y%m%dT%H%M")}
        if tickers:
            params["tickers"] = ",".join(tickers)
        if topics:
            params["topics"] = ",".join(topics)
        url = AV_URL + "?" + urllib.parse.urlencode(params)
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
        except Exception as e:
            print(f"  [av] window {lo.date()}..{hi.date()} ERR {e}")
            calls += 1
            continue
        calls += 1
        feed = d.get("feed")
        if feed is None:
            print(f"  [av] window {lo.date()}..{hi.date()}: {d.get('Information') or d.get('Note') or list(d.keys())}")
            time.sleep(sleep)
            continue
        for it in feed:
            ts = it.get("ticker_sentiment", [])
            tks = [t.get("ticker") for t in ts]
            # average relevance across mentioned tickers (or the queried ones)
            rels = [float(t.get("relevance_score", 0) or 0) for t in ts]
            arts.append(Article(
                source="alphavantage", kind="news", title=it.get("title", ""),
                url=it.get("url", ""), published=_parse_av_time(it.get("time_published", "")),
                authors=it.get("authors", []) or [], summary=it.get("summary", ""),
                tickers=[t for t in tks if t], sentiment=float(it.get("overall_sentiment_score", 0) or 0),
                relevance=(sum(rels) / len(rels) if rels else None),
                fetched_at=now.timestamp(),
                extra={"sentiment_label": it.get("overall_sentiment_label"), "topics":
                       [t.get("topic") for t in it.get("topics", [])]}))
        print(f"  [av] window {lo.date()}..{hi.date()}: {len(feed)} articles (call {calls}/{max_calls})")
        time.sleep(sleep)
    return arts


def alpaca_news(tickers, *, days=30, max_pages=10):
    from alpca.config import load_config
    from alpca.data.news import fetch_alpaca_news
    cfg = load_config()
    if not cfg.has_credentials:
        print("  [alpaca] no credentials, skipping")
        return []
    end = datetime.now(timezone.utc) - timedelta(minutes=20)
    start = end - timedelta(days=days)
    out = []
    for sym in tickers:
        raw = fetch_alpaca_news(cfg, sym, start=start, end=end, max_pages=max_pages)
        for a in raw:
            hl = a.get("headline") or a.get("title") or ""
            url = a.get("url") or (a.get("source_url") or "")
            ts = a.get("created_at") or a.get("updated_at")
            pub = None
            if isinstance(ts, (int, float)):
                pub = float(ts)
            out.append(Article(source="alpaca", kind="news", title=hl, url=url or f"alpaca:{hl[:60]}",
                               summary=a.get("summary", "") or "", tickers=[sym], extra={"raw_keys": list(a.keys())[:8]}))
        print(f"  [alpaca] {sym}: {len(raw)} articles")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--av-tickers", default="")
    ap.add_argument("--av-topics", default="")
    ap.add_argument("--av-months", type=int, default=2)
    ap.add_argument("--alpaca-tickers", default="")
    ap.add_argument("--alpaca-days", type=int, default=30)
    ap.add_argument("--max-calls", type=int, default=6, help="AV free tier ~25/day, shared with earnings job")
    ap.add_argument("--root", default="data/research")
    args = ap.parse_args()

    corpus = Corpus(args.root)
    arts = []
    if args.av_tickers or args.av_topics:
        print("[harvest] AlphaVantage NEWS_SENTIMENT...")
        arts += av_news(_av_key(),
                        tickers=[t.strip() for t in args.av_tickers.split(",") if t.strip()] or None,
                        topics=[t.strip() for t in args.av_topics.split(",") if t.strip()] or None,
                        months=args.av_months, max_calls=args.max_calls)
    if args.alpaca_tickers:
        print("[harvest] Alpaca news...")
        arts += alpaca_news([t.strip() for t in args.alpaca_tickers.split(",") if t.strip()],
                            days=args.alpaca_days)
    added = corpus.add_articles(arts)
    print(f"\n[done] harvested {len(arts)} articles, {added} new -> {corpus.articles_path}")
    print(json.dumps(corpus.stats(), indent=2))


if __name__ == "__main__":
    main()
