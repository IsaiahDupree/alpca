"""
alpca.research — the strategy-discovery pipeline.

Harvest articles/strategy write-ups from every provider we have access to, normalize them into
one corpus, extract structured + testable strategy specs, map each to an implementable backtest,
and run it through the same honest validation battery (out-of-universe + out-of-regime + cost +
DSR) that gates every other edge in this repo.

Architecture (hybrid — by necessity):
  - API sources (AlphaVantage NEWS_SENTIMENT, Alpaca/Benzinga news) are pure HTTP -> harvested by
    `scripts/harvest_news.py` (scriptable, schedulable).
  - Web / Reddit / Perplexity sources come through the agent's tools (WebSearch/WebFetch/MCP) ->
    the agent writes normalized records into the SAME corpus via `corpus.add_*`.

Everything lands in data/research/ as append-only JSONL so the corpus is reproducible and diffable.
"""

from .corpus import Article, StrategySpec, Corpus  # noqa: F401
