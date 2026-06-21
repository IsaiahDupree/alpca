"""
Strategy-spec extraction — turn article/research text into STRUCTURED, testable StrategySpecs,
and map each against the edges we've already tested so the pipeline surfaces only NOVEL ones.

Uses the OpenAI API (key in .env) via urllib (no SDK dependency). JSON-mode response, validated
into StrategySpec. The extractor is told our venue constraints and our existing case list so it can
(a) write a precise rule and (b) flag whether the idea is already-tested or novel.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import List, Optional

from .corpus import Article, StrategySpec

# Short catalog of what we've ALREADY tested (so the extractor can dedup). Keep terse.
ALREADY_TESTED = (
    "cointegration pairs (DEPLOYED), short-vol/VRP (DEPLOYED), single-asset trend/breakout/MR (beta), "
    "cross-sectional momentum, short-term reversal, naive pairs+Kalman, crypto pairs, Avellaneda-Stoikov MM, "
    "PCA/eigenportfolio stat-arb, TSMOM, crypto funding tilt, news/sentiment alt-data, PEAD/EAR-PEAD/SUE-PEAD, "
    "calendar seasonality (turn-of-month, pre-FOMC), overnight reversal, lead-lag, gap reversion, "
    "short-interest tilt, 52-week-high, accruals, value composite, BAB/low-vol, gross profitability, "
    "asset-growth, net-issuance, ROA, MAX, idio-vol, residual-mom, vol-managed-mom, mid-cap value/momentum, "
    "value+momentum combo, formulaic-alpha zoo (Alpha101/158/191), meta-labeling"
)

VENUE = ("Alpaca PAPER; ~1.2s fills; IEX/SIP top-of-book (no L2 depth); no maker rebates; price-taker; "
         "~2bps equity per leg; adverse-selection borrow on shorts; daily/minute bars + AV news-sentiment + "
         "AV/EDGAR fundamentals + FINRA short-interest available; 10.5yr SIP daily history (2016-).")

SCHEMA_HINT = {
    "name": "str", "asset_class": "equity|etf|crypto|multi", "style": "market-neutral|directional|factor|event|seasonal|carry|other",
    "signal_rule": "precise entry/exit", "direction": "long|short|long-short|market-neutral",
    "universe": "str", "rebalance": "str", "holding": "str", "data_needs": ["str"],
    "claimed_metric": "str", "claimed_period": "str", "citation": "str",
    "maps_to": "<existing edge name from the tested list> OR 'novel'",
    "feasible_on_alpaca": "true|false", "notes": "why novel / why infeasible / key risk",
}


def _openai_key() -> str:
    for line in (Path(__file__).resolve().parents[2] / ".env").read_text().splitlines():
        if line.startswith("OPENAI_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("no OPENAI_API_KEY in .env")


def _chat_json(prompt: str, *, model="gpt-4o-mini", max_tokens=1500) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a quant research analyst. Extract testable trading "
             "strategies as strict JSON. Be precise and skeptical; do not invent metrics."},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request("https://api.openai.com/v1/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {_openai_key()}",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        out = json.load(r)
    return json.loads(out["choices"][0]["message"]["content"])


def extract_specs(article: Article, *, model="gpt-4o-mini") -> List[StrategySpec]:
    """Extract 0..N strategy specs from one research/community article."""
    text = (article.text or article.summary or article.title)[:8000]
    prompt = (
        f"VENUE CONSTRAINTS: {VENUE}\n\n"
        f"ALREADY-TESTED EDGES (mark maps_to to one of these if the idea is the same, else 'novel'):\n{ALREADY_TESTED}\n\n"
        f"ARTICLE TITLE: {article.title}\nSOURCE: {article.source}\nURL: {article.url}\n\n"
        f"ARTICLE TEXT:\n{text}\n\n"
        "Extract every DISTINCT, IMPLEMENTABLE trading strategy described or strongly implied. "
        "Return JSON: {\"strategies\": [ {spec}, ... ]} where each spec has exactly these keys: "
        f"{json.dumps(SCHEMA_HINT)}. "
        "If the article describes no concrete implementable strategy, return {\"strategies\": []}. "
        "Set feasible_on_alpaca=false for anything needing L2/HFT/options/intl-shorting we lack."
    )
    try:
        data = _chat_json(prompt, model=model)
    except Exception as e:
        return [StrategySpec(name=f"[extract-error] {article.title[:50]}", asset_class="equity",
                             style="other", signal_rule="", direction="long",
                             citation=article.url, source_ids=[article.id], status="rejected",
                             notes=f"extraction failed: {e}")]
    specs = []
    for s in data.get("strategies", []):
        if not s.get("name") or not s.get("signal_rule"):
            continue
        specs.append(StrategySpec(
            name=s.get("name", "")[:120], asset_class=s.get("asset_class", "equity"),
            style=s.get("style", "other"), signal_rule=s.get("signal_rule", ""),
            direction=s.get("direction", "long"), universe=s.get("universe", ""),
            rebalance=s.get("rebalance", ""), holding=s.get("holding", ""),
            data_needs=s.get("data_needs", []) or [], claimed_metric=s.get("claimed_metric", ""),
            claimed_period=s.get("claimed_period", ""), citation=s.get("citation", "") or article.url,
            source_ids=[article.id], maps_to=s.get("maps_to", "novel"),
            feasible_on_alpaca=s.get("feasible_on_alpaca"), notes=s.get("notes", ""),
            status="extracted"))
    return specs
