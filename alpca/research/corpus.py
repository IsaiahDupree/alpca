"""
The research corpus — normalized storage for harvested articles and the strategy specs mined
from them. Append-only JSONL, content-hashed dedup, provider-agnostic.

Two record types, two files under data/research/:
  - articles.jsonl  : raw harvested items (news, research write-ups, community posts)
  - specs.jsonl     : structured, testable strategy specs extracted from articles

Both are plain dicts on disk (forward-compatible); the dataclasses below are the in-code shape.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def _hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


@dataclass
class Article:
    """One harvested item from any provider."""
    source: str                      # e.g. "alphavantage", "alpaca", "alphaarchitect.com", "reddit:algotrading"
    kind: str                        # "news" | "research" | "community"
    title: str
    url: str
    published: Optional[float] = None  # epoch seconds (best-effort)
    authors: List[str] = field(default_factory=list)
    summary: str = ""
    text: str = ""                   # full text if fetched (research/community); else summary
    tickers: List[str] = field(default_factory=list)
    sentiment: Optional[float] = None  # provider sentiment score if any (AV)
    relevance: Optional[float] = None  # provider ticker-relevance if any (AV)
    fetched_at: Optional[float] = None
    extra: Dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return _hash(self.source, self.url or self.title)

    def to_row(self) -> dict:
        d = asdict(self)
        d["id"] = self.id
        return d


@dataclass
class StrategySpec:
    """A structured, testable strategy distilled from one or more articles."""
    name: str
    asset_class: str                 # "equity" | "etf" | "crypto" | "multi"
    style: str                       # "market-neutral" | "directional" | "factor" | "event" | "seasonal" | "carry" | ...
    signal_rule: str                 # plain-language but precise entry/exit rule
    direction: str                   # "long" | "short" | "long-short" | "market-neutral"
    universe: str = ""               # claimed universe (e.g. "S&P 500", "mid-cap")
    rebalance: str = ""              # "daily" | "monthly" | "event-driven" | ...
    holding: str = ""               # claimed holding period
    data_needs: List[str] = field(default_factory=list)   # ["daily bars", "earnings surprise", "short interest", "news sentiment"]
    claimed_metric: str = ""         # e.g. "Sharpe 1.2", "8%/yr alpha"
    claimed_period: str = ""         # e.g. "1990-2015 US"
    citation: str = ""               # source title / DOI / URL
    source_ids: List[str] = field(default_factory=list)   # Article.id list
    maps_to: str = "novel"           # existing primitive name OR "novel"
    feasible_on_alpaca: Optional[bool] = None             # given venue constraints (no L2, paper short frictions, etc.)
    status: str = "extracted"        # extracted | mapped | implemented | validated | rejected
    verdict: str = ""                # filled by the validation stage
    notes: str = ""

    @property
    def id(self) -> str:
        return _hash(self.name, self.signal_rule)

    def to_row(self) -> dict:
        d = asdict(self)
        d["id"] = self.id
        return d


class Corpus:
    """Append-only, deduped JSONL store for Articles and StrategySpecs."""

    def __init__(self, root: str | Path = "data/research"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.articles_path = self.root / "articles.jsonl"
        self.specs_path = self.root / "specs.jsonl"

    # ---- load ----
    def _load(self, path: Path) -> Dict[str, dict]:
        out: Dict[str, dict] = {}
        if path.exists():
            for line in path.open():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("id"):
                    out[r["id"]] = r       # later wins -> idempotent updates
        return out

    def load_articles(self) -> Dict[str, dict]:
        return self._load(self.articles_path)

    def load_specs(self) -> Dict[str, dict]:
        return self._load(self.specs_path)

    # ---- add (dedup by id; returns # newly added) ----
    def add_articles(self, arts: List[Article]) -> int:
        have = set(self.load_articles())
        n = 0
        with self.articles_path.open("a") as f:
            for a in arts:
                if a.id in have:
                    continue
                f.write(json.dumps(a.to_row()) + "\n")
                have.add(a.id)
                n += 1
        return n

    def add_specs(self, specs: List[StrategySpec]) -> int:
        have = set(self.load_specs())
        n = 0
        with self.specs_path.open("a") as f:
            for s in specs:
                if s.id in have:
                    continue
                f.write(json.dumps(s.to_row()) + "\n")
                have.add(s.id)
                n += 1
        return n

    def update_spec(self, spec_id: str, **fields) -> bool:
        """Rewrite specs.jsonl with `fields` merged into the matching spec (status/verdict updates)."""
        specs = self.load_specs()
        if spec_id not in specs:
            return False
        specs[spec_id].update(fields)
        with self.specs_path.open("w") as f:
            for r in specs.values():
                f.write(json.dumps(r) + "\n")
        return True

    def stats(self) -> dict:
        arts = self.load_articles()
        specs = self.load_specs()
        by_source: Dict[str, int] = {}
        by_kind: Dict[str, int] = {}
        for a in arts.values():
            by_source[a.get("source", "?")] = by_source.get(a.get("source", "?"), 0) + 1
            by_kind[a.get("kind", "?")] = by_kind.get(a.get("kind", "?"), 0) + 1
        by_status: Dict[str, int] = {}
        for s in specs.values():
            by_status[s.get("status", "?")] = by_status.get(s.get("status", "?"), 0) + 1
        return {"n_articles": len(arts), "n_specs": len(specs),
                "articles_by_source": by_source, "articles_by_kind": by_kind,
                "specs_by_status": by_status}
