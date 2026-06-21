"""
Mine strategy specs from the research corpus: run the LLM extractor over every research/community
article, write structured StrategySpecs, and print a dedup triage (novel vs already-tested,
feasible-on-Alpaca vs not). This is the bridge from "obtained their articles" to "testable edges".

Run:
  .venv/bin/python scripts/mine_strategies.py                 # mine all research/community articles
  .venv/bin/python scripts/mine_strategies.py --kinds research --limit 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpca.research.corpus import Article, Corpus  # noqa: E402
from alpca.research.extract import extract_specs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/research")
    ap.add_argument("--kinds", default="research,community", help="article kinds to mine")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    corpus = Corpus(args.root)
    kinds = {k.strip() for k in args.kinds.split(",")}
    arts = [Article(**{k: v for k, v in r.items() if k != "id"})
            for r in corpus.load_articles().values() if r.get("kind") in kinds]
    arts = arts[:args.limit]
    print(f"[mine] {len(arts)} {'/'.join(kinds)} articles to mine with {args.model}")

    all_specs = []
    for a in arts:
        specs = extract_specs(a, model=args.model)
        all_specs.extend(specs)
        for s in specs:
            tag = "NOVEL" if s.maps_to == "novel" else f"~{s.maps_to}"
            feas = "feasible" if s.feasible_on_alpaca else "INFEASIBLE"
            print(f"  · {s.name[:60]:60} [{tag}] [{feas}]")
    added = corpus.add_specs(all_specs)

    novel = [s for s in all_specs if s.maps_to == "novel" and s.feasible_on_alpaca]
    print(f"\n[mine] extracted {len(all_specs)} specs, {added} new")
    print(f"[mine] NOVEL + feasible (validation queue): {len(novel)}")
    for s in novel:
        print(f"    -> {s.name}  ({s.style}, {s.direction}; needs: {', '.join(s.data_needs)})")
    print("\n" + json.dumps(corpus.stats(), indent=2))


if __name__ == "__main__":
    main()
