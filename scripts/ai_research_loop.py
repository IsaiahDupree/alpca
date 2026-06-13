"""
AI research loop — the unattended hunt, gated by the honest harness.

Each iteration:
  1. detect the current market regime  (deterministic; alpca/ai/regime.py)
  2. PROPOSE a market-neutral strategy config suited to the regime  (OpenAI medium, or heuristic
     fallback; constrained to known tested templates — the model configures, never codes)
  3. RUN it through the harness on the MAIN universe + a DISJOINT fresh universe (real fresh-symbol
     holdout) + per-year regime stability + cost + DSR
  4. GATE it  (deterministic falsification rail w/ veto + live Haiku per-regime verdict)
  5. LOG the proposal, result, and decision; flag any GO survivor for the combiner / forward-track

The model proposes; ONLY the data validates. A GO requires the fresh-symbol holdout + regime
robustness + cost survival + DSR AND a Haiku concurrence — so the loop cannot ship an overfit edge.

Run: .venv/bin/python scripts/ai_research_loop.py --iterations 4
     (add --no-ai to use the deterministic heuristic proposer; needs no OpenAI key)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import alpca.config  # noqa: F401,E402  (auto-loads .env into the environment)
from alpca.ai.router import AIRouter  # noqa: E402
from alpca.ai.regime import detect_regime  # noqa: E402
from alpca.ai.strategy_generator import propose, run_proposal  # noqa: E402
from alpca.ai.strategy_gate import gate  # noqa: E402


def _load(cache: Path):
    bars = {}
    for p in cache.glob("*_1day_bars.jsonl"):
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if rows:
            bars[p.name.split("_1day_")[0]] = rows
    return bars


def _load_funds(fdir: Path):
    funds = {}
    if fdir.exists():
        for p in fdir.glob("*_fund.json"):
            rows = json.loads(p.read_text())
            if rows:
                funds[p.name.replace("_fund.json", "")] = rows
    return funds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--cache-fresh", default="/Volumes/My Passport/AlpcaData/cache_fresh")
    ap.add_argument("--fundamentals", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar")
    ap.add_argument("--fundamentals-fresh", default="/Volumes/My Passport/AlpcaData/fundamentals_edgar_fresh")
    ap.add_argument("--iterations", type=int, default=4)
    ap.add_argument("--no-ai", action="store_true", help="deterministic heuristic proposer only")
    ap.add_argument("--log", default="data/ai_research_log.jsonl")
    args = ap.parse_args()

    bars_main = _load(Path(args.cache))
    bars_fresh = _load(Path(args.cache_fresh))
    fund_main = _load_funds(Path(args.fundamentals))
    fund_fresh = _load_funds(Path(args.fundamentals_fresh))
    spy = bars_main.get("SPY") or bars_main.get("QQQ")
    if not bars_main or not spy:
        print("[fail] need a universe + SPY in --cache"); return 1
    router = None if args.no_ai else AIRouter()
    avail = router.available() if router else {"openai": False, "haiku": False}
    print(f"[loop] universe {len(bars_main)} (+{len(bars_fresh)} fresh) · funds {len(fund_main)} "
          f"(+{len(fund_fresh)} fresh) · models {avail} · {args.iterations} iterations\n")

    regime = detect_regime(spy)
    print(f"REGIME: {regime.as_prompt()}\n")
    tried, survivors = [], []
    logf = Path(args.log)
    for i in range(1, args.iterations + 1):
        cfg = propose(regime.as_prompt(), regime.label, router, tried)
        tried.append(f"{cfg['strategy_type']}{cfg['params']}")
        result = run_proposal(cfg, bars_main, bars_fresh, fund_main=fund_main, fund_fresh=fund_fresh)
        # gate: deterministic rail + (live Haiku if available; else a no-AI stub that defers to the rail)
        if router and router.available().get("haiku"):
            d = gate(result, spy, router)
        else:
            from alpca.ai.strategy_gate import falsification_gate
            fg = falsification_gate(result)
            d = {"regime": regime.label, "falsification_pass": fg.passed,
                 "falsification_reasons": fg.reasons,
                 "haiku": {"verdict": "GO" if fg.passed else "NO-GO", "rationale": "no-AI: rail only"},
                 "decision": "GO" if fg.passed else "NO-GO"}
        entry = {"iter": i, "regime": regime.label, "config": cfg, "result": result, "gate": d}
        with logf.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        fh = result["fresh_holdout_sharpe"]
        print(f"#{i} [{cfg['source']}] {cfg['strategy_type']} {cfg['params']}")
        print(f"    sharpe {result['sharpe']:+.2f} | OOS {result['oos_sharpe']:+.2f} | "
              f"fresh-holdout {fh:+.2f} | DSR {result['dsr']:.2f} | +{sum(1 for s in result['per_year'].values() if s>0)}/{len(result['per_year'])}yr")
        print(f"    rail={'PASS' if d['falsification_pass'] else 'FAIL'} · haiku={d['haiku'].get('verdict')} "
              f"· DECISION={d['decision']}  ({cfg.get('rationale','')[:60]})")
        if d["decision"] == "GO":
            survivors.append(entry)
        print()

    print("=" * 60)
    print(f"[loop] done. {len(survivors)}/{args.iterations} GO survivors "
          f"(would go to the combiner + forward paper-track). log: {logf}")
    if not survivors:
        print("  No survivor — expected: the bar (fresh-symbol + out-of-regime + cost + DSR) is brutal "
              "and almost nothing clears it. The loop's job is to reject cheaply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
