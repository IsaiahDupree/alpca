"""
The honest truth table — run EVERY registered strategy through the bulletproof
evaluation harness (vs buy-and-hold, statistical significance, regime stability, OOS)
and print a ranked verdict. This is the definitive, self-honest assessment: no strategy
is "good" unless it clears significance + stability + OOS, not just a positive backtest.

Run:
  .venv/bin/python scripts/truth_table.py --symbol SPY --timeframe 1day \
      --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import evaluate  # noqa: E402
from alpca.strategies.registry import available  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--timeframe", default="1day")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--ppy", type=float, default=252.0)
    ap.add_argument("--out", default="data/truth_table.json")
    args = ap.parse_args()

    path = Path(args.cache) / f"{args.symbol}_{args.timeframe}_bars.jsonl"
    if not path.exists():
        print(f"[fail] no bars at {path}", file=sys.stderr)
        return 1
    bars = [json.loads(l) for l in path.open() if l.strip()]
    print(f"[ok] {args.symbol} {args.timeframe}: {len(bars)} bars  (buy-and-hold is the benchmark)\n")

    rows = []
    for name in available():
        try:
            r = evaluate(name, bars, periods_per_year=args.ppy)
        except Exception as e:
            print(f"  {name}: eval failed ({e})")
            continue
        rows.append(r)
    rows.sort(key=lambda r: -r.strat_sharpe)

    if rows:
        bh = rows[0]
        print(f"{'strategy':<22}{'ret':>7}{'Sharpe':>7}{'sig':>4}{'stbl':>5}{'a':>5}  verdict")
        print("-" * 92)
        for r in rows:
            sig = "Y" if r.significant else "."
            stb = "Y" if r.stable else "."
            beat = ">B&H" if r.beats_sharpe else ""
            print(f"{r.name:<22}{r.strat_return*100:>6.0f}%{r.strat_sharpe:>7.2f}{sig:>4}{stb:>5}{beat:>5}  {r.verdict[:46]}")
        print(f"\nbenchmark: buy-and-hold {args.symbol} = {rows[0].bh_return*100:+.0f}% ret, "
              f"Sharpe {rows[0].bh_sharpe:.2f}, maxDD {rows[0].bh_maxdd*100:.0f}%")
        genuine = [r.name for r in rows if "GENUINE" in r.verdict]
        riskred = [r.name for r in rows if "RISK-REDUCED" in r.verdict]
        print(f"\nGENUINE (beats B&H risk-adj + significant + stable + OOS): {genuine or 'NONE'}")
        print(f"RISK-REDUCED exposure (better Sharpe, not a market-beater):  {riskred or 'none'}")

    Path(args.out).write_text(json.dumps([{
        "name": r.name, "return": r.strat_return, "sharpe": r.strat_sharpe,
        "bh_sharpe": r.bh_sharpe, "significant": r.significant, "stable": r.stable,
        "beats_sharpe": r.beats_sharpe, "beta": r.beta, "alpha": r.alpha,
        "sharpe_pvalue": r.sharpe_pvalue, "verdict": r.verdict,
    } for r in rows], indent=2))
    print(f"\n[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
