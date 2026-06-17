"""
Cache the pairs-basket walk-forward OOS daily returns to disk so downstream combiner/portfolio scripts
read them instantly instead of re-running the ~2-minute delisting-aware walk-forward every time.

Writes {date(YYYY-MM-DD), asof(epoch), ret} rows + a header with the WF Sharpe and config, to
data/pairs_wf_returns.json (gitignored — it's derived data). Re-run whenever the large-cap bars refresh.

Run: .venv/bin/python scripts/cache_pairs_wf.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.pairs import delisting_aware_walkforward  # noqa: E402


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--largecap", default="/Volumes/My Passport/AlpcaData/cache_largecap_sip")
    ap.add_argument("--out", default="data/pairs_wf_returns.json")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--max-adf", type=float, default=-2.86)
    args = ap.parse_args()

    bars = _load(args.largecap)
    print(f"[cache] running delisting-aware walk-forward on {len(bars)} large-caps (deployed config)...")
    r = delisting_aware_walkforward(bars, train=252, test=63, top_n=args.top_n, max_adf=args.max_adf)
    rows = [{"date": time.strftime("%Y-%m-%d", time.gmtime(int(t))), "asof": int(t), "ret": ret}
            for t, ret in zip(r.dates, r.daily_returns)]
    out = {"wf_sharpe": round(r.sharpe, 4), "n_windows": r.n_windows, "n_days": len(rows),
           "config": {"top_n": args.top_n, "max_adf": args.max_adf, "train": 252, "test": 63},
           "returns": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[cache] WF Sharpe {r.sharpe:.3f} · {len(rows)} OOS days -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
