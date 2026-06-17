"""
Canonical DEPLOYED-PORTFOLIO backtest — the reference the live forward track is validated against.

Ties the codified weights (`alpca/live/portfolio.py`) to the real leg returns: pairs (cached walk-forward
OOS, `cache_pairs_wf.py`) + short-vol (long-SVXY) blended via the SAME `combine_tracks` the live reporter
uses, at the SAME `deployed_weights()`. So the backtested book and the live book are computed by identical
code — the live curve has a precise benchmark, not a vibe. Saves the reference to
data/deployed_portfolio_backtest.json.

Run: .venv/bin/python scripts/backtest_deployed_portfolio.py   (after cache_pairs_wf.py)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.live.portfolio import deployed_weights, combine_tracks, DEPLOYED  # noqa: E402
from alpca.backtest.evaluation import sharpe_of, max_drawdown_of, deflated_sharpe_ratio  # noqa: E402

PPY = 252.0


def _eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


def _svxy_returns(vol_cache):
    b = sorted([json.loads(l) for l in (Path(vol_cache) / "SVXY_1day_bars.jsonl").open() if l.strip()],
               key=lambda x: int(x["timestamp"]))
    return {int(b[i]["timestamp"]): float(b[i]["close"]) / float(b[i - 1]["close"]) - 1.0
            for i in range(1, len(b)) if float(b[i - 1]["close"]) > 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-cache", default="data/pairs_wf_returns.json")
    ap.add_argument("--vol-cache", default="/Volumes/My Passport/AlpcaData/cache_vol")
    ap.add_argument("--out", default="data/deployed_portfolio_backtest.json")
    args = ap.parse_args()

    pc = Path(args.pairs_cache)
    if not pc.exists():
        print(f"[err] {pc} missing — run scripts/cache_pairs_wf.py first."); return 1
    pj = json.loads(pc.read_text())
    pairs = {int(r["asof"]): r["ret"] for r in pj["returns"]}
    svxy = _svxy_returns(args.vol_cache)
    print(f"[ok] pairs WF {pj['wf_sharpe']} ({len(pairs)} days) · short-vol ({len(svxy)} days)\n")

    # GUARD against silent degradation: if the diversifier barely overlaps the core, the headline
    # lift is computed against a near-zero-weight sleeve and is meaningless.
    overlap = len(set(pairs) & set(svxy))
    cov = overlap / max(1, min(len(pairs), len(svxy)))
    if cov < 0.5:
        print(f"[WARN] pairs<>short-vol overlap only {overlap} days ({cov:.0%} of the shorter series) — "
              f"the diversifier is barely present; the lift below is NOT trustworthy. Refresh cache_vol/pairs.")

    w = deployed_weights()
    print("=== DEPLOYED WEIGHTS ===")
    for s in DEPLOYED:
        print(f"  {s.name:>10} [{s.role}] {w.get(s.name,0):.0%}")
    # the codified portfolio uses the SAME combine_tracks as the live reporter
    book = combine_tracks({"pairs": pairs, "short_vol": svxy}, weights=w)
    eq = book.equity_curve
    by = {}
    for t, x in zip(book.dates, book.daily_returns):
        by.setdefault(time.gmtime(t).tm_year, []).append(x)
    yr = {y: round(sharpe_of(_eq(v), PPY), 2) for y, v in sorted(by.items()) if len(v) >= 20}
    sh = sharpe_of(eq, PPY); dd = max_drawdown_of(eq)
    dsr = deflated_sharpe_ratio(eq, n_trials=100, sharpe_variance=1e-4)
    # pairs-alone reference (the core, for the lift)
    pa = sharpe_of(_eq([pairs[t] for t in sorted(pairs)]), PPY)
    print(f"\n=== CANONICAL DEPLOYED BOOK ({book.n_days} days) ===")
    print(f"  Sharpe {sh:.3f} · maxDD {dd*100:.1f}% · DSR {dsr:.2f} · per-year {yr}")
    print(f"  vs pairs-core alone {pa:.3f}  ->  lift {sh-pa:+.3f}")
    print("  ^ this is the benchmark the LIVE forward track is measured against (identical combine code).")
    out = {"weights": w, "sharpe": round(sh, 4), "max_drawdown": round(dd, 4), "dsr": round(dsr, 3),
           "pairs_core_sharpe": round(pa, 4), "lift": round(sh - pa, 4), "per_year": yr,
           "n_days": book.n_days, "as_of": time.strftime("%Y-%m-%d")}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
