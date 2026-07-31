"""
Deep-history extension of scripts/test_seasonality.py — tests the calendar seasonality
sleeves (turn-of-month, pre-FOMC) on the Yahoo-Finance deep-history cache (SPY back to
1993-01-29, QQQ back to 1999-03-10), which is long enough to test the pre-2011-vs-post-2011
DECAY claim that scripts/test_seasonality.py explicitly could NOT test (its cache is only
~2021-2026).

Unlike the existing script's fractional 70/30 IS/OOS split, this splits by ACTUAL CALENDAR
DATE at 2011-01-01: everything before is "pre-2011", everything from 2011-01-01 onward is
"post-2011". Also reports a cross-check leg restricted to the 2021-2026 window (same window
the existing script tests), to sanity-check this script's numbers against the existing
script's numbers on the overlapping data.

Run: .venv/bin/python scripts/test_seasonality_deep.py \
    --cache "/Volumes/My Passport/AlpcaData/cache_seasonality_deep"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import (  # noqa: E402
    buy_and_hold, sharpe_of, sharpe_pvalue, sharpe_tstat, max_drawdown_of)
from alpca.backtest.seasonality import (  # noqa: E402
    backtest_seasonal, pre_fomc_position, turn_of_month_position)

PPY = 252.0
SPLIT_DATE = "2011-01-01"
CROSSCHECK_START = "2021-01-01"  # matches the ~5yr window test_seasonality.py's cache covers


def _bar_date(b: dict) -> str:
    return datetime.fromtimestamp(float(b["timestamp"]), timezone.utc).strftime("%Y-%m-%d")


def segment_stats(bars: list, position: list, name: str, cost_bps: float) -> dict:
    """Backtest `position` on `bars` and return the full stat set (Sharpe/t/p/exposure/maxDD)."""
    if len(bars) < 3:
        return {"n_bars": len(bars), "insufficient_data": True}
    r = backtest_seasonal(bars, position, name=name, cost_bps=cost_bps, periods_per_year=PPY)
    return {
        "n_bars": len(bars),
        "sharpe": r.sharpe,
        "tstat": sharpe_tstat(r.equity_curve),
        "pval": sharpe_pvalue(r.equity_curve),
        "exposure": r.exposure,
        "maxdd": r.max_drawdown,
        "total_return": r.total_return,
    }


def slice_bars_and_position(bars: list, position: list, start: str = None, end: str = None) -> tuple:
    """Slice bars+position (parallel lists) to [start, end) by calendar date string."""
    idx = [i for i, b in enumerate(bars)
           if (start is None or _bar_date(b) >= start) and (end is None or _bar_date(b) < end)]
    if not idx:
        return [], []
    lo, hi = idx[0], idx[-1] + 1
    return bars[lo:hi], position[lo:hi]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache_seasonality_deep")
    ap.add_argument("--symbols", default="SPY,QQQ")
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--out", default="data/seasonality_deep_results.json")
    args = ap.parse_args()
    cache = Path(args.cache)
    out = {}

    for sym in args.symbols.split(","):
        p = cache / f"{sym}_1day_bars.jsonl"
        if not p.exists():
            print(f"[skip] {sym}: {p} not found")
            continue
        bars = [json.loads(l) for l in p.open() if l.strip()]
        first_date, last_date = _bar_date(bars[0]), _bar_date(bars[-1])
        bh_full = buy_and_hold(bars, PPY)
        print(f"\n===== {sym} ({len(bars)} bars, {first_date} -> {last_date}) =====")
        print(f"  buy-and-hold (full):  Sharpe {bh_full.sharpe:.4f}  ret {bh_full.total_return*100:+.1f}%")

        legs = {}
        for name, pos in (("turn-of-month", turn_of_month_position(bars)),
                          ("pre-FOMC", pre_fomc_position(bars))):
            full = segment_stats(bars, pos, name, args.cost_bps)

            pre_bars, pre_pos = slice_bars_and_position(bars, pos, end=SPLIT_DATE)
            post_bars, post_pos = slice_bars_and_position(bars, pos, start=SPLIT_DATE)
            pre = segment_stats(pre_bars, pre_pos, f"{name}-pre2011", args.cost_bps)
            post = segment_stats(post_bars, post_pos, f"{name}-post2011", args.cost_bps)

            xc_bars, xc_pos = slice_bars_and_position(bars, pos, start=CROSSCHECK_START)
            crosscheck = segment_stats(xc_bars, xc_pos, f"{name}-crosscheck-2021-2026", args.cost_bps)

            print(f"\n  -- {name} --")
            print(f"    FULL       ({full.get('n_bars')} bars) Sharpe {full.get('sharpe', float('nan')):.4f}  "
                  f"t {full.get('tstat', float('nan')):.4f}  p {full.get('pval', float('nan')):.4f}  "
                  f"exposure {full.get('exposure', float('nan'))*100:.2f}%  maxDD {full.get('maxdd', float('nan'))*100:.2f}%")
            print(f"    PRE-2011   ({pre.get('n_bars')} bars) Sharpe {pre.get('sharpe', float('nan')):.4f}  "
                  f"t {pre.get('tstat', float('nan')):.4f}  p {pre.get('pval', float('nan')):.4f}  "
                  f"exposure {pre.get('exposure', float('nan'))*100:.2f}%  maxDD {pre.get('maxdd', float('nan'))*100:.2f}%")
            print(f"    POST-2011  ({post.get('n_bars')} bars) Sharpe {post.get('sharpe', float('nan')):.4f}  "
                  f"t {post.get('tstat', float('nan')):.4f}  p {post.get('pval', float('nan')):.4f}  "
                  f"exposure {post.get('exposure', float('nan'))*100:.2f}%  maxDD {post.get('maxdd', float('nan'))*100:.2f}%")
            print(f"    CROSSCHECK (2021-2026 window, {crosscheck.get('n_bars')} bars) "
                  f"Sharpe {crosscheck.get('sharpe', float('nan')):.4f}  "
                  f"t {crosscheck.get('tstat', float('nan')):.4f}  p {crosscheck.get('pval', float('nan')):.4f}  "
                  f"exposure {crosscheck.get('exposure', float('nan'))*100:.2f}%  maxDD {crosscheck.get('maxdd', float('nan'))*100:.2f}%")

            legs[name] = {"full": full, "pre_2011": pre, "post_2011": post, "crosscheck_2021_2026": crosscheck}

        out[sym] = {
            "first_date": first_date, "last_date": last_date, "n_bars": len(bars),
            "buy_hold_sharpe_full": bh_full.sharpe,
            "split_date": SPLIT_DATE,
            "legs": legs,
        }

    print("\nNOTE: cross-check the CROSSCHECK-2021-2026 numbers above against a fresh run of "
          "scripts/test_seasonality.py --cache \"/Volumes/My Passport/AlpcaData/cache\" — "
          "they test overlapping calendar windows but are NOT identical methodology (this "
          "script uses a calendar-date split with no train/test leakage; the existing script "
          "uses a 70/30 fractional IS/OOS split of the whole 5yr window), so expect the same "
          "ballpark, not exact equality.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[done] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
