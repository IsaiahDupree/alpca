"""
Deploy the modest SECOND-EDGE candidate (mid-cap vol-managed momentum, borrow-free long/index-hedged
form) as a SHADOW FORWARD PAPER TRACK — the one test that settles whether the ~0.4 backtest survives
live, with ZERO survivorship or borrow-estimation bias (a forward track trades the universe as it
exists going forward, including names that later delist).

Each run: (1) mark the PRIOR logged book to today's prices -> realized return appended to a live OOS
curve; (2) if a rebalance is due (monthly), recompute the top-quintile momentum winners + SPY hedge,
else CARRY the prior winners forward (momentum is slow — hold between rebalances); (3) size small on
the HONEST Sharpe (~0.23, Case 45 — NOT the cherry-picked 1.35) and log. No broker orders.

HONESTY: this is a MODEST edge (long/index-hedge ~0.23, borrow-free; the L/S ~0.43 needs borrow on the
short leg). Sized tiny — a forward experiment on a marginal, uncorrelated-with-pairs (ρ=−0.03) sleeve,
let the live track speak. Run daily (needs cache_midcap_sip + SPY refreshed):
  .venv/bin/python scripts/deploy_momentum_paper.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.live.momentum_portfolio import compute_momentum_book, size_book  # noqa: E402
from alpca.backtest.evaluation import sharpe_of  # noqa: E402

PPY = 252.0
REBALANCE_DAYS = 21


def _load(cache: Path):
    out = {}
    for p in cache.glob("*_1day_bars.jsonl"):
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if rows:
            out[p.name.split("_1day_")[0]] = rows
    return out


def _last_close(bars, sym):
    rows = bars.get(sym)
    return float(rows[-1]["close"]) if rows else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache_midcap_sip")
    ap.add_argument("--spy-cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--track", default="data/momentum_forward_track.jsonl")
    ap.add_argument("--sleeve-sharpe", type=float, default=0.23, help="HONEST long/index-hedge Sharpe "
                    "(Case 45 representative PIT — NOT the cherry-picked 1.35)")
    ap.add_argument("--ann-vol", type=float, default=0.12)
    ap.add_argument("--target-vol", type=float, default=0.04)
    ap.add_argument("--kelly", type=float, default=0.5)
    ap.add_argument("--cap", type=float, default=0.5)
    args = ap.parse_args()
    cache, spy_cache, track = Path(args.cache), Path(args.spy_cache), Path(args.track)

    bars = _load(cache)
    spy_bars = _load(spy_cache).get("SPY", [])
    if not bars:
        print(f"[deploy] no mid-cap bars in {cache} — run the SIP refresh first."); return 1
    print(f"[deploy] mid-cap universe {len(bars)} · SPY {'ok' if spy_bars else 'MISSING'}. "
          f"Sizing on HONEST long/index-hedge Sharpe {args.sleeve_sharpe} (Case 45) — tiny forward track.\n")

    # all symbols whose latest close we can mark (mid-caps + SPY)
    markable = dict(bars)
    if spy_bars:
        markable["SPY"] = spy_bars

    # ---- 1. mark prior book to today's prices (realized OOS return) ----
    prior = None
    if track.exists():
        lines = [l for l in track.read_text().splitlines() if l.strip()]
        if lines:
            prior = json.loads(lines[-1])
    realized = None
    if prior and prior.get("sized"):
        r = 0.0
        for sym, w in prior["sized"].items():
            p0 = prior["prices"].get(sym)
            p1 = _last_close(markable, sym)
            if p0 and p1:
                r += w * (p1 / p0 - 1.0)
        realized = r
        print(f"[track] prior book ({prior['date']}) marked to today: realized return {r*100:+.3f}%")

    # ---- 2. rebalance due? monthly. else carry prior winners forward ----
    today_asof = max(int(rows[-1]["timestamp"]) for rows in bars.values())
    last_rebal = prior.get("rebalance_asof") if prior else None
    due = (prior is None) or (last_rebal is None) or \
          (today_asof - last_rebal >= REBALANCE_DAYS * 86400 - 43200)   # ~21 trading-ish days
    if due:
        book = compute_momentum_book(bars, spy_bars, top_frac=0.2, lookback=120, skip=21, vol_window=60)
        sized = size_book(book, sleeve_sharpe=args.sleeve_sharpe, ann_vol=args.ann_vol,
                          target_vol=args.target_vol, kelly_fraction=args.kelly, cap=args.cap)
        rebalance_asof = today_asof
        winners = sorted(book.longs)
        print(f"[REBALANCE] new top-quintile winners ({book.n_winners}) + SPY hedge {book.spy_weight:+.0f}")
    else:
        # carry the prior winners + hedge forward, unchanged weights
        sized = dict(prior["sized"])
        rebalance_asof = last_rebal
        winners = [s for s in sized if s != "SPY"]
        days_since = (today_asof - last_rebal) / 86400
        print(f"[HOLD] carrying {len(winners)} winners + SPY hedge forward ({days_since:.0f}d since rebalance)")

    gross = sum(abs(w) for w in sized.values())
    if sized:
        longs_str = ", ".join(f"{s} {w:+.3f}" for s, w in sorted(sized.items()) if s != "SPY")
        print(f"[book] gross {gross:.2f}× · SPY hedge {sized.get('SPY', 0):+.3f}")
        print(f"   winners: {longs_str}")
    else:
        print("   FLAT (insufficient data for momentum signal).")

    # ---- 3. append today's entry ----
    entry = {
        "date": time.strftime("%Y-%m-%d", time.gmtime(today_asof)),
        "asof": today_asof, "n_winners": len(winners),
        "sized": sized, "gross": gross, "rebalance_asof": rebalance_asof,
        "prices": {s: _last_close(markable, s) for s in sized},
        "sleeve_sharpe_ref": args.sleeve_sharpe, "realized_prev": realized,
    }
    with track.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    # ---- 4. report the live forward curve ----
    rets = [json.loads(l).get("realized_prev") for l in track.read_text().splitlines() if l.strip()]
    rets = [x for x in rets if isinstance(x, (int, float))]
    if len(rets) >= 2:
        eq = [1.0]
        for x in rets:
            eq.append(eq[-1] * (1 + x))
        print(f"\n[forward-track] {len(rets)} marked periods · cumulative {(eq[-1]-1)*100:+.2f}% · "
              f"live Sharpe {sharpe_of(eq, PPY):.2f} (the number that actually matters)")
    else:
        print(f"\n[forward-track] logged {entry['date']}; need ≥2 entries to mark a realized return. "
              f"Run daily to build the live OOS curve.")
    print(f"[done] appended to {track}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
