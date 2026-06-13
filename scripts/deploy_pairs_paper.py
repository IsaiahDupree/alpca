"""
Deploy the ONE validated edge (cointegrated-pairs basket) as a SHADOW FORWARD PAPER TRACK.
Each run: (1) mark the PRIOR logged book to today's prices -> realized return, appended to a live
out-of-sample curve; (2) compute today's target book, size it conservatively, log it. No broker
orders — this is the gold-standard forward track (a live OOS record beats any backtest number),
risk-free, and it adjudicates whether the modest WF edge survives going forward.

HONESTY: the basket's honest walk-forward Sharpe is ~0.83 at the CONCENTRATED top-10 with a 5% ADF
screen (the earlier "0.29" was an over-diversified top-24 that diluted the edge into weak pairs). Its
static 60/40 OOS is negative, and on a flat start most pairs are inactive (thin/concentrated book).
So this is sized SMALL with a diversification guard — a forward experiment on a marginal edge, not a
high-conviction bet. Expectations: near-zero, let the live track speak.

Run daily:  .venv/bin/python scripts/deploy_pairs_paper.py --cache "/Volumes/My Passport/AlpcaData/cache"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.live.pairs_portfolio import compute_pairs_book, size_book  # noqa: E402
from alpca.backtest.evaluation import sharpe_of  # noqa: E402

PPY = 252.0


def _load_universe(cache: Path):
    syms = sorted(p.name.split("_1day_")[0] for p in cache.glob("*_1day_bars.jsonl"))
    bars = {}
    for s in syms:
        rows = [json.loads(l) for l in (cache / f"{s}_1day_bars.jsonl").open() if l.strip()]
        if rows:
            bars[s] = rows
    return bars


def _last_close(bars, sym):
    rows = bars.get(sym)
    return float(rows[-1]["close"]) if rows else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--track", default="data/pairs_forward_track.jsonl")
    ap.add_argument("--wf-sharpe", type=float, default=0.83, help="honest walk-forward Sharpe to size on "
                    "(top-10 concentrated + 5%% ADF screen; the 0.29 was an over-diversified top-24)")
    ap.add_argument("--ann-vol", type=float, default=0.048, help="WF-implied annual vol of the basket")
    ap.add_argument("--target-vol", type=float, default=0.05)
    ap.add_argument("--kelly", type=float, default=0.5)
    ap.add_argument("--target-pairs", type=int, default=6, help="diversification baseline for the guard")
    ap.add_argument("--cap", type=float, default=1.0)
    args = ap.parse_args()
    cache, track = Path(args.cache), Path(args.track)

    bars = _load_universe(cache)
    print(f"[deploy] universe {len(bars)} symbols. Sizing on WALK-FORWARD Sharpe {args.wf_sharpe} "
          f"(top-10 + 5%% ADF screen) — small forward paper-track, NOT a conviction bet.\n")

    # ---- 1. mark the most recent prior book to today's prices (realized OOS return) ----
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
            p1 = _last_close(bars, sym)
            if p0 and p1:
                r += w * (p1 / p0 - 1.0)
        realized = r
        print(f"[track] prior book ({prior['date']}) marked to today: realized return {r*100:+.3f}%")

    # ---- 2. compute + size today's book ----
    book = compute_pairs_book(bars, train=378, top_n=10, lookback=60, entry_z=2.0, exit_z=0.5,
                              max_half_life=30, min_half_life=3, max_adf=-2.86,   # validated config (WF 0.83)
                              prior_state=(prior.get("state") if prior else None))
    # diversification guard: a thin book (few active pairs) is sized down vs a full basket
    div = min(1.0, book.n_active / max(1, args.target_pairs))
    sized = size_book(book, basket_sharpe=args.wf_sharpe, ann_vol=args.ann_vol,
                      target_vol=args.target_vol, kelly_fraction=args.kelly, cap=args.cap)
    sized = {s: w * div for s, w in sized.items()}
    gross = sum(abs(w) for w in sized.values())

    print(f"[book] active pairs {book.n_active}/{len(book.targets)} · diversification guard ×{div:.2f} "
          f"· sized gross exposure {gross:.2f}×")
    for t in book.targets:
        if t.state:
            print(f"   {t.a:>5}/{t.b:<5} z={t.z:+.2f} state={t.state:+d} (half-life {t.half_life}d)")
    if sized:
        print("   target weights: " + ", ".join(f"{s} {w:+.3f}" for s, w in sorted(sized.items())))
    else:
        print("   FLAT today (no pair past its entry band).")

    # ---- 3. append today's entry to the forward track ----
    entry = {
        "date": time.strftime("%Y-%m-%d", time.gmtime(book.asof)) if book.asof else "?",
        "asof": book.asof, "n_active": book.n_active,
        "state": book.state, "sized": sized, "gross": gross,
        "prices": {s: _last_close(bars, s) for s in sized},
        "wf_sharpe_ref": args.wf_sharpe, "realized_prev": realized,
    }
    with track.open("a") as f:
        f.write(json.dumps(entry) + "\n")

    # ---- 4. accumulate + report the live forward curve ----
    rets = [json.loads(l)["realized_prev"] for l in track.read_text().splitlines() if l.strip()]
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
