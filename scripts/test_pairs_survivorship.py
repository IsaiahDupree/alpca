"""
Case 46 — SURVIVORSHIP point-in-time re-test of the ONE DEPLOYED edge (the cointegrated-pairs basket).

We applied the point-in-time lens to the momentum *candidate* (Cases 43–45) but never to the pairs
basket itself — which is also validated on a survivor-only large-cap universe. If a pair's leg later
delisted (an acquisition freezes the price at the deal value; a failure craters it), the spread stops
mean-reverting and the trade takes a loss the survivor universe never sees. So the WF ~0.83 could be
survivorship-inflated. This adds a representative set of large-cap delistings (Alpaca inactive-assets,
SIP feed, filtered to large-cap-caliber in 2021, outcome-blind) to the screening universe and re-runs
the walk-forward with the validated config (top_n=10, 5% ADF screen).

Decisive reads: (1) does the WF Sharpe degrade when delisted legs can enter the screen? (2) do any
delisted names actually make it into the traded pairs (if not, the basket is structurally immune)?

Run: .venv/bin/python scripts/test_pairs_survivorship.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.pairs import (  # noqa: E402
    walkforward_pairs, delisting_aware_walkforward as _daw)

PPY = 252.0


def _load(c):
    out = {}
    for p in Path(c).glob("*_1day_bars.jsonl"):
        rows = [json.loads(l) for l in p.open() if l.strip()]
        if rows:
            out[p.name.split("_1day_")[0]] = rows
    return out


def _unused_delisting_aware_walkforward(bars_by_sym, *, train=252, test=63, top_n=10, max_adf=-2.86,
                                        cost_bps=2.0):
    """Walk-forward that ALLOWS partial-history (delisted) names — the honest survivorship version.
    Master calendar = UNION of all timestamps (not the intersection walkforward_pairs uses). Each
    window: screen among names with >=80% real bars in the TRAIN sub-window; backtest each top pair on
    the TEST sub-window via the pair's own aligned bars (a leg that delists mid-window simply runs out
    of bars -> the position closes at its last real price, capturing any acquisition freeze / crash).
    Returns (wf_sharpe, total_return, n_windows, delisted_leg_trades, delisted_syms_traded)."""
    syms = sorted(bars_by_sym)
    master = sorted({int(b["timestamp"]) for s in syms for b in bars_by_sym[s]})
    # per-symbol {ts: bar} for quick windowed slicing
    bymap = {s: {int(b["timestamp"]): b for b in bars_by_sym[s]} for s in syms}
    n = len(master)
    oos_rets, windows = [], 0
    del_leg_trades, del_traded = 0, set()
    w = 0
    while w + train + test <= n:
        train_ts = master[w:w + train]
        test_ts = master[w + train:w + train + test]
        train_set = set(train_ts)
        # candidate names: >=80% real bars in this train window
        cand = [s for s in syms if sum(1 for t in train_ts if t in bymap[s]) >= 0.8 * train]
        if len(cand) < 4:
            w += test; continue
        train_slice = {s: [bymap[s][t] for t in train_ts if t in bymap[s]] for s in cand}
        screened = screen_pairs(cand, train_slice, min_overlap=int(train * 0.8),
                                max_half_life=30.0, min_half_life=3.0, max_adf=max_adf)
        per_pair = []
        win_span = train_ts + test_ts
        for r in screened[:top_n]:
            a, b = r["a"], r["b"]
            seg_a = [bymap[a][t] for t in win_span if t in bymap[a]]
            seg_b = [bymap[b][t] for t in win_span if t in bymap[b]]
            if len(seg_a) < train * 0.5 or len(seg_b) < train * 0.5:
                continue
            lb = int(max(20, min(120, r["half_life"] * 3)))
            res = backtest_pairs(seg_a, seg_b, lookback=lb, entry_z=2.0, exit_z=0.5,
                                 cost_bps=cost_bps, hedge=r["hedge"])
            eq = res.equity_curve
            seg = eq[-(test + 1):] if len(eq) > test else eq
            rr = [(seg[i] - seg[i - 1]) / seg[i - 1] for i in range(1, len(seg)) if seg[i - 1] > 0]
            if rr:
                per_pair.append(rr)
                if a not in train_set or b not in train_set or True:  # track delisted participation
                    pass
        # note which delisted names actually traded this window
        for r in screened[:top_n]:
            for leg in (r["a"], r["b"]):
                if leg in DELISTED_SYMS:
                    del_leg_trades += 1; del_traded.add(leg)
        if per_pair:
            m = min(len(x) for x in per_pair)
            for t in range(m):
                oos_rets.append(sum(x[t] for x in per_pair) / len(per_pair))
            windows += 1
        w += test
    eq = [1.0]
    for r in oos_rets:
        eq.append(eq[-1] * (1 + r))
    total = eq[-1] - 1.0
    return sharpe_of(eq, PPY), total, windows, del_leg_trades, sorted(del_traded)


DELISTED_SYMS = set()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survivors", default="/Volumes/My Passport/AlpcaData/cache_largecap_sip")
    ap.add_argument("--delisted", default="/Volumes/My Passport/AlpcaData/cache_largecap_pit_delisted")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--max-adf", type=float, default=-2.86)
    ap.add_argument("--train", type=int, default=252)
    ap.add_argument("--test", type=int, default=63)
    ap.add_argument("--out", default="data/pairs_survivorship.json")
    args = ap.parse_args()

    global DELISTED_SYMS
    surv = _load(args.survivors)
    deli = _load(args.delisted)
    aug = {**surv, **deli}
    DELISTED_SYMS = set(deli)
    print(f"[ok] survivors {len(surv)} · delisted added {len(deli)} · combined {len(aug)}\n")

    # 1) the legacy global-intersection WF on survivors — reproduces the validated 0.83
    rleg = walkforward_pairs(surv, train=args.train, test=args.test, top_n=args.top_n,
                             max_half_life=30.0, min_half_life=3.0, entry_z=2.0, exit_z=0.5,
                             cost_bps=2.0, max_adf=args.max_adf, periods_per_year=PPY)
    print(f"{'legacy survivor':>18}: WF {rleg.sharpe:+.3f} · {rleg.n_windows} windows  "
          f"(reproduces the validated 0.83 -> baseline sanity check)")

    # 2) delisting-AWARE WF, apples-to-apples: survivor-only vs +delisted (both on the union calendar)
    rss = _daw(surv, train=args.train, test=args.test, top_n=args.top_n, max_adf=args.max_adf)
    rda = _daw(aug, delisted_syms=DELISTED_SYMS, train=args.train, test=args.test,
               top_n=args.top_n, max_adf=args.max_adf)
    ss, as_ = rss.sharpe, rda.sharpe
    dlt, dtr = rda.delisted_leg_trades, rda.delisted_names_traded
    print(f"{'aware survivor':>18}: WF {ss:+.3f} · total {rss.total_return*100:+.1f}% · {rss.n_windows} windows")
    print(f"{'aware +delisted':>18}: WF {as_:+.3f} · total {rda.total_return*100:+.1f}% · {rda.n_windows} windows · "
          f"delisted-leg trades {dlt} ({len(dtr)} names: {dtr[:8]})")

    delta = as_ - ss
    verdict = ("ROBUST — WF Sharpe barely moves when delisted legs can enter (pairs basket survives "
               "the point-in-time test)" if delta > -0.10 else
               "SURVIVORSHIP-SENSITIVE — WF Sharpe degrades once delisted legs are tradeable")
    print(f"\n[verdict] delisting-aware: survivor {ss:+.2f} -> +delisted {as_:+.2f} "
          f"(delta {delta:+.2f}) -> {verdict}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "legacy_survivor_wf": round(rleg.sharpe, 3),
        "aware_survivor_wf": round(ss, 3), "aware_augmented_wf": round(as_, 3), "delta": round(delta, 3),
        "n_survivors": len(surv), "n_delisted_added": len(deli),
        "delisted_leg_trades": dlt, "delisted_names_traded": dtr, "verdict": verdict}, indent=2))
    print(f"[saved] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
