"""
OUT-OF-UNIVERSE / symbol-generalization test for EAR-PEAD — the overfit check we had not run.
Run the EXACT fixed a-priori params (thr 2.0, hold 40, trailing 126d hedge, beta_hedged) on symbol
sets the aggregate result was not tuned to. No re-tuning, ever.

Two modes (auto-selected):
  A. TRUE HOLDOUT — if a disjoint `earnings_av_holdout` set exists (fresh symbols from a later AV
     quota / the nightly job), run TRAIN vs HOLDOUT vs pooled.
  B. SUBSET GENERALIZATION (fallback, no new data) — partition the available symbols into disjoint
     halves, and resample many random ~half-size subsets. If a RANDOM subset usually works, the
     edge is not carried by a few cherry-picked names — a real (if weaker) symbol-overfit check.

Run: .venv/bin/python scripts/test_ear_pead_holdout.py
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.evaluation import sharpe_of  # noqa: E402
from alpca.backtest.ear_pead import backtest_ear_pead  # noqa: E402

PPY = 252.0
PARAMS = dict(hold=40, entry_thr=2.0, mode="beta_hedged", hedge_window=126, cost_bps=2.0, periods_per_year=PPY)


def _eq(daily, start=100_000.0):
    eq = [start]
    for x in daily:
        eq.append(eq[-1] * (1 + x))
    return eq


def load_set(cache, edir):
    bars_by, events_by = {}, {}
    p = Path(edir)
    if not p.exists():
        return bars_by, events_by
    for ef in p.glob("*_earnings.json"):
        sym = ef.name.replace("_earnings.json", "")
        bf = cache / f"{sym}_1day_bars.jsonl"
        ev = json.loads(ef.read_text())
        if ev and bf.exists():
            bars_by[sym] = [json.loads(l) for l in bf.open() if l.strip()]
            events_by[sym] = ev
    return bars_by, events_by


def run_on(bars_by, events_by, syms, bench):
    b = {s: bars_by[s] for s in syms}
    e = {s: events_by[s] for s in syms}
    return backtest_ear_pead(b, e, bench_bars=bench, **PARAMS)


def per_year(res):
    by = {}
    for ep, r in zip(res.dates, res.daily_returns):
        by.setdefault(time.gmtime(ep).tm_year, []).append(r)
    return {y: round(sharpe_of(_eq(rr), PPY), 2) for y, rr in by.items() if len(rr) >= 30}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache")
    ap.add_argument("--train", default="/Volumes/My Passport/AlpcaData/earnings_av")
    ap.add_argument("--holdout", default="/Volumes/My Passport/AlpcaData/earnings_av_holdout")
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()
    cache = Path(args.cache)
    bench = [json.loads(l) for l in (cache / "SPY_1day_bars.jsonl").open() if l.strip()]
    tr_bars, tr_ev = load_set(cache, args.train)
    ho_bars, ho_ev = load_set(cache, args.holdout)
    print(f"FROZEN a-priori params: {PARAMS}\n")

    if len(ho_ev) >= 8:
        # ---- MODE A: true fresh-symbol holdout ----
        overlap = set(tr_ev) & set(ho_ev)
        print(f"MODE A — TRUE HOLDOUT. TRAIN {len(tr_ev)}, HOLDOUT {len(ho_ev)}, overlap {len(overlap)} "
              f"({'DISJOINT' if not overlap else 'NOT DISJOINT'})\n")
        pooled_b, pooled_e = {**tr_bars, **ho_bars}, {**tr_ev, **ho_ev}
        print(f"{'set':>10}{'symbols':>9}{'events':>8}{'Sharpe':>8}{'beta':>7}")
        print("-" * 42)
        res = {}
        for nm, syms in (("TRAIN", list(tr_ev)), ("HOLDOUT", list(ho_ev)), ("pooled", list(pooled_e))):
            src_b = pooled_b if nm == "pooled" else (tr_bars if nm == "TRAIN" else ho_bars)
            src_e = pooled_e if nm == "pooled" else (tr_ev if nm == "TRAIN" else ho_ev)
            r = run_on(src_b, src_e, syms, bench)
            res[nm] = r
            print(f"{nm:>10}{len(syms):>9}{r.n_events_used:>8}{r.sharpe:>8.2f}{r.beta:>7.2f}")
        print(f"\nHOLDOUT per-year: {per_year(res['HOLDOUT'])}")
        tr, ho = res["TRAIN"].sharpe, res["HOLDOUT"].sharpe
        verdict = ("GENERALIZES (not overfit to the original set)" if ho > 0.3 and abs(tr - ho) < 0.45
                   else "PARTIAL" if ho > 0.2 else "FAILS — overfit")
        print(f"\nVERDICT: TRAIN {tr:.2f} vs HOLDOUT {ho:.2f} -> {verdict}")
        return 0

    # ---- MODE B: subset generalization on the available symbols (no new data) ----
    syms = sorted(tr_ev)
    print(f"MODE B — SUBSET GENERALIZATION (no fresh holdout yet; AV quota/nightly job will supply it).\n"
          f"  {len(syms)} symbols available.\n")
    rng = random.Random(12)

    # disjoint split-half by a neutral hash (name), fixed params on each independent half
    half = sorted(syms, key=lambda s: (hash(("ear", s)) & 0xffffffff))
    a, b = half[: len(half) // 2], half[len(half) // 2:]
    ra, rb = run_on(tr_bars, tr_ev, a, bench), run_on(tr_bars, tr_ev, b, bench)
    print(f"DISJOINT SPLIT-HALF (each half excludes the other's names):")
    print(f"  half A ({len(a)} syms): Sharpe {ra.sharpe:.2f}   per-year {per_year(ra)}")
    print(f"  half B ({len(b)} syms): Sharpe {rb.sharpe:.2f}   per-year {per_year(rb)}")
    print(f"  -> both positive: {'YES' if ra.sharpe > 0 and rb.sharpe > 0 else 'NO'}\n")

    # resample many random subsets -> distribution of hedged Sharpe
    k = max(8, len(syms) // 2)
    sh = []
    for _ in range(args.draws):
        sub = rng.sample(syms, k)
        sh.append(run_on(tr_bars, tr_ev, sub, bench).sharpe)
    sh.sort()
    pos = sum(1 for x in sh if x > 0) / len(sh)
    p10, p50, p90 = (sh[int(len(sh) * q)] for q in (0.1, 0.5, 0.9))
    print(f"RANDOM SUBSET RESAMPLING ({args.draws} draws of {k} symbols, fixed params):")
    print(f"  Sharpe distribution  p10 {p10:.2f} | median {p50:.2f} | p90 {p90:.2f}")
    print(f"  fraction of random subsets with POSITIVE hedged Sharpe: {pos*100:.0f}%")
    print(f"  -> {'ROBUST: a random subset usually works -> not carried by a few names' if pos > 0.8 and p50 > 0.3 else 'FRAGILE: depends on which names -> symbol-selection risk'}\n")

    print("=" * 60)
    print("NOTE: this is symbol-SUBSET generalization, not fresh symbols. The TRUE fresh-symbol "
          "holdout (MODE A) runs automatically once earnings_av_holdout fills (AV quota resets / the "
          "nightly avearnings job adds disjoint names).")
    Path("data/ear_pead_holdout_results.json").write_text(json.dumps(
        {"mode": "subset", "split_half": {"A": ra.sharpe, "B": rb.sharpe},
         "resample": {"draws": args.draws, "k": k, "p10": p10, "median": p50, "p90": p90, "frac_positive": pos}},
        indent=2))
    print("[done] wrote data/ear_pead_holdout_results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
