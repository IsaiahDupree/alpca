"""
GAUNTLET — out-of-universe / fresh-symbol holdout for the MID-CAP cointegrated-pairs mechanism.

The ONE validated edge is a cointegrated-pairs market-neutral basket on LARGE-caps (WF ~0.83,
survivorship-stamped). Question: does the SAME mechanism (mean-reversion of cointegrated price
residuals) generalize to a DISJOINT MID-CAP universe, survive the delisting-aware PIT bar, AND is
its return stream uncorrelated with the large-cap book (=> genuine 2nd leg)?

Pairs form WITHIN a universe, so the honest out-of-universe analog is a DISJOINT-SYMBOL split:
  - Split the FULL mid-cap PIT universe (304 survivors + 75 delistings) into two DISJOINT halves
    by symbol via a deterministic md5(symbol) parity hash (independent of cap/sector/alphabet).
    Each half keeps its OWN survivors + its OWN delistings -> each half is itself a valid PIT
    universe, so the delisting-aware machinery runs unchanged within each half.
  - Build the pairs book SEPARATELY within each half (delisting_aware_walkforward, validated
    cadence). Pairs are only ever screened among names in that half — NEVER across halves.
  - selection_sharpe = PIT Sharpe of the book built among half A.
  - holdout_sharpe   = PIT Sharpe of the book built among the DISJOINT half B.
    If the mechanism is REAL it must be POSITIVE in BOTH halves (a mechanism, not a fit to names).
    B << A or B negative => the edge was specific to the names in A => overfit_to_selection.

Also computes, on the FULL PIT book (data/pairs_wf_returns_midcap.json):
  - per_year PIT Sharpe (regime stability; 2022 = large-cap pairs' worst year, -0.9)
  - DSR with an honest n_trials reflecting the cadence search (~tens of configs)
  - correlation vs the large-cap PIT book (data/pairs_wf_returns.json)

Validated cadence (do NOT deviate): train=252, test=63, top_n=10, max_adf=-2.86, entry_z=2.0,
exit_z=0.5, cost_bps=2.0 (library defaults).

Writes data/gauntlet_pairs_midcap_disjoint.json. Run in BACKGROUND (O(n^2) pairs, ~minutes):
  .venv/bin/python scripts/_gauntlet_pairs_midcap_disjoint.py
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.pairs import delisting_aware_walkforward  # noqa: E402
from alpca.backtest.evaluation import (  # noqa: E402
    sharpe_of, max_drawdown_of, deflated_sharpe_ratio,
)

SURV = "/Volumes/My Passport/AlpcaData/cache_midcap_sip"
DELS = "/Volumes/My Passport/AlpcaData/cache_midcap_pit_delisted"
LARGECAP_BOOK = "data/pairs_wf_returns.json"
OUT = "data/gauntlet_pairs_midcap_disjoint.json"
PPY = 252.0
CADENCE = dict(train=252, test=63, top_n=10, max_adf=-2.86,
               entry_z=2.0, exit_z=0.5, cost_bps=2.0)


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def _half(sym: str) -> int:
    """Deterministic disjoint split: parity of the md5 hash of the symbol name. Independent of
    cap rank, sector, and alphabetical order — a fresh-symbol holdout, not a thematic one."""
    return int(hashlib.md5(sym.encode()).hexdigest(), 16) & 1


def _eq(rets):
    eq = [1.0]
    for r in rets:
        eq.append(eq[-1] * (1.0 + r))
    return eq


def _sharpe(rets):
    return sharpe_of(_eq(rets), PPY) if len(rets) >= 4 else 0.0


def _run(bars, del_names, label):
    print(f"[gauntlet] {label}: walk-forward on {len(bars)} names "
          f"({len(del_names)} delisted) ...", flush=True)
    t0 = time.time()
    r = delisting_aware_walkforward(bars, delisted_syms=del_names or None, **CADENCE)
    print(f"[gauntlet] {label}: Sharpe {r.sharpe:.4f} · DD {r.max_drawdown:.4f} · "
          f"{r.n_windows} windows · {len(r.daily_returns)} OOS days · {time.time()-t0:.0f}s "
          f"· delisted-traded={len(r.delisted_names_traded)}", flush=True)
    return r


def _per_year(dates, rets):
    by_year = defaultdict(list)
    for t, ret in zip(dates, rets):
        y = time.strftime("%Y", time.gmtime(int(t)))
        by_year[y].append(ret)
    out = {}
    for y in sorted(by_year):
        out[y] = round(_sharpe(by_year[y]), 3) if len(by_year[y]) >= 4 else None
    return out


def _corr(a_dates, a_rets, b_path):
    """Correlation of the mid-cap PIT book with the large-cap PIT book on shared dates."""
    try:
        bk = json.loads(Path(b_path).read_text())
    except Exception as e:  # noqa: BLE001
        return None, 0, str(e)
    bmap = {row["asof"]: row["ret"] for row in bk["returns"]}
    amap = {int(t): r for t, r in zip(a_dates, a_rets)}
    shared = sorted(set(amap) & set(bmap))
    if len(shared) < 30:
        return None, len(shared), "too few shared days"
    xa = [amap[t] for t in shared]
    xb = [bmap[t] for t in shared]
    sa, sb = statistics.pstdev(xa), statistics.pstdev(xb)
    if sa <= 0 or sb <= 0:
        return None, len(shared), "zero variance"
    ma, mb = statistics.fmean(xa), statistics.fmean(xb)
    cov = sum((xa[i] - ma) * (xb[i] - mb) for i in range(len(shared))) / len(shared)
    return cov / (sa * sb), len(shared), None


def main() -> int:
    surv = _load(SURV)
    dels = _load(DELS)
    del_all = set(dels)
    union = {**surv, **dels}
    print(f"[gauntlet] full mid-cap PIT universe: {len(union)} names "
          f"({len(surv)} survivors + {len(dels)} delistings)", flush=True)

    # --- deterministic disjoint split by symbol (md5 parity) ---------------------------------------
    halfA = {s: b for s, b in union.items() if _half(s) == 0}
    halfB = {s: b for s, b in union.items() if _half(s) == 1}
    delA = {s for s in del_all if _half(s) == 0}
    delB = {s for s in del_all if _half(s) == 1}
    assert set(halfA).isdisjoint(set(halfB)), "halves overlap!"
    print(f"[gauntlet] split: half A = {len(halfA)} names ({len(delA)} delisted), "
          f"half B = {len(halfB)} names ({len(delB)} delisted); "
          f"disjoint={set(halfA).isdisjoint(set(halfB))}", flush=True)

    # --- FULL PIT book (regime/DSR/correlation reference) ------------------------------------------
    rp = _run(union, del_all, "FULL-PIT")
    per_year = _per_year(rp.dates, rp.daily_returns)
    # honest n_trials: cadence search ~ tens of configs (train{252,315,378} x top_n{6,10,15} x
    # max_adf grid x entry/exit) -> use 30 as an honest mid estimate of distinct configs tried.
    n_trials = 30
    full_eq = _eq(rp.daily_returns)
    dsr = deflated_sharpe_ratio(full_eq, n_trials=n_trials, sharpe_variance=1.0 / max(1, len(rp.daily_returns)))
    corr, n_shared, corr_err = _corr(rp.dates, rp.daily_returns, LARGECAP_BOOK)

    # --- DISJOINT halves ---------------------------------------------------------------------------
    rA = _run(halfA, delA, "HALF-A(selection)")
    rB = _run(halfB, delB, "HALF-B(holdout)")

    out = {
        "cadence": CADENCE,
        "split_method": "deterministic md5(symbol) parity (bit 0). Survivors AND delistings hashed "
                        "identically; each half keeps its own survivors + its own delistings, so "
                        "each half is itself a valid PIT universe. Pairs screened only within a half.",
        "universe": {"survivors": len(surv), "delistings": len(dels), "total": len(union),
                     "halfA_names": len(halfA), "halfA_delisted": len(delA),
                     "halfB_names": len(halfB), "halfB_delisted": len(delB)},
        "full_pit": {"sharpe": round(rp.sharpe, 4), "max_drawdown": round(rp.max_drawdown, 6),
                     "n_windows": rp.n_windows, "n_days": len(rp.daily_returns),
                     "delisted_traded": len(rp.delisted_names_traded)},
        "per_year": per_year,
        "dsr": round(dsr, 4), "n_trials": n_trials,
        "corr_vs_largecap": (None if corr is None else round(corr, 4)),
        "corr_n_shared_days": n_shared, "corr_err": corr_err,
        "selection_sharpe": round(rA.sharpe, 4),
        "selection_max_drawdown": round(rA.max_drawdown, 6),
        "selection_n_days": len(rA.daily_returns),
        "selection_per_year": _per_year(rA.dates, rA.daily_returns),
        "holdout_sharpe": round(rB.sharpe, 4),
        "holdout_max_drawdown": round(rB.max_drawdown, 6),
        "holdout_n_days": len(rB.daily_returns),
        "holdout_per_year": _per_year(rB.dates, rB.daily_returns),
        "holdout_positive": bool(rB.sharpe > 0),
    }

    # verdict
    pos_years = sum(1 for v in per_year.values() if v is not None and v > 0)
    tot_years = sum(1 for v in per_year.values() if v is not None)
    majority_pos = pos_years > tot_years / 2.0
    both_pos = rA.sharpe > 0 and rB.sharpe > 0
    if not majority_pos:
        verdict = "regime_unstable"
    elif both_pos and rB.sharpe > 0:
        verdict = "generalizes"
    else:
        verdict = "overfit_to_selection"
    out["pos_years"] = pos_years
    out["tot_years"] = tot_years
    out["majority_positive_years"] = majority_pos
    out["verdict"] = verdict

    Path(OUT).write_text(json.dumps(out, indent=2))
    print(f"[gauntlet] === selection(A) {rA.sharpe:.4f} | holdout(B) {rB.sharpe:.4f} | "
          f"full-PIT {rp.sharpe:.4f} | DSR {dsr:.4f} | corr-vs-largecap {corr} | "
          f"pos-years {pos_years}/{tot_years} -> {verdict.upper()} ===", flush=True)
    print(f"[gauntlet] wrote {OUT}", flush=True)
    print("[gauntlet] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
