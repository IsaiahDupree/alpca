"""
Mid-cap cointegrated-pairs basket — SURVIVOR vs DELISTING-AWARE PIT walk-forward.

Mirrors scripts/cache_pairs_wf.py but on a DISJOINT MID-CAP universe to test whether the
mean-reversion-of-cointegrated-residuals mechanism (the ONE validated large-cap edge, WF ~0.83,
survivorship-stamped +0.83->+0.94 in Case 46) GENERALIZES, and crucially whether it SURVIVES the
delisting-aware point-in-time bar — the same bar that killed mid-cap VALUE and mid-cap MOMENTUM as
survivorship artifacts.

Two runs on the VALIDATED cadence (train=252, test=63, top_n=10, max_adf=-2.86, entry_z=2.0,
exit_z=0.5, cost_bps=2.0 — all library defaults):
  1. SURVIVOR: delisting_aware_walkforward(survivor_bars)  [on survivor-only == walkforward_pairs]
  2. PIT:      delisting_aware_walkforward({**survivor, **delisted}, delisted_syms=delisted_names)

Writes:
  data/pairs_wf_returns_midcap.json           — PIT daily-return stream (Gauntlet input; mirrors
                                                 data/pairs_wf_returns.json format exactly)
  data/pairs_wf_returns_midcap_survivor.json  — survivor daily-return stream

THE DECISIVE TEST: delta = pit_sharpe - survivor_sharpe.
  delta < -0.3 or pit<=0  -> SURVIVORSHIP ARTIFACT (the edge was an illusion of dropping dead names).
  delta ~0 or positive    -> REAL (like large-cap pairs which went +0.83->+0.94).

Run: .venv/bin/python scripts/cache_pairs_wf_midcap.py
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


def _rows(r):
    return [{"date": time.strftime("%Y-%m-%d", time.gmtime(int(t))), "asof": int(t), "ret": ret}
            for t, ret in zip(r.dates, r.daily_returns)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survivor", default="/Volumes/My Passport/AlpcaData/cache_midcap_sip")
    ap.add_argument("--delisted", default="/Volumes/My Passport/AlpcaData/cache_midcap_pit_delisted")
    ap.add_argument("--out-pit", default="data/pairs_wf_returns_midcap.json")
    ap.add_argument("--out-survivor", default="data/pairs_wf_returns_midcap_survivor.json")
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--max-adf", type=float, default=-2.86)
    args = ap.parse_args()

    cfg = {"top_n": args.top_n, "max_adf": args.max_adf, "train": 252, "test": 63}

    # ---- 1. SURVIVOR run --------------------------------------------------------------------------
    surv = _load(args.survivor)
    print(f"[midcap] SURVIVOR walk-forward on {len(surv)} survivor mid-caps (validated cadence)...",
          flush=True)
    t0 = time.time()
    rs = delisting_aware_walkforward(surv, train=252, test=63, top_n=args.top_n, max_adf=args.max_adf)
    print(f"[midcap] SURVIVOR Sharpe {rs.sharpe:.4f} · DD {rs.max_drawdown:.4f} · "
          f"{rs.n_windows} windows · {len(rs.daily_returns)} OOS days · {time.time()-t0:.0f}s",
          flush=True)

    surv_rows = _rows(rs)
    surv_out = {"wf_sharpe": round(rs.sharpe, 4), "n_windows": rs.n_windows, "n_days": len(surv_rows),
                "max_drawdown": round(rs.max_drawdown, 6), "config": cfg, "returns": surv_rows}
    Path(args.out_survivor).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_survivor).write_text(json.dumps(surv_out, indent=2))
    print(f"[midcap] wrote {args.out_survivor}", flush=True)

    # ---- 2. PIT (delisting-aware) run -------------------------------------------------------------
    dels = _load(args.delisted)
    del_names = set(dels)
    union = {**surv, **dels}
    print(f"[midcap] PIT walk-forward on {len(union)} names "
          f"({len(surv)} survivors + {len(dels)} delistings)...", flush=True)
    t0 = time.time()
    rp = delisting_aware_walkforward(union, delisted_syms=del_names,
                                     train=252, test=63, top_n=args.top_n, max_adf=args.max_adf)
    print(f"[midcap] PIT Sharpe {rp.sharpe:.4f} · DD {rp.max_drawdown:.4f} · "
          f"{rp.n_windows} windows · {len(rp.daily_returns)} OOS days · {time.time()-t0:.0f}s",
          flush=True)
    print(f"[midcap] delisted names that ACTUALLY TRADED: {len(rp.delisted_names_traded)} "
          f"(leg-trades={rp.delisted_leg_trades}) -> {rp.delisted_names_traded[:20]}", flush=True)

    pit_rows = _rows(rp)
    pit_out = {"wf_sharpe": round(rp.sharpe, 4), "n_windows": rp.n_windows, "n_days": len(pit_rows),
               "max_drawdown": round(rp.max_drawdown, 6),
               "delisted_traded": len(rp.delisted_names_traded),
               "delisted_leg_trades": rp.delisted_leg_trades,
               "delisted_names_traded": rp.delisted_names_traded,
               "config": cfg, "returns": pit_rows}
    Path(args.out_pit).write_text(json.dumps(pit_out, indent=2))
    print(f"[midcap] wrote {args.out_pit}", flush=True)

    # ---- 3. Verdict -------------------------------------------------------------------------------
    delta = rp.sharpe - rs.sharpe
    if rp.sharpe > 0.2 and delta > -0.3:
        verdict = "real_survives_pit"
    elif rs.sharpe > 0.2:
        verdict = "survivorship_artifact"
    else:
        verdict = "dead_on_arrival"
    print(f"[midcap] === SURVIVOR {rs.sharpe:.4f} | PIT {rp.sharpe:.4f} | DELTA {delta:+.4f} "
          f"-> {verdict.upper()} ===", flush=True)
    print("[midcap] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
