"""
GAUNTLET: cadence-robustness map of the mid-cap cointegrated-pairs PIT edge.

Mirrors scripts/cache_pairs_wf_midcap.py universe loading EXACTLY (survivor SIP cache UNION
PIT-delisted cache with delisted_syms passed). Re-runs delisting_aware_walkforward varying ONE knob
at a time off the VALIDATED cadence (train=252, test=63, top_n=10, max_adf=-2.86, entry_z=2.0,
exit_z=0.5, cost_bps=2.0):

  - train in {189, 252, 315, 378}
  - top_n in {6, 10, 15}
  - max_adf = -2.86 (screen ON) vs None (screen OFF)

The large-cap edge is ROBUST to top_n but FRAGILE to train (252 robust, 315->0.30, 378 INVERTS to
-1.42). Question: is mid-cap PIT in a robust basin (positive across the neighborhood) or a knife-edge
lucky point (positive ONLY at exactly 252/top10)?

Also computes correlation of the BASE-POINT (252/top10/adf-on) PIT return stream vs the cached
large-cap book (data/pairs_wf_returns.json) on the date-intersection, to assess 2nd-leg eligibility.

Writes data/midcap_pairs_cadence_gauntlet.json incrementally (after EACH run) so polling sees progress.

Run: .venv/bin/python scripts/_midcap_pairs_cadence_gauntlet.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alpca.backtest.pairs import delisting_aware_walkforward  # noqa: E402

SURV = "/Volumes/My Passport/AlpcaData/cache_midcap_sip"
DELS = "/Volumes/My Passport/AlpcaData/cache_midcap_pit_delisted"
LARGECAP_BOOK = "data/pairs_wf_returns.json"
OUT = "data/midcap_pairs_cadence_gauntlet.json"


def _load(c):
    return {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in Path(c).glob("*_1day_bars.jsonl")}


def _corr(a_rows, b_rows):
    """Pearson corr of two {date->ret} return streams on the date intersection."""
    a = {r["date"]: r["ret"] for r in a_rows}
    b = {r["date"]: r["ret"] for r in b_rows}
    common = sorted(set(a) & set(b))
    if len(common) < 30:
        return None, len(common)
    xs = [a[d] for d in common]
    ys = [b[d] for d in common]
    n = len(common)
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None, n
    return cov / (vx ** 0.5 * vy ** 0.5), n


def main() -> int:
    t_all = time.time()
    # ---- Universe (mirror cache_pairs_wf_midcap.py) ------------------------------------------------
    surv = _load(SURV)
    dels = _load(DELS)
    del_names = set(dels)
    union = {**surv, **dels}
    print(f"[gauntlet] universe: {len(surv)} survivors + {len(dels)} delistings = {len(union)} names",
          flush=True)

    book = json.loads(Path(LARGECAP_BOOK).read_text())["returns"]

    # ---- Sweep plan: one knob at a time off the validated base ------------------------------------
    BASE = {"train": 252, "test": 63, "top_n": 10, "max_adf": -2.86}
    runs = []
    runs.append(("base", dict(BASE)))
    for tr in (189, 315, 378):
        runs.append((f"train{tr}", {**BASE, "train": tr}))
    for tn in (6, 15):
        runs.append((f"topn{tn}", {**BASE, "top_n": tn}))
    runs.append(("adf_off", {**BASE, "max_adf": None}))

    results = []
    out = {"status": "running", "universe": {"survivors": len(surv), "delistings": len(dels),
                                             "total": len(union)},
           "base_cadence": BASE, "runs": results}

    def flush_out(extra=None):
        if extra:
            out.update(extra)
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        Path(OUT).write_text(json.dumps(out, indent=2))

    flush_out()

    base_rows = None
    for label, cfg in runs:
        print(f"[gauntlet] >>> {label}: {cfg}", flush=True)
        t0 = time.time()
        r = delisting_aware_walkforward(
            union, delisted_syms=del_names,
            train=cfg["train"], test=cfg["test"], top_n=cfg["top_n"], max_adf=cfg["max_adf"],
            entry_z=2.0, exit_z=0.5, cost_bps=2.0,
        )
        dt = time.time() - t0
        rows = [{"date": time.strftime("%Y-%m-%d", time.gmtime(int(t))), "ret": ret}
                for t, ret in zip(r.dates, r.daily_returns)]
        rec = {"label": label, "config": cfg, "pit_sharpe": round(r.sharpe, 4),
               "max_drawdown": round(r.max_drawdown, 6), "n_windows": r.n_windows,
               "n_days": len(rows), "delisted_traded": len(r.delisted_names_traded),
               "secs": round(dt, 1)}
        if label == "base":
            base_rows = rows
            c, n = _corr(rows, book)
            rec["corr_vs_largecap"] = (round(c, 4) if c is not None else None)
            rec["corr_overlap_days"] = n
        results.append(rec)
        print(f"[gauntlet]     PIT Sharpe {r.sharpe:+.4f} · DD {r.max_drawdown:.4f} · "
              f"{r.n_windows}w · {len(rows)}d · {dt:.0f}s"
              + (f" · corr_vs_largecap {rec.get('corr_vs_largecap')}" if label == "base" else ""),
              flush=True)
        flush_out()

    # ---- Robustness verdict ----------------------------------------------------------------------
    by = {r["label"]: r["pit_sharpe"] for r in results}
    train_vals = {189: by.get("train189"), 252: by.get("base"), 315: by.get("train315"),
                  378: by.get("train378")}
    topn_vals = {6: by.get("topn6"), 10: by.get("base"), 15: by.get("topn15")}
    train_pos = [v for v in train_vals.values() if v is not None]
    topn_pos = [v for v in topn_vals.values() if v is not None]
    # knife-edge: positive ONLY at base, negative/<=0 everywhere else in the neighborhood
    neighborhood = [by.get(k) for k in ("train189", "train315", "topn6", "topn15") if by.get(k) is not None]
    knife_edge = (by.get("base", 0) > 0) and all((v is None or v <= 0) for v in neighborhood)
    robust = (sum(1 for v in train_pos if v > 0) >= 2) and (sum(1 for v in topn_pos if v > 0) >= 2)
    flush_out({"status": "done",
               "train_sweep": {str(k): v for k, v in train_vals.items()},
               "topn_sweep": {str(k): v for k, v in topn_vals.items()},
               "adf_on": by.get("base"), "adf_off": by.get("adf_off"),
               "knife_edge": knife_edge, "robust": robust})
    print("[gauntlet] TRAIN SWEEP:", {k: v for k, v in train_vals.items()}, flush=True)
    print("[gauntlet] TOPN SWEEP :", {k: v for k, v in topn_vals.items()}, flush=True)
    print(f"[gauntlet] ADF on {by.get('base')} | ADF off {by.get('adf_off')}", flush=True)
    print(f"[gauntlet] robust={robust} knife_edge={knife_edge} · total {time.time()-t_all:.0f}s",
          flush=True)
    print("[gauntlet] DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
