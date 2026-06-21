"""
Case 60 — OUT-OF-REGIME test of the deployed pairs basket on 10.5 years of SIP history.

Everything in the program was tuned/measured on the 5yr window (2021-06 -> 2026-06). The SIP
feed (which the account has, though config defaults to iex) serves full daily history back to
2016 — so we can now run the EXACT deployed pairs config on 2016-2020, a regime the edge has
NEVER seen (the 2018 vol-spike and 2020 COVID crash included). This is a true out-of-regime
holdout for our one deployed equity edge.

Runs walkforward_pairs at the deployed config (train252/test63/top10/ADF<=-2.86/2bps) on the
10yr cache, then splits the OOS daily returns by calendar year and into:
  - OUT-OF-REGIME  2016-2020 (never in any tuning window)
  - IN-REGIME      2021-2026 (the window everything was measured on)

Verdict: the deployed edge HOLDS out-of-regime if 2016-2020 Sharpe is clearly positive and the
per-year record isn't carried by one lucky year.

Run: .venv/bin/python scripts/test_pairs_out_of_regime.py [--cache PATH]
Writes: data/pairs_out_of_regime.json
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# delisting_aware_walkforward (the deployed/validated path) uses the UNION calendar + per-window
# 80%-availability screening, so it handles the RAGGED 10yr panel (recent IPOs have no 2016 data);
# walkforward_pairs' strict timestamp-intersection collapses to 0 windows on a ragged panel.
from alpca.backtest.pairs import delisting_aware_walkforward  # noqa: E402
from alpca.backtest.evaluation import sharpe_of  # noqa: E402

PPY = 252.0


def sharpe_from_rets(rets):
    eq = [1.0]
    for r in rets:
        eq.append(eq[-1] * (1 + r))
    return sharpe_of(eq, PPY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/Volumes/My Passport/AlpcaData/cache_sip_10y")
    ap.add_argument("--out", default="data/pairs_out_of_regime.json")
    args = ap.parse_args()

    cache = Path(args.cache)
    bars = {p.name.split("_1day_")[0]: [json.loads(l) for l in p.open() if l.strip()]
            for p in cache.glob("*_1day_bars.jsonl")}
    spans = [int(b["timestamp"]) for v in bars.values() for b in v]
    print(f"[ok] {len(bars)} symbols · {time.strftime('%Y-%m', time.gmtime(min(spans)))} -> "
          f"{time.strftime('%Y-%m', time.gmtime(max(spans)))}")

    r = delisting_aware_walkforward(bars, train=252, test=63, top_n=10, max_adf=-2.86,
                                    entry_z=2.0, exit_z=0.5, cost_bps=2.0, periods_per_year=PPY)
    print(f"[ok] full-period WF Sharpe {r.sharpe:.3f} · {r.n_windows} windows · "
          f"{len(r.daily_returns)} OOS days")

    # per-calendar-year + regime split
    by_year = defaultdict(list)
    for ts, ret in zip(r.dates, r.daily_returns):
        by_year[time.strftime("%Y", time.gmtime(int(ts)))].append(ret)
    per_year = {y: round(sharpe_from_rets(by_year[y]), 3) for y in sorted(by_year)}

    oor = [ret for ts, ret in zip(r.dates, r.daily_returns)
           if int(time.strftime("%Y", time.gmtime(int(ts)))) <= 2020]
    inr = [ret for ts, ret in zip(r.dates, r.daily_returns)
           if int(time.strftime("%Y", time.gmtime(int(ts)))) >= 2021]
    oor_sh = round(sharpe_from_rets(oor), 3) if oor else None
    inr_sh = round(sharpe_from_rets(inr), 3) if inr else None

    print("\nper-year OOS Sharpe:")
    for y, s in per_year.items():
        tag = "  (out-of-regime)" if int(y) <= 2020 else "  (in-regime/tuning)"
        print(f"  {y}: {s:+.2f}{tag}")
    print(f"\nOUT-OF-REGIME 2016-2020 : Sharpe {oor_sh}  ({len(oor)} days)")
    print(f"IN-REGIME     2021-2026 : Sharpe {inr_sh}  ({len(inr)} days)")

    pos_oor_years = sum(1 for y, s in per_year.items() if int(y) <= 2020 and s > 0)
    n_oor_years = sum(1 for y in per_year if int(y) <= 2020)
    holds = (oor_sh is not None and oor_sh > 0.3 and pos_oor_years >= max(1, n_oor_years - 1))
    verdict = ("HOLDS OUT-OF-REGIME — deployed pairs edge persists on never-seen 2016-2020 "
               "(incl. 2018 vol-spike + 2020 COVID)"
               if holds else
               "WEAKER OUT-OF-REGIME — 2016-2020 does not clearly confirm the edge; investigate")

    print("\n" + "=" * 78)
    print(f"VERDICT: {verdict}")

    out = {"case": 60, "name": "Pairs basket out-of-regime (2016-2020) on 10.5yr SIP history",
           "n_symbols": len(bars), "full_sharpe": round(r.sharpe, 3), "n_windows": r.n_windows,
           "per_year": per_year, "out_of_regime_2016_2020": oor_sh, "in_regime_2021_2026": inr_sh,
           "pos_oor_years": pos_oor_years, "n_oor_years": n_oor_years,
           "holds_out_of_regime": bool(holds), "verdict": verdict}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[meta] wrote {args.out}")


if __name__ == "__main__":
    main()
