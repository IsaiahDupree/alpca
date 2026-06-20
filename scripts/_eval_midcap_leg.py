"""
Gauntlet: is the mid-cap PIT pairs stream a genuine 2nd leg vs the large-cap book?

- Date-JOIN mid-cap PIT returns (data/pairs_wf_returns_midcap.json) with the large-cap book
  (data/pairs_wf_returns.json) on `asof` (epoch).
- Pearson rho on the overlapping window.
- evaluate_leg_candidate(midcap, largecap, book_label="largecap_pairs").
- Inverse-vol combiner lift = blended Sharpe - BETTER single sleeve, BOTH measured on the
  IDENTICAL overlapping date set (no in-sample/OOS window mismatch).

Writes nothing to library source. Pure read + report.
"""
import json
import statistics
import math

from alpca.backtest.leg_gate import evaluate_leg_candidate
from alpca.backtest.combine import (
    evaluate_combo, inverse_vol_weights, correlation as combine_corr,
)
from alpca.backtest.evaluation import sharpe_of

PPY = 252.0


def load_stream(path):
    d = json.load(open(path))
    return {int(r["asof"]): float(r["ret"]) for r in d["returns"]}, d


def pearson(a, b):
    n = len(a)
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    sa, sb = statistics.pstdev(a), statistics.pstdev(b)
    if sa <= 0 or sb <= 0:
        return 0.0
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / n
    return cov / (sa * sb)


def eq(daily, s=1.0):
    e = [s]
    for x in daily:
        e.append(e[-1] * (1 + x))
    return e


mid, mid_meta = load_stream("data/pairs_wf_returns_midcap.json")
big, big_meta = load_stream("data/pairs_wf_returns.json")

# ---- 1. date JOIN on asof ----
common = sorted(set(mid) & set(big))
overlap_days = len(common)
mv = [mid[t] for t in common]
bv = [big[t] for t in common]

# ---- 2. Pearson rho on overlap ----
rho = pearson(mv, bv)
rho_lib = combine_corr(mv, bv)  # sanity cross-check via library

# ---- single-sleeve Sharpe on the SAME overlap window (both legs) ----
mid_sh = sharpe_of(eq(mv), PPY)
big_sh = sharpe_of(eq(bv), PPY)
better_single = max(mid_sh, big_sh)

# ---- 3. leg gate ----
verdict = evaluate_leg_candidate(mid, big, book_label="largecap_pairs")

# ---- 4. inverse-vol combiner lift on identical window ----
streams = {"largecap_pairs": bv, "candidate": mv}
rep = evaluate_combo(streams, ppy=PPY)
blended_sh = rep.invvol_sharpe
# honest lift = blend minus the BETTER single sleeve, same window for both
lift_vs_better = blended_sh - better_single
lift_vs_book = blended_sh - big_sh  # what the gate uses (vs book = largecap)

# inverse-vol weights for transparency
ivw = inverse_vol_weights(streams)

# confirm both single-sleeve Sharpes were computed on the identical date set
lift_same_window = (len(mv) == len(bv) == overlap_days)

print(json.dumps({
    "overlap_days": overlap_days,
    "mid_n_days": mid_meta["n_days"],
    "big_n_days": big_meta["n_days"],
    "mid_wf_sharpe_full": mid_meta["wf_sharpe"],
    "big_wf_sharpe_full": big_meta["wf_sharpe"],
    "mid_delisted_traded": mid_meta.get("delisted_traded"),
    "mid_delisted_names": mid_meta.get("delisted_names_traded"),
    "mid_config": mid_meta["config"],
    "big_config": big_meta["config"],
    "rho_pearson": round(rho, 6),
    "rho_lib_crosscheck": round(rho_lib, 6),
    "mid_sharpe_overlap": round(mid_sh, 4),
    "big_sharpe_overlap": round(big_sh, 4),
    "better_single_sleeve": round(better_single, 4),
    "blended_invvol_sharpe": round(blended_sh, 4),
    "equalweight_sharpe": round(rep.equalweight_sharpe, 4),
    "lift_vs_better_single": round(lift_vs_better, 4),
    "lift_vs_book_largecap": round(lift_vs_book, 4),
    "invvol_weights": {k: round(v, 4) for k, v in ivw.items()},
    "lift_same_window": lift_same_window,
    "gate_passed": verdict.passed,
    "gate_checks": verdict.checks,
    "gate_candidate_sharpe": verdict.candidate_sharpe,
    "gate_book_sharpe": verdict.book_sharpe,
    "gate_rho": verdict.rho,
    "gate_combined_sharpe": verdict.combined_sharpe,
    "gate_lift": verdict.lift,
    "gate_loo_positive_frac": verdict.loo_positive_frac,
    "gate_ex_recent_lift": verdict.ex_recent_lift,
    "gate_reasons": verdict.reasons,
}, indent=2))
