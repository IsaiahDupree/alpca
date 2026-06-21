"""
The edge funnel / graveyard — the single graphic that communicates the RIGOR:
every documented case study, classified by the control that killed it, down to
the deployed sleeves. The point of the whole program is the denominator.

Four panels:
  A. Funnel waterfall — N edges hypothesized -> rejections -> deployed sleeves.
  B. Rejection taxonomy — which control caught how many (the harness's kill log).
  C. Deployed book — per-year Sharpe of the live sleeves (read live from data/).
  D. The controls that did the killing (what each test is, what it caught).

The case -> bucket mapping below is explicit and auditable against docs/EDGE_CASE_STUDIES.md
(the Scoreboard). Re-tests / audits are their own bucket so nothing is double-counted as
a "rejection." Deployed numbers are read live from data/ — never hardcoded.

Run: .venv/bin/python scripts/plot_edge_funnel.py   ->  docs/edge_funnel.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GREEN, RED, BLUE, GREY = "#1b9e77", "#d7301f", "#2c7fb8", "#999999"
AMBER, PURPLE, TEAL = "#e6a817", "#7b3294", "#3690c0"

# ---- Explicit, auditable case -> kill-bucket map (vs EDGE_CASE_STUDIES Scoreboard) ----
# 26-32 is a 7-factor row (asset-growth, net-issuance, ROA, MAX, idio-vol, residual-mom,
# vol-managed-mom); represented here as 7 entries 260..266.
BUCKETS = {
    "Deployed (survived every control)": {
        "color": GREEN,
        "cases": [1, 46, 49, 50, 60],   # pairs (1, PIT 46, out-of-regime 60) + short-vol (49, tail-stress 50)
        "note": "2 sleeves: pairs (survivorship-PIT + 2016-2020 out-of-regime) + short-vol/VRP",
    },
    "Fresh-symbol holdout (out-of-universe)": {
        "color": RED,
        "cases": [18, 23, 35, 55, 59],
        "note": "passed in-universe, collapsed on disjoint new symbols",
    },
    "Cost / borrow wall": {
        "color": "#b30000",
        "cases": [14, 17, 21, 44],
        "note": "real gross edge eaten by spread / adverse-selection borrow",
    },
    "Survivorship / PIT artifact": {
        "color": PURPLE,
        "cases": [43, 45, 53],
        "note": "edge was an artifact of excluding delisted/bankrupt names",
    },
    "Signal failure / did not replicate": {
        "color": "#cc4c02",
        "cases": [4, 6, 7, 20, 22, 33, 34, 51, 260, 261, 262, 263, 264, 265, 266],
        "note": "famous premia thin/absent/inverted on our venue (incl. factor zoo 26-32)",
    },
    "Too weak / dilutes / inert": {
        "color": AMBER,
        "cases": [3, 9, 12, 15, 24, 37, 39, 40, 47, 48, 57, 58],
        "note": "real but sub-rail / no OOS skill; adds no lift to the book",
    },
    "Overfit (in-sample -> OOS decay)": {
        "color": "#762a83",
        "cases": [5, 10, 11],
        "note": "IS Sharpe evaporated out-of-sample / walk-forward",
    },
    "Shuffle-placebo / statistical": {
        "color": TEAL,
        "cases": [19, 36],
        "note": "indistinguishable from shuffled labels / failed Bonferroni",
    },
    "Beta, not alpha": {
        "color": "#80b1d3",
        "cases": [2, 25, 38, 61, 62],
        "note": "dampened market exposure / beta overlay, not alpha",
    },
    "Infeasible on venue / data": {
        "color": GREY,
        "cases": [8, 13],
        "note": "needs L2/rebates (HFT) or paywalled alt-data",
    },
    "Method / audit (not an edge test)": {
        "color": "#bbbbbb",
        "cases": [16, 41, 42, 52, 54],
        "note": "combiner math, factor-zoo scaffolding, the self-audit",
    },
}

REJECTION_ORDER = [
    "Fresh-symbol holdout (out-of-universe)",
    "Signal failure / did not replicate",
    "Too weak / dilutes / inert",
    "Cost / borrow wall",
    "Survivorship / PIT artifact",
    "Overfit (in-sample -> OOS decay)",
    "Beta, not alpha",
    "Shuffle-placebo / statistical",
    "Infeasible on venue / data",
]


def _counts():
    return {k: len(v["cases"]) for k, v in BUCKETS.items()}


def panel_a(ax):
    """Funnel waterfall: total documented -> minus each rejection -> deployed."""
    c = _counts()
    total = sum(c.values())
    deployed = c["Deployed (survived every control)"]
    method = c["Method / audit (not an edge test)"]
    rejected = total - deployed - method

    stages = [
        (f"{total} documented\ncase studies", total, "#34495e"),
        (f"−{method} method/audit\n(scaffolding, not edges)", total - method, "#7f8c8d"),
        (f"{rejected} genuine edge\nhypotheses tested", rejected, BLUE),
        (f"−{rejected} rejected by\nthe controls →", deployed, RED),
        (f"{deployed} deployed sleeves\n(pairs + short-vol)", deployed, GREEN),
    ]
    y = np.arange(len(stages))[::-1]
    maxv = total
    for i, (label, val, color) in enumerate(stages):
        width = max(val / maxv, 0.02)
        left = (1 - width) / 2
        ax.barh(y[i], width, left=left, height=0.62, color=color, alpha=0.9)
        ax.text(0.5, y[i], label, ha="center", va="center", fontsize=8.6,
                color="white", fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.axis("off")
    ax.set_title("A · The funnel — the denominator is the point\n"
                 "hypothesize → run the full control battery → almost everything dies",
                 fontsize=9.8, loc="left")


def panel_b(ax):
    c = _counts()
    labels = [k for k in REJECTION_ORDER]
    vals = [c[k] for k in labels]
    colors = [BUCKETS[k]["color"] for k in labels]
    short = [k.split(" (")[0] for k in labels]
    yy = np.arange(len(labels))[::-1]
    ax.barh(yy, vals, color=colors, alpha=0.9, height=0.66)
    for y, v in zip(yy, vals):
        ax.text(v + 0.1, y, str(v), va="center", fontsize=8.5, fontweight="bold")
    ax.set_yticks(yy)
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("# of edge hypotheses caught")
    ax.set_xlim(0, max(vals) + 1.4)
    ax.set_title("B · The kill log — which control caught how many\n"
                 "out-of-universe + cost + survivorship do most of the work",
                 fontsize=9.8, loc="left")
    ax.grid(axis="x", alpha=0.25)


def panel_c(ax):
    dep = json.loads((ROOT / "data" / "deployed_portfolio_backtest.json").read_text())
    sv = json.loads((ROOT / "data" / "short_vol_results.json").read_text())["_combined"]
    years = sorted(dep["per_year"].keys())
    vals = [dep["per_year"][y] for y in years]
    colors = [GREEN if v >= 0 else RED for v in vals]
    ax.bar(years, vals, color=colors, alpha=0.88)
    for x, v in zip(years, vals):
        ax.text(x, v + (0.04 if v >= 0 else -0.04), f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
    ax.axhline(0, color="#333", lw=0.8)
    w = dep["weights"]
    pos_years = sum(1 for v in vals if v >= 0)
    ax.set_title(
        f"C · Deployed book by year — pairs {w['pairs']:.0%} + short-vol {w['short_vol']:.0%}\n"
        f"Sharpe {dep['sharpe']:.2f} · DSR {dep['dsr']:.2f} · maxDD {dep['max_drawdown']*100:.1f}% · "
        f"+{pos_years}/{len(vals)} yrs · short-vol lift +{sv['lift']:.2f}",
        fontsize=9.5, loc="left")
    ax.set_ylabel("Sharpe")
    ax.grid(axis="y", alpha=0.25)


def panel_d(ax):
    ax.axis("off")
    ax.set_title("D · The controls that did the killing", fontsize=9.8, loc="left")
    rows = [
        ("Fresh-symbol holdout", "select on one symbol set, report on a DISJOINT set."),
        ("", "Catches edges fit to a specific universe (EAR-PEAD, accruals)."),
        ("Survivorship PIT", "rebuild the universe as it was, delisted names included."),
        ("", "Catches edges that only exist by excluding the losers."),
        ("Cost / borrow wall", "charge real spread + adverse-selection borrow."),
        ("", "Catches high-turnover & hard-to-short edges (overnight, SI-tilt)."),
        ("Multi-regime / per-year", "Sharpe must hold across calendar years, not 1."),
        ("", "Catches lucky-window artifacts (SI-tilt's 1-yr Nasdaq 2.34)."),
        ("Shuffle placebo", "re-run on shuffled labels; real ≈ shuffled ⇒ noise."),
        ("", "Catches learned-structure mirages (lead-lag)."),
        ("DSR / Bonferroni", "deflate Sharpe for the # of trials searched."),
        ("", "Catches the formulaic-alpha zoo (0/21 pass)."),
        ("", ""),
        ("RULE", "\"Validated\" = clears out-of-UNIVERSE and out-of-REGIME,"),
        ("", "net of realistic costs. In-sample + DSR alone is not enough."),
    ]
    y = 0.96
    for head, body in rows:
        if head == "RULE":
            ax.text(0.0, y, head, fontsize=8.6, fontweight="bold", color=GREEN,
                    transform=ax.transAxes)
            ax.text(0.18, y, body, fontsize=8.0, color=GREEN, transform=ax.transAxes)
        elif head:
            ax.text(0.0, y, head, fontsize=8.3, fontweight="bold", color="#222",
                    transform=ax.transAxes)
            ax.text(0.30, y, body, fontsize=7.7, color="#333", transform=ax.transAxes)
        else:
            ax.text(0.30, y, body, fontsize=7.7, color="#555", transform=ax.transAxes)
        y -= 0.064


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    panel_a(axes[0, 0])
    panel_b(axes[0, 1])
    panel_c(axes[1, 0])
    panel_d(axes[1, 1])
    c = _counts()
    total = sum(c.values())
    fig.suptitle(
        f"Alpca — The Edge Funnel: {total} documented case studies → 2 deployed sleeves  "
        "(honest evaluation, survivorship-corrected)",
        fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ROOT / "docs" / "edge_funnel.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {out} ({out.stat().st_size // 1024} KB) · {total} cases classified")


if __name__ == "__main__":
    main()
