"""
The holdout collapse — the entire overfit phenomenon in one chart.

X = headline Sharpe (in-sample / lucky-window / pre-cost).
Y = honest Sharpe (out-of-sample / fresh-symbol / multi-regime / post-cost).
The diagonal y = x is "if it generalized." Almost everything falls FAR below it,
into the red "collapsed" zone. Only the deployed sleeves stay in the green band.

Reads data/edge_records.json (the curated, auditable headline-vs-honest table).
Output: docs/holdout_collapse.png

Run: .venv/bin/python scripts/plot_holdout_collapse.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GREEN, RED, AMBER, GREY = "#1b9e77", "#d7301f", "#e6a817", "#999999"

KILL_COLOR = {
    "deployed": GREEN,
    "too_weak": AMBER,
    "fresh_holdout": RED,
    "cost_wall": "#b30000",
    "survivorship": "#7b3294",
    "overfit": "#762a83",
    "placebo": "#3690c0",
    "stat": "#3690c0",
    "signal_fail": "#cc4c02",
    "beta": "#80b1d3",
}
KILL_LABEL = {
    "deployed": "Deployed (survived)",
    "too_weak": "Too weak / dilutes",
    "fresh_holdout": "Fresh-symbol holdout",
    "cost_wall": "Cost / borrow wall",
    "survivorship": "Survivorship / PIT",
    "overfit": "Overfit (IS→OOS)",
    "placebo": "Shuffle placebo",
    "stat": "DSR / Bonferroni",
    "signal_fail": "Signal failure",
    "beta": "Beta, not alpha",
}


def main():
    data = json.loads((ROOT / "data" / "edge_records.json").read_text())
    recs = data["records"]

    fig, ax = plt.subplots(figsize=(11.5, 9))

    lo, hi = -2.0, 2.6
    # zones
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], color=GREEN, alpha=0.05)   # above-diagonal (impossible-ish)
    ax.axhspan(0, hi, xmin=0, xmax=1, color=GREEN, alpha=0.04)
    ax.axhspan(lo, 0, xmin=0, xmax=1, color=RED, alpha=0.05)
    ax.plot([lo, hi], [lo, hi], color=GREY, lw=1.4, ls="--", zorder=1,
            label="y = x  (if it generalized)")
    ax.axhline(0, color="#333", lw=1.0)
    ax.axvline(0, color="#333", lw=0.8)

    seen = set()
    for r in recs:
        c = KILL_COLOR.get(r["kill"], GREY)
        x, y = r["is_sharpe"], r["oos_sharpe"]
        lbl = KILL_LABEL.get(r["kill"]) if r["kill"] not in seen else None
        seen.add(r["kill"])
        marker = "*" if r["survives"] else "o"
        size = 360 if r["survives"] else 130
        ax.scatter(x, y, s=size, c=c, marker=marker, edgecolors="white",
                   linewidths=1.0, zorder=5, label=lbl)
        # draw the collapse as a downward arrow from diagonal to actual
        ax.plot([x, x], [x, y], color=c, lw=0.8, alpha=0.35, zorder=2)
        dy = 0.10 if y >= 0 else -0.16
        ax.annotate(r["name"], (x, y), xytext=(x + 0.04, y + dy),
                    fontsize=7.3, color="#222")

    # callout for the average collapse among the killed
    killed = [r for r in recs if not r["survives"]]
    avg_drop = np.mean([r["is_sharpe"] - r["oos_sharpe"] for r in killed])
    ax.text(0.02, 0.04,
            f"Among the {len(killed)} killed edges, the average headline→honest\n"
            f"collapse is {avg_drop:.2f} Sharpe — and {sum(1 for r in killed if r['oos_sharpe'] < 0)} "
            f"of {len(killed)} go outright NEGATIVE.",
            transform=ax.transAxes, fontsize=8.6, color=RED, style="italic",
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff5f5", ec=RED, lw=0.8))

    ax.text(0.98, 0.97, "★ = deployed sleeve\n● = rejected candidate",
            transform=ax.transAxes, fontsize=8.2, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f7f7f7", ec=GREY, lw=0.7))

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Headline Sharpe  (in-sample / lucky-window / pre-cost)", fontsize=10)
    ax.set_ylabel("Honest Sharpe  (out-of-sample / fresh-symbol / multi-regime / post-cost)",
                  fontsize=10)
    ax.set_title("Alpca — The Holdout Collapse\n"
                 "every documented edge plotted headline vs honest: the gap below the diagonal IS the overfit",
                 fontsize=12, fontweight="bold", loc="left")
    ax.legend(fontsize=7.8, loc="lower right", framealpha=0.9, ncol=1)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = ROOT / "docs" / "holdout_collapse.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {out} ({out.stat().st_size // 1024} KB) · {len(recs)} edges plotted")


if __name__ == "__main__":
    main()
