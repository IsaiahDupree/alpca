"""
Visualize the strategy landscape — the honest map for defining a high-profit-per-day bot.
Four panels:
  A. In-sample vs out-of-sample Sharpe — the overfit catch (only pairs survives the drop).
  B. Per-calendar-year Sharpe heatmap — regime dependence (why one good year ≠ an edge).
  C. Profit-per-day vs Sharpe at half-Kelly, with the daily-noise band — the real ceiling.
  D. Takeaways — what actually defines a high-profit-per-day algo.

Numbers are the session's committed results (EDGE_CASE_STUDIES). Output: docs/strategy_landscape.png

Run: .venv/bin/python scripts/plot_strategy_landscape.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

GREEN, RED, GREY, BLUE = "#1b9e77", "#d7301f", "#999999", "#2c7fb8"

# ---- A: in-sample (headline) vs out-of-sample (truth) ----
# (name, IS label, IS Sharpe, OOS label, OOS Sharpe, survives?)
LADDER = [
    ("Pairs basket",       "IS 1.78",     1.78, "walk-fwd OOS 0.54",   0.54, True),
    ("EAR-PEAD",           "train 0.68",  0.68, "fresh holdout −0.52", -0.52, False),
    ("Short-int / borrow", "1yr 2.34",    2.34, "6yr net −0.42",       -0.42, False),
    ("Overnight reversal", "gross 0.93",  0.93, "@2bps −0.41",         -0.41, False),
    ("Lead-lag",           "gross 0.27",  0.27, "walk-fwd −1.02",      -1.02, False),
]

# ---- B: per-calendar-year Sharpe (regime dependence) ----
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
PERYEAR = {
    "EAR-PEAD (train 40)":  [1.80, 0.19, 0.44, 0.05, 0.81, 1.55],
    "EAR-PEAD (fresh 19)":  [0.64, 0.74, -0.61, -1.20, -0.41, -2.25],
    "Short-int (FINRA 6yr)":[-1.09, -2.38, 1.14, -1.12, 0.87, 0.28],
}


def panel_a(ax):
    n = len(LADDER)
    y = np.arange(n)[::-1]
    h = 0.38
    for i, (name, il, isv, ol, oos, surv) in enumerate(LADDER):
        yy = y[i]
        ax.barh(yy + h / 2, isv, height=h, color=GREY, alpha=0.7)
        ax.barh(yy - h / 2, oos, height=h, color=(GREEN if surv else RED))
        ax.text(isv + (0.05 if isv >= 0 else -0.05), yy + h / 2, il, va="center",
                ha="left" if isv >= 0 else "right", fontsize=7.5, color="#444")
        ax.text(oos + (0.05 if oos >= 0 else -0.05), yy - h / 2, ol, va="center",
                ha="left" if oos >= 0 else "right", fontsize=7.5,
                color=(GREEN if surv else RED), fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([x[0] for x in LADDER], fontsize=9)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Sharpe ratio")
    ax.set_title("A · In-sample looked great — out-of-sample tells the truth\n"
                 "grey = headline/IS · colored = honest OOS holdout (green survives, red collapses)",
                 fontsize=9.5, loc="left")
    ax.set_xlim(-3.0, 3.2)
    ax.grid(axis="x", alpha=0.25)


def panel_b(ax):
    rows = list(PERYEAR)
    mat = np.array([PERYEAR[r] for r in rows])
    cmap = LinearSegmentedColormap.from_list("rg", [RED, "#ffffbf", GREEN])
    im = ax.imshow(mat, cmap=cmap, vmin=-2.4, vmax=2.4, aspect="auto")
    ax.set_xticks(range(len(YEARS)))
    ax.set_xticklabels(YEARS, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8.5)
    for i in range(len(rows)):
        for j in range(len(YEARS)):
            ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=7.5,
                    color="black")
    ax.set_title("B · Per-calendar-year Sharpe — regimes matter\n"
                 "a strong single year (e.g. EAR-PEAD train 2021/2026) ≠ a durable edge",
                 fontsize=9.5, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Sharpe")


def panel_c(ax):
    S = np.linspace(0.0, 1.6, 100)
    bps_half_kelly = 0.375 * S ** 2 / 252.0 * 1e4          # half-Kelly geometric growth -> bps/day
    noise_bps = S / math.sqrt(252) * 1e4                    # daily vol at ~Sharpe-proportional vol target
    ax.plot(S, bps_half_kelly, color=BLUE, lw=2.2, label="expected edge (bps/day, ½-Kelly)")
    ax.plot(S, noise_bps, color=GREY, lw=1.6, ls="--", label="daily noise (1σ, bps/day)")
    ax.fill_between(S, 0, noise_bps, color=GREY, alpha=0.12)
    # mark the only validated edge
    s0 = 0.54
    ax.axvline(s0, color=GREEN, lw=1.3, ls=":")
    ax.scatter([s0], [0.375 * s0 ** 2 / 252 * 1e4], color=GREEN, zorder=5)
    ax.annotate("pairs basket\n(only validated edge)\n~1.5 bps/day vs ~34 bps noise",
                xy=(s0, 0.375 * s0 ** 2 / 252 * 1e4), xytext=(0.62, 18),
                fontsize=7.8, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    ax.set_xlabel("annualized Sharpe of the sleeve")
    ax.set_ylabel("bps / day")
    ax.set_title("C · Profit-per-day reality — edge ≪ noise\n"
                 "profit/day = Sharpe² × sizing, NOT trade frequency (noise ≈ 15–25× the edge)",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 105)


def panel_d(ax):
    ax.axis("off")
    ax.set_title("D · What actually defines a high-profit-per-day algo", fontsize=9.5, loc="left")
    lines = [
        ("1. A REAL edge that GENERALIZES.", "Must clear out-of-universe (new symbols) AND"),
        ("", "out-of-regime (new years) holdouts. In-sample"),
        ("", "Sharpe, DSR, even per-year stability can all"),
        ("", "pass on an overfit edge — 2 of 3 candidates this"),
        ("", "session did, then failed the clean test."),
        ("2. Then SIZE it (½-Kelly).", "Profit/day scales with Sharpe²×risk-budget."),
        ("", "More good uncorrelated legs > one leg harder."),
        ("3. NOT frequency.", "Every high-turnover sleeve (overnight, lead-lag,"),
        ("", "gap) died to the spread/borrow cost wall."),
        ("4. Diversify across legs.", "Combiner math is real but edge-supply-limited:"),
        ("", "needs several genuinely-uncorrelated edges."),
        ("", ""),
        ("STATUS:", "1 validated edge (pairs, OOS ~0.5). The honest"),
        ("", "ceiling today is ~1–2 bps/day — small but real."),
    ]
    y = 0.93
    for head, body in lines:
        if head:
            ax.text(0.0, y, head, fontsize=8.6, fontweight="bold", color="#222", transform=ax.transAxes)
        ax.text(0.30, y, body, fontsize=8.0, color="#333", transform=ax.transAxes)
        y -= 0.066


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    panel_a(axes[0, 0]); panel_b(axes[0, 1]); panel_c(axes[1, 0]); panel_d(axes[1, 1])
    fig.suptitle("Alpca — Strategy Landscape & the Profit-Per-Day Reality  (Session 31, honest eval)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = Path("docs/strategy_landscape.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {out} ({out.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
