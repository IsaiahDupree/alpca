"""
Deployed-portfolio results graphic — the honest "what actually ships" picture.
Three panels:
  A. Deployed portfolio Sharpe by calendar year (regime stability — positive 5 of 6 years).
  B. 6-leg correlation heatmap (avg |corr| = 0.038 — the diversification that makes the combiner work).
  C. Honest return translation (Sharpe 0.99 -> ~9.7%/yr total, 16x daily noise-to-edge).

Numbers are read live from data/deployed_portfolio_backtest.json and data/combine_results.json.
Output: docs/deployed_results.png

Run: .venv/bin/python scripts/plot_deployed_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREEN, RED, BLUE, GREY = "#1b9e77", "#d7301f", "#2c7fb8", "#999999"
ROOT = Path(__file__).resolve().parent.parent

dep = json.loads((ROOT / "data" / "deployed_portfolio_backtest.json").read_text())
comb = json.loads((ROOT / "data" / "combine_results.json").read_text())


def panel_a(ax):
    years = sorted(dep["per_year"].keys())
    vals = [dep["per_year"][y] for y in years]
    colors = [GREEN if v >= 0 else RED for v in vals]
    ax.bar(years, vals, color=colors, alpha=0.85)
    for x, v in zip(years, vals):
        ax.text(x, v + (0.05 if v >= 0 else -0.05), f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=8, fontweight="bold")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_title(f"A. Deployed portfolio — Sharpe by year\n"
                 f"full-period Sharpe {dep['sharpe']:.2f}  |  DSR {dep['dsr']:.2f}  |  "
                 f"max DD {dep['max_drawdown']*100:.1f}%", fontsize=9.5)
    ax.set_ylabel("Sharpe")
    ax.grid(axis="y", alpha=0.25)


def panel_b(ax):
    names = comb["corr_names"]
    M = np.array(comb["corr_matrix"])
    short = [n.replace(" (beta)", "").replace(" (MN)", "") for n in names]
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticklabels(short, fontsize=7.5)
    for i in range(len(short)):
        for j in range(len(short)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5, color="#222" if abs(M[i, j]) < 0.6 else "white")
    ax.set_title(f"B. Leg correlation matrix\navg |corr| = {comb['avg_abs_corr']:.3f} "
                 f"(near-zero -> real diversification)", fontsize=9.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def panel_c(ax):
    t = comb["translation"]
    ax.axis("off")
    lines = [
        ("Combiner (inverse-vol, 6 legs)", "", "header"),
        ("Annualized Sharpe", f"{t['sharpe_annual']:.2f}", "good"),
        ("Annualized vol", f"{t['ann_vol']*100:.1f}%", "neutral"),
        ("Expected total return / yr", f"{t['expected_total_annual']*100:.1f}%", "good"),
        ("Expected excess / yr", f"{t['expected_excess_annual']*100:.1f}%", "neutral"),
        ("Daily noise-to-edge ratio", f"{t['noise_to_edge_ratio']:.1f}x", "warn"),
        ("", "", "spacer"),
        ("Honest read: modest, market-neutral,", "", "note"),
        ("uncorrelated to beta. Per-DAY targeting", "", "note"),
        ("is noise-mining (16x noise). Edge is", "", "note"),
        ("real but small — size with half-Kelly.", "", "note"),
    ]
    y = 0.95
    for label, val, kind in lines:
        if kind == "spacer":
            y -= 0.04
            continue
        if kind == "header":
            ax.text(0.0, y, label, fontsize=10, fontweight="bold", transform=ax.transAxes)
        elif kind == "note":
            ax.text(0.0, y, label, fontsize=8, style="italic", color="#555",
                    transform=ax.transAxes)
        else:
            color = {"good": GREEN, "warn": RED, "neutral": "#333"}[kind]
            ax.text(0.0, y, label, fontsize=8.5, transform=ax.transAxes)
            ax.text(1.0, y, val, fontsize=8.5, fontweight="bold", color=color,
                    ha="right", transform=ax.transAxes)
        y -= 0.085
    ax.set_title("C. Honest return translation", fontsize=9.5, loc="left")


fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
panel_a(axes[0])
panel_b(axes[1])
panel_c(axes[2])
fig.suptitle("Alpca — Deployed Portfolio Results (read live from data/)", fontsize=12,
             fontweight="bold", y=1.02)
fig.tight_layout()
out = ROOT / "docs" / "deployed_results.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"wrote {out}")
