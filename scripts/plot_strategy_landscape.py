"""
Visualize the strategy landscape — the honest map for defining a high-profit-per-day bot.
Four panels:
  A. In-sample vs honest Sharpe — the overfit catch (now data-driven, all major candidates).
  B. Per-calendar-year Sharpe — the DEPLOYED book (stable) vs killed examples (regime-dependent).
  C. Profit-per-day vs Sharpe at half-Kelly, with the daily-noise band — the real ceiling.
  D. Takeaways — what actually defines a high-profit-per-day algo.

Data-driven: panel A reads data/edge_records.json; panels B & C read the live deployed numbers
from data/deployed_portfolio_backtest.json. Nothing hardcoded except the killed-example regime
rows (kept for contrast) and the narrative. Output: docs/strategy_landscape.png

Run: .venv/bin/python scripts/plot_strategy_landscape.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
GREEN, RED, GREY, BLUE, AMBER = "#1b9e77", "#d7301f", "#999999", "#2c7fb8", "#e6a817"

REC = json.loads((ROOT / "data" / "edge_records.json").read_text())["records"]
DEP = json.loads((ROOT / "data" / "deployed_portfolio_backtest.json").read_text())

# killed-example regime rows kept for contrast (committed per-year results)
YEARS = [2021, 2022, 2023, 2024, 2025, 2026]
KILLED_PERYEAR = {
    "EAR-PEAD (fresh 19)":   [0.64, 0.74, -0.61, -1.20, -0.41, -2.25],
    "Short-int (FINRA 6yr)": [-1.09, -2.38, 1.14, -1.12, 0.87, 0.28],
}


def panel_a(ax):
    recs = sorted(REC, key=lambda r: r["oos_sharpe"])
    n = len(recs)
    y = np.arange(n)
    h = 0.40
    for i, r in enumerate(recs):
        col = GREEN if r["survives"] else RED
        ax.barh(y[i] + h / 2, r["is_sharpe"], height=h, color=GREY, alpha=0.65)
        ax.barh(y[i] - h / 2, r["oos_sharpe"], height=h, color=col)
        ax.text(r["is_sharpe"] + (0.04 if r["is_sharpe"] >= 0 else -0.04), y[i] + h / 2,
                r["headline"], va="center", ha="left" if r["is_sharpe"] >= 0 else "right",
                fontsize=6.6, color="#555")
        ax.text(r["oos_sharpe"] + (0.04 if r["oos_sharpe"] >= 0 else -0.04), y[i] - h / 2,
                r["honest"], va="center", ha="left" if r["oos_sharpe"] >= 0 else "right",
                fontsize=6.6, color=col, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in recs], fontsize=7.2)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Sharpe ratio")
    ax.set_title("A · Headline looked great — honest tells the truth\n"
                 "grey = headline/IS · colored = honest OOS/fresh/post-cost (green survives, red collapses)",
                 fontsize=9.3, loc="left")
    ax.set_xlim(-2.6, 3.0)
    ax.grid(axis="x", alpha=0.25)


def panel_b(ax):
    dep_years = sorted(DEP["per_year"].keys())
    rows = ["DEPLOYED book"] + list(KILLED_PERYEAR)
    mat = [[DEP["per_year"][y] for y in dep_years]]
    for k in KILLED_PERYEAR:
        mat.append(KILLED_PERYEAR[k])
    mat = np.array(mat)
    cmap = LinearSegmentedColormap.from_list("rg", [RED, "#ffffbf", GREEN])
    im = ax.imshow(mat, cmap=cmap, vmin=-2.4, vmax=2.4, aspect="auto")
    ax.set_xticks(range(len(dep_years)))
    ax.set_xticklabels(dep_years, fontsize=8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=8.3)
    for i in range(len(rows)):
        for j in range(len(dep_years)):
            ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=7.5,
                    color="black")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), len(dep_years), 1, fill=False,
                               edgecolor=GREEN, lw=2.2))
    ax.set_title("B · Per-year Sharpe — deployed (stable) vs killed (regime-dependent)\n"
                 "the survivor is positive in 5/6 years; the rejects swing on one lucky window",
                 fontsize=9.3, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Sharpe")


def panel_c(ax):
    S = np.linspace(0.0, 1.6, 100)
    bps_half_kelly = 0.375 * S ** 2 / 252.0 * 1e4
    noise_bps = S / math.sqrt(252) * 1e4
    ax.plot(S, bps_half_kelly, color=BLUE, lw=2.2, label="expected edge (bps/day, ½-Kelly)")
    ax.plot(S, noise_bps, color=GREY, lw=1.6, ls="--", label="daily noise (1σ, bps/day)")
    ax.fill_between(S, 0, noise_bps, color=GREY, alpha=0.12)
    s0 = DEP["sharpe"]
    edge0 = 0.375 * s0 ** 2 / 252 * 1e4
    noise0 = s0 / math.sqrt(252) * 1e4
    ax.axvline(s0, color=GREEN, lw=1.3, ls=":")
    ax.scatter([s0], [edge0], color=GREEN, zorder=5)
    ax.annotate(f"deployed book (pairs+short-vol)\nSharpe {s0:.2f} → ~{edge0:.1f} bps/day\nvs ~{noise0:.0f} bps daily noise",
                xy=(s0, edge0), xytext=(min(s0 + 0.05, 1.0), 22),
                fontsize=7.8, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1))
    ax.set_xlabel("annualized Sharpe of the book")
    ax.set_ylabel("bps / day")
    ax.set_title("C · Profit-per-day reality — edge ≪ noise\n"
                 "profit/day = Sharpe² × sizing, NOT trade frequency (noise ≈ 15–25× the edge)",
                 fontsize=9.3, loc="left")
    ax.legend(fontsize=7.8, loc="upper left")
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 1.6)
    ax.set_ylim(0, 105)


def panel_d(ax):
    ax.axis("off")
    ax.set_title("D · What actually defines a high-profit-per-day algo", fontsize=9.3, loc="left")
    n_total = 62
    n_surv = sum(1 for r in REC if r["survives"] and r["kill"] == "deployed")
    lines = [
        ("1. A REAL edge that GENERALIZES.", "Must clear out-of-universe (new symbols) AND"),
        ("", "out-of-regime (new years), net of cost. IS Sharpe,"),
        ("", "DSR, even per-year stability can ALL pass an overfit"),
        ("", "edge — only the fresh-symbol + survivorship holdout"),
        ("", "catches it. (See panel A: nearly all collapse.)"),
        ("2. Then SIZE it (½-Kelly).", "Profit/day scales with Sharpe²×risk-budget."),
        ("", "More good uncorrelated legs > one leg harder."),
        ("3. NOT frequency.", "Every high-turnover sleeve (overnight, lead-lag,"),
        ("", "gap) died to the spread/borrow cost wall."),
        ("4. Diversify across legs.", "Combiner math is real but edge-supply-limited —"),
        ("", "short-vol is the first leg that actually LIFTED."),
        ("", ""),
        ("STATUS:", f"{n_total} documented cases → {n_surv} deployed sleeves"),
        ("", f"(pairs + short-vol). Book Sharpe {DEP['sharpe']:.2f}, DSR {DEP['dsr']:.2f},"),
        ("", f"maxDD {DEP['max_drawdown']*100:.0f}%. Honest ceiling ~1–2 bps/day — small but real."),
    ]
    y = 0.95
    for head, body in lines:
        if head:
            ax.text(0.0, y, head, fontsize=8.4, fontweight="bold", color="#222", transform=ax.transAxes)
        ax.text(0.32, y, body, fontsize=7.8, color="#333", transform=ax.transAxes)
        y -= 0.063


def main():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    panel_a(axes[0, 0]); panel_b(axes[0, 1]); panel_c(axes[1, 0]); panel_d(axes[1, 1])
    fig.suptitle("Alpca — Strategy Landscape & the Profit-Per-Day Reality  (62 cases, honest eval, data-driven)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = ROOT / "docs" / "strategy_landscape.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[done] wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
