"""
Case 61 — validate the MINED "conference-driven return drift" strategy (harvested from Quantpedia
via the research pipeline) through our honest battery.

Claim (Quantpedia, 2011-2025): long AAPL/GOOGL/MSFT from D-2 before to D+2 after their flagship
conferences (WWDC / Google I/O / MS Build) → Sharpe 0.97. Market-neutral variant: long stock /
short SPY in the window.

The decisive control: is it ALPHA or just BETA? Three mega-cap tech names ripped 2016-2026, so a
long-only "edge" that holds them a few days/year may simply be compressed tech beta. The beta-hedged
(long stock − short SPY) variant is the alpha test. Also: per-year stability, cost, and the
small-sample caveat (~3 conferences/yr × 3 names = very few independent bets).

Data: 4 tickers from Alpaca SIP (2016-2026), local data/cache_conf. Conference dates are publicly
SCHEDULED (announced months ahead) → no lookahead.

Run: .venv/bin/python scripts/validate_conference_drift.py
Writes: data/conference_drift_results.json
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alpca.backtest.evaluation import sharpe_of, max_drawdown_of  # noqa: E402

PPY = 252.0
CACHE = Path("data/cache_conf")
COST = 2.0 / 1e4

# Scheduled conference START dates (UTC), publicly known in advance.
CONFERENCES = {
    "AAPL": ["2016-06-13", "2017-06-05", "2018-06-04", "2019-06-03", "2020-06-22",
             "2021-06-07", "2022-06-06", "2023-06-05", "2024-06-10", "2025-06-09"],   # WWDC
    "GOOGL": ["2016-05-18", "2017-05-17", "2018-05-08", "2019-05-07", "2021-05-18",
              "2022-05-11", "2023-05-10", "2024-05-14", "2025-05-20"],                  # Google I/O (2020 cancelled)
    "MSFT": ["2016-03-30", "2017-05-10", "2018-05-07", "2019-05-06", "2020-05-19",
             "2021-05-25", "2022-05-24", "2023-05-23", "2024-05-21", "2025-05-19"],    # MS Build
}
PRE, POST = 2, 4          # hold from start-2 trading days to start+4 (covers a ~3-day conf + D+2)


def load(sym):
    bars = [json.loads(l) for l in (CACHE / f"{sym}_1day_bars.jsonl").open() if l.strip()]
    bars.sort(key=lambda b: int(b["timestamp"]))
    return bars


def main():
    syms = ["AAPL", "GOOGL", "MSFT", "SPY"]
    bars = {s: load(s) for s in syms}
    # common calendar
    common = sorted(set.intersection(*[set(int(b["timestamp"]) for b in bars[s]) for s in syms]))
    idx = {t: i for i, t in enumerate(common)}
    T = len(common)
    px = {s: [next(float(b["close"]) for b in bars[s] if int(b["timestamp"]) == t) for t in common]
          for s in syms}   # small N, fine
    ret = {s: [0.0] + [(px[s][i] - px[s][i - 1]) / px[s][i - 1] if px[s][i - 1] > 0 else 0.0
                       for i in range(1, T)] for s in syms}
    dstr = [time.strftime("%Y-%m-%d", time.gmtime(t)) for t in common]

    # build per-stock in-window position mask (no lookahead: scheduled dates)
    pos = {s: [0] * T for s in ("AAPL", "GOOGL", "MSFT")}
    n_events = 0
    for s, dates in CONFERENCES.items():
        for d in dates:
            # nearest trading-day index on/after the scheduled start
            k = next((i for i, ds in enumerate(dstr) if ds >= d), None)
            if k is None:
                continue
            n_events += 1
            for j in range(max(0, k - PRE), min(T, k + POST + 1)):
                pos[s][j] = 1

    def book(hedged: bool):
        daily, prev_w = [], {}
        days_in = 0
        for i in range(1, T):
            active = [s for s in ("AAPL", "GOOGL", "MSFT") if pos[s][i - 1] == 1]
            w = {}
            if active:
                days_in += 1
                for s in active:
                    w[s] = 1.0 / len(active)
                if hedged:
                    w["SPY"] = -1.0       # dollar-neutral: long basket, short index
            # gross return
            r = sum(w.get(s, 0.0) * ret[s][i] for s in syms)
            # turnover cost on weight changes
            allk = set(w) | set(prev_w)
            turn = sum(abs(w.get(s, 0.0) - prev_w.get(s, 0.0)) for s in allk)
            r -= turn * COST
            daily.append(r)
            prev_w = w
        eq = [1.0]
        for r in daily:
            eq.append(eq[-1] * (1 + r))
        return daily, eq, days_in

    def peryear(daily):
        by = defaultdict(list)
        for i, r in enumerate(daily):
            by[dstr[i + 1][:4]].append(r)
        out = {}
        for y in sorted(by):
            e = [1.0]
            for r in by[y]:
                e.append(e[-1] * (1 + r))
            out[y] = round(sharpe_of(e, PPY), 2)
        return out

    res = {}
    for name, hedged in (("long_only", False), ("beta_hedged", True)):
        daily, eq, days_in = book(hedged)
        res[name] = {"sharpe": round(sharpe_of(eq, PPY), 3),
                     "total_return": round(eq[-1] - 1, 4),
                     "max_drawdown": round(max_drawdown_of(eq), 4),
                     "exposure": round(days_in / (T - 1), 3),
                     "per_year": peryear(daily)}

    # control: buy-and-hold the 3 names, always invested (what beta gives)
    bh_daily = [sum(ret[s][i] for s in ("AAPL", "GOOGL", "MSFT")) / 3 for i in range(1, T)]
    bh_eq = [1.0]
    for r in bh_daily:
        bh_eq.append(bh_eq[-1] * (1 + r))
    bh_sharpe = round(sharpe_of(bh_eq, PPY), 3)

    lo, bh = res["long_only"]["sharpe"], res["beta_hedged"]["sharpe"]
    pos_years = sum(1 for v in res["beta_hedged"]["per_year"].values() if v > 0)
    n_years = len(res["beta_hedged"]["per_year"])
    is_alpha = bh > 0.3 and pos_years >= n_years - 2
    verdict = ("CANDIDATE — beta-hedged drift is positive and reasonably stable (small-sample caveat)"
               if is_alpha else
               "REJECT — the long-only 'edge' is compressed mega-cap tech BETA; the alpha (beta-hedged) leg "
               "does not clear the bar / is too thin a sample")

    print(f"events: {n_events}  ·  trading days {T}  ·  2016-2026 SIP")
    print(f"{'variant':<14}{'Sharpe':>8}{'ret':>9}{'maxDD':>8}{'expo':>7}")
    for k in ("long_only", "beta_hedged"):
        r = res[k]
        print(f"{k:<14}{r['sharpe']:>8.2f}{r['total_return']*100:>8.0f}%{r['max_drawdown']*100:>7.1f}%{r['exposure']*100:>6.0f}%")
    print(f"{'B&H 3 names':<14}{bh_sharpe:>8.2f}   (always-invested control = the beta)")
    print(f"\nbeta-hedged per-year: {res['beta_hedged']['per_year']}")
    print(f"\nVERDICT: {verdict}")

    out = {"case": 61, "name": "Conference-driven return drift (mined from Quantpedia)",
           "n_events": n_events, "long_only": res["long_only"], "beta_hedged": res["beta_hedged"],
           "buy_hold_3names_sharpe": bh_sharpe, "claimed": "Sharpe 0.97 (2011-2025, long-only)",
           "is_alpha": bool(is_alpha), "verdict": verdict, "window": f"D-{PRE}..D+{POST}",
           "source": "https://quantpedia.com/an-empirical-analysis-of-conference-driven-return-drift-in-tech-stocks/"}
    Path("data/conference_drift_results.json").write_text(json.dumps(out, indent=2))
    print("[meta] wrote data/conference_drift_results.json")


if __name__ == "__main__":
    main()
