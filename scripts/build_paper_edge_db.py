"""
Continuously document the LIVE paper edge in a database.

Every forward-track run logs each sleeve's realized OOS return (`realized_prev`) to its jsonl ledger;
`report_forward_track.py` computes the edge but only PRINTS it. This persists it: it ingests all the
paper-strat ledgers into SQLite (data/paper_edge.db), computes per-sleeve + combined edge stats
(reusing the deployed-portfolio weights), and APPENDS a timestamped snapshot each run — so the edge is
documented as it bakes, not just viewed once. Also writes a committed summary (data/paper_edge_summary.json,
force-tracked) + a human-readable docs/PAPER_EDGE.md so the live edge is in git too.

Tables:
  observations(strategy, asof, date, realized_ret)            -- every realized mark (idempotent upsert)
  edge_snapshots(snapshot_ts, strategy, n_marks, cum_pct, sharpe, maxdd_pct, last_date, funded)  -- time series
  edge_current(strategy, ...latest...)                        -- newest per strategy

Run: .venv/bin/python scripts/build_paper_edge_db.py   (also invoked by forward_track.sh after each track)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from alpca.live.portfolio import DEPLOYED, deployed_weights, combine_tracks  # noqa: E402
from alpca.backtest.evaluation import sharpe_of, max_drawdown_of  # noqa: E402

PPY = 252.0
TRACKS = {"pairs": "data/pairs_forward_track.jsonl",
          "short_vol": "data/shortvol_forward_track.jsonl",
          "momentum": "data/momentum_forward_track.jsonl"}


def read_track(path: str):
    """asof -> realized_ret for every mark that carries a realized return; plus total mark count."""
    p = ROOT / path
    out, n_marks = {}, 0
    if not p.exists():
        return out, 0
    for line in p.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        n_marks += 1
        r = e.get("realized_prev")
        if isinstance(r, (int, float)) and e.get("asof"):
            out[int(e["asof"])] = r
    return out, n_marks


def curve_stats(daily):
    if len(daily) < 2:
        return None
    eq = [1.0]
    for x in daily:
        eq.append(eq[-1] * (1 + x))
    return {"cum_pct": (eq[-1] - 1) * 100, "sharpe": sharpe_of(eq, PPY),
            "maxdd_pct": max_drawdown_of(eq) * 100, "n": len(daily)}


def ensure_schema(con):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS observations(
      strategy TEXT, asof INTEGER, date TEXT, realized_ret REAL,
      PRIMARY KEY(strategy, asof));
    CREATE TABLE IF NOT EXISTS edge_snapshots(
      snapshot_ts INTEGER, strategy TEXT, n_marks INTEGER, n_realized INTEGER,
      cum_pct REAL, sharpe REAL, maxdd_pct REAL, last_date TEXT, funded INTEGER);
    CREATE TABLE IF NOT EXISTS edge_current(
      strategy TEXT PRIMARY KEY, n_marks INTEGER, n_realized INTEGER,
      cum_pct REAL, sharpe REAL, maxdd_pct REAL, last_date TEXT, funded INTEGER, updated_ts INTEGER);
    """)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "paper_edge.db"))
    ap.add_argument("--tracks", default=None)
    args = ap.parse_args()
    tracks = json.loads(args.tracks) if args.tracks else TRACKS
    now = int(time.time())

    w = deployed_weights()
    track_returns = {k: read_track(p)[0] for k, p in tracks.items()}
    marks = {k: read_track(p)[1] for k, p in tracks.items()}

    con = sqlite3.connect(args.db)
    ensure_schema(con)

    rows_summary = {}

    def record(strategy, ret_by_asof, n_marks_total, funded):
        asofs = sorted(ret_by_asof)
        for a in asofs:
            con.execute("INSERT OR REPLACE INTO observations(strategy,asof,date,realized_ret) VALUES(?,?,?,?)",
                        (strategy, a, time.strftime("%Y-%m-%d", time.gmtime(a)), float(ret_by_asof[a])))
        st = curve_stats([ret_by_asof[a] for a in asofs])
        last_date = time.strftime("%Y-%m-%d", time.gmtime(asofs[-1])) if asofs else None
        cum = round(st["cum_pct"], 4) if st else None
        shp = round(st["sharpe"], 4) if st else None
        dd = round(st["maxdd_pct"], 4) if st else None
        con.execute("INSERT INTO edge_snapshots VALUES(?,?,?,?,?,?,?,?,?)",
                    (now, strategy, n_marks_total, len(asofs), cum, shp, dd, last_date, int(funded)))
        con.execute("INSERT OR REPLACE INTO edge_current VALUES(?,?,?,?,?,?,?,?,?)",
                    (strategy, n_marks_total, len(asofs), cum, shp, dd, last_date, int(funded), now))
        rows_summary[strategy] = {"n_marks": n_marks_total, "n_realized": len(asofs),
                                  "cum_pct": cum, "sharpe": shp, "maxdd_pct": dd,
                                  "last_date": last_date, "funded": bool(funded)}

    for k in tracks:
        record(k, track_returns.get(k, {}), marks.get(k, 0), w.get(k, 0) > 0)

    # combined deployed book (funded sleeves at deployed weights)
    book = combine_tracks(track_returns)
    comb = {int(d): r for d, r in zip(book.dates, book.daily_returns)}
    record("_combined", comb, len(comb), True)

    con.commit()
    n_snap = con.execute("SELECT COUNT(*) FROM edge_snapshots").fetchone()[0]
    con.close()

    summary = {"updated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(now)),
               "funded_weights": {k: round(v, 3) for k, v in book.weights.items()},
               "snapshots_logged": n_snap, "edge": rows_summary}
    (ROOT / "data" / "paper_edge_summary.json").write_text(json.dumps(summary, indent=2))

    # human-readable doc (committed)
    lines = ["# Alpca — Live Paper Edge (auto-generated)", "",
             f"_Updated {summary['updated']} · {n_snap} snapshots logged in `data/paper_edge.db` · "
             "regenerate: `.venv/bin/python scripts/build_paper_edge_db.py`_", "",
             "The realized out-of-sample edge from the deployed paper strategies, documented continuously "
             "(every forward-track run appends a snapshot). `_combined` = the funded book at deployed weights "
             "— the number that adjudicates the program.", "",
             "| Sleeve | Funded | Realized marks | Cum % | Live Sharpe | Max DD % | Through |",
             "|---|---|---:|---:|---:|---:|---|"]
    order = ["_combined"] + [k for k in tracks]
    for k in order:
        r = rows_summary.get(k)
        if not r:
            continue
        nm = f"{r['n_realized']}/{r['n_marks']}"
        fmt = lambda v: ("—" if v is None else f"{v:+.2f}")
        lines.append(f"| {'**'+k+'**' if k=='_combined' else k} | {'✅' if r['funded'] else 'probation'} | "
                     f"{nm} | {fmt(r['cum_pct'])} | {fmt(r['sharpe'])} | {fmt(r['maxdd_pct'])} | {r['last_date'] or '—'} |")
    lines += ["", "_Marks accumulate daily via `com.alpca.forwardtrack` (Mon–Fri 17:15 ET). "
              "Need ≥2 realized marks to score a sleeve. NO capital at risk — pure forward validation._"]
    (ROOT / "docs" / "PAPER_EDGE.md").write_text("\n".join(lines) + "\n")

    print(f"[paper-edge] {n_snap} total snapshots · {len(rows_summary)} strategies updated")
    for k in order:
        r = rows_summary.get(k)
        if r:
            print(f"  {k:>10}: {r['n_realized']}/{r['n_marks']} realized · cum {r['cum_pct']}% · "
                  f"Sharpe {r['sharpe']} · maxDD {r['maxdd_pct']}% (funded={r['funded']})")
    print(f"[paper-edge] -> {args.db} + data/paper_edge_summary.json + docs/PAPER_EDGE.md")


if __name__ == "__main__":
    main()
