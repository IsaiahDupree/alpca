# Alpca — Live Paper Edge (auto-generated)

_Updated 2026-07-17 21:17:51 UTC · 125 snapshots logged in `data/paper_edge.db` · regenerate: `.venv/bin/python scripts/build_paper_edge_db.py`_

The realized out-of-sample edge from the deployed paper strategies, documented continuously (every forward-track run appends a snapshot). `_combined` = the funded book at deployed weights — the number that adjudicates the program.

| Sleeve | Funded | Realized marks | Cum % | Live Sharpe | Max DD % | Through |
|---|---|---:|---:|---:|---:|---|
| **_combined** | ✅ | 22/22 | -0.64 | -1.78 | -1.47 | 2026-07-17 |
| pairs | ✅ | 13/27 | -0.71 | -2.35 | -1.60 | 2026-07-16 |
| short_vol | ✅ | 22/26 | +0.09 | +0.56 | -0.40 | 2026-07-17 |
| momentum | probation | 22/26 | -2.25 | -5.05 | -2.66 | 2026-07-17 |
| revision | probation | 0/19 | — | — | — | — |
| composite | probation | 16/17 | +0.00 | +0.00 | +0.00 | 2026-07-17 |

_Marks accumulate daily via `com.alpca.forwardtrack` (Mon–Fri 17:15 ET). Need ≥2 realized marks to score a sleeve. NO capital at risk — pure forward validation._
