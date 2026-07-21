# Alpca — Live Paper Edge (auto-generated)

_Updated 2026-07-21 21:18:57 UTC · 137 snapshots logged in `data/paper_edge.db` · regenerate: `.venv/bin/python scripts/build_paper_edge_db.py`_

The realized out-of-sample edge from the deployed paper strategies, documented continuously (every forward-track run appends a snapshot). `_combined` = the funded book at deployed weights — the number that adjudicates the program.

| Sleeve | Funded | Realized marks | Cum % | Live Sharpe | Max DD % | Through |
|---|---|---:|---:|---:|---:|---|
| **_combined** | ✅ | 24/24 | -0.63 | -1.66 | -1.47 | 2026-07-21 |
| pairs | ✅ | 13/29 | -0.71 | -2.35 | -1.60 | 2026-07-16 |
| short_vol | ✅ | 24/28 | +0.31 | +1.74 | -0.40 | 2026-07-21 |
| momentum | probation | 24/28 | -1.10 | -1.84 | -2.66 | 2026-07-21 |
| revision | probation | 0/21 | — | — | — | — |
| composite | probation | 18/19 | +0.00 | +0.00 | +0.00 | 2026-07-21 |

_Marks accumulate daily via `com.alpca.forwardtrack` (Mon–Fri 17:15 ET). Need ≥2 realized marks to score a sleeve. NO capital at risk — pure forward validation._
