# Alpca — Live Paper Edge (auto-generated)

_Updated 2026-07-30 21:18:01 UTC · 179 snapshots logged in `data/paper_edge.db` · regenerate: `.venv/bin/python scripts/build_paper_edge_db.py`_

The realized out-of-sample edge from the deployed paper strategies, documented continuously (every forward-track run appends a snapshot). `_combined` = the funded book at deployed weights — the number that adjudicates the program.

| Sleeve | Funded | Realized marks | Cum % | Live Sharpe | Max DD % | Through |
|---|---|---:|---:|---:|---:|---|
| **_combined** | ✅ | 31/31 | -1.53 | -2.64 | -1.55 | 2026-07-30 |
| pairs | ✅ | 18/36 | -1.68 | -3.50 | -1.68 | 2026-07-30 |
| short_vol | ✅ | 31/35 | +0.19 | +0.76 | -0.48 | 2026-07-30 |
| momentum | probation | 31/35 | -3.40 | -3.55 | -4.30 | 2026-07-30 |
| revision | probation | 0/28 | — | — | — | — |
| composite | probation | 25/26 | +0.00 | +0.00 | +0.00 | 2026-07-30 |

_Marks accumulate daily via `com.alpca.forwardtrack` (Mon–Fri 17:15 ET). Need ≥2 realized marks to score a sleeve. NO capital at risk — pure forward validation._
