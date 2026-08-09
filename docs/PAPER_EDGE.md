# Alpca — Live Paper Edge (auto-generated)

_Updated 2026-08-07 21:20:07 UTC · 215 snapshots logged in `data/paper_edge.db` · regenerate: `.venv/bin/python scripts/build_paper_edge_db.py`_

The realized out-of-sample edge from the deployed paper strategies, documented continuously (every forward-track run appends a snapshot). `_combined` = the funded book at deployed weights — the number that adjudicates the program.

| Sleeve | Funded | Realized marks | Cum % | Live Sharpe | Max DD % | Through |
|---|---|---:|---:|---:|---:|---|
| **_combined** | ✅ | 37/37 | -0.90 | -1.29 | -1.77 | 2026-08-07 |
| pairs | ✅ | 23/42 | -1.02 | -1.69 | -1.94 | 2026-08-07 |
| short_vol | ✅ | 37/41 | +0.49 | +1.72 | -0.48 | 2026-08-07 |
| momentum | probation | 37/41 | -2.53 | -2.28 | -4.30 | 2026-08-07 |
| revision | probation | 0/34 | — | — | — | — |
| composite | probation | 31/32 | +7.02 | +3.41 | -3.43 | 2026-08-07 |

_Marks accumulate daily via `com.alpca.forwardtrack` (Mon–Fri 17:15 ET). Need ≥2 realized marks to score a sleeve. NO capital at risk — pure forward validation._
