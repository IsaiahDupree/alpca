# Alpca — Live Paper Edge (auto-generated)

_Updated 2026-08-13 21:21:00 UTC · 239 snapshots logged in `data/paper_edge.db` · regenerate: `.venv/bin/python scripts/build_paper_edge_db.py`_

The realized out-of-sample edge from the deployed paper strategies, documented continuously (every forward-track run appends a snapshot). `_combined` = the funded book at deployed weights — the number that adjudicates the program.

| Sleeve | Funded | Realized marks | Cum % | Live Sharpe | Max DD % | Through |
|---|---|---:|---:|---:|---:|---|
| **_combined** | ✅ | 41/41 | -1.32 | -1.76 | -1.77 | 2026-08-13 |
| pairs | ✅ | 27/46 | -1.49 | -2.25 | -1.94 | 2026-08-13 |
| short_vol | ✅ | 41/45 | +0.64 | +2.12 | -0.48 | 2026-08-13 |
| momentum | probation | 41/45 | -2.37 | -1.90 | -4.30 | 2026-08-13 |
| revision | probation | 0/38 | — | — | — | — |
| composite | probation | 35/36 | +23.29 | +6.55 | -3.43 | 2026-08-13 |

_Marks accumulate daily via `com.alpca.forwardtrack` (Mon–Fri 17:15 ET). Need ≥2 realized marks to score a sleeve. NO capital at risk — pure forward validation._
