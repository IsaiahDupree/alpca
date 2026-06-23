# Alpca — Live Paper Edge (auto-generated)

_Updated 2026-06-23 00:25:51 UTC · 4 snapshots logged in `data/paper_edge.db` · regenerate: `.venv/bin/python scripts/build_paper_edge_db.py`_

The realized out-of-sample edge from the deployed paper strategies, documented continuously (every forward-track run appends a snapshot). `_combined` = the funded book at deployed weights — the number that adjudicates the program.

| Sleeve | Funded | Realized marks | Cum % | Live Sharpe | Max DD % | Through |
|---|---|---:|---:|---:|---:|---|
| **_combined** | ✅ | 5/5 | -0.33 | -7.63 | -0.35 | 2026-06-22 |
| pairs | ✅ | 3/8 | -0.37 | -11.22 | -0.37 | 2026-06-22 |
| short_vol | ✅ | 5/7 | +0.11 | +2.36 | -0.19 | 2026-06-22 |
| momentum | probation | 5/7 | -0.25 | -1.74 | -1.03 | 2026-06-22 |

_Marks accumulate daily via `com.alpca.forwardtrack` (Mon–Fri 17:15 ET). Need ≥2 realized marks to score a sleeve. NO capital at risk — pure forward validation._
