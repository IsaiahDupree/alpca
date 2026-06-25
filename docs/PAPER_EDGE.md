# Alpca — Live Paper Edge (auto-generated)

_Updated 2026-06-24 21:17:38 UTC · 23 snapshots logged in `data/paper_edge.db` · regenerate: `.venv/bin/python scripts/build_paper_edge_db.py`_

The realized out-of-sample edge from the deployed paper strategies, documented continuously (every forward-track run appends a snapshot). `_combined` = the funded book at deployed weights — the number that adjudicates the program.

| Sleeve | Funded | Realized marks | Cum % | Live Sharpe | Max DD % | Through |
|---|---|---:|---:|---:|---:|---|
| **_combined** | ✅ | 7/7 | -0.27 | -4.81 | -0.40 | 2026-06-24 |
| pairs | ✅ | 5/11 | -0.29 | -5.69 | -0.40 | 2026-06-24 |
| short_vol | ✅ | 7/10 | -0.09 | -1.42 | -0.40 | 2026-06-24 |
| momentum | probation | 7/10 | -1.45 | -16.78 | -1.45 | 2026-06-24 |
| revision | probation | 0/3 | — | — | — | — |

_Marks accumulate daily via `com.alpca.forwardtrack` (Mon–Fri 17:15 ET). Need ≥2 realized marks to score a sleeve. NO capital at risk — pure forward validation._
