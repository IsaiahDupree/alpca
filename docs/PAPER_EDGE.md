# Alpca — Live Paper Edge (auto-generated)

_Updated 2026-06-23 00:40:34 UTC · 8 snapshots logged in `data/paper_edge.db` · regenerate: `.venv/bin/python scripts/build_paper_edge_db.py`_

The realized out-of-sample edge from the deployed paper strategies, documented continuously (every forward-track run appends a snapshot). `_combined` = the funded book at deployed weights — the number that adjudicates the program.

| Sleeve | Funded | Realized marks | Cum % | Live Sharpe | Max DD % | Through |
|---|---|---:|---:|---:|---:|---|
| **_combined** | ✅ | 5/5 | -0.33 | -7.65 | -0.35 | 2026-06-22 |
| pairs | ✅ | 3/9 | -0.37 | -11.22 | -0.37 | 2026-06-22 |
| short_vol | ✅ | 5/8 | +0.10 | +2.11 | -0.19 | 2026-06-22 |
| momentum | probation | 5/8 | -1.03 | -16.26 | -1.03 | 2026-06-22 |

_Marks accumulate daily via `com.alpca.forwardtrack` (Mon–Fri 17:15 ET). Need ≥2 realized marks to score a sleeve. NO capital at risk — pure forward validation._
