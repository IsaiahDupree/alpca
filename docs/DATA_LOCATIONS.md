# Alpca — Data Locations (where everything lives)

Authoritative map of where Alpca's data lives, what's the source of truth, and how to regenerate
anything. Written 2026-06-22 after the device cleanup that archived ~80 GB of *other* projects'
trading data to the external drive (Alpca's caches were preserved untouched).

## TL;DR
- **Source of truth = git** (`github.com/IsaiahDupree/alpca`). All code, docs, committed result
  JSONs, and the whitelisted research catalogs are pushed. A fresh clone + a few download scripts
  reproduces everything.
- **Bulk market data = external drive** `/Volumes/My Passport/AlpcaData/` (**655 MB**, 30 caches).
  100% regenerable from Alpaca SIP / SEC EDGAR / FINRA / AlphaVantage. **Not** in git (too large /
  third-party), **not** part of the device-cleanup archive.
- **The cleanup's 82 GB `trading-data-archive`** on the same drive belongs to **other** projects
  (HFT-work, hft-live, poly2dollar) — *not* Alpca. Listed here only so the boundary is explicit.

## 1. External drive — `/Volumes/My Passport/AlpcaData/` (bulk caches, regenerable)

All regenerable via `scripts/` (mostly `download_data.py --feed sip`). The expensive-to-rebuild ones
are flagged. Feed `sip` = full history to 2016; `iex` = ~5yr only.

| Path (under `AlpcaData/`) | Size | What it is | Rebuild cost |
|---|---:|---|---|
| `cache_sip_10y` | 71M | **10.5yr SIP daily bars** (2016–) — the out-of-regime backbone | cheap (`download_data.py --feed sip --years 10`) |
| `cache` | 278M | 5yr IEX daily bars, ~300 symbols — most backtests | cheap |
| `cache_largecap_sip` | 36M | SIP daily, large-cap pairs universe | cheap |
| `cache_midcap_sip` / `cache_midcap` | 53M / 50M | mid-cap universes (SIP / legacy IEX) | cheap |
| `cache_smallcap` | 18M | small-cap (tested & rejected) | cheap |
| `cache_delisted_sip` | 39M | **survivorship-aware** universe (1,702 incl. delisted) — the clean broad test set | cheap |
| `cache_*_pit_delisted`, `cache_delisted_all` | ~7M | point-in-time delisted caches (survivorship correction) | cheap |
| `cache_vol` | 1.1M | VIX/UVXY/VXX — short-vol sleeve | cheap |
| `cache_xasset` | 2.7M | TLT/IEF/BND/RSP/MTUM — cross-asset | cheap |
| `cache_fresh` | 5.3M | recent 1-min bars + NBBO for live calibration | cheap (weekly) |
| `insider/insider_buys.jsonl` | 15M | **SEC bulk Form-4** open-market buys (2016–2026, 111,526 rows) | medium (`build_insider_signal.py`, ~10 min) |
| `fundamentals_edgar*` | ~3M | SEC EDGAR fundamentals (large/mid/small + delisted) | medium (`download_fundamentals_edgar.py`) |
| `short_interest_finra*` | ~5M | FINRA consolidated short-interest, 9+ yrs | medium |
| `earnings_av` / `earnings_av_holdout` | 708K / 208K | **AlphaVantage earnings** (PEAD universe) | ⚠️ **EXPENSIVE** — AV free tier = 25 req/day; took ~8 days to fill. **Preserve; don't delete.** |
| `earnings` | 828K | legacy earnings dates/EPS | cheap |
| `crypto` / `crypto_hourly` | 4.8M / 59M | crypto OHLCV (tested & rejected) | cheap |
| `sic_codes.json` | 16K | ticker → SIC industry map | static |

**The one preserve-at-all-costs item is `earnings_av*`** (rate-limit-bound rebuild). Everything else
re-downloads in minutes.

Local mirror used when the drive is unmounted: a few tickers can be pulled fresh into
`repo:data/cache_conf/` (e.g. the conference-drift + bond-ETF validations did this).

## 2. Local repo — `Alpca/data/` (runtime + committed results)

`data/*` is **gitignored** except an explicit whitelist (see `.gitignore`). Force-tracked in git:

| File | What it is |
|---|---|
| `data/edge_records.json` | headline-vs-honest Sharpe table driving the scatter/landscape graphics |
| `data/research/specs.jsonl` | mined strategy specs (web + KB) |
| `data/research/harvest_catalog.json` | 176-strategy web-harvest triage (Case 62) |
| `data/research/trading_knowledge_coverage.json` | 217-spec trading-knowledge coverage (Case 64) |
| `data/microstructure_deadbands.json`, `data/ofi_deadbands.json` | calibrated fill-model thresholds |

Untracked-but-local: `data/cache/` (1-min live-session bars), `data/cache_conf/` (drive-independent
validation pulls), `data/research/articles.jsonl` (bulky news corpus), and per-case result JSONs
(all regenerable by their `scripts/*.py`).

## 3. The cleanup archive (NOT Alpca)

`/Volumes/My Passport/trading-data-archive` (**82 GB**) holds historical data moved off-Mac from
**HFT-work / hft-live / poly2dollar** during the 2026-06-21/22 device cleanup. `HFT-work/data` is now
a **symlink** → that archive. **None of this is Alpca data** — it's recorded here only to mark the
boundary so future cleanups don't conflate the two.

## 4. Live paper-strategy fleet (macOS launchd, PAPER only)

All loaded/armed; they write forward-track ledgers to `data/*_forward_track.jsonl`. PAPER by default;
live is double-gated (`ALPACA_PAPER=0` **and** `ALPACA_LIVE_CONFIRMED=I_UNDERSTAND`).

| launchd job | Schedule (ET) | What it runs |
|---|---|---|
| `com.alpca.calibration` | Mon–Fri 09:36 | fill-model calibration to real paper fills |
| `com.alpca.livesession` | Mon–Fri 09:50 | hardened paper session (donchian-adx / SPY) |
| `com.alpca.swing` | Mon–Fri 09:50 | swing paper session (QQQ, holds overnight) |
| `com.alpca.forwardtrack` | Mon–Fri 17:15 | **deployed book** forward paper-track (pairs + short-vol + momentum) |
| `com.alpca.avearnings` | Daily 18:15 | AlphaVantage earnings pull (25/day, fills PEAD universe) |
| `com.alpca.discovery` | Fri 17:30 | market-neutral discovery re-screen (research, no orders) |
| `com.alpca.airesearch` | Sat 11:00 | AI research loop (proposes → harness-backtests → gates; no orders) |

Runner scripts: `scripts/{run_live_session,run_swing,run_calibration_pipeline,deploy_pairs_paper,deploy_shortvol_paper}.py`
(launchd shims live under `~/Library/Application Support/Alpca/*.sh`). Disable any job:
`launchctl bootout gui/$(id -u)/com.alpca.<name>`.

## Regenerate-from-scratch recipe
```bash
git clone github.com/IsaiahDupree/alpca && cd alpca
python3 -m venv .venv && .venv/bin/pip install -e . pytest matplotlib python-docx
# market data (drive optional — pick an --out):
.venv/bin/python scripts/download_data.py --symbols <universe> --timeframe 1day --years 10 --feed sip --out <cache>
.venv/bin/python scripts/build_insider_signal.py        # SEC Form-4
# earnings: let com.alpca.avearnings trickle (AV 25/day) — do NOT delete earnings_av if it exists
.venv/bin/python -m pytest tests/                        # 3,467 tests
```
