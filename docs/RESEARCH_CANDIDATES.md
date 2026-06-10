# Alpca — Research Candidate Edges (second deep scout pass)

A ranked register of **untested** candidate edges surfaced by a deep survey of open-source
quant repos / papers, each screened against our venue (Alpaca PAPER, ~1.2s taker fills,
top-of-book, ~2bps equity / ~10bps crypto, daily/low-turnover) **and** our harness
(beat buy-and-hold, Sharpe t-stat significance, regime stability, select-IS/report-OOS).

This is a *to-test* list, not a results document — see `EDGE_CASE_STUDIES.md` for what has
already been built and rejected. Hard venue fact that kills several otherwise-attractive
ideas: **Alpaca spot crypto is BTC/USD + ETH/USD only, long-only, no shorting** — so any
cross-sectional crypto long-short basket is infeasible here regardless of published Sharpe.

## Register

| # | Candidate | Data (free?) | Feasibility | Honest expectation |
|---|-----------|--------------|-------------|--------------------|
| 1 | Overnight-vs-intraday premium (buy-close/sell-open) | Alpaca bars ✓ | **MARGINAL→INFEASIBLE** | Largely a bid-ask-bounce / open-print **artifact**; dies to taker cost twice daily |
| 2 | Pre-FOMC announcement drift | SPY + FOMC calendar ✓ | **FEASIBLE** (~8 trades/yr) | Real but **decaying post-2011**; risk-reduced tilt, not standalone alpha |
| 3 | Turn-of-month (TOM) effect | SPY bars ✓ | **FEASIBLE** (~12 rt/yr) | Old, arbitraged; diversifying overlay, likely < B&H absolute return |
| 4 | **PEAD long-short** (earnings surprise) | Finnhub free + Alpaca ✓ | **MARGINAL→FEASIBLE** | The one genuinely fundamental, market-neutral, diversifying shot; decayed since ~2005 |
| 5 | Betting-Against-Beta / low-vol | Alpaca bars ✓ | **MARGINAL** | Un-levered = risk-reduced beta (like rsi-mr), not a beater; needs leverage we lack |
| 6 | Vol-target / risk-parity overlay | Alpaca bars ✓ | **FEASIBLE** as overlay | Drawdown insurance on the beta sleeve; **already half-rejected as alpha** (Kim 2016) |
| 7 | VIX term-structure regime timing (SVXY) | CBOE/Yahoo ^VIX/^VIX3M ✓ | **MARGINAL** | Short-vol tail trap — Sharpe-test actively **misleads**; only with a CVaR/tail gate |
| 8 | **Meta-labeling** (triple-barrier) on our basket | none (runs on signals) ✓ | **FEASIBLE**, low-risk | Can only *filter* our 0.54 basket; fewer trades may wash out the gain, downside bounded |
| 9 | ETF relative-value via Box-Tiao / min-half-life | Alpaca sector ETFs ✓ | **MARGINAL** | Same family we rejected; optimizing half-life is a **stronger overfit magnet** |
| 10 | Cross-asset / lead-lag (BTC→ETH, crypto X-section) | Alpaca BTC/ETH ✓ | **INFEASIBLE** | Needs a coin cross-section + shorting; 2 long-only coins too narrow — **dead** |
| 11 | Cross-sectional crypto factor/reversal basket | needs coin X-section ✗ | **INFEASIBLE** | Long-only 2-coin venue cannot implement it — **dead** |

## Top 3 to build next (scout's ranking, by expected value)

1. **PEAD cross-sectional long-short** (free Finnhub earnings surprise + Alpaca bars).
   *Why it might survive where price-only strategies didn't:* the only **fundamental,
   event-driven, dollar-neutral** idea here — orthogonal to both long-beta and our price-
   mean-reversion pairs basket, so it **diversifies** the one survivor instead of duplicating
   it. Most-replicated anomaly in finance, 30-yr backtests, free data now exists. Real risks:
   post-2005 decay, paper-shorting realism. Build L/S, report strictly OOS, judge the short
   leg separately.

2. **Seasonality overlay basket** = pre-FOMC drift + turn-of-month (+ optional overnight).
   *Why it might survive:* our prior failures all died to **overtrading costs** — these die
   the opposite way (~20 trades/yr total), so cost is a non-issue. Honest framing: a
   **risk-reduced beta overlay** (cash-parked most days), cheap and fast to falsify; test the
   "disappearing post-2011" claim head-on (pre-2011 IS vs post-2011 OOS).

3. **Meta-labeling (mlfinlab triple-barrier) on the surviving 84-symbol basket.**
   *Why it might survive:* doesn't need a new edge — it **amplifies** the one we've proven
   OOS (0.54) by filtering low-conviction trades, with purged-CV machinery built to resist
   the overfit that killed Avellaneda-Lee. Lowest-risk build; upside is lifting our sole
   deployable market-neutral strategy. Honest null: fewer trades + estimation noise may wash
   out the precision gain on an already-thin edge.

## Explicitly dead ends (do not spend time)

- **Cross-crypto / lead-lag / crypto-factor baskets** (#10, #11) — need a coin cross-section
  and shorting we don't have on Alpaca spot.
- **Naive-to-fancy ETF cointegration reselection** (#9) — same family already rejected;
  half-life optimization is a stronger overfit magnet.
- **Standalone overnight buy-close/sell-open** (#1) — largely a bid-ask / open-auction-print
  artifact that evaporates at a taker fill with cost. (Correction to an earlier suggestion:
  this is **not** a high-promise pick.)
- **Short-vol VIX harvesting** (#7) — executable via SVXY but its fat negative-skew tail is
  exactly what a Sharpe-significance test over-rewards; only with an explicit tail/CVaR gate.

## Notes on data access (verified)

- **Binance `fapi`** funding/derivatives: **geo-blocked (HTTP 451) from US IPs.** Use Kraken
  Futures (works) for funding.
- **Finnhub free tier** `/calendar/earnings`: EPS estimate + actual + surprise %, US-accessible
  — the free substitute for WRDS/IBES that makes PEAD buildable.
- **CBOE / Yahoo `^VIX`, `^VIX3M`**: free term-structure proxy for VIX-curve signals.
