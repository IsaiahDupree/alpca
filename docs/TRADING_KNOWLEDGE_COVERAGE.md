# Alpca × Trading-Knowledge — Coverage Map & Cross-Ecosystem Synthesis

We ingested the entire **`trading-knowledge`** corpus (the `knowledge` MCP provider —
github.com/IsaiahDupree/Trading-Knowledge): strategy references distilled from **19 algorithmic-
trading books** (`books/`), per-strategy aggregations (`strategies/`), and — most valuable — every
edge **actually implemented and forward-tested in our own five codebases** (`code-strategies/`:
TradingBot2/lastminute, HFT-work, polymarket-weather, kalshi, LiquidationBot), plus the
`LIVE-LANE-MAP` that reconciles the book shelf against real forward tracks.

A 32-agent ingestion workflow read all 50 reachable docs, extracted **217 strategy specs**, mapped
each against Alpca's 63 tested cases and the venue, and surfaced any Alpaca-feasible gap.

## The headline: the corpus is covered, and the two programs agree

| Bucket | Count | Meaning |
|---|---:|---|
| **Maps to an Alpca case already tested** | **126** | Momentum, MR, pairs, factors, sentiment, RSI/Bollinger/breakout, etc. — already run through our battery (mostly beta/rejected). |
| **Structural / venue-specific (not Alpaca-feasible)** | **43** | Maker rebates, complete-set merge-maker, Dutch-book arb, oracle-lag, options spreads, VWAP/TWAP, HFT-scalping, on-chain liquidation keeping. Real edges *elsewhere*; impossible on a price-taker equity venue. |
| **Infeasible (other)** | **27** | Need L2 depth, options, ms requoting, or data we lack. |
| **"Novel + feasible + directional"** | **19** | All **plausibility ≤ 0.35** — and on inspection: backtest infra already built (queue-fill model), verification machinery (Council v2), or re-labelings of momentum/fair-value/MR/LSTM/TD3 we've already rejected. **No genuine new alpha.** |

**Net: of 217 published/implemented strategies across the full corpus, ZERO surface a new
Alpaca-deployable alpha.** Everything is (a) already tested here, (b) a structural edge specific to
the prediction-market/crypto/DeFi venues, or (c) a low-conviction relabel. The deployed Alpca book
(pairs + short-vol) stands.

## Cross-ecosystem confirmation (this is the real value)

Two independent research programs — Alpca (equities) and the `code-strategies` stack (prediction
markets / crypto / DeFi) — built with the **same anti-overfitting discipline**, converge on the
same conclusions:

1. **Directional retail strategies get debunked at honest fills.** The shelf's backbone is
   mean-reversion (11/19 books); their forward tracks killed directional MR (penny-sniper −77%/$),
   exactly as Alpca found single-asset directional = beta (Case 2). Both programs **sell** the tails
   the books tell you to buy.
2. **The one directional survivor matches.** HFT-work's **cross-sectional momentum survived the
   overfit gauntlet at OOS Sharpe ~0.68** — *the same number Alpca found independently* for
   cross-sectional momentum (Case 3). Two venues, one edge, one magnitude. Strong evidence it's real
   (and real-but-modest).
3. **The edges that actually pay are STRUCTURAL — and not on Alpaca's venue.** Their confirmed
   winners are maker/structural: complete-set merge-maker (the capital path), reward-farming
   (+$749/6d), and the **weather oracle-lag** (+$0.1855/ct, t=8.65 — the only proven *forecasting*
   edge in the whole stack). None is portable to a price-taker equity venue with no rebates/L2/oracle.
4. **The transferable asset is the discipline, not a strategy.** Both programs independently
   concluded the real edge is the 5-control verification machinery (OOS-only, structural>fitted,
   block-perm, forward>backtest) — which is exactly Alpca's `VERIFICATION_CONTROLS.md`.

## What this means for Alpca

- **No new edge to add from the corpus.** The hunt for a 3rd Alpca leg is not blocked by *missing
  ideas* — we've now exhausted both the public literature (Case 62, 176 web strategies) and the
  user's entire private corpus (217 specs). The bar is the venue, not the idea supply.
- **The maker/structural lesson reframes the ceiling.** The user's real money is made on the
  *maker/structural* side (rebates, complete-set, oracle-lag) — categorically unavailable on Alpaca.
  Alpca's honest ceiling remains market-neutral (pairs + short-vol, ~0.5 Sharpe), and that is
  consistent with the broader program's finding that directional price alpha is thin everywhere.
- **Cross-sectional momentum (~0.68) deserves a second look as a deployable leg** — it's the one
  directional edge confirmed in *both* programs. Alpca tested it (Case 3, modest/config-sensitive);
  given the independent HFT-work confirmation, a hardened market-neutral cross-sectional-momentum
  sleeve (z-scored, trend-gated, the HFT-work construction) on the 10.5yr SIP universe is the
  highest-conviction remaining build.

## Provider now wired in
The `knowledge` MCP (`trading-knowledge`, 60 docs) is now an active source in the research pipeline
(`docs/RESEARCH_PIPELINE.md`), alongside AlphaVantage, Alpaca, SEC EDGAR/Form-4, FINRA, and the web
publishers. Coverage catalog: `data/research/trading_knowledge_coverage.json`.

_Generated by the `ingest-trading-knowledge` workflow (32 agents, ~1.2M tokens, 50 docs → 217 specs)._
