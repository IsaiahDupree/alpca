# Alpca — Research Pipeline (harvest → extract → validate published edges)

A system that turns *other people's published strategies* into *honestly-validated edges*. It
harvests articles from every data/research provider we have access to, distils them into
structured testable strategy specs, dedups against the 60+ edges we've already tested, and runs the
novel + venue-feasible ones through the same battery (out-of-universe + out-of-regime + cost + DSR)
that gates everything else in this repo.

The point is leverage: the world publishes thousands of strategy claims; almost all collapse under
honest testing (see `EDGE_CASE_STUDIES.md`). This pipeline lets us ingest those claims at scale and
let the harness do the killing — so a real edge, if one exists, surfaces without us hand-coding each.

## Architecture

```
                 ┌─ AlphaVantage NEWS_SENTIMENT ─┐   (HTTP, scriptable)
  HARVEST        ├─ Alpaca / Benzinga news ──────┤ → scripts/harvest_news.py
  (providers) ───┤                               │
                 ├─ Quant publishers (web) ──────┤   (agent tools: WebSearch/WebFetch)
                 ├─ Reddit r/algotrading,r/quant ┤   (reddit MCP)
                 └─ Perplexity synthesis ────────┘   (perplexity)
                                │
                                ▼
                 data/research/articles.jsonl        (Article: source,kind,title,url,text,sentiment,...)
                                │
  EXTRACT        alpca/research/extract.py  (OpenAI, JSON-mode)  → scripts/mine_strategies.py
  (LLM)          • precise signal_rule, universe, direction, data_needs, claimed_metric
                 • maps_to: dedup against our ALREADY_TESTED list (else "novel")
                 • feasible_on_alpaca: venue filter (no L2/HFT/options/intl-shorting)
                                │
                                ▼
                 data/research/specs.jsonl           (StrategySpec, status: extracted→…)
                                │
  TRIAGE         keep:  maps_to == "novel"  AND  feasible_on_alpaca
                                │
                                ▼
  VALIDATE       implement (map to an existing primitive, or code a new one) → run the battery:
                 vs B&H · beta decomposition · OOS · out-of-regime (10.5yr SIP) · cost/borrow · DSR
                                │
                                ▼
  CASE STUDY     EDGE_CASE_STUDIES.md + spec.status = validated/rejected + verdict
```

**Why hybrid harvest.** API providers (AlphaVantage, Alpaca) are pure HTTP → `harvest_news.py`
(scriptable, schedulable). Web / Reddit / Perplexity come through the agent's tools → the agent
writes normalized `Article`s into the *same* corpus via `alpca.research.corpus`. One corpus, many
intakes.

## Components

| Piece | What it does |
|---|---|
| `alpca/research/corpus.py` | `Article` + `StrategySpec` dataclasses; append-only, content-hashed, deduped JSONL store (`data/research/`) |
| `alpca/research/extract.py` | OpenAI extractor (urllib, no SDK): article text → structured specs, with dedup + venue-feasibility flags |
| `scripts/harvest_news.py` | Harvest AlphaVantage NEWS_SENTIMENT (+ Alpaca news) → corpus. Rate-limit-aware (AV ~25/day, shared with the earnings job) |
| `scripts/mine_strategies.py` | Run the extractor over research/community articles → specs; prints novel-vs-tested + feasible triage |
| `scripts/validate_conference_drift.py` | Example end-to-end validation of a mined edge (Case 61) — the template for validating future specs |

## Data sources — what we actually have (verified 2026-06)

| Provider | Access | Use |
|---|---|---|
| **AlphaVantage NEWS_SENTIMENT** | free key (`.env`) | articles + sentiment + per-ticker relevance, history via `time_from` (1000/window); ~25 calls/day |
| **Alpaca news** (Benzinga) | paper key | real-time headlines, shallow history; `alpca/data/news.py` |
| **Alpaca SIP bars** | account has it (config defaults to `iex`) | **10.5yr daily history (2016-)** — the out-of-regime backbone; `download_data.py --feed sip` |
| **WebSearch / WebFetch** | agent tools | quant publishers: Quantpedia, Alpha Architect, SSRN, arXiv q-fin, QuantConnect |
| **Reddit MCP** | connected | r/algotrading, r/quant practitioner ideas |
| **Perplexity / knowledge MCP** | connected | synthesis + internal KB |
| ~~RapidAPI~~ | subbed to tiktok/reddit scrapers only | **not** finance — unusable here |
| ~~AlphaVantage `TIME_SERIES_DAILY` full~~ | now premium | deep daily history gated — use Alpaca SIP / Stooq / Tiingo |

## Run it

```bash
# 1. Harvest API news (modest call budget — AV cap is shared with the earnings job)
.venv/bin/python scripts/harvest_news.py --av-topics financial_markets,economy_macro --av-months 2 --max-calls 3

# 2. (agent step) harvest quant-publisher / Reddit articles via tools → corpus.add_articles(...)

# 3. Mine specs from research/community articles
.venv/bin/python scripts/mine_strategies.py --kinds research,community

# 4. Validate a surfaced novel+feasible spec (template: conference drift = Case 61)
.venv/bin/python scripts/validate_conference_drift.py
```

## Track record (this is the whole point)

| Mined edge | Source | Verdict |
|---|---|---|
| Conference-driven return drift (AAPL/GOOGL/MSFT) | Quantpedia | ❌ Case 61 — beta dressed as alpha; long-only 0.63 < B&H 1.06; hedged 0.54 is a 29-event recency artifact |
| Market-neutral conference variant | Quantpedia | ❌ flagged infeasible (single-name/index short framing) at extraction |
| 11-factor Lasso market-neutral (NYSE) | arXiv 2412.12350 | ⏳ extracted; flagged infeasible at extraction (40-name shorting + fundamentals + adverse borrow) — revisit if the long-only / index-hedged form is tractable |

The corpus currently holds **2,000+ news articles** (AlphaVantage) plus the harvested research
write-ups. As the queue grows, every novel + feasible spec gets the same honest battery — and the
denominator (how many we killed) is the credibility, exactly as in the main case-study log.
