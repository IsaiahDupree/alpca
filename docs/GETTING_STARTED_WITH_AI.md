# Getting Started with Alpca + an AI Coding IDE

This repo is built to be driven by an AI coding agent. It already ships a `CLAUDE.md`
(agent context) and a `TRADING_POLICY.md` (the anti-overfit / pro-edge stance the agent
must follow). Point any modern AI IDE at the repo root and it will pick those up.

---

## 0. Clone + create the environment (once)

```bash
git clone https://github.com/IsaiahDupree/alpca.git
cd alpca
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env     # fill ALPACA_API_KEY / ALPACA_SECRET_KEY (PAPER keys)
```

Everything runs **offline + paper** by default. You do **not** need API keys to run the
backtests, the harness, or the 3,300+ tests — only to place paper orders. No live trading
happens unless you explicitly double-gate it (see README → Safety).

---

## 1. Connect it to your AI IDE

### Claude Code (CLI)
```bash
cd alpca
claude            # launches in the repo; auto-loads CLAUDE.md + TRADING_POLICY.md
```
That's it — `CLAUDE.md` is the project context and is read on every session. Try:
> "Read the README and docs/STATE_OF_THE_PROGRAM.md, then summarize which edges actually
> survived out-of-sample and why the rest were rejected."

### OpenAI Codex / Codex CLI
Open the folder as your workspace. Codex reads `CLAUDE.md` and `README.md` as context.
If your Codex setup uses `AGENTS.md`, symlink it: `ln -s CLAUDE.md AGENTS.md`.

### Cursor
`File → Open Folder → alpca`. Cursor indexes the repo automatically. Add `CLAUDE.md`
and `TRADING_POLICY.md` to `.cursorrules` context (or just @-mention them in chat).

### Windsurf / Cline / Continue / any MCP-aware IDE
Open the folder. Reference `CLAUDE.md` for the map and `TRADING_POLICY.md` for the stance.
The repo is plain Python with a clean package layout (`alpca/`), so any agent can navigate it.

**The one rule for the agent:** before it proposes any strategy or trade, have it read
`TRADING_POLICY.md`. It encodes the non-negotiables — out-of-sample + walk-forward +
realistic costs, no look-ahead, no martingale, and an advocate check paired with every
skeptic check. This is what keeps the agent from fooling you with a pretty backtest.

---

## 2. The 60-second orientation for the agent (and you)

Ask your AI IDE to run these, in order — they tell the whole honest story:

```bash
# (a) Judge ONE strategy honestly — vs buy-and-hold, significance, stability, OOS:
python -c "import json; from alpca.backtest.evaluation import evaluate; \
  bars=[json.loads(l) for l in open('data/cache/SPY_1day_bars.jsonl')]; \
  print(evaluate('rsi-mr', bars).render())"

# (b) Run the WHOLE 34-strategy registry through the harness -> ranked truth table:
python scripts/truth_table.py --symbol SPY --timeframe 1day --cache data/cache

# (c) Market-neutral discovery on a universe (basket + OOS + walk-forward):
python scripts/discover_universe.py --cache data/cache

# (d) Run the offline test suite (no creds, no network):
pytest -q
```

---

## 3. Where everything lives

| You want…                              | Look at…                                            |
|----------------------------------------|-----------------------------------------------------|
| The honest one-page summary            | `docs/STATE_OF_THE_PROGRAM.md`                      |
| Every edge we tested + why it passed/failed | `docs/EDGE_CASE_STUDIES.md` (61 case studies)  |
| The evaluation harness (the point)     | `alpca/backtest/evaluation.py`                      |
| The 34 strategies                      | `alpca/strategies/`                                 |
| Market-neutral pairs / cross-sectional | `alpca/backtest/pairs.py`, `cross_sectional.py`     |
| Realistic fills / costs / latency      | `alpca/execution/`                                  |
| Untested ideas worth chasing next      | `docs/RESEARCH_CANDIDATES.md`                       |
| How the autonomous research loop works | `docs/AI_RESEARCH_LOOP.md`                          |

---

## 4. The honest results in one paragraph

Across 34 price-only strategies on 5 years of daily data, **none beat buy-and-hold** — the
good-looking ones are risk-reduced *beta*, not *alpha*. The only thing that survived rigorous
out-of-sample + walk-forward + realistic-cost testing is **market-neutral**: a cointegrated-
pairs basket (in-sample Sharpe 1.78 → honest **walk-forward OOS ~0.54**, −5% max drawdown).
A 6-leg combiner (pairs + EAR-PEAD-hedged + cross-sectional + calendar overlays) runs at
**Sharpe ~0.99, ~9.7%/yr, near-zero cross-leg correlation (avg |corr| 0.038)** — modest, real,
and uncorrelated to the market. Many candidates that looked great in-sample (EAR-PEAD,
short-interest tilt, overnight reversal) were **killed by out-of-universe / out-of-regime
holdouts** — and the case studies document exactly how, so you can trust the survivors.
See the two graphs: `docs/strategy_landscape.png` (the overfit catch) and
`docs/deployed_results.png` (what actually ships).
