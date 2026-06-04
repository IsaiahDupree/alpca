"""
Runner-driven backtest — full order-book lifecycle + BacktestResult analytics.

`run_backtest` (engine.py) uses the simple next-bar-open model. This wrapper runs
a strategy through the SAME LiveRunner machinery the live bot uses — so RESTING
limit/stop orders, DAY/GTC expiry, stop triggers, partial fills, and cancel-
replace all apply — then returns a `BacktestResult` with the usual analytics
(total_return / Sharpe / max_drawdown / win_rate / n_trades).

Use this to backtest resting-order strategies (e.g. Donchian `entry="stop"`).
Offline: SimAdapter + OpenOrderBook, no credentials.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from alpca.backtest.engine import BacktestResult
from alpca.config import RiskConfig
from alpca.data.feed import ReplayBarSource
from alpca.execution.adapters.sim import SimAdapter
from alpca.execution.fills import FillModel
from alpca.execution.open_orders import OpenOrderBook
from alpca.execution.router import ExecutionRouter
from alpca.risk.risk_engine import RiskEngine
from alpca.runtime.runner import LiveRunner
from alpca.strategies.base import Strategy


def backtest_resting(
    strategy: Strategy,
    bars: List[Dict[str, float]],
    *,
    starting_equity: float = 100_000.0,
    target_notional_pct: float = 0.95,
    fill_model: Optional[FillModel] = None,
    risk_config: Optional[RiskConfig] = None,
    require_regular_hours: bool = False,
    settlement=None,
    pdt=None,
    borrow=None,
    allow_short: bool = False,
    seed: int = 0,
) -> BacktestResult:
    """
    Run `strategy` over `bars` through the runner + open-order book, return a
    BacktestResult. The fill model is shared by the book (resting fills) and the
    SimAdapter (market fills) so both paths cost identically.

    `allow_short`: when True (and no explicit `risk_config` overrides it), the
    risk engine permits shorts — required for a long/short strategy to actually
    trade. Left False, a short-seeking strategy's SELLs are rejected and the
    backtest silently records 0 trades, so this flag is surfaced explicitly.
    `borrow`: an optional BorrowFeeLedger to charge daily short-borrow fees.
    """
    fm = fill_model or FillModel(half_spread_bps=1.0, impact_coef_bps=8.0,
                                 participation_cap=0.10, min_tick=0.01)
    # NOTE: disable the orders/min rate limit for offline backtests. It is a live
    # broker-courtesy guardrail keyed on WALL-CLOCK time (time.monotonic); a fast
    # replay submits thousands of orders inside one real minute and would trip it
    # after ~60 orders (~30 round trips), silently capping every backtest. A 1-min
    # strategy never approaches 60 orders/min in live trading, so removing it here
    # changes no realistic behavior — it just stops the artifact.
    cfg = risk_config or RiskConfig(max_order_notional=1e12, max_concentration_pct=1.0,
                                    allow_short=allow_short, max_orders_per_min=10**9)
    risk = RiskEngine(cfg)
    router = ExecutionRouter(SimAdapter(seed=seed, sleep=False, fill_model=fm),
                             risk, None, fill_timeout_s=1.0)
    book = OpenOrderBook(fm)
    runner = LiveRunner(strategy, bars[0].get("symbol", strategy.name) if bars else strategy.name,
                        router, starting_equity=starting_equity,
                        target_notional_pct=target_notional_pct,
                        require_regular_hours=require_regular_hours,
                        settlement=settlement, pdt=pdt, borrow=borrow,
                        open_order_book=book)

    asyncio.run(runner.run(ReplayBarSource(bars)))

    # surface the fill model's headline cost on the result for provenance
    return runner.to_result(slippage_bps=fm.half_spread_bps)
