"""
Calendar / event-clock seasonality sleeves — turn-of-month and pre-FOMC drift.

These are NOT alpha generators on their own (they're cash-parked most days). Their value is
that their PnL is on an EVENT CLOCK, exogenous to price-driven strategies, so they correlate
~0 with the market-neutral pairs basket and the beta sleeve — the rare genuinely-uncorrelated
leg that lifts a *combined* Sharpe (see backtest/combine.py). Honest framing: risk-reduced,
time-diversifying overlay. Both die the OPPOSITE way to overtrading (~12-20 trades/yr), so
costs are negligible.

  - turn_of_month: long an index ETF the last `days_before` + first `days_after` trading days
    of each month (month-end index/pension flows), flat otherwise.
  - pre_fomc: long the day(s) before scheduled FOMC announcements (Lucca-Moench drift). Pass
    the announcement dates; FOMC_ANNOUNCEMENTS has the 2019-2026 scheduled dates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class SeasonalResult:
    name: str
    equity_curve: List[float]
    total_return: float
    sharpe: float
    max_drawdown: float
    n_days: int
    exposure: float                 # fraction of days in-market
    periods_per_year: float
    daily_returns: List[float] = field(default_factory=list)


# Scheduled FOMC announcement dates (second day of each meeting), 2019-2026.
FOMC_ANNOUNCEMENTS = [
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18",
    "2019-10-30", "2019-12-11", "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10",
    "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16", "2021-01-27", "2021-03-17",
    "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21",
    "2022-11-02", "2022-12-14", "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13", "2024-01-31", "2024-03-20",
    "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17",
    "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
]


def _date(ts) -> datetime:
    return datetime.fromtimestamp(float(ts), timezone.utc)


def turn_of_month_position(bars: List[dict], *, days_before: int = 4, days_after: int = 3) -> List[float]:
    """1.0 on the last `days_before` + first `days_after` trading days of each month, else 0."""
    n = len(bars)
    months = [(_date(b["timestamp"]).year, _date(b["timestamp"]).month) for b in bars]
    pos = [0.0] * n
    # index within each month + month length in trading days
    for i in range(n):
        # days from the start of this month-run
        start = i
        while start > 0 and months[start - 1] == months[i]:
            start -= 1
        end = i
        while end < n - 1 and months[end + 1] == months[i]:
            end += 1
        in_first = (i - start) < days_after
        in_last = (end - i) < days_before
        if in_first or in_last:
            pos[i] = 1.0
    return pos


def pre_fomc_position(bars: List[dict], announcements: Optional[List[str]] = None,
                      *, days_before: int = 1) -> List[float]:
    """1.0 on the `days_before` trading days immediately preceding each FOMC announcement."""
    ann = set(announcements or FOMC_ANNOUNCEMENTS)
    dates = [_date(b["timestamp"]).strftime("%Y-%m-%d") for b in bars]
    pos = [0.0] * len(bars)
    ann_idx = [i for i, d in enumerate(dates) if d in ann]
    for ai in ann_idx:
        for k in range(1, days_before + 1):
            if ai - k >= 0:
                pos[ai - k] = 1.0
    return pos


def backtest_seasonal(bars: List[dict], position: List[float], *, name: str = "seasonal",
                      cost_bps: float = 2.0, starting_equity: float = 100_000.0,
                      periods_per_year: float = 252.0) -> SeasonalResult:
    """Equity curve for a 0/1 long position series on `bars` (held into next day's return)."""
    from alpca.backtest.evaluation import max_drawdown_of, sharpe_of
    closes = [float(b["close"]) for b in bars]
    eq = [starting_equity]
    daily, prev = [], 0.0
    inmkt = 0
    for t in range(1, len(closes)):
        r = (closes[t] - closes[t - 1]) / closes[t - 1] if closes[t - 1] else 0.0
        p = position[t - 1]
        turn = abs(p - prev)
        ret = p * r - turn * (cost_bps / 10_000.0)
        eq.append(eq[-1] * (1 + ret))
        daily.append(ret)
        inmkt += 1 if p != 0 else 0
        prev = p
    return SeasonalResult(
        name=name, equity_curve=eq, total_return=(eq[-1] - eq[0]) / eq[0],
        sharpe=sharpe_of(eq, periods_per_year), max_drawdown=max_drawdown_of(eq),
        n_days=len(daily), exposure=inmkt / len(daily) if daily else 0.0,
        periods_per_year=periods_per_year, daily_returns=daily)
