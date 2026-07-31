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


# Scheduled FOMC announcement dates (second day of each meeting), 1994-2026.
FOMC_ANNOUNCEMENTS = [
    "1994-02-04", "1994-03-22", "1994-05-17", "1994-07-06", "1994-08-16", "1994-09-27",
    "1994-11-15", "1994-12-20",
    "1995-02-01", "1995-03-28", "1995-05-23", "1995-07-06", "1995-08-22", "1995-09-26",
    "1995-11-15", "1995-12-19",
    "1996-01-31", "1996-03-26", "1996-05-21", "1996-07-03", "1996-08-20", "1996-09-24",
    "1996-11-13", "1996-12-17",
    "1997-02-05", "1997-03-25", "1997-05-20", "1997-07-02", "1997-08-19", "1997-09-30",
    "1997-11-12", "1997-12-16",
    "1998-02-04", "1998-03-31", "1998-05-19", "1998-07-01", "1998-08-18", "1998-09-29",
    "1998-11-17", "1998-12-22",
    "1999-02-03", "1999-03-30", "1999-05-18", "1999-06-30", "1999-08-24", "1999-10-05",
    "1999-11-16", "1999-12-21",
    "2000-02-02", "2000-03-21", "2000-05-16", "2000-06-28", "2000-08-22", "2000-10-03",
    "2000-11-15", "2000-12-19",
    "2001-01-31", "2001-03-20", "2001-05-15", "2001-06-27", "2001-08-21", "2001-10-02",
    "2001-11-06", "2001-12-11",
    "2002-01-30", "2002-03-19", "2002-05-07", "2002-06-26", "2002-08-13", "2002-09-24",
    "2002-11-06", "2002-12-10",
    "2003-01-29", "2003-03-18", "2003-05-06", "2003-06-25", "2003-08-12", "2003-09-16",
    "2003-10-28", "2003-12-09",
    "2004-01-28", "2004-03-16", "2004-05-04", "2004-06-30", "2004-08-10", "2004-09-21",
    "2004-11-10", "2004-12-14",
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30", "2005-08-09", "2005-09-20",
    "2005-11-01", "2005-12-13",
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29", "2006-08-08", "2006-09-20",
    "2006-10-25", "2006-12-12",
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28", "2007-08-07", "2007-09-18",
    "2007-10-31", "2007-12-11",
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25", "2008-08-05", "2008-09-16",
    "2008-10-29", "2008-12-16",
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24", "2009-08-12", "2009-09-23",
    "2009-11-04", "2009-12-16",
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10", "2010-09-21",
    "2010-11-03", "2010-12-14",
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09", "2011-09-21",
    "2011-11-02", "2011-12-13",
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01", "2012-09-13",
    "2012-10-24", "2012-12-12",
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31", "2013-09-18",
    "2013-10-30", "2013-12-18",
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17",
    "2014-10-29", "2014-12-17",
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17",
    "2015-10-28", "2015-12-16",
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21",
    "2016-11-02", "2016-12-14",
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20",
    "2017-11-01", "2017-12-13",
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26",
    "2018-11-08", "2018-12-19",
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
