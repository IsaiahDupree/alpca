"""
NYSE session classifier — local market-hours knowledge.

Classifies an epoch-seconds instant into a trading session in America/New_York:
  REGULAR     09:30–16:00 ET (early-close days end 13:00)
  PRE_MARKET  04:00–09:30 ET
  AFTER_HOURS 16:00–20:00 ET
  CLOSED_WEEKEND / CLOSED_HOLIDAY / CLOSED

A naive backtest treats every bar as tradeable; real Alpaca rejects (or queues)
market orders outside regular hours, and overnight gaps mean you can't fill
between a 16:00 close and the next 09:30 open. This is the local approximation —
Alpaca's get_clock remains authoritative for live submission.

Holiday/early-close lists are hand-maintained (2024–2027) and MUST be updated
yearly; after the table's range, days are treated as regular weekdays (logged
caveat in docs/REALISM.md).
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

try:
    from zoneinfo import ZoneInfo
    _ET = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover - zoneinfo always present on 3.9+
    _ET = None


class Session(str, Enum):
    REGULAR = "REGULAR"
    PRE_MARKET = "PRE_MARKET"
    AFTER_HOURS = "AFTER_HOURS"
    CLOSED_WEEKEND = "CLOSED_WEEKEND"
    CLOSED_HOLIDAY = "CLOSED_HOLIDAY"
    CLOSED = "CLOSED"

    @property
    def is_open(self) -> bool:
        return self in (Session.REGULAR, Session.PRE_MARKET, Session.AFTER_HOURS)

    @property
    def is_regular(self) -> bool:
        return self == Session.REGULAR


# Full-day NYSE holidays (YYYY-MM-DD), 2024–2027.
NYSE_HOLIDAYS = {
    # 2024
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27",
    "2024-06-19", "2024-07-04", "2024-09-02", "2024-11-28", "2024-12-25",
    # 2025
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26",
    "2025-06-19", "2025-07-04", "2025-09-01", "2025-11-27", "2025-12-25",
    # 2026
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25",
    "2026-06-19", "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    # 2027
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31",
    "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
}

# Half-days: regular session ends 13:00 ET (day before July 4, day after
# Thanksgiving, Christmas Eve when a weekday).
NYSE_EARLY_CLOSES = {
    "2024-07-03", "2024-11-29", "2024-12-24",
    "2025-07-03", "2025-11-28", "2025-12-24",
    "2026-11-27", "2026-12-24",
    "2027-11-26",
}

# Table coverage — outside this, holidays are unknown (treated as weekdays).
_KNOWN_YEARS = range(2024, 2028)


def _to_et(epoch_s: float) -> datetime:
    dt = datetime.fromtimestamp(epoch_s, tz=timezone.utc)
    return dt.astimezone(_ET) if _ET is not None else dt


def session_at(epoch_s: float) -> Session:
    et = _to_et(epoch_s)
    day = et.strftime("%Y-%m-%d")

    if et.weekday() >= 5:  # Sat/Sun
        return Session.CLOSED_WEEKEND
    if day in NYSE_HOLIDAYS:
        return Session.CLOSED_HOLIDAY

    minutes = et.hour * 60 + et.minute
    open_min = 9 * 60 + 30          # 09:30
    close_min = (13 * 60) if day in NYSE_EARLY_CLOSES else (16 * 60)  # 13:00 / 16:00
    pre_min = 4 * 60               # 04:00
    post_end = 20 * 60             # 20:00

    if open_min <= minutes < close_min:
        return Session.REGULAR
    if pre_min <= minutes < open_min:
        return Session.PRE_MARKET
    if close_min <= minutes < post_end:
        return Session.AFTER_HOURS
    return Session.CLOSED


def session_date(epoch_s: float) -> str:
    """The America/New_York calendar date (YYYY-MM-DD) of an instant. Two bars
    share a 'trading session' iff they share this date."""
    return _to_et(epoch_s).strftime("%Y-%m-%d")


def is_regular_hours(epoch_s: float) -> bool:
    return session_at(epoch_s).is_regular


def is_tradeable(epoch_s: float, *, allow_extended: bool = False) -> bool:
    s = session_at(epoch_s)
    if s.is_regular:
        return True
    return allow_extended and s.is_open


def calendar_covers(epoch_s: float) -> bool:
    """False if the instant's year is outside the hand-maintained holiday table."""
    return _to_et(epoch_s).year in _KNOWN_YEARS
