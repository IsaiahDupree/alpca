from datetime import datetime, timezone

from alpca.data.calendar import (
    Session,
    calendar_covers,
    is_regular_hours,
    is_tradeable,
    session_at,
)


def _et(y, mo, d, h, mi):
    """epoch seconds for a wall-clock America/New_York time."""
    from zoneinfo import ZoneInfo
    return datetime(y, mo, d, h, mi, tzinfo=ZoneInfo("America/New_York")).timestamp()


def test_regular_hours_weekday():
    # Tue 2025-06-17 10:30 ET -> regular
    ts = _et(2025, 6, 17, 10, 30)
    assert session_at(ts) == Session.REGULAR
    assert is_regular_hours(ts)
    assert is_tradeable(ts)


def test_premarket_and_afterhours():
    assert session_at(_et(2025, 6, 17, 5, 0)) == Session.PRE_MARKET
    assert session_at(_et(2025, 6, 17, 18, 0)) == Session.AFTER_HOURS
    # not tradeable by default, tradeable with allow_extended
    assert not is_tradeable(_et(2025, 6, 17, 5, 0))
    assert is_tradeable(_et(2025, 6, 17, 5, 0), allow_extended=True)


def test_overnight_closed():
    assert session_at(_et(2025, 6, 17, 2, 0)) == Session.CLOSED


def test_weekend_closed():
    # Sat 2025-06-21
    assert session_at(_et(2025, 6, 21, 12, 0)) == Session.CLOSED_WEEKEND


def test_holiday_closed():
    # Independence Day 2025-07-04 (Friday)
    assert session_at(_et(2025, 7, 4, 12, 0)) == Session.CLOSED_HOLIDAY


def test_early_close_half_day():
    # 2025-07-03 is an early close (13:00). 14:00 should be AFTER_HOURS, not REGULAR.
    assert session_at(_et(2025, 7, 3, 12, 30)) == Session.REGULAR
    assert session_at(_et(2025, 7, 3, 14, 0)) == Session.AFTER_HOURS


def test_calendar_coverage_flag():
    assert calendar_covers(_et(2025, 6, 17, 10, 0))
    # 2030 is outside the hand-maintained table
    assert not calendar_covers(datetime(2030, 6, 17, 14, 0, tzinfo=timezone.utc).timestamp())
