from datetime import datetime
from zoneinfo import ZoneInfo

from phase1.market_clock import (
    get_session_state,
    is_cash_market_open,
    get_expiration_close_dt,
    compute_time_to_expiry_years,
)

NY = ZoneInfo("America/New_York")


def test_weekend_is_closed():
    ts = datetime(2026, 3, 14, 12, 0, tzinfo=NY)  # Saturday
    state = get_session_state("NYSE", ts)
    assert state.is_open is False


def test_regular_session_is_open():
    ts = datetime(2026, 3, 13, 10, 0, tzinfo=NY)  # Friday 10:00 AM ET
    state = get_session_state("NYSE", ts)
    assert state.is_open is True


def test_after_close_is_closed():
    ts = datetime(2026, 3, 13, 16, 30, tzinfo=NY)  # Friday 4:30 PM ET
    state = get_session_state("NYSE", ts)
    assert state.is_open is False


def test_holiday_is_closed():
    ts = datetime(2026, 1, 1, 12, 0, tzinfo=NY)  # New Year's Day
    assert is_cash_market_open(ts) is False


def test_compute_time_to_expiry_years_decreases_into_close():
    t1 = datetime(2026, 3, 20, 10, 0, tzinfo=NY)
    t2 = datetime(2026, 3, 20, 15, 0, tzinfo=NY)

    y1, _ = compute_time_to_expiry_years("2026-03-20", ts=t1)
    y2, _ = compute_time_to_expiry_years("2026-03-20", ts=t2)

    assert y1 > y2 > 0


def test_compute_time_to_expiry_zero_after_close_without_floor():
    t = datetime(2026, 3, 20, 17, 0, tzinfo=NY)

    y, close_dt = compute_time_to_expiry_years("2026-03-20", ts=t, floor=None)

    assert close_dt.tzinfo is not None
    assert y == 0.0
