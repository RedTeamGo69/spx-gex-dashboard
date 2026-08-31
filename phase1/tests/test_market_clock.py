from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from phase1.market_clock import (
    get_session_state,
    is_cash_market_open,
    get_expiration_close_dt,
    compute_time_to_expiry_years,
    trading_days_between,
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


@pytest.mark.parametrize(
    ("root", "expected_day", "expected_hour", "expected_minute"),
    [
        ("SPX", 19, 17, 0),    # AM-settled; prior-session Cboe curb close
        ("SPXW", 20, 16, 0),   # PM-settled
        ("XSP", 20, 16, 0),    # PM-settled
        ("NDX", 19, 16, 15),   # AM-settled; prior regular session
        ("NDXP", 20, 16, 0),   # PM-settled
    ],
)
def test_root_aware_final_trading_timestamp(
        root, expected_day, expected_hour, expected_minute):
    _, close_dt = compute_time_to_expiry_years(
        "2026-03-20",
        ts=datetime(2026, 3, 19, 12, 0, tzinfo=NY),
        root=root,
        floor=None,
    )

    assert close_dt.day == expected_day
    assert (close_dt.hour, close_dt.minute) == (expected_hour, expected_minute)


@pytest.mark.parametrize("root", ["SPXW", "XSP", "NDXP"])
def test_pm_settled_roots_expire_at_four_not_four_fifteen(root):
    before, _ = compute_time_to_expiry_years(
        "2026-03-20",
        ts=datetime(2026, 3, 20, 15, 59, tzinfo=NY),
        root=root,
        floor=None,
    )
    at_close, _ = compute_time_to_expiry_years(
        "2026-03-20",
        ts=datetime(2026, 3, 20, 16, 0, tzinfo=NY),
        root=root,
        floor=None,
    )
    after, _ = compute_time_to_expiry_years(
        "2026-03-20",
        ts=datetime(2026, 3, 20, 16, 5, tzinfo=NY),
        root=root,
        floor=None,
    )

    assert before > 0
    assert at_close == 0.0
    assert after == 0.0


@pytest.mark.parametrize("root", ["SPXW", "XSP", "NDXP"])
def test_pm_root_shortened_session_uses_one_pm_close(root):
    _, close_dt = compute_time_to_expiry_years(
        "2026-11-27",
        ts=datetime(2026, 11, 27, 12, 0, tzinfo=NY),
        root=root,
        floor=None,
    )

    assert close_dt.date().isoformat() == "2026-11-27"
    assert (close_dt.hour, close_dt.minute) == (13, 0)


@pytest.mark.parametrize(
    ("root", "live_time", "dead_time"),
    [
        ("SPX", (16, 59), (17, 0)),
        ("NDX", (16, 14), (16, 15)),
    ],
)
def test_am_roots_stop_on_the_prior_session(root, live_time, dead_time):
    live, _ = compute_time_to_expiry_years(
        "2026-03-20",
        ts=datetime(2026, 3, 19, *live_time, tzinfo=NY),
        root=root,
        floor=None,
    )
    dead, _ = compute_time_to_expiry_years(
        "2026-03-20",
        ts=datetime(2026, 3, 19, *dead_time, tzinfo=NY),
        root=root,
        floor=None,
    )

    assert live > 0
    assert dead == 0.0


def test_good_friday_expiration_uses_thursday_close():
    """Good Friday 2026-04-03: market closed. Expiration should use Thursday's close."""
    # From Thursday 2 PM, expiring Good Friday
    ts = datetime(2026, 4, 2, 14, 0, tzinfo=NY)
    close_dt = get_expiration_close_dt("2026-04-03")

    # Close should be on Thursday, not Friday
    assert close_dt.weekday() == 3  # Thursday
    assert close_dt.day == 2  # April 2

    # T should be small (only remaining Thursday trading hours)
    T, _ = compute_time_to_expiry_years("2026-04-03", ts=ts, floor=0.0)
    # Should be ~2.25 hours / (252*6.5 hours/year) ≈ 0.00137 years
    assert T < 0.005  # less than ~1.8 days
    assert T > 0.0    # still positive (before close)


# ── trading_days_between ──────────────────────────────────────────────────────

def test_trading_days_between_full_week():
    assert trading_days_between("2026-03-09", "2026-03-13") == 5


def test_trading_days_between_holiday_week():
    # MLK Monday 2026-01-19: the Mon→Fri window has only 4 sessions
    assert trading_days_between("2026-01-19", "2026-01-23") == 4


def test_trading_days_between_same_day_session():
    assert trading_days_between("2026-03-11", "2026-03-11") == 1


def test_trading_days_between_weekend_only():
    assert trading_days_between("2026-03-14", "2026-03-15") == 0


def test_trading_days_between_inverted_window():
    assert trading_days_between("2026-03-13", "2026-03-09") == 0


def test_trading_days_between_accepts_date_objects():
    from datetime import date
    assert trading_days_between(date(2026, 3, 9), date(2026, 3, 13)) == 5
