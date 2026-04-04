from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd
import pandas_market_calendars as mcal

from phase1.config import (
    NY_TZ,
    CASH_CALENDAR,
    OPTIONS_CALENDAR,
    USE_TRADING_TIME,
    TRADING_HOURS_PER_YEAR,
)


@dataclass(frozen=True)
class SessionState:
    calendar_name: str
    now_ny: datetime
    is_open: bool
    market_open: pd.Timestamp | None
    market_close: pd.Timestamp | None


def now_ny() -> datetime:
    return datetime.now(NY_TZ)


# --- Fix #11: Cache calendar objects and schedule lookups ---

@lru_cache(maxsize=8)
def get_calendar(calendar_name: str):
    return mcal.get_calendar(calendar_name)


_schedule_cache: dict[tuple[str, str, str], pd.DataFrame] = {}


def get_schedule(calendar_name: str, start_date: str, end_date: str) -> pd.DataFrame:
    key = (calendar_name, start_date, end_date)
    if key not in _schedule_cache:
        cal = get_calendar(calendar_name)
        _schedule_cache[key] = cal.schedule(start_date=start_date, end_date=end_date)
    return _schedule_cache[key]


def clear_schedule_cache():
    """Call between runs if calendar data may have changed."""
    _schedule_cache.clear()


def get_session_state(calendar_name: str, ts: datetime | None = None) -> SessionState:
    ts = ts or now_ny()
    ts_ny = ts.astimezone(NY_TZ)

    day = ts_ny.date().isoformat()
    sched = get_schedule(calendar_name, day, day)

    if sched.empty:
        return SessionState(
            calendar_name=calendar_name,
            now_ny=ts_ny,
            is_open=False,
            market_open=None,
            market_close=None,
        )

    market_open = sched.iloc[0]["market_open"]
    market_close = sched.iloc[0]["market_close"]

    is_open = market_open <= pd.Timestamp(ts_ny) < market_close

    return SessionState(
        calendar_name=calendar_name,
        now_ny=ts_ny,
        is_open=is_open,
        market_open=market_open,
        market_close=market_close,
    )


def is_cash_market_open(ts: datetime | None = None) -> bool:
    return get_session_state(CASH_CALENDAR, ts).is_open


def is_options_market_open(ts: datetime | None = None) -> bool:
    return get_session_state(OPTIONS_CALENDAR, ts).is_open


def get_calendar_snapshot(ts: datetime | None = None) -> dict:
    ts_ny = (ts or now_ny()).astimezone(NY_TZ)

    cash_state = get_session_state(CASH_CALENDAR, ts_ny)
    options_state = get_session_state(OPTIONS_CALENDAR, ts_ny)

    return {
        "now_ny": ts_ny.isoformat(),
        "cash_calendar": CASH_CALENDAR,
        "options_calendar": OPTIONS_CALENDAR,
        "cash_market_open": cash_state.is_open,
        "options_market_open": options_state.is_open,
        "cash_market_open_time": cash_state.market_open.isoformat() if cash_state.market_open is not None else None,
        "cash_market_close_time": cash_state.market_close.isoformat() if cash_state.market_close is not None else None,
        "options_market_open_time": options_state.market_open.isoformat() if options_state.market_open is not None else None,
        "options_market_close_time": options_state.market_close.isoformat() if options_state.market_close is not None else None,
    }

def _ensure_ny(ts: datetime) -> datetime:
    return ts.astimezone(NY_TZ) if ts.tzinfo is not None else ts.replace(tzinfo=NY_TZ)


def get_expiration_close_dt(expiration_str: str, calendar_name: str = OPTIONS_CALENDAR) -> datetime:
    """
    Return the scheduled market close in New York time for a given expiration date.

    If the expiration date falls on a holiday (schedule is empty), we look backward
    to find the previous trading day's close — that's the actual last trading time
    for options expiring on the holiday. This correctly handles Good Friday, etc.
    """
    sched = get_schedule(calendar_name, expiration_str, expiration_str)

    if not sched.empty:
        close_ts = sched.iloc[0]["market_close"]
        close_dt = close_ts.to_pydatetime() if hasattr(close_ts, "to_pydatetime") else close_ts
        return _ensure_ny(close_dt)

    # Expiration falls on a holiday — find the previous trading day's close.
    # Options expiring on Good Friday, for example, effectively expire at
    # Thursday's market close.
    exp_date = datetime.strptime(expiration_str, "%Y-%m-%d").date()
    lookback_start = (exp_date - timedelta(days=7)).isoformat()
    lookback_sched = get_schedule(calendar_name, lookback_start, expiration_str)

    if not lookback_sched.empty:
        close_ts = lookback_sched.iloc[-1]["market_close"]
        close_dt = close_ts.to_pydatetime() if hasattr(close_ts, "to_pydatetime") else close_ts
        return _ensure_ny(close_dt)

    # Ultimate fallback if no trading days found in lookback window
    return datetime.strptime(expiration_str, "%Y-%m-%d").replace(
        tzinfo=NY_TZ,
        hour=16,
        minute=0,
        second=0,
        microsecond=0,
    )


# --- Fix #10: Trading-time T ---

def _compute_trading_hours_to_expiry(
    ts_ny: datetime,
    close_dt: datetime,
    calendar_name: str,
) -> float:
    """
    Count only trading hours between now and expiration close.

    For each session in the calendar between ts and close_dt:
    - Current session: remaining hours from ts to session close
    - Middle sessions: full session hours (open to close)
    - Expiration session: open to close_dt

    Returns total trading hours.
    """
    if close_dt <= ts_ny:
        return 0.0

    start_date = ts_ny.date().isoformat()
    end_date = close_dt.date().isoformat()

    sched = get_schedule(calendar_name, start_date, end_date)

    if sched.empty:
        # Fallback: estimate as calendar time assuming 6.5 hr/day
        cal_days = max((close_dt - ts_ny).total_seconds() / 86400.0, 0.0)
        return cal_days * 6.5 / 7.0 * 5.0  # rough weekday adjustment

    total_hours = 0.0
    ts_pd = pd.Timestamp(ts_ny)
    close_pd = pd.Timestamp(close_dt)

    for _, row in sched.iterrows():
        sess_open = row["market_open"]
        sess_close = row["market_close"]

        # Effective start: max(ts, session open)
        eff_start = max(ts_pd, sess_open)
        # Effective end: min(close_dt, session close)
        eff_end = min(close_pd, sess_close)

        if eff_end > eff_start:
            total_hours += (eff_end - eff_start).total_seconds() / 3600.0

    return max(total_hours, 0.0)


def compute_time_to_expiry_years(
    expiration_str: str,
    ts: datetime | None = None,
    calendar_name: str = OPTIONS_CALENDAR,
    floor: float | None = None,
):
    """
    Compute time to expiry in years.

    If USE_TRADING_TIME is True (recommended for 0DTE accuracy), counts only
    trading hours between now and expiration close and converts using
    TRADING_HOURS_PER_YEAR.

    Otherwise, falls back to calendar-time computation.

    Returns:
        (T_years, expiration_close_dt_ny)
    """
    ts_ny = _ensure_ny(ts or now_ny())
    close_dt = get_expiration_close_dt(expiration_str, calendar_name=calendar_name)

    if USE_TRADING_TIME:
        trading_hours = _compute_trading_hours_to_expiry(ts_ny, close_dt, calendar_name)
        years = trading_hours / TRADING_HOURS_PER_YEAR
    else:
        years = max((close_dt - ts_ny).total_seconds() / (365.25 * 24 * 3600), 0.0)

    if floor is not None:
        years = max(years, floor)

    return years, close_dt
