"""Exchange-session boundaries shared by the finder and the forward study."""
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


@dataclass(frozen=True)
class Session:
    day: date
    open: datetime
    close: datetime


@dataclass(frozen=True)
class TradingWeek:
    monday: date
    sessions: tuple[Session, ...]

    @property
    def capture_start(self):
        return self.sessions[0].open + timedelta(minutes=15)

    @property
    def capture_end(self):
        return self.sessions[0].open + timedelta(minutes=45)

    @property
    def evaluation_close(self):
        return self.sessions[-1].close

    def admits(self, now: datetime) -> bool:
        return self.capture_start <= now < self.capture_end


@lru_cache(maxsize=512)
def trading_week(day: date) -> TradingWeek:
    monday = day - timedelta(days=day.weekday())
    schedule = mcal.get_calendar("NYSE").schedule(
        start_date=monday, end_date=monday + timedelta(days=4))
    sessions = tuple(Session(idx.date(), row.market_open.to_pydatetime(),
                             row.market_close.to_pydatetime())
                     for idx, row in schedule.iterrows())
    if not sessions:
        raise ValueError(f"No exchange sessions in week {monday}")
    return TradingWeek(monday, sessions)


def listed_week_expiration(available: list[str], week: TradingWeek) -> str | None:
    """Require the actual final session, never a nearby next-week contract."""
    target = week.sessions[-1].day.isoformat()
    return target if target in available else None
