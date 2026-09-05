# =============================================================================
# event_calendars.py
# Hardcoded FOMC, CPI, and NFP event dates plus event flag builder.
#
# Separated from data_collector.py for easier maintenance — these date
# lists are updated every year and have high churn.
# =============================================================================

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

log = logging.getLogger(__name__)


# Known 2016-2026 FOMC meeting dates (statement/decision days).
# 2016-2019 were backfilled (2026-07) for the history-depth experiment —
# the training window can now extend to ~10y without event_count silently
# reading 0 on pre-2020 weeks.
FOMC_DATES = [
    # 2016
    "2016-01-27","2016-03-16","2016-04-27","2016-06-15","2016-07-27",
    "2016-09-21","2016-11-02","2016-12-14",
    # 2017
    "2017-02-01","2017-03-15","2017-05-03","2017-06-14","2017-07-26",
    "2017-09-20","2017-11-01","2017-12-13",
    # 2018
    "2018-01-31","2018-03-21","2018-05-02","2018-06-13","2018-08-01",
    "2018-09-26","2018-11-08","2018-12-19",
    # 2019
    "2019-01-30","2019-03-20","2019-05-01","2019-06-19","2019-07-31",
    "2019-09-18","2019-10-30","2019-12-11",
    # 2020
    "2020-01-29","2020-03-03","2020-03-15","2020-04-29","2020-06-10",
    "2020-07-29","2020-09-16","2020-11-05","2020-12-16",
    # 2021
    "2021-01-27","2021-03-17","2021-04-28","2021-06-16","2021-07-28",
    "2021-09-22","2021-11-03","2021-12-15",
    # 2022
    "2022-01-26","2022-03-16","2022-05-04","2022-06-15","2022-07-27",
    "2022-09-21","2022-11-02","2022-12-14",
    # 2023
    "2023-02-01","2023-03-22","2023-05-03","2023-06-14","2023-07-26",
    "2023-09-20","2023-11-01","2023-12-13",
    # 2024
    "2024-01-31","2024-03-20","2024-05-01","2024-06-12","2024-07-31",
    "2024-09-18","2024-11-07","2024-12-18",
    # 2025
    "2025-01-29","2025-03-19","2025-05-07","2025-06-18","2025-07-30",
    "2025-09-17","2025-10-29","2025-12-10",
    # 2026 (decision days — second day of each scheduled meeting, per the
    # Fed's published 2026 calendar: Jan 27-28, Mar 17-18, Apr 28-29,
    # Jun 16-17, Jul 28-29, Sep 15-16, Oct 27-28, Dec 8-9)
    "2026-01-28","2026-03-18","2026-04-29","2026-06-17","2026-07-29",
    "2026-09-16","2026-10-28","2026-12-09",
]

CPI_DATES = [
    # 2016-2019: official monthly-release dates per the FRED release
    # calendar (release_id=10), verified 2026-07. The annual
    # seasonal-revision releases (e.g. 2017-02-13, 2019-02-11) are
    # deliberately excluded — only the 8:30 ET monthly prints move vol.
    # 2016
    "2016-01-20","2016-02-19","2016-03-16","2016-04-14","2016-05-17",
    "2016-06-16","2016-07-15","2016-08-16","2016-09-16","2016-10-18",
    "2016-11-17","2016-12-15",
    # 2017
    "2017-01-18","2017-02-15","2017-03-15","2017-04-14","2017-05-12",
    "2017-06-14","2017-07-14","2017-08-11","2017-09-14","2017-10-13",
    "2017-11-15","2017-12-13",
    # 2018
    "2018-01-12","2018-02-14","2018-03-13","2018-04-11","2018-05-10",
    "2018-06-12","2018-07-12","2018-08-10","2018-09-13","2018-10-11",
    "2018-11-14","2018-12-12",
    # 2019
    "2019-01-11","2019-02-13","2019-03-12","2019-04-10","2019-05-10",
    "2019-06-12","2019-07-11","2019-08-13","2019-09-12","2019-10-10",
    "2019-11-13","2019-12-11",
    # 2020
    "2020-01-14","2020-02-13","2020-03-11","2020-04-10","2020-05-12",
    "2020-06-10","2020-07-14","2020-08-12","2020-09-11","2020-10-13",
    "2020-11-12","2020-12-10",
    # 2021
    "2021-01-13","2021-02-10","2021-03-10","2021-04-13","2021-05-12",
    "2021-06-10","2021-07-13","2021-08-11","2021-09-14","2021-10-13",
    "2021-11-10","2021-12-10",
    # 2022
    "2022-01-12","2022-02-10","2022-03-10","2022-04-12","2022-05-11",
    "2022-06-10","2022-07-13","2022-08-10","2022-09-13","2022-10-13",
    "2022-11-10","2022-12-13",
    # 2023
    "2023-01-12","2023-02-14","2023-03-14","2023-04-12","2023-05-10",
    "2023-06-13","2023-07-12","2023-08-10","2023-09-13","2023-10-12",
    "2023-11-14","2023-12-12",
    # 2024
    "2024-01-11","2024-02-13","2024-03-12","2024-04-10","2024-05-15",
    "2024-06-12","2024-07-11","2024-08-14","2024-09-11","2024-10-10",
    "2024-11-13","2024-12-11",
    # 2025
    "2025-01-15","2025-02-12","2025-03-12","2025-04-10","2025-05-13",
    "2025-06-11","2025-07-15","2025-08-12","2025-09-10","2025-10-14",
    "2025-11-12","2025-12-10",
    # 2026 (BLS revised 2026 schedule — Dec'25 CPI landed Jan 13 and
    # Jan'26 CPI on Feb 13, not the originally-penciled Jan 14 / Feb 11)
    "2026-01-13","2026-02-13","2026-03-11","2026-04-10","2026-05-12",
    "2026-06-10","2026-07-14","2026-08-12","2026-09-11","2026-10-14",
    "2026-11-10","2026-12-10",
]

NFP_DATES = [
    # 2016-2019: official Employment Situation release dates per the FRED
    # release calendar (release_id=50), verified 2026-07.
    # 2016
    "2016-01-08","2016-02-05","2016-03-04","2016-04-01","2016-05-06",
    "2016-06-03","2016-07-08","2016-08-05","2016-09-02","2016-10-07",
    "2016-11-04","2016-12-02",
    # 2017
    "2017-01-06","2017-02-03","2017-03-10","2017-04-07","2017-05-05",
    "2017-06-02","2017-07-07","2017-08-04","2017-09-01","2017-10-06",
    "2017-11-03","2017-12-08",
    # 2018
    "2018-01-05","2018-02-02","2018-03-09","2018-04-06","2018-05-04",
    "2018-06-01","2018-07-06","2018-08-03","2018-09-07","2018-10-05",
    "2018-11-02","2018-12-07",
    # 2019
    "2019-01-04","2019-02-01","2019-03-08","2019-04-05","2019-05-03",
    "2019-06-07","2019-07-05","2019-08-02","2019-09-06","2019-10-04",
    "2019-11-01","2019-12-06",
    # 2020
    "2020-01-10","2020-02-07","2020-03-06","2020-04-03","2020-05-08",
    "2020-06-05","2020-07-02","2020-08-07","2020-09-04","2020-10-02",
    "2020-11-06","2020-12-04",
    # 2021
    "2021-01-08","2021-02-05","2021-03-05","2021-04-02","2021-05-07",
    "2021-06-04","2021-07-02","2021-08-06","2021-09-03","2021-10-08",
    "2021-11-05","2021-12-03",
    # 2022
    "2022-01-07","2022-02-04","2022-03-04","2022-04-01","2022-05-06",
    "2022-06-03","2022-07-08","2022-08-05","2022-09-02","2022-10-07",
    "2022-11-04","2022-12-02",
    # 2023
    "2023-01-06","2023-02-03","2023-03-10","2023-04-07","2023-05-05",
    "2023-06-02","2023-07-07","2023-08-04","2023-09-01","2023-10-06",
    "2023-11-03","2023-12-08",
    # 2024
    "2024-01-05","2024-02-02","2024-03-08","2024-04-05","2024-05-03",
    "2024-06-07","2024-07-05","2024-08-02","2024-09-06","2024-10-04",
    "2024-11-01","2024-12-06",
    # 2025
    "2025-01-10","2025-02-07","2025-03-07","2025-04-04","2025-05-02",
    "2025-06-06","2025-07-03","2025-08-01","2025-09-05","2025-10-03",
    "2025-11-07","2025-12-05",
    # 2026 (Jan'26 jobs report was released Wed Feb 11 — schedule shifted
    # after the late-2025 data disruption — then back to first-Friday
    # cadence; Jun'26 data lands Thu Jul 2 ahead of the July-4th holiday)
    "2026-01-09","2026-02-11","2026-03-06","2026-04-03","2026-05-08",
    "2026-06-05","2026-07-02","2026-08-07","2026-09-04","2026-10-02",
    "2026-11-06","2026-12-04",
]


# ── Tier-2 calendars (display/warning only — NEVER model features) ──────────
# PPI and PCE deliberately do not feed event_flags / model_features: the HAR
# event features and spread buffer multipliers stay tier-1 (FOMC/CPI/NFP/OpEx)
# only, so these lists start at 2026 rather than backfilling to 2016. They
# stay live through _EVENT_SOURCES, which drives the Spread Finder's
# calendar-staleness banner. Extend annually like the lists above.
PPI_DATES = [
    # 2026 BLS schedule (verified 2026-07 via bls.gov/schedule/news_release/
    # ppi.htm + the release archives ppi_02272026/ppi_04142026/ppi_05132026/
    # ppi_06112026). 13 calendar-2026 releases: the late-2025 data disruption
    # folded Oct'25 data into the Nov'25 print (Jan 14) and the cadence
    # caught up over Jan-Feb. All 8:30 ET.
    "2026-01-14","2026-01-30","2026-02-27","2026-03-18","2026-04-14",
    "2026-05-13","2026-06-11","2026-07-15","2026-08-13","2026-09-10",
    "2026-10-15","2026-11-13","2026-12-15",
]

PCE_DATES = [
    # 2026 Personal Income & Outlays (PCE price index) releases, verified
    # 2026-07 via bea.gov/news/schedule (all 8:30 ET). BEA does not list
    # past dates retroactively, so the list starts at the June print.
    "2026-06-25","2026-07-30","2026-08-26","2026-09-30","2026-10-29",
    "2026-11-25","2026-12-23",
]

def _get_week_start(date_str: str) -> str:
    """Given any date string, return the Monday of that week as ISO string."""
    dt = pd.to_datetime(date_str)
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def _warn_if_calendar_stale(name: str, dates: list[str], horizon_days: int = 45) -> None:
    """Log loudly when a hardcoded calendar is close to running out.

    These lists need a manual refresh every year; when they lapse, event
    flags silently read 0 for upcoming weeks and the spread buffer stops
    widening for FOMC/CPI/NFP — exactly the weeks where it matters most.
    """
    if not dates:
        return
    last = pd.to_datetime(max(dates))
    days_left = (last - pd.Timestamp(datetime.today().date())).days
    if days_left < horizon_days:
        log.warning(
            f"{name} calendar runs out on {last.date()} ({days_left} days away) — "
            f"extend range_finder/event_calendars.py with the next published dates "
            f"or upcoming event weeks will be flagged as event-free."
        )


def _third_friday(year: int, month: int) -> str:
    """Monthly OpEx date: the 3rd Friday of the given month, ISO string.

    When the 3rd Friday is an exchange holiday (Good Friday is the recurring
    case — April OpEx has landed on it), index options roll to the prior
    trading session, so the effective OpEx is Thursday. Rolls back off any
    NYSE holiday; degrades to the raw 3rd Friday if the calendar is
    unavailable (offline / import failure)."""
    first_day = datetime(year, month, 1)
    first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
    third = first_friday + timedelta(weeks=2)
    try:
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("XNYS")
        sched = nyse.schedule(
            start_date=(third - timedelta(days=4)).strftime("%Y-%m-%d"),
            end_date=third.strftime("%Y-%m-%d"))
        sessions = [d.date() for d in sched.index]
        if sessions and third.date() not in sessions:
            return sessions[-1].strftime("%Y-%m-%d")  # last session ≤ 3rd Fri
    except Exception:
        pass
    return third.strftime("%Y-%m-%d")


def calendar_staleness_warnings(horizon_days: int = 45) -> list[str]:
    """UI-facing companion to _warn_if_calendar_stale: return one message per
    hardcoded macro calendar that runs out within horizon_days, so a banner
    can surface what the log-only warning hides on Streamlit Cloud. Empty
    list == every calendar has runway."""
    out: list[str] = []
    today = pd.Timestamp(datetime.today().date())
    for name, dates in _EVENT_SOURCES.items():
        if not dates:
            continue
        last = pd.to_datetime(max(dates))
        days_left = (last - today).days
        if days_left < horizon_days:
            out.append(
                f"{name.upper()} calendar ends {last.date()} ({days_left}d away) — "
                f"upcoming {name.upper()} weeks will flag as event-free until "
                f"range_finder/event_calendars.py is extended.")
    return out


_EVENT_SOURCES = {  # name → hardcoded date list (opex is computed, not listed)
    "fomc": FOMC_DATES,
    "cpi": CPI_DATES,
    "nfp": NFP_DATES,
    "ppi": PPI_DATES,
    "pce": PCE_DATES,
}

# events_for_week() (plus EventInfo / EVENT_TIERS / RELEASE_TIMES_ET) lived
# here to build the Monday Cockpit and Pre-Flight event strips. Both tabs
# were removed 2026-08, so the per-event read model went with them. The two
# live entry points are build_event_flags (tier-1 FOMC/CPI/NFP/OpEx → the
# model's event_flags) and calendar_staleness_warnings (all five lists → the
# Spread Finder banner), which is what keeps PPI/PCE earning their place.


def build_event_flags(conn) -> int:
    """
    Generate event flag rows for all weeks that have FOMC, CPI, or NFP events.
    Opex (monthly options expiration = 3rd Friday) is computed programmatically.

    Returns number of rows upserted.
    """
    now = datetime.now(timezone.utc).isoformat()

    _warn_if_calendar_stale("FOMC", FOMC_DATES)
    _warn_if_calendar_stale("CPI", CPI_DATES)
    _warn_if_calendar_stale("NFP", NFP_DATES)

    flags = event_flag_rows()

    # Upsert into DB
    cur = conn.cursor()
    rows_written = 0
    for ws, f in flags.items():
        event_count = f["has_fomc"] + f["has_cpi"] + f["has_nfp"] + f["has_opex"]
        cur.execute("""
            INSERT INTO event_flags (
                week_start, has_fomc, has_cpi, has_nfp, has_opex, event_count, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(week_start) DO UPDATE SET
                has_fomc    = excluded.has_fomc,
                has_cpi     = excluded.has_cpi,
                has_nfp     = excluded.has_nfp,
                has_opex    = excluded.has_opex,
                event_count = excluded.event_count,
                updated_at  = excluded.updated_at
        """, (ws, f["has_fomc"], f["has_cpi"], f["has_nfp"], f["has_opex"], event_count, now))
        rows_written += 1

    conn.commit()
    log.info(f"Event flags: {rows_written} weeks flagged")
    return rows_written


def event_flag_rows(as_of=None) -> dict:
    """Pure calendar inputs shared by persistence and the forward tester."""
    # Map week_start → flags
    flags: dict[str, dict] = {}

    def mark(date_str: str, field: str):
        ws = _get_week_start(date_str)
        if ws not in flags:
            flags[ws] = {"has_fomc": 0, "has_cpi": 0, "has_nfp": 0, "has_opex": 0}
        flags[ws][field] = 1

    for d in FOMC_DATES:
        mark(d, "has_fomc")
    for d in CPI_DATES:
        mark(d, "has_cpi")
    for d in NFP_DATES:
        mark(d, "has_nfp")

    # Monthly opex: 3rd Friday of each month, 2016 through the END of next
    # year (2016+ matches the backfilled FOMC/CPI/NFP calendars above).
    # 3rd Fridays are deterministic, so flagging future weeks is
    # exact — the old `third_friday <= today` filter meant the UPCOMING
    # opex week (the one the HAR forecast and buffer logic actually care
    # about) was never flagged until after it had already passed.
    today = as_of or datetime.today()
    for year in range(2016, today.year + 2):
        for month in range(1, 13):
            mark(_third_friday(year, month), "has_opex")

    for f in flags.values():
        f["event_count"] = sum(f.values())
    return flags
