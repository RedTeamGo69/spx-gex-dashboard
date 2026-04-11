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


# Known 2020-2026 FOMC meeting dates
FOMC_DATES = [
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
    # 2026
    "2026-01-28","2026-03-18","2026-04-29","2026-06-17",
]

CPI_DATES = [
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
    # 2026
    "2026-01-14","2026-02-11","2026-03-11","2026-04-10",
]

NFP_DATES = [
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
    # 2026
    "2026-01-09","2026-02-06","2026-03-06","2026-04-03",
]


def _get_week_start(date_str: str) -> str:
    """Given any date string, return the Monday of that week as ISO string."""
    dt = pd.to_datetime(date_str)
    monday = dt - timedelta(days=dt.weekday())
    return monday.strftime("%Y-%m-%d")


def build_event_flags(conn) -> int:
    """
    Generate event flag rows for all weeks that have FOMC, CPI, or NFP events.
    Opex (monthly options expiration = 3rd Friday) is computed programmatically.

    Returns number of rows upserted.
    """
    now = datetime.now(timezone.utc).isoformat()

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

    # Monthly opex: 3rd Friday of each month from 2020 to present
    year = 2020
    today = datetime.today()
    while year <= today.year:
        for month in range(1, 13):
            first_day = datetime(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)
            if third_friday <= today:
                mark(third_friday.strftime("%Y-%m-%d"), "has_opex")
        year += 1

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
