# =============================================================================
# data_collector.py
# Weekly SPX Range Prediction Model — Data Collection Module
#
# Pulls and stores historical SPX OHLC, VIX, and FRED macro data.
# All data is persisted to SQLite for use by downstream modules.
#
# Data Sources:
#   - yfinance  : SPX weekly OHLC, VIX weekly close
#   - FRED API  : 10Y Treasury yield, 2Y Treasury yield, Fed Funds rate
#   - SQLite    : local storage (weekly_data.db)
# =============================================================================

import math
import os
import sqlite3
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# =============================================================================
# CONFIG
# =============================================================================

# Read FRED key from Streamlit secrets or environment
FRED_API_KEY = ""
try:
    import streamlit as st
    FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")
except Exception:
    pass
if not FRED_API_KEY:
    FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# Database location — sits in the range_finder directory
DB_PATH = Path(__file__).parent / "weekly_data.db"

# How many years of history to pull on initial load
HISTORY_YEARS = 5

# FRED series used as macro features
FRED_SERIES = {
    "treasury_10y": "DGS10",       # 10-Year Treasury Constant Maturity Rate
    "treasury_2y":  "DGS2",        # 2-Year Treasury Constant Maturity Rate
    "fed_funds":    "DFF",         # Federal Funds Effective Rate
}

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# DATABASE SETUP
# =============================================================================

def init_db(db_path: Path = DB_PATH):
    """
    Create (or connect to) the database and initialize the schema.
    Prefers Postgres (via DATABASE_URL) and falls back to SQLite.

    Tables:
        weekly_spx  — SPX and VIX weekly OHLC + derived range metrics
        macro_daily — Daily FRED macro series (joined to weekly during feature build)
        event_flags — Manual or scraped FOMC/CPI/NFP week flags
    """
    from range_finder.db import get_connection, init_all_tables
    conn = get_connection()
    init_all_tables(conn)
    return conn


# =============================================================================
# SPX + VIX DATA
# =============================================================================

def fetch_spx_vix(years: int = HISTORY_YEARS) -> pd.DataFrame:
    """
    Pull weekly SPX and VIX OHLC from yfinance.

    yfinance weekly bars run Monday open → Friday close.
    VIX close is aligned to the same week as SPX — in the feature builder
    this gets lagged by one week (you observe Friday's VIX before next week opens).

    Returns a single DataFrame with prefixed columns: spx_*, vix_*
    """
    end   = datetime.today()
    start = end - timedelta(weeks=years * 52 + 4)   # small buffer for alignment

    log.info(f"Fetching SPX weekly OHLC from {start.date()} to {end.date()}")
    spx_raw = yf.download("^GSPC", start=start, end=end, interval="1wk", progress=False, timeout=60)

    if spx_raw.empty:
        raise RuntimeError("yfinance returned empty SPX data — market may be closed or network issue")

    log.info(f"Fetching VIX weekly OHLC from {start.date()} to {end.date()}")
    vix_raw = yf.download("^VIX", start=start, end=end, interval="1wk", progress=False, timeout=60)

    # yfinance sometimes returns a MultiIndex — flatten it
    if isinstance(spx_raw.columns, pd.MultiIndex):
        spx_raw.columns = spx_raw.columns.get_level_values(0)
    if isinstance(vix_raw.columns, pd.MultiIndex):
        vix_raw.columns = vix_raw.columns.get_level_values(0)

    # Rename and prefix
    spx = spx_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    spx.columns = ["spx_open", "spx_high", "spx_low", "spx_close", "spx_volume"]

    vix = vix_raw[["Open", "High", "Low", "Close"]].copy()
    vix.columns = ["vix_open", "vix_high", "vix_low", "vix_close"]

    # Merge on date index
    df = spx.join(vix, how="inner")
    df.index.name = "week_start"
    df.index = pd.to_datetime(df.index).normalize()   # strip time component

    # Derived range metrics — BUG FIX: use math.log directly instead of inline __import__
    df["range_pts"]   = df["spx_high"] - df["spx_low"]
    df["range_pct"]   = df["range_pts"] / df["spx_open"]
    df["log_range"]   = df["range_pct"].apply(lambda x: pd.NA if x <= 0 else math.log(x))
    df["spx_return"]  = (df["spx_close"] - df["spx_open"]) / df["spx_open"]

    # Approximate week_end (Friday = Monday + 4 days)
    df["week_end"] = df.index + timedelta(days=4)

    df.dropna(subset=["spx_open", "spx_close", "range_pct"], inplace=True)

    log.info(f"SPX/VIX: {len(df)} weekly rows collected")
    return df


def save_spx_vix(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """
    Upsert weekly SPX/VIX rows into the database.
    Returns the number of rows written.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows_written = 0

    cur = conn.cursor()
    for week_start, row in df.iterrows():
        cur.execute("""
            INSERT INTO weekly_spx (
                week_start, week_end,
                spx_open, spx_high, spx_low, spx_close, spx_volume,
                vix_open, vix_high, vix_low, vix_close,
                range_pts, range_pct, log_range, spx_return,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(week_start) DO UPDATE SET
                week_end    = excluded.week_end,
                spx_open    = excluded.spx_open,
                spx_high    = excluded.spx_high,
                spx_low     = excluded.spx_low,
                spx_close   = excluded.spx_close,
                spx_volume  = excluded.spx_volume,
                vix_open    = excluded.vix_open,
                vix_high    = excluded.vix_high,
                vix_low     = excluded.vix_low,
                vix_close   = excluded.vix_close,
                range_pts   = excluded.range_pts,
                range_pct   = excluded.range_pct,
                log_range   = excluded.log_range,
                spx_return  = excluded.spx_return,
                updated_at  = excluded.updated_at
        """, (
            week_start.strftime("%Y-%m-%d"),
            row["week_end"].strftime("%Y-%m-%d") if pd.notna(row["week_end"]) else None,
            _safe(row, "spx_open"),   _safe(row, "spx_high"),
            _safe(row, "spx_low"),    _safe(row, "spx_close"),
            _safe(row, "spx_volume"),
            _safe(row, "vix_open"),   _safe(row, "vix_high"),
            _safe(row, "vix_low"),    _safe(row, "vix_close"),
            _safe(row, "range_pts"),  _safe(row, "range_pct"),
            _safe(row, "log_range"),  _safe(row, "spx_return"),
            now,
        ))
        rows_written += 1

    conn.commit()
    log.info(f"SPX/VIX: {rows_written} rows upserted into weekly_spx")
    return rows_written


# =============================================================================
# FRED MACRO DATA
# =============================================================================

def fetch_fred_macro(years: int = HISTORY_YEARS) -> pd.DataFrame:
    """
    Pull daily macro series from FRED.

    Series pulled:
        DGS10  — 10-Year Treasury yield
        DGS2   — 2-Year Treasury yield
        DFF    — Federal Funds Effective Rate

    Yield spread (10y - 2y) is computed here. The feature builder will
    resample this to weekly frequency and align it to SPX weeks.
    """
    from fredapi import Fred

    fred  = Fred(api_key=FRED_API_KEY)
    end   = datetime.today()
    start = end - timedelta(days=years * 365 + 30)

    frames = {}
    for col_name, series_id in FRED_SERIES.items():
        log.info(f"Fetching FRED series: {series_id} ({col_name})")
        s = fred.get_series(series_id, observation_start=start, observation_end=end)
        frames[col_name] = s

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df.index = pd.to_datetime(df.index).normalize()

    # Forward-fill small gaps (FRED has occasional missing business days)
    df.ffill(inplace=True)
    df.dropna(inplace=True)

    # Derived feature
    df["yield_spread"] = df["treasury_10y"] - df["treasury_2y"]

    log.info(f"FRED macro: {len(df)} daily rows collected")
    return df


def save_fred_macro(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """
    Upsert daily FRED macro rows into the database.
    Returns the number of rows written.
    """
    now = datetime.now(timezone.utc).isoformat()
    rows_written = 0

    cur = conn.cursor()
    for date, row in df.iterrows():
        cur.execute("""
            INSERT INTO macro_daily (
                date, treasury_10y, treasury_2y, yield_spread, fed_funds, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(date) DO UPDATE SET
                treasury_10y  = excluded.treasury_10y,
                treasury_2y   = excluded.treasury_2y,
                yield_spread  = excluded.yield_spread,
                fed_funds     = excluded.fed_funds,
                updated_at    = excluded.updated_at
        """, (
            date.strftime("%Y-%m-%d"),
            _safe(row, "treasury_10y"),
            _safe(row, "treasury_2y"),
            _safe(row, "yield_spread"),
            _safe(row, "fed_funds"),
            now,
        ))
        rows_written += 1

    conn.commit()
    log.info(f"FRED macro: {rows_written} rows upserted into macro_daily")
    return rows_written


# =============================================================================
# EVENT FLAGS  (FOMC / CPI / NFP)
# =============================================================================

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


def build_event_flags(conn: sqlite3.Connection) -> int:
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


# =============================================================================
# WEEKLY UPDATE  (call this every Friday evening)
# =============================================================================

def update_weekly(conn: sqlite3.Connection) -> None:
    """
    Incremental update — pulls the last 8 weeks of data and upserts.
    Use this every Friday after market close instead of the full initial load.
    """
    log.info("Running incremental weekly update...")
    df_spx = fetch_spx_vix(years=0.2)    # ~10 weeks
    save_spx_vix(conn, df_spx)
    df_macro = fetch_fred_macro(years=0.2)
    save_fred_macro(conn, df_macro)
    build_event_flags(conn)
    log.info("Weekly update complete.")


# =============================================================================
# UTILITY
# =============================================================================

def _safe(row: pd.Series, col: str):
    """Return float or None — SQLite doesn't like numpy scalars or pd.NA."""
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def get_weekly_spx(conn: sqlite3.Connection, limit: int = None) -> pd.DataFrame:
    """
    Convenience reader — returns weekly_spx as a DataFrame sorted by week_start.
    """
    query = "SELECT * FROM weekly_spx ORDER BY week_start ASC"
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, conn, parse_dates=["week_start", "week_end"])
    df.set_index("week_start", inplace=True)
    return df


def get_macro_daily(conn: sqlite3.Connection) -> pd.DataFrame:
    """Returns macro_daily as a DataFrame indexed by date."""
    df = pd.read_sql_query(
        "SELECT * FROM macro_daily ORDER BY date ASC", conn, parse_dates=["date"]
    )
    df.set_index("date", inplace=True)
    return df


def get_event_flags(conn: sqlite3.Connection) -> pd.DataFrame:
    """Returns event_flags as a DataFrame indexed by week_start."""
    df = pd.read_sql_query(
        "SELECT * FROM event_flags ORDER BY week_start ASC", conn, parse_dates=["week_start"]
    )
    df.set_index("week_start", inplace=True)
    return df


def print_summary(conn: sqlite3.Connection) -> None:
    """Print a quick data health summary to console."""
    cur = conn.cursor()

    spx_count = cur.execute("SELECT COUNT(*) FROM weekly_spx").fetchone()[0]
    spx_range = cur.execute(
        "SELECT MIN(week_start), MAX(week_start) FROM weekly_spx"
    ).fetchone()

    macro_count = cur.execute("SELECT COUNT(*) FROM macro_daily").fetchone()[0]
    event_count = cur.execute(
        "SELECT SUM(event_count) FROM event_flags"
    ).fetchone()[0]

    print("\n" + "=" * 55)
    print("  DATA COLLECTOR — DATABASE SUMMARY")
    print("=" * 55)
    print(f"  weekly_spx  : {spx_count:>5} rows  ({spx_range[0]} → {spx_range[1]})")
    print(f"  macro_daily : {macro_count:>5} rows")
    print(f"  event_flags : {event_count:>5} total event-weeks flagged")
    print("=" * 55 + "\n")
