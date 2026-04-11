# =============================================================================
# data_collector.py
# Weekly SPX Range Prediction Model — Data Collection Module
#
# Pulls and stores historical SPX OHLC, VIX, and FRED macro data to Postgres
# (via range_finder.db.get_connection()) for use by downstream modules.
#
# Data Sources:
#   - yfinance  : SPX weekly OHLC, VIX weekly close
#   - FRED API  : 10Y Treasury yield, 2Y Treasury yield, Fed Funds rate
# =============================================================================

import math
import os
import logging
from datetime import datetime, timedelta, timezone

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

# How many years of history to pull on initial load
HISTORY_YEARS = 6

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

def init_db():
    """
    Connect to Postgres (via DATABASE_URL) and ensure all range-finder tables
    exist.

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


def save_spx_vix(conn, df: pd.DataFrame) -> int:
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


def save_fred_macro(conn, df: pd.DataFrame) -> int:
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
# EVENT FLAGS  (FOMC / CPI / NFP) — re-exported from event_calendars.py
# =============================================================================

from range_finder.event_calendars import (  # noqa: F401
    FOMC_DATES,
    CPI_DATES,
    NFP_DATES,
    _get_week_start,
    build_event_flags,
)


# =============================================================================
# WEEKLY UPDATE  (call this every Friday evening)
# =============================================================================

def update_weekly(conn) -> None:
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
    """Return float or None — psycopg2 doesn't like numpy scalars or pd.NA."""
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def get_weekly_spx(conn, limit: int = None) -> pd.DataFrame:
    """
    Convenience reader — returns weekly_spx as a DataFrame sorted by week_start.
    """
    query = "SELECT * FROM weekly_spx ORDER BY week_start ASC"
    params = None
    if limit:
        query += " LIMIT ?"
        params = (int(limit),)
    df = pd.read_sql_query(query, conn, params=params, parse_dates=["week_start", "week_end"])
    df.set_index("week_start", inplace=True)
    return df


def get_macro_daily(conn) -> pd.DataFrame:
    """Returns macro_daily as a DataFrame indexed by date."""
    df = pd.read_sql_query(
        "SELECT * FROM macro_daily ORDER BY date ASC", conn, parse_dates=["date"]
    )
    df.set_index("date", inplace=True)
    return df


def get_event_flags(conn) -> pd.DataFrame:
    """Returns event_flags as a DataFrame indexed by week_start."""
    df = pd.read_sql_query(
        "SELECT * FROM event_flags ORDER BY week_start ASC", conn, parse_dates=["week_start"]
    )
    df.set_index("week_start", inplace=True)
    return df


def print_summary(conn) -> None:
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
