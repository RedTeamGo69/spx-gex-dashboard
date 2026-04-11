"""
Historical GEX snapshot tracking — Postgres only.

DATABASE_URL must be set via Streamlit secrets or as an environment variable.
psycopg2 must be installed. If either is missing the module raises a clear
error at import / first-call time rather than silently degrading.
"""
from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
_logger = logging.getLogger(__name__)


# ── Connection string resolution ──

_pg_conn_str = None

try:
    import streamlit as st
    _pg_conn_str = st.secrets.get("DATABASE_URL", "")
except Exception:
    pass

if not _pg_conn_str:
    _pg_conn_str = os.environ.get("DATABASE_URL", "")


def _require_postgres():
    """Raise a clear error if DATABASE_URL is missing or psycopg2 is unavailable."""
    if not _pg_conn_str:
        raise RuntimeError(
            "DATABASE_URL is not set. This app requires Postgres — set DATABASE_URL "
            "in Streamlit secrets or as an environment variable."
        )
    try:
        import psycopg2  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "psycopg2 is not installed. This app requires Postgres — "
            "`pip install psycopg2-binary`."
        ) from e


# ── Postgres helpers ──

def _pg_get_connection():
    _require_postgres()
    import psycopg2
    conn = psycopg2.connect(_pg_conn_str, sslmode="require")
    conn.autocommit = True
    return conn


def _pg_ensure_table():
    conn = _pg_get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS gex_snapshots (
                id SERIAL PRIMARY KEY,
                timestamp TEXT NOT NULL,
                date TEXT NOT NULL,
                minute_key TEXT NOT NULL,
                ticker TEXT NOT NULL DEFAULT 'SPX',
                spot REAL NOT NULL,
                zero_gamma REAL NOT NULL,
                is_true_crossing BOOLEAN NOT NULL DEFAULT TRUE,
                call_wall REAL,
                put_wall REAL,
                regime TEXT,
                net_gex REAL,
                expected_move_pts REAL,
                confidence_score REAL,
                freshness_score REAL,
                coverage_ratio REAL,
                pc_ratio REAL,
                gex_ratio REAL,
                call_iv REAL,
                put_iv REAL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_snapshots_date ON gex_snapshots(date)
        """)
        # Migration: add ticker column if missing (existing tables)
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE gex_snapshots ADD COLUMN ticker TEXT NOT NULL DEFAULT 'SPX';
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """)
        # Migration: drop old unique constraint on minute_key alone, add composite
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE gex_snapshots DROP CONSTRAINT IF EXISTS gex_snapshots_minute_key_key;
            EXCEPTION WHEN undefined_object THEN NULL;
            END $$
        """)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_ticker_minute
            ON gex_snapshots(ticker, minute_key)
        """)
    finally:
        conn.close()


def _pg_save_snapshot(row):
    conn = _pg_get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO gex_snapshots
               (timestamp, date, minute_key, ticker, spot, zero_gamma, is_true_crossing,
                call_wall, put_wall, regime, net_gex, expected_move_pts,
                confidence_score, freshness_score, coverage_ratio,
                pc_ratio, gex_ratio, call_iv, put_iv)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (ticker, minute_key) DO NOTHING""",
            (
                row["timestamp"], row["date"], row["minute_key"], row["ticker"],
                row["spot"], row["zero_gamma"], row["is_true_crossing"],
                row["call_wall"], row["put_wall"], row["regime"],
                row["net_gex"], row["expected_move_pts"],
                row["confidence_score"], row["freshness_score"],
                row["coverage_ratio"], row["pc_ratio"], row["gex_ratio"],
                row["call_iv"], row["put_iv"],
            ),
        )
    finally:
        conn.close()


def _pg_get_daily_summary(days, ticker="SPX"):
    """Return first (open) and last (close) snapshot per day for a given ticker."""
    conn = _pg_get_connection()
    try:
        cutoff = (datetime.now(NY_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
        cur = conn.cursor()
        # Get first and last snapshot id per day
        cur.execute(
            """SELECT * FROM gex_snapshots
               WHERE id IN (
                   SELECT MIN(id) FROM gex_snapshots WHERE date >= %s AND ticker = %s GROUP BY date
                   UNION
                   SELECT MAX(id) FROM gex_snapshots WHERE date >= %s AND ticker = %s GROUP BY date
               )
               ORDER BY date DESC, id ASC""",
            (cutoff, ticker, cutoff, ticker),
        )
        cols = [desc[0] for desc in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]

        # Tag each row as 'open' or 'close'
        from itertools import groupby
        tagged = []
        for _date, group in groupby(rows, key=lambda r: r["date"]):
            group_list = list(group)
            if len(group_list) == 1:
                group_list[0]["scan_type"] = "open"
                tagged.append(group_list[0])
            else:
                group_list[0]["scan_type"] = "open"
                for mid in group_list[1:-1]:
                    mid["scan_type"] = "intraday"
                group_list[-1]["scan_type"] = "close"
                tagged.extend(group_list)
        return tagged
    finally:
        conn.close()


def _pg_get_zero_gamma_trend(days, ticker="SPX"):
    conn = _pg_get_connection()
    try:
        cutoff = (datetime.now(NY_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
        cur = conn.cursor()
        cur.execute(
            """SELECT date, zero_gamma, spot FROM gex_snapshots
               WHERE date >= %s AND ticker = %s
               ORDER BY timestamp ASC""",
            (cutoff, ticker),
        )
        return cur.fetchall()
    finally:
        conn.close()


def _pg_get_history(days, ticker="SPX"):
    conn = _pg_get_connection()
    try:
        cutoff = (datetime.now(NY_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM gex_snapshots
               WHERE date >= %s AND ticker = %s
               ORDER BY timestamp DESC""",
            (cutoff, ticker),
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


# ── Initialize Postgres table on import (best-effort; clear error if misconfigured) ──

try:
    _pg_ensure_table()
except Exception as e:
    _logger.warning(f"Failed to initialize Postgres table on import: {e}")


# ── Public API ──

def get_backend():
    """Legacy compatibility shim. Always returns 'postgres' now."""
    return "postgres"


def _to_float(v):
    """Convert numpy/pandas numeric types to plain Python float for psycopg2."""
    if v is None:
        return None
    return float(v)


def _build_row(spot, levels, regime_info, stats, confidence_info, staleness_info, em_analysis=None, ticker="SPX"):
    now = datetime.now(NY_TZ)
    em_pts = None
    if em_analysis:
        em_data = em_analysis.get("expected_move", {})
        em_pts = _to_float(em_data.get("expected_move_pts"))

    return {
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "minute_key": now.strftime("%Y-%m-%d %H:%M"),
        "ticker": ticker,
        "spot": _to_float(spot),
        "zero_gamma": _to_float(levels.get("zero_gamma", 0)),
        "is_true_crossing": bool(levels.get("zero_gamma_is_true_crossing", True)),
        "call_wall": _to_float(levels.get("call_wall")),
        "put_wall": _to_float(levels.get("put_wall")),
        "regime": str(regime_info.get("regime")) if regime_info.get("regime") else None,
        "net_gex": _to_float(stats.get("net_gex", 0)),
        "expected_move_pts": em_pts,
        "confidence_score": _to_float(confidence_info.get("score")),
        "freshness_score": _to_float(staleness_info.get("freshness_score")),
        "coverage_ratio": _to_float(stats.get("coverage_ratio")),
        "pc_ratio": _to_float(stats.get("pc_ratio")),
        "gex_ratio": _to_float(stats.get("gex_ratio")),
        "call_iv": _to_float(stats.get("call_iv")),
        "put_iv": _to_float(stats.get("put_iv")),
    }


def save_snapshot(spot, levels, regime_info, stats, confidence_info, staleness_info, em_analysis=None, ticker="SPX"):
    """Save a GEX snapshot to Postgres. Deduplicates by (ticker, minute)."""
    row = _build_row(spot, levels, regime_info, stats, confidence_info, staleness_info, em_analysis, ticker=ticker)
    _pg_save_snapshot(row)


def check_db_connection():
    """Diagnostic: test the Postgres connection and return status info."""
    try:
        conn = _pg_get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM gex_snapshots")
        count = cur.fetchone()[0]
        cur.execute("SELECT MIN(date), MAX(date) FROM gex_snapshots")
        min_date, max_date = cur.fetchone()
        cur.execute("SELECT timestamp, spot, zero_gamma FROM gex_snapshots ORDER BY id DESC LIMIT 3")
        recent = cur.fetchall()
        conn.close()
        return {
            "ok": True,
            "total_rows": count,
            "date_range": f"{min_date} to {max_date}" if min_date else "empty",
            "recent": recent,
            "conn_str_prefix": _pg_conn_str[:40] + "..." if _pg_conn_str else "none",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def save_em_snapshot(em_data, date_str, ticker="SPX", em_type="daily"):
    """Persist EM snapshot to Postgres so it survives across sessions on the same day."""
    conn = _pg_get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS em_snapshots (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL DEFAULT 'SPX',
                em_type TEXT NOT NULL DEFAULT 'daily',
                em_pts REAL,
                em_pct REAL,
                upper_level REAL,
                lower_level REAL,
                anchor_spot REAL,
                straddle_strike REAL,
                captured_at TEXT,
                PRIMARY KEY (ticker, date, em_type)
            )
        """)
        # Migrations for older schemas
        for col, typedef in [
            ("ticker", "TEXT NOT NULL DEFAULT 'SPX'"),
            ("em_pct", "REAL"),
            ("em_type", "TEXT NOT NULL DEFAULT 'daily'"),
            ("anchor_spot", "REAL"),
        ]:
            cur.execute(f"""
                DO $$ BEGIN
                    ALTER TABLE em_snapshots ADD COLUMN {col} {typedef};
                EXCEPTION WHEN duplicate_column THEN NULL;
                END $$
            """)
        # Migrate PK from old (date)-only to (ticker, date, em_type)
        cur.execute("""
            DO $$ BEGIN
                IF EXISTS (
                    SELECT 1 FROM pg_constraint
                    WHERE conname = 'em_snapshots_pkey'
                      AND conrelid = 'em_snapshots'::regclass
                      AND array_length(conkey, 1) = 1
                ) THEN
                    ALTER TABLE em_snapshots DROP CONSTRAINT em_snapshots_pkey;
                    ALTER TABLE em_snapshots ADD PRIMARY KEY (ticker, date, em_type);
                END IF;
            END $$
        """)
        # Drop old index and create new composite one
        cur.execute("DROP INDEX IF EXISTS idx_em_ticker_date")
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_em_ticker_date_type
            ON em_snapshots(ticker, date, em_type)
        """)
        # Compute anchor_spot from EM range midpoint
        upper = em_data.get("upper_level")
        lower = em_data.get("lower_level")
        anchor_spot = round((upper + lower) / 2, 2) if upper is not None and lower is not None else None
        cur.execute(
            """INSERT INTO em_snapshots (date, ticker, em_type, em_pts, em_pct, upper_level, lower_level, anchor_spot, straddle_strike, captured_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (ticker, date, em_type) DO NOTHING""",
            (
                date_str, ticker, em_type,
                em_data.get("expected_move_pts"),
                em_data.get("expected_move_pct"),
                upper, lower, anchor_spot,
                em_data.get("straddle", {}).get("strike"),
                datetime.now(NY_TZ).isoformat(),
            ),
        )
    finally:
        conn.close()


def get_em_snapshot(date_str, ticker="SPX", em_type="daily"):
    """Retrieve persisted EM snapshot for a given ticker/type/date, if any."""
    try:
        conn = _pg_get_connection()
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT em_pts, em_pct, upper_level, lower_level, anchor_spot, straddle_strike, captured_at "
                "FROM em_snapshots WHERE date = %s AND ticker = %s AND em_type = %s",
                (date_str, ticker, em_type),
            )
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    "expected_move_pts": row[0],
                    "expected_move_pct": row[1],
                    "upper_level": row[2],
                    "lower_level": row[3],
                    "anchor_spot": row[4],
                    "straddle": {"strike": row[5]},
                    "captured_at": row[6],
                }
        except Exception:
            # Fallback: older schema without em_type/anchor_spot
            conn = _pg_get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT em_pts, em_pct, upper_level, lower_level, straddle_strike, captured_at "
                "FROM em_snapshots WHERE date = %s AND ticker = %s",
                (date_str, ticker),
            )
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    "expected_move_pts": row[0],
                    "expected_move_pct": row[1],
                    "upper_level": row[2],
                    "lower_level": row[3],
                    "anchor_spot": None,
                    "straddle": {"strike": row[4]},
                    "captured_at": row[5],
                }
    except Exception:
        pass
    return None


def get_weekly_em_date_key(now):
    """Return Monday's date string for the current trading week."""
    days_since_monday = now.weekday()  # 0=Mon
    if hasattr(now, 'date'):
        monday = (now - timedelta(days=days_since_monday)).date()
    else:
        monday = now - timedelta(days=days_since_monday)
    return monday.strftime("%Y-%m-%d")


def get_monthly_em_date_key(now):
    """
    Return the OpEx-cycle key: the Monday following the most recent standard
    3rd-Friday OpEx (strictly before today). Stable across the whole cycle.

    The cycle runs from the Monday-after-OpEx through the NEXT 3rd Friday.
    On the 3rd Friday itself, the day is still in the *old* cycle (its
    standard options settle that morning) — the new cycle begins the Monday
    after. If that Monday is a market holiday the cron's freeze-day check
    will fire on Tuesday instead, but the DB key remains the Monday date so
    save and restore agree.
    """
    from datetime import date as _date, timedelta as _td
    import calendar as _cal

    today = now.date() if hasattr(now, 'date') else now

    # Walk back from today through up to 3 months to find the most recent
    # 3rd Friday that is strictly before today.
    year, month = today.year, today.month
    third_fri = None
    for _ in range(4):
        first_weekday = _cal.weekday(year, month, 1)  # 0=Mon
        first_fri_day = 1 + (4 - first_weekday) % 7
        candidate = _date(year, month, first_fri_day + 14)
        if candidate < today:
            third_fri = candidate
            break
        # Walk back one month
        if month == 1:
            year, month = year - 1, 12
        else:
            month -= 1

    if third_fri is None:
        # Extreme fallback (shouldn't happen in practice)
        return today.replace(day=1).strftime("%Y-%m-%d")

    # 3rd Friday is always a Friday, so Monday-after = +3 days.
    cycle_open_mon = third_fri + _td(days=3)
    return cycle_open_mon.strftime("%Y-%m-%d")


def get_history(days=30, ticker="SPX"):
    """Get historical snapshots, most recent first."""
    return _pg_get_history(days, ticker=ticker)


def get_zero_gamma_trend(days=14, ticker="SPX"):
    """Get zero gamma values over time."""
    return _pg_get_zero_gamma_trend(days, ticker=ticker)


def get_daily_summary(days=30, ticker="SPX"):
    """Get first + last snapshot per day. Returns list of dicts."""
    return _pg_get_daily_summary(days, ticker=ticker)
