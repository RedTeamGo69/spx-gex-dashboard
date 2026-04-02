"""
Historical GEX snapshot tracking.

Storage backends (auto-detected):
1. Neon Postgres — when DATABASE_URL is set in st.secrets or env vars
   Persistent across sessions and deploys (recommended for Streamlit Cloud)
2. In-session fallback — st.session_state only
   Data lost on page refresh (works everywhere, zero config)
"""
from __future__ import annotations

import os
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")
_logger = logging.getLogger(__name__)

# ── Backend detection ──
_backend = "session"  # default fallback
_pg_conn_str = None

# Check for Neon/Postgres connection string
try:
    import streamlit as st
    _pg_conn_str = st.secrets.get("DATABASE_URL", "")
except Exception:
    pass

if not _pg_conn_str:
    _pg_conn_str = os.environ.get("DATABASE_URL", "")

if _pg_conn_str:
    try:
        import psycopg2
        _backend = "postgres"
    except ImportError:
        _logger.warning("DATABASE_URL set but psycopg2 not installed. Falling back to session storage.")
        _pg_conn_str = None


# ── Postgres backend ──

def _pg_get_connection():
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


# ── Session-state backend ──

def _session_get_store():
    import streamlit as st
    if "gex_history" not in st.session_state:
        st.session_state["gex_history"] = []
    return st.session_state["gex_history"]


def _session_save_snapshot(row):
    store = _session_get_store()
    # Dedup by minute_key
    if store and store[-1].get("minute_key") == row["minute_key"]:
        return
    store.append(row)
    # Keep last 500 snapshots in memory
    if len(store) > 500:
        del store[:-500]


def _session_get_daily_summary(days):
    store = _session_get_store()
    if not store:
        return []
    # Group by date, take first and last per day
    first_by_date = {}
    last_by_date = {}
    for row in store:
        d = row["date"]
        if d not in first_by_date:
            first_by_date[d] = row
        last_by_date[d] = row
    tagged = []
    for d in sorted(first_by_date.keys(), reverse=True)[:days]:
        first = {**first_by_date[d], "scan_type": "open"}
        tagged.append(first)
        if last_by_date[d] is not first_by_date[d]:
            last = {**last_by_date[d], "scan_type": "close"}
            tagged.append(last)
    return tagged


def _session_get_zero_gamma_trend(days):
    store = _session_get_store()
    return [(r["date"], r["zero_gamma"], r["spot"]) for r in store]


def _session_get_history(days):
    store = _session_get_store()
    return list(reversed(store[-days * 50:]))


# ── Initialize Postgres table if needed ──

if _backend == "postgres":
    try:
        _pg_ensure_table()
    except Exception as e:
        _logger.warning(f"Failed to initialize Postgres table: {e}. Falling back to session storage.")
        _backend = "session"


# ── Public API ──

def get_backend():
    """Return the active backend name: 'postgres' or 'session'."""
    return _backend


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
        "is_true_crossing": bool(levels.get("is_true_crossing", True)),
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
    """Save a GEX snapshot. Deduplicates by (ticker, minute). Raises on error."""
    row = _build_row(spot, levels, regime_info, stats, confidence_info, staleness_info, em_analysis, ticker=ticker)
    if _backend == "postgres":
        _pg_save_snapshot(row)
    else:
        _session_save_snapshot(row)


def check_db_connection():
    """Diagnostic: test the Postgres connection and return status info."""
    if _backend != "postgres":
        return {"ok": False, "error": "Not using Postgres backend", "backend": _backend}
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


def save_em_snapshot(em_data, date_str, ticker="SPX"):
    """Persist EM snapshot to Postgres so it survives across sessions on the same day."""
    if _backend != "postgres":
        return
    conn = _pg_get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS em_snapshots (
                date TEXT NOT NULL,
                ticker TEXT NOT NULL DEFAULT 'SPX',
                em_pts REAL,
                em_pct REAL,
                upper_level REAL,
                lower_level REAL,
                straddle_strike REAL,
                captured_at TEXT,
                PRIMARY KEY (ticker, date)
            )
        """)
        # Migration: add ticker column and update PK if missing
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE em_snapshots ADD COLUMN ticker TEXT NOT NULL DEFAULT 'SPX';
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """)
        cur.execute("""
            DO $$ BEGIN
                ALTER TABLE em_snapshots ADD COLUMN em_pct REAL;
            EXCEPTION WHEN duplicate_column THEN NULL;
            END $$
        """)
        # Try creating the composite unique index (covers both old single-PK and new schemas)
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_em_ticker_date ON em_snapshots(ticker, date)
        """)
        cur.execute(
            """INSERT INTO em_snapshots (date, ticker, em_pts, em_pct, upper_level, lower_level, straddle_strike, captured_at)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (ticker, date) DO NOTHING""",
            (
                date_str, ticker,
                em_data.get("expected_move_pts"),
                em_data.get("expected_move_pct"),
                em_data.get("upper_level"),
                em_data.get("lower_level"),
                em_data.get("straddle", {}).get("strike"),
                datetime.now(NY_TZ).isoformat(),
            ),
        )
    finally:
        conn.close()


def get_em_snapshot(date_str, ticker="SPX"):
    """Retrieve today's persisted EM snapshot for a given ticker, if any."""
    if _backend != "postgres":
        return None
    try:
        conn = _pg_get_connection()
        cur = conn.cursor()
        try:
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
                    "straddle": {"strike": row[4]},
                    "captured_at": row[5],
                }
        except Exception:
            # Fallback: ticker column may not exist yet on older schemas
            conn = _pg_get_connection()
            cur = conn.cursor()
            cur.execute(
                "SELECT em_pts, upper_level, lower_level, straddle_strike, captured_at "
                "FROM em_snapshots WHERE date = %s",
                (date_str,),
            )
            row = cur.fetchone()
            conn.close()
            if row:
                return {
                    "expected_move_pts": row[0],
                    "expected_move_pct": None,
                    "upper_level": row[1],
                    "lower_level": row[2],
                    "straddle": {"strike": row[3]},
                    "captured_at": row[4],
                }
    except Exception:
        pass
    return None


def get_history(days=30, ticker="SPX"):
    """Get historical snapshots, most recent first."""
    if _backend == "postgres":
        return _pg_get_history(days, ticker=ticker)
    return _session_get_history(days)


def get_zero_gamma_trend(days=14, ticker="SPX"):
    """Get zero gamma values over time."""
    if _backend == "postgres":
        return _pg_get_zero_gamma_trend(days, ticker=ticker)
    return _session_get_zero_gamma_trend(days)


def get_daily_summary(days=30, ticker="SPX"):
    """Get first + last snapshot per day. Returns list of dicts."""
    if _backend == "postgres":
        return _pg_get_daily_summary(days, ticker=ticker)
    return _session_get_daily_summary(days)
