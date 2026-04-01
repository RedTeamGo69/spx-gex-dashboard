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
                minute_key TEXT NOT NULL UNIQUE,
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
    finally:
        conn.close()


def _pg_save_snapshot(row):
    conn = _pg_get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO gex_snapshots
               (timestamp, date, minute_key, spot, zero_gamma, is_true_crossing,
                call_wall, put_wall, regime, net_gex, expected_move_pts,
                confidence_score, freshness_score, coverage_ratio,
                pc_ratio, gex_ratio, call_iv, put_iv)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (minute_key) DO NOTHING""",
            (
                row["timestamp"], row["date"], row["minute_key"],
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


def _pg_get_daily_summary(days):
    conn = _pg_get_connection()
    try:
        cutoff = (datetime.now(NY_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM gex_snapshots
               WHERE id IN (
                   SELECT MAX(id) FROM gex_snapshots
                   WHERE date >= %s
                   GROUP BY date
               )
               ORDER BY date DESC""",
            (cutoff,),
        )
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _pg_get_zero_gamma_trend(days):
    conn = _pg_get_connection()
    try:
        cutoff = (datetime.now(NY_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
        cur = conn.cursor()
        cur.execute(
            """SELECT date, zero_gamma, spot FROM gex_snapshots
               WHERE date >= %s
               ORDER BY timestamp ASC""",
            (cutoff,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def _pg_get_history(days):
    conn = _pg_get_connection()
    try:
        cutoff = (datetime.now(NY_TZ) - timedelta(days=days)).strftime("%Y-%m-%d")
        cur = conn.cursor()
        cur.execute(
            """SELECT * FROM gex_snapshots
               WHERE date >= %s
               ORDER BY timestamp DESC""",
            (cutoff,),
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
    # Group by date, take last per day
    by_date = {}
    for row in store:
        by_date[row["date"]] = row
    result = sorted(by_date.values(), key=lambda r: r["date"], reverse=True)
    return result[:days]


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


def _build_row(spot, levels, regime_info, stats, confidence_info, staleness_info, em_analysis=None):
    now = datetime.now(NY_TZ)
    em_pts = None
    if em_analysis:
        em_data = em_analysis.get("expected_move", {})
        em_pts = em_data.get("expected_move_pts")

    return {
        "timestamp": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "minute_key": now.strftime("%Y-%m-%d %H:%M"),
        "spot": spot,
        "zero_gamma": levels.get("zero_gamma", 0),
        "is_true_crossing": bool(levels.get("is_true_crossing", True)),
        "call_wall": levels.get("call_wall"),
        "put_wall": levels.get("put_wall"),
        "regime": regime_info.get("regime"),
        "net_gex": stats.get("net_gex", 0),
        "expected_move_pts": em_pts,
        "confidence_score": confidence_info.get("score"),
        "freshness_score": staleness_info.get("freshness_score"),
        "coverage_ratio": stats.get("coverage_ratio"),
        "pc_ratio": stats.get("pc_ratio"),
        "gex_ratio": stats.get("gex_ratio"),
        "call_iv": stats.get("call_iv"),
        "put_iv": stats.get("put_iv"),
    }


def save_snapshot(spot, levels, regime_info, stats, confidence_info, staleness_info, em_analysis=None):
    """Save a GEX snapshot. Deduplicates by minute. Raises on error."""
    row = _build_row(spot, levels, regime_info, stats, confidence_info, staleness_info, em_analysis)
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


def save_em_snapshot(em_data, date_str):
    """Persist EM snapshot to Postgres so it survives across sessions on the same day."""
    if _backend != "postgres":
        return
    conn = _pg_get_connection()
    try:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS em_snapshots (
                date TEXT PRIMARY KEY,
                em_pts REAL,
                upper_level REAL,
                lower_level REAL,
                straddle_strike REAL,
                captured_at TEXT
            )
        """)
        cur.execute(
            """INSERT INTO em_snapshots (date, em_pts, upper_level, lower_level, straddle_strike, captured_at)
               VALUES (%s, %s, %s, %s, %s, %s)
               ON CONFLICT (date) DO NOTHING""",
            (
                date_str,
                em_data.get("expected_move_pts"),
                em_data.get("upper_level"),
                em_data.get("lower_level"),
                em_data.get("straddle", {}).get("strike"),
                datetime.now(NY_TZ).isoformat(),
            ),
        )
    finally:
        conn.close()


def get_em_snapshot(date_str):
    """Retrieve today's persisted EM snapshot, if any."""
    if _backend != "postgres":
        return None
    try:
        conn = _pg_get_connection()
        cur = conn.cursor()
        cur.execute("SELECT em_pts, upper_level, lower_level, straddle_strike, captured_at FROM em_snapshots WHERE date = %s", (date_str,))
        row = cur.fetchone()
        conn.close()
        if row:
            return {
                "expected_move_pts": row[0],
                "upper_level": row[1],
                "lower_level": row[2],
                "straddle": {"strike": row[3]},
                "captured_at": row[4],
            }
    except Exception:
        pass
    return None


def get_history(days=30):
    """Get historical snapshots, most recent first."""
    if _backend == "postgres":
        return _pg_get_history(days)
    return _session_get_history(days)


def get_zero_gamma_trend(days=14):
    """Get zero gamma values over time."""
    if _backend == "postgres":
        return _pg_get_zero_gamma_trend(days)
    return _session_get_zero_gamma_trend(days)


def get_daily_summary(days=30):
    """Get one row per day (latest snapshot). Returns list of dicts."""
    if _backend == "postgres":
        return _pg_get_daily_summary(days)
    return _session_get_daily_summary(days)
