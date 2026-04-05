# =============================================================================
# db.py — Dual-backend connection layer (Postgres / SQLite)
#
# Auto-detects Neon Postgres via DATABASE_URL in Streamlit secrets or env vars.
# Falls back to local SQLite for development.
#
# All range_finder modules receive a connection object from get_connection().
# The PGConnectionWrapper translates SQLite-style '?' placeholders to '%s'
# so existing queries work unchanged.
# =============================================================================

import os
import re
import sqlite3
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend detection (mirrors phase1/gex_history.py pattern)
# ---------------------------------------------------------------------------

_backend = "sqlite"
_pg_conn_str = None

try:
    import streamlit as st
    _pg_conn_str = st.secrets.get("DATABASE_URL", "")
except Exception:
    pass

if not _pg_conn_str:
    _pg_conn_str = os.environ.get("DATABASE_URL", "")

if _pg_conn_str:
    try:
        import psycopg2  # noqa: F401
        _backend = "postgres"
    except ImportError:
        log.warning("DATABASE_URL set but psycopg2 not installed — falling back to SQLite")
        _pg_conn_str = None
        _backend = "sqlite"


def get_backend() -> str:
    """Return the active backend name: 'postgres' or 'sqlite'."""
    return _backend


# ---------------------------------------------------------------------------
# SQLite → Postgres query translation
# ---------------------------------------------------------------------------

def _translate_query(sql: str) -> str:
    """Convert SQLite-style '?' placeholders to Postgres '%s'."""
    return sql.replace("?", "%s")


def _to_float(v):
    """Convert numpy/pandas types to plain Python float for psycopg2."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


# ---------------------------------------------------------------------------
# Postgres connection wrapper
# ---------------------------------------------------------------------------

class PGCursor:
    """Wraps a psycopg2 cursor, translating ? → %s in queries."""

    def __init__(self, real_cursor):
        self._cur = real_cursor

    def execute(self, sql, params=None):
        sql = _translate_query(sql)
        # Convert numpy/pandas types in params
        if params:
            params = tuple(_to_float(p) if isinstance(p, (int, float)) or p is None
                          else p for p in params)
        return self._cur.execute(sql, params)

    def executescript(self, sql):
        """Postgres doesn't have executescript — execute statements individually."""
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                self._cur.execute(_translate_query(stmt))

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def description(self):
        return self._cur.description

    def close(self):
        return self._cur.close()


class PGConnectionWrapper:
    """
    Wraps a psycopg2 connection to behave like sqlite3.Connection.

    Key differences handled:
    - ? → %s placeholder translation
    - executescript → split-and-execute
    - Auto-reconnect when Neon serverless drops idle connections
    - pd.read_sql_query works natively with psycopg2 connections
    """

    def __init__(self, conn_str: str):
        self._conn_str = conn_str
        self._conn = None
        self._connect()

    def _connect(self):
        """Establish a fresh Postgres connection."""
        import psycopg2
        self._conn = psycopg2.connect(self._conn_str, sslmode="require")
        self._conn.autocommit = False

    def _ensure_alive(self):
        """Reconnect if the underlying connection has been closed or dropped."""
        try:
            if self._conn is None or self._conn.closed:
                log.info("Postgres connection lost — reconnecting...")
                self._connect()
                return
            # Lightweight check — will raise if connection is dead
            self._conn.cursor().execute("SELECT 1")
        except Exception:
            log.info("Postgres connection stale — reconnecting...")
            try:
                self._conn.close()
            except Exception:
                pass
            self._connect()

    def cursor(self):
        self._ensure_alive()
        return PGCursor(self._conn.cursor())

    def execute(self, sql, params=None):
        cur = self.cursor()
        cur.execute(sql, params)
        return cur

    def executescript(self, sql):
        cur = self.cursor()
        cur.executescript(sql)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    # Allow pd.read_sql_query to work — it calls conn.cursor() internally
    # but also needs the raw connection for the DB-API interface
    def __getattr__(self, name):
        return getattr(self._conn, name)


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

_SQLITE_DB_PATH = Path(__file__).parent / "weekly_data.db"


def get_connection():
    """
    Return a database connection — Postgres if available, else SQLite.

    Postgres tables use the 'rf_' prefix to avoid collisions with
    existing GEX dashboard tables.
    """
    if _backend == "postgres":
        # PGConnectionWrapper handles connection lifecycle including
        # auto-reconnect when Neon serverless drops idle connections
        wrapped = PGConnectionWrapper(_pg_conn_str)
        log.info("Range finder connected to Postgres")
        return wrapped
    else:
        conn = sqlite3.connect(_SQLITE_DB_PATH, check_same_thread=False)
        log.info(f"Range finder connected to SQLite: {_SQLITE_DB_PATH}")
        return conn


def init_all_tables(conn) -> None:
    """
    Create all range finder tables if they don't exist.
    Works on both Postgres and SQLite.
    """
    cur = conn.cursor()

    # --- weekly_spx ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weekly_spx (
            week_start      TEXT PRIMARY KEY,
            week_end        TEXT,
            spx_open        REAL,
            spx_high        REAL,
            spx_low         REAL,
            spx_close       REAL,
            spx_volume      REAL,
            vix_open        REAL,
            vix_high        REAL,
            vix_low         REAL,
            vix_close       REAL,
            range_pts       REAL,
            range_pct       REAL,
            log_range       REAL,
            spx_return      REAL,
            updated_at      TEXT
        )
    """)

    # --- macro_daily ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS macro_daily (
            date            TEXT PRIMARY KEY,
            treasury_10y    REAL,
            treasury_2y     REAL,
            yield_spread    REAL,
            fed_funds       REAL,
            updated_at      TEXT
        )
    """)

    # --- event_flags ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS event_flags (
            week_start      TEXT PRIMARY KEY,
            has_fomc        INTEGER DEFAULT 0,
            has_cpi         INTEGER DEFAULT 0,
            has_nfp         INTEGER DEFAULT 0,
            has_opex        INTEGER DEFAULT 0,
            event_count     INTEGER DEFAULT 0,
            updated_at      TEXT
        )
    """)

    # --- model_features ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_features (
            week_start          TEXT PRIMARY KEY,
            log_range           REAL,
            range_pct           REAL,
            har_d1              REAL,
            har_w               REAL,
            har_m               REAL,
            vix_close           REAL,
            vix_implied_range   REAL,
            vix9d_close         REAL,
            vix3m_close         REAL,
            vix_ts_slope        REAL,
            vix_wk_ratio        REAL,
            hv5                 REAL,
            hv10                REAL,
            hv20                REAL,
            hv_ratio            REAL,
            gex                 REAL,
            gex_flag            INTEGER,
            yield_spread        REAL,
            fed_funds           REAL,
            spx_return_lag1     REAL,
            abs_return_lag1     REAL,
            has_fomc            INTEGER,
            has_cpi             INTEGER,
            has_nfp             INTEGER,
            has_opex            INTEGER,
            event_count         INTEGER,
            updated_at          TEXT
        )
    """)

    # --- gex_inputs ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gex_inputs (
            week_start  TEXT PRIMARY KEY,
            gex         REAL,
            notes       TEXT,
            updated_at  TEXT
        )
    """)

    # --- spread_log ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS spread_log (
            week_start          TEXT PRIMARY KEY,
            generated_at        TEXT,
            spx_ref_close       REAL,
            point_pct           REAL,
            upper_pct           REAL,
            effective_range_pct REAL,
            call_short          REAL,
            call_long           REAL,
            put_short           REAL,
            put_long            REAL,
            wing_width_used     INTEGER,
            buffer_pct          REAL,
            event_count         INTEGER,
            gex_flag            INTEGER,
            warnings            TEXT,
            actual_high         REAL,
            actual_low          REAL,
            actual_range_pct    REAL,
            call_breached       INTEGER,
            put_breached        INTEGER,
            outcome             TEXT,
            pnl_pts             REAL,
            updated_at          TEXT
        )
    """)

    # --- saved_models (replaces pickle files) ---
    if _backend == "postgres":
        cur.execute("""
            CREATE TABLE IF NOT EXISTS saved_models (
                model_name      TEXT PRIMARY KEY,
                model_data      BYTEA NOT NULL,
                fitted_at       TEXT,
                updated_at      TEXT
            )
        """)
    else:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS saved_models (
                model_name      TEXT PRIMARY KEY,
                model_data      BLOB NOT NULL,
                fitted_at       TEXT,
                updated_at      TEXT
            )
        """)

    # --- weekly_setup (Monday open freeze for spread finder) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weekly_setup (
            week_start      TEXT NOT NULL,
            ticker          TEXT NOT NULL DEFAULT 'SPX',
            monday_open     REAL,
            monday_vix      REAL,
            captured_at     TEXT,
            PRIMARY KEY (week_start, ticker)
        )
    """)

    conn.commit()
    log.info(f"All range finder tables initialized ({_backend})")
