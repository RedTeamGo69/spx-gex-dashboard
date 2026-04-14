# =============================================================================
# db.py — Postgres connection layer for range finder tables
#
# Postgres is required. DATABASE_URL must be set via Streamlit secrets or an
# environment variable. The PGConnectionWrapper translates sqlite-style '?'
# placeholders to '%s' so existing range_finder queries work unchanged —
# that's purely a convenience for the query authors, NOT a sqlite fallback.
# =============================================================================

import os
import logging

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Connection string resolution
# ---------------------------------------------------------------------------

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


def get_backend() -> str:
    """Return the active backend name. Always 'postgres' now that sqlite is removed."""
    return "postgres"


# ---------------------------------------------------------------------------
# Placeholder translation (sqlite-style '?' → Postgres '%s')
#
# This is NOT a sqlite compatibility shim — the range_finder modules simply
# use '?' placeholders for historical reasons, and rewriting every query to
# '%s' would be a lot of churn with no behavioral benefit. The wrapper below
# does the translation on the fly.
# ---------------------------------------------------------------------------

def _translate_query(sql: str) -> str:
    return sql.replace("?", "%s")


def _to_float(v):
    """Convert numpy/pandas numeric types to plain Python floats for psycopg2."""
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
        if params:
            params = tuple(
                _to_float(p) if isinstance(p, (int, float)) or p is None else p
                for p in params
            )
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
    Wraps a psycopg2 connection to present the small sqlite-like surface that
    the range_finder modules expect (execute / executescript / cursor / commit).

    Also handles Neon serverless dropping idle connections by lazily
    reconnecting on the next use.
    """

    def __init__(self, conn_str: str):
        self._conn_str = conn_str
        self._conn = None
        self._connect()

    def _connect(self):
        import psycopg2
        self._conn = psycopg2.connect(self._conn_str, sslmode="require")
        self._conn.autocommit = False

    def _ensure_alive(self):
        try:
            if self._conn is None or self._conn.closed:
                log.info("Postgres connection lost — reconnecting...")
                self._connect()
                return
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

def get_connection():
    """Return a Postgres connection wrapped for placeholder translation."""
    _require_postgres()
    wrapped = PGConnectionWrapper(_pg_conn_str)
    log.info("Range finder connected to Postgres")
    return wrapped


def init_all_tables(conn) -> None:
    """Create all range finder tables if they don't exist (Postgres DDL)."""
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
            garch_vol           REAL,
            high_vol_regime     INTEGER,
            gex                 REAL,
            gex_flag            INTEGER,
            gex_normalized      REAL,
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

    # Schema migrations for columns added after the initial release
    for col, ctype in [
        ("garch_vol", "REAL"),
        ("high_vol_regime", "INTEGER"),
        ("gex_normalized", "REAL"),
    ]:
        try:
            cur.execute(f"ALTER TABLE model_features ADD COLUMN IF NOT EXISTS {col} {ctype}")
        except Exception:
            pass

    # --- gex_inputs ---
    # Composite PK (week_start, ticker) so SPX and XSP runs don't stomp on
    # each other's rows. Historically the table was keyed on week_start
    # alone; the migration block below adds the ticker column and swaps
    # the PK in place. The HAR feature builder only consumes SPX rows
    # (it normalizes by spx_open²), but XSP dashboard runs still write
    # here via save_gex_to_range_finder and need a non-colliding slot.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gex_inputs (
            week_start  TEXT NOT NULL,
            ticker      TEXT NOT NULL DEFAULT 'SPX',
            gex         REAL,
            notes       TEXT,
            updated_at  TEXT,
            PRIMARY KEY (week_start, ticker)
        )
    """)
    # Migration: add ticker column on legacy tables and rebuild the PK.
    cur.execute("""
        DO $$ BEGIN
            ALTER TABLE gex_inputs ADD COLUMN ticker TEXT NOT NULL DEFAULT 'SPX';
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$
    """)
    cur.execute("""
        DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'gex_inputs_pkey'
                  AND conrelid = 'gex_inputs'::regclass
                  AND array_length(conkey, 1) = 1
            ) THEN
                ALTER TABLE gex_inputs DROP CONSTRAINT gex_inputs_pkey;
                ALTER TABLE gex_inputs ADD PRIMARY KEY (week_start, ticker);
            END IF;
        END $$
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

    # --- saved_models (Postgres BYTEA — replaces pickle files) ---
    cur.execute("""
        CREATE TABLE IF NOT EXISTS saved_models (
            model_name      TEXT PRIMARY KEY,
            model_data      BYTEA NOT NULL,
            fitted_at       TEXT,
            updated_at      TEXT
        )
    """)

    # One-time cleanup: M5_garch was removed from MODEL_SPECS because it
    # was strictly dominated by M6_regime on live data.  Drop any orphan
    # saved fit for it so it doesn't sit in the DB as dead state — the
    # dashboard can't reach it anyway now that it's off the dropdown.
    # DELETE is idempotent; safe to run on every init_all_tables call.
    cur.execute("DELETE FROM saved_models WHERE model_name = 'M5_garch'")

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
    log.info("All range finder tables initialized (postgres)")
