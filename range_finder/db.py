# =============================================================================
# db.py — Postgres connection layer for range finder tables
#
# Postgres is required. DATABASE_URL must be set via Streamlit secrets or an
# environment variable. The PGConnectionWrapper translates sqlite-style '?'
# placeholders to '%s' so existing range_finder queries work unchanged —
# that's purely a convenience for the query authors, NOT a sqlite fallback.
# =============================================================================

import os
import time
import logging

log = logging.getLogger(__name__)

# Minimum seconds between proactive `SELECT 1` liveness probes inside
# _ensure_alive. The Spread Finder render calls cursor() ~5-10 times per
# pass (directly + via pd.read_sql_query); without throttling, every one
# of those becomes a Postgres roundtrip just to confirm the connection
# is alive. The per-cursor() exception fallback below catches the rare
# case where Neon dropped the connection between probes.
_ALIVE_PROBE_INTERVAL_SECONDS = 60.0


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

    @property
    def rowcount(self):
        """Rows affected by the last execute (UPDATE/DELETE/INSERT).

        psycopg2 and sqlite3 agree on the semantics callers rely on:
        an UPDATE whose WHERE matches nothing reports 0.
        """
        return self._cur.rowcount

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
        self._last_alive_check_ts: float | None = None
        self._connect()

    def _connect(self):
        import psycopg2
        self._conn = psycopg2.connect(self._conn_str, sslmode="require")
        self._last_alive_check_ts = time.monotonic()
        # autocommit=True so read-only paths (e.g. saved_models / model_features
        # SELECTs from the Spread Finder) don't leave the connection sitting in
        # `idle in transaction` state. Neon will not auto-suspend a compute as
        # long as any pooled connection has an open transaction, which kept the
        # endpoint warm 19+ hrs/day. All writes in this layer are single-
        # statement UPSERTs with ON CONFLICT DO UPDATE (see data_collector.py,
        # feature_builder.py, model_persistence.py, spread_persistence.py), so
        # statement-level autocommit is safe — the trailing `conn.commit()`
        # calls become harmless no-ops.
        self._conn.autocommit = True

    def _ensure_alive(self):
        # Cheap local check first (no roundtrip): if the wrapper has no
        # connection or psycopg2 already marked it closed, reconnect.
        if self._conn is None or self._conn.closed:
            log.info("Postgres connection lost — reconnecting...")
            self._connect()
            return

        # Proactive `SELECT 1` probe — but throttled. Doing it on every
        # cursor() call meant 5-10 extra SELECTs per Spread Finder render,
        # which kept Neon's compute warm. The per-cursor() exception
        # fallback in cursor() handles the rare case where Neon drops
        # the connection between probes.
        now = time.monotonic()
        if self._last_alive_check_ts is not None and \
                (now - self._last_alive_check_ts) < _ALIVE_PROBE_INTERVAL_SECONDS:
            return
        try:
            cur = self._conn.cursor()
            try:
                cur.execute("SELECT 1")
            finally:
                cur.close()
            self._last_alive_check_ts = now
        except Exception:
            log.info("Postgres connection stale — reconnecting...")
            try:
                self._conn.close()
            except Exception:
                pass
            self._connect()

    def cursor(self):
        self._ensure_alive()
        try:
            return PGCursor(self._conn.cursor())
        except Exception as e:
            # Throttled `_ensure_alive` may have skipped the probe and the
            # connection died in the meantime. Reconnect transparently
            # and retry once before bubbling the error up.
            log.info(f"Postgres cursor() failed ({e!r}) — reconnecting and retrying...")
            try:
                self._conn.close()
            except Exception:
                pass
            self._connect()
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


# Arbitrary 64-bit key so every job serializes on the SAME advisory lock.
_INIT_ADVISORY_LOCK_KEY = 776699


def _acquire_init_advisory_lock(conn) -> None:
    """Block until the session owns the schema-init advisory lock.

    ``PGCursor`` intentionally retains its historical numeric adaptation,
    which turns Python integers into floats. PostgreSQL does not implicitly
    resolve ``pg_advisory_lock(double precision)`` to the bigint overload, so
    this one overload-sensitive call casts explicitly instead of changing
    parameter semantics for every range-finder query.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT pg_advisory_lock(CAST(? AS bigint))",
        (_INIT_ADVISORY_LOCK_KEY,),
    )
    if cur.fetchone() is None:
        raise RuntimeError("pg_advisory_lock returned no result row")


def _release_init_advisory_lock(conn) -> None:
    """Release the session lock and prove PostgreSQL reported success."""
    cur = conn.cursor()
    cur.execute(
        "SELECT pg_advisory_unlock(CAST(? AS bigint))",
        (_INIT_ADVISORY_LOCK_KEY,),
    )
    row = cur.fetchone()
    if not row or row[0] is not True:
        raise RuntimeError("pg_advisory_unlock reported that the lock was not held")

def init_all_tables(conn) -> None:
    """Create all range finder tables if they don't exist (Postgres DDL).

    Serialized behind a session-level Postgres advisory lock. Eight matrix
    jobs fire this concurrently every Monday; without serialization their
    CREATE / ALTER / migration DDL raced on Neon's throttled free tier,
    where the pile-up stretched a normally-instant init to 15+ minutes and
    delayed the 9:30 snapshot. pg_advisory_lock makes the others wait for
    the first to finish (after which every statement is a cheap no-op via
    IF NOT EXISTS / duplicate-column guards). Postgres is the only supported
    backend, so failure to acquire or release this serialization mechanism is
    a hard initialization error rather than an unlocked best-effort fallback.
    """
    try:
        _acquire_init_advisory_lock(conn)
    except Exception as exc:
        raise RuntimeError(
            "Failed to acquire Postgres schema init lock"
        ) from exc
    try:
        _init_all_tables_body(conn)
    finally:
        try:
            _release_init_advisory_lock(conn)
        except Exception as exc:
            raise RuntimeError(
                "Failed to release Postgres schema init lock"
            ) from exc


def _init_all_tables_body(conn) -> None:
    """The actual DDL body — see init_all_tables for the locking wrapper."""
    cur = conn.cursor()

    # --- gamma_level_history ---
    # This is an archival research table, not Range Finder application state.
    # It lives in this explicit bootstrap path so the hot save path never runs
    # DDL. The helper executes one CREATE TABLE IF NOT EXISTS and adds no
    # speculative secondary indexes; its composite PK covers the intended
    # per-ticker/session backtest reads.
    from phase1.gamma_level_history import init_gamma_level_history_schema
    init_gamma_level_history_schema(conn)

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

    # --- weekly_underlying ---
    # Per-ticker weekly OHLC + vol-proxy OHLC for tickers that own their HAR
    # pipeline (QQQ / AMZN / AMD — see phase1.ticker_config.uses_own_har).
    # SPX/XSP continue to use weekly_spx above; they share SPX history and
    # don't need a row here. The vol-proxy columns store either VIX (for
    # stocks) or VXN (for QQQ) as configured in ticker_config.vol_proxy_yf.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS weekly_underlying (
            ticker            TEXT NOT NULL,
            week_start        TEXT NOT NULL,
            week_end          TEXT,
            open              REAL,
            high              REAL,
            low               REAL,
            close             REAL,
            volume            REAL,
            vol_proxy_open    REAL,
            vol_proxy_high    REAL,
            vol_proxy_low     REAL,
            vol_proxy_close   REAL,
            range_pts         REAL,
            range_pct         REAL,
            log_range         REAL,
            return_pct        REAL,
            updated_at        TEXT,
            PRIMARY KEY (ticker, week_start)
        )
    """)

    # --- earnings_flags ---
    # Per-ticker weekly earnings flag for the single-stock earnings-week gate
    # in the Spread Finder (see phase1.ticker_config.has_single_name_earnings).
    # Index ETFs / cash indexes never set this; only AMZN/AMD will populate it.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS earnings_flags (
            ticker          TEXT NOT NULL,
            week_start      TEXT NOT NULL,
            has_earnings    INTEGER NOT NULL DEFAULT 0,
            earnings_date   TEXT,
            updated_at      TEXT,
            PRIMARY KEY (ticker, week_start)
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
    # Composite PK (week_start, ticker) so each ticker that owns a HAR pipeline
    # (QQQ / AMZN / AMD — see phase1.ticker_config.uses_own_har) stores its own
    # feature rows without colliding with SPX. SPX/XSP share the SPX rows
    # (XSP rides on SPX features at 1/10 scale — see ticker_config xsp_scale_to_spx).
    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_features (
            week_start          TEXT NOT NULL,
            ticker              TEXT NOT NULL DEFAULT 'SPX',
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
            has_earnings        INTEGER DEFAULT 0,
            path_source_week    TEXT,
            updated_at          TEXT,
            PRIMARY KEY (week_start, ticker)
        )
    """)

    # Schema migrations for columns added after the initial release
    for col, ctype in [
        ("high_vol_regime", "INTEGER"),
        ("gex_normalized", "REAL"),
        ("has_earnings", "INTEGER DEFAULT 0"),
        ("path_source_week", "TEXT"),
    ]:
        try:
            cur.execute(f"ALTER TABLE model_features ADD COLUMN IF NOT EXISTS {col} {ctype}")
        except Exception:
            pass

    # Migration: add ticker column on legacy single-PK deploys and rebuild PK.
    # Legacy rows become ticker='SPX' (accurate — every pre-migration row was
    # SPX-derived). Idempotent via IF NOT EXISTS + pg_constraint probe so it's
    # safe to re-run on every init_all_tables() call.
    cur.execute("""
        DO $$ BEGIN
            ALTER TABLE model_features ADD COLUMN ticker TEXT NOT NULL DEFAULT 'SPX';
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$
    """)
    cur.execute("""
        DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'model_features_pkey'
                  AND conrelid = 'model_features'::regclass
                  AND COALESCE(array_length(conkey, 1), 0) = 1
            ) THEN
                ALTER TABLE model_features DROP CONSTRAINT model_features_pkey;
                ALTER TABLE model_features ADD PRIMARY KEY (week_start, ticker);
            END IF;
        END $$
    """)

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
    # The forecast-vs-realized record behind the PI-coverage calibration
    # audit (range_finder/calibration.py): the Monday cron logs the plan
    # (model_name + point/lower/upper forecast bounds + strikes), then the
    # next Monday's run scores the expired week's actual high/low/range.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS spread_log (
            week_start          TEXT NOT NULL,
            ticker              TEXT NOT NULL DEFAULT 'SPX',
            model_name          TEXT,
            generated_at        TEXT,
            spx_ref_close       REAL,
            point_pct           REAL,
            lower_pct           REAL,
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
            updated_at          TEXT,
            PRIMARY KEY (week_start, ticker)
        )
    """)

    # Migration: calibration columns for deploys that pre-date them.
    for col, ctype in [
        ("model_name", "TEXT"),
        ("lower_pct", "REAL"),
    ]:
        try:
            cur.execute(f"ALTER TABLE spread_log ADD COLUMN IF NOT EXISTS {col} {ctype}")
        except Exception:
            pass

    # Migration: add ticker on legacy single-PK deploys and rebuild the PK
    # (same pattern as model_features/gex_inputs above). Legacy rows become
    # ticker='SPX' — accurate, every pre-migration plan was SPX.
    cur.execute("""
        DO $$ BEGIN
            ALTER TABLE spread_log ADD COLUMN ticker TEXT NOT NULL DEFAULT 'SPX';
        EXCEPTION WHEN duplicate_column THEN NULL;
        END $$
    """)
    cur.execute("""
        DO $$ BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'spread_log_pkey'
                  AND conrelid = 'spread_log'::regclass
                  AND COALESCE(array_length(conkey, 1), 0) = 1
            ) THEN
                ALTER TABLE spread_log DROP CONSTRAINT spread_log_pkey;
                ALTER TABLE spread_log ADD PRIMARY KEY (week_start, ticker);
            END IF;
        END $$
    """)

    # --- saved_models (Postgres BYTEA — replaces pickle files) ---
    # PK is (model_name, ticker) so SPX and XSP keep independent fits.
    # Sharing the key across tickers used to cause cross-contamination:
    # saving M4_full on XSP would overwrite the SPX fit of the same spec,
    # and the Spread Finder would then silently load XSP-trained weights
    # into the SPX view after a ticker-switch round trip.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS saved_models (
            model_name      TEXT NOT NULL,
            ticker          TEXT NOT NULL DEFAULT 'SPX',
            model_data      BYTEA NOT NULL,
            fitted_at       TEXT,
            updated_at      TEXT,
            PRIMARY KEY (model_name, ticker)
        )
    """)

    # Migration for existing deploys that pre-date the composite PK: add
    # the ticker column (legacy rows become ticker='SPX', which is
    # accurate — every pre-migration fit was SPX) and swap the PK from
    # (model_name) to (model_name, ticker). Idempotent via IF NOT EXISTS
    # and a pg_constraint probe, so safe to run on every init.
    cur.execute("""
        ALTER TABLE saved_models
        ADD COLUMN IF NOT EXISTS ticker TEXT NOT NULL DEFAULT 'SPX'
    """)
    cur.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM pg_constraint
                WHERE conname = 'saved_models_pkey'
                  AND conrelid = 'saved_models'::regclass
                  AND COALESCE(array_length(conkey, 1), 0) = 1
            ) THEN
                ALTER TABLE saved_models DROP CONSTRAINT saved_models_pkey;
                ALTER TABLE saved_models ADD PRIMARY KEY (model_name, ticker);
            END IF;
        END $$
    """)

    # One-time cleanup: specs removed from MODEL_SPECS leave orphan fits in
    # saved_models that the dashboard can no longer reach (they're off the
    # dropdown). Drop them so they don't sit in the DB as dead state.
    #   - M5_garch: HAR + GARCH(1,1), removed earlier (dominated by the VIX specs).
    #   - M6_regime: removed 2026-06 after walk-forward OOS showed its regime
    #     interaction terms hurt out-of-sample R² (see har_model.MODEL_SPECS).
    # DELETE is idempotent; safe to run on every init_all_tables call.
    cur.execute("DELETE FROM saved_models WHERE model_name IN ('M5_garch', 'M6_regime')")

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

    # --- interval_calibration (conformal λ per ticker/spec) ---
    # Written ONLY by the offline adoption gate (walkforward --conformal
    # --persist); read by conformal.maybe_apply_conformal, which is a no-op
    # unless conformal.CONFORMAL_ENABLED is flipped on. See conformal.py.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS interval_calibration (
            ticker          TEXT NOT NULL,
            model_name      TEXT NOT NULL,
            lambda          REAL NOT NULL,
            n_obs           INTEGER,
            method          TEXT,
            updated_at      TEXT,
            PRIMARY KEY (ticker, model_name)
        )
    """)

    # The 0DTE / daily-cadence tables (daily_spx, event_flags_daily,
    # daily_model_features, forecast_log_daily, spread_log_daily) are no
    # longer created here: the 0DTE finder was deliberately removed (2026-07)
    # after a 1,036-session audit showed its VRP verdict had no predictive
    # edge. Their historical rows are still in Neon — drop manually if the
    # storage matters.
    #
    # The Monday Cockpit / Pre-Flight tables (cockpit_verdicts, preflight_log)
    # and the Public.com fill-history tables that fed Pre-Flight's behavioral
    # gate (strategy_history, trade_category_stats, fill_review_queue) went
    # further: both tabs and the whole public_api package were removed
    # 2026-08, and the five tables were DROPPED outright at the user's request
    # (2026-08-07) rather than left as orphaned history. Re-creating any of
    # them means restoring the deleted modules from git history first.

    conn.commit()
    log.info("All range finder tables initialized (postgres)")
