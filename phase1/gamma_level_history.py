"""Cheap, forward-only archival of derived Gamma levels for research.

This table deliberately stores only already-computed levels and compact quality
diagnostics. Raw chains and per-strike frames are expensive, unnecessary for
the intended backtests, and must never enter this persistence path. Rows are
first-observation-wins within 30-minute NYSE RTH buckets and are retained
indefinitely: about 13 rows/session/ticker, or roughly 3.3k rows/year/ticker.

Schema initialization is explicit and separate from steady-state saves. A save
opens one short-lived connection, executes one parameterized INSERT with
``ON CONFLICT DO NOTHING``, and closes the connection. No pre-read or liveness
probe is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
import logging
import threading
from typing import Callable

from phase1.config import CASH_CALENDAR, NY_TZ
from phase1.market_clock import get_session_state

log = logging.getLogger(__name__)

BUCKET_MINUTES = 30

GAMMA_LEVEL_HISTORY_DDL = """
    CREATE TABLE IF NOT EXISTS gamma_level_history (
        captured_at                 TIMESTAMPTZ NOT NULL,
        session_date                DATE NOT NULL,
        bucket_start                TIME NOT NULL,
        ticker                      TEXT NOT NULL,
        spot                        DOUBLE PRECISION NOT NULL,
        zero_gamma                  DOUBLE PRECISION NOT NULL,
        call_wall                   DOUBLE PRECISION,
        put_wall                    DOUBLE PRECISION,
        gamma_regime                TEXT NOT NULL
                                    CHECK (gamma_regime IN
                                           ('positive', 'negative', 'transition', 'unknown')),
        net_gex                     DOUBLE PRECISION,
        call_wall_gex               DOUBLE PRECISION,
        put_wall_gex                DOUBLE PRECISION,
        zero_gamma_is_true_crossing BOOLEAN,
        zero_gamma_method           TEXT,
        expected_move_pts           DOUBLE PRECISION,
        confidence_score            DOUBLE PRECISION,
        freshness_score             DOUBLE PRECISION,
        coverage_ratio              DOUBLE PRECISION,
        PRIMARY KEY (ticker, session_date, bucket_start)
    )
"""

_INSERT_SQL = """
    INSERT INTO gamma_level_history (
        captured_at, session_date, bucket_start, ticker,
        spot, zero_gamma, call_wall, put_wall, gamma_regime, net_gex,
        call_wall_gex, put_wall_gex, zero_gamma_is_true_crossing,
        zero_gamma_method, expected_move_pts, confidence_score,
        freshness_score, coverage_ratio
    ) VALUES (
        %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s,
        %s, %s, %s,
        %s, %s, %s,
        %s, %s
    )
    ON CONFLICT (ticker, session_date, bucket_start) DO NOTHING
"""


@dataclass(frozen=True)
class GammaArchiveBucket:
    """One NYSE RTH bucket plus the actual UTC observation timestamp."""

    captured_at_utc: datetime
    session_date: date
    bucket_start: time


def gamma_archive_bucket(captured_at: datetime) -> GammaArchiveBucket | None:
    """Map an aware timestamp to its 30-minute NYSE RTH bucket.

    Buckets are anchored to the exchange calendar's actual open. Weekends,
    holidays, pre-market, the exact close, and after-hours return ``None``.
    This intentionally does not create a 16:00 bucket: the archive records
    contemporaneous in-session levels, not a synthetic completed-session row.
    """
    if captured_at.tzinfo is None or captured_at.utcoffset() is None:
        raise ValueError("captured_at must be timezone-aware")

    captured_ny = captured_at.astimezone(NY_TZ)
    session = get_session_state(CASH_CALENDAR, captured_ny)
    if not session.is_open or session.market_open is None:
        return None

    market_open_ny = session.market_open.to_pydatetime().astimezone(NY_TZ)
    elapsed_minutes = int((captured_ny - market_open_ny).total_seconds() // 60)
    bucket_number = elapsed_minutes // BUCKET_MINUTES
    bucket_dt = market_open_ny + timedelta(minutes=bucket_number * BUCKET_MINUTES)

    return GammaArchiveBucket(
        captured_at_utc=captured_at.astimezone(timezone.utc),
        session_date=captured_ny.date(),
        bucket_start=bucket_dt.time().replace(tzinfo=None),
    )


def _get_connection():
    # Keep the existing phase1 Postgres/secrets resolution in one place while
    # avoiding any connection or schema side effect at module import time.
    from phase1.gex_history import _pg_get_connection

    return _pg_get_connection()


def init_gamma_level_history_schema(conn=None) -> None:
    """Create the archive table with exactly one DDL statement.

    Call this from an explicit bootstrap/migration path. It is intentionally
    never called by :func:`save_gamma_level_snapshot`.
    """
    owns_connection = conn is None
    conn = conn or _get_connection()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(GAMMA_LEVEL_HISTORY_DDL)
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
        if owns_connection:
            conn.close()


def _to_float(value):
    if value is None:
        return None
    return float(value)


def _normalize_regime(value: str | None) -> str:
    normalized = str(value or "").strip().lower()
    if "positive" in normalized:
        return "positive"
    if "negative" in normalized:
        return "negative"
    if "zero" in normalized or "transition" in normalized or "neutral" in normalized:
        return "transition"
    return "unknown"


# Avoid repeated conflict INSERTs from Streamlit reruns within one process.
# The database primary key remains authoritative across workers/process starts.
_completed_bucket_keys: set[tuple[str, date, time]] = set()
_completed_bucket_lock = threading.Lock()


def save_gamma_level_snapshot(
    *,
    captured_at: datetime,
    ticker: str,
    spot: float,
    zero_gamma: float,
    call_wall: float | None,
    put_wall: float | None,
    gamma_regime: str | None,
    net_gex: float | None,
    call_wall_gex: float | None = None,
    put_wall_gex: float | None = None,
    zero_gamma_is_true_crossing: bool | None = None,
    zero_gamma_method: str | None = None,
    expected_move_pts: float | None = None,
    confidence_score: float | None = None,
    freshness_score: float | None = None,
    coverage_ratio: float | None = None,
    connection_factory: Callable | None = None,
) -> bool:
    """Persist one already-computed observation; return whether it was inserted.

    Out-of-session observations are skipped without opening Postgres. Within a
    process, an already-completed bucket is also skipped locally. Across
    processes, the primary key and ``ON CONFLICT DO NOTHING`` preserve the
    first observation without a SELECT-before-INSERT race.
    """
    bucket = gamma_archive_bucket(captured_at)
    if bucket is None:
        return False

    ticker_key = str(ticker or "").strip().upper()
    if not ticker_key:
        raise ValueError("ticker is required")

    key = (ticker_key, bucket.session_date, bucket.bucket_start)
    with _completed_bucket_lock:
        if key in _completed_bucket_keys:
            return False

    params = (
        bucket.captured_at_utc,
        bucket.session_date,
        bucket.bucket_start,
        ticker_key,
        _to_float(spot),
        _to_float(zero_gamma),
        _to_float(call_wall),
        _to_float(put_wall),
        _normalize_regime(gamma_regime),
        _to_float(net_gex),
        _to_float(call_wall_gex),
        _to_float(put_wall_gex),
        None if zero_gamma_is_true_crossing is None else bool(zero_gamma_is_true_crossing),
        str(zero_gamma_method).strip() if zero_gamma_method else None,
        _to_float(expected_move_pts),
        _to_float(confidence_score),
        _to_float(freshness_score),
        _to_float(coverage_ratio),
    )

    conn = (connection_factory or _get_connection)()
    cur = None
    try:
        cur = conn.cursor()
        cur.execute(_INSERT_SQL, params)
        inserted = cur.rowcount == 1
    finally:
        if cur is not None:
            try:
                cur.close()
            except Exception:
                pass
        conn.close()

    # A successful conflict no-op also proves the bucket already exists, so
    # later reruns in this process need not wake Neon again.
    with _completed_bucket_lock:
        _completed_bucket_keys.add(key)
    return inserted


def gamma_level_snapshot_kwargs(
    *,
    captured_at: datetime,
    ticker: str,
    spot: float,
    levels: dict,
    regime_info: dict,
    stats: dict,
    confidence_info: dict,
    staleness_info: dict,
    em_analysis: dict | None = None,
) -> dict:
    """Build the one-row save arguments from already-computed engine values."""
    expected_move = (em_analysis or {}).get("expected_move") or {}
    return {
        "captured_at": captured_at,
        "ticker": ticker,
        "spot": spot,
        "zero_gamma": levels.get("zero_gamma"),
        "call_wall": levels.get("call_wall"),
        "put_wall": levels.get("put_wall"),
        "gamma_regime": regime_info.get("regime"),
        "net_gex": levels.get("net_gex", stats.get("net_gex")),
        "call_wall_gex": levels.get("call_wall_gex"),
        "put_wall_gex": levels.get("put_wall_gex"),
        "zero_gamma_is_true_crossing": levels.get("zero_gamma_is_true_crossing"),
        "zero_gamma_method": levels.get("zero_gamma_method"),
        "expected_move_pts": expected_move.get("expected_move_pts"),
        "confidence_score": confidence_info.get("score"),
        "freshness_score": staleness_info.get("freshness_score"),
        "coverage_ratio": stats.get("coverage_ratio"),
    }


def archive_computed_gamma_levels(
    *,
    captured_at: datetime | str,
    ticker: str,
    spot: float,
    levels: dict,
    regime_info: dict,
    stats: dict,
    confidence_info: dict,
    staleness_info: dict,
    em_analysis: dict | None = None,
) -> bool:
    """Extract the narrow archive payload and isolate all persistence errors."""
    try:
        if isinstance(captured_at, str):
            captured_at = datetime.fromisoformat(captured_at)
        return save_gamma_level_snapshot(**gamma_level_snapshot_kwargs(
            captured_at=captured_at,
            ticker=ticker,
            spot=spot,
            levels=levels,
            regime_info=regime_info,
            stats=stats,
            confidence_info=confidence_info,
            staleness_info=staleness_info,
            em_analysis=em_analysis,
        ))
    except Exception as exc:
        log.warning("Gamma archive save failed: %s", exc)
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_gamma_level_history_schema()
    log.info("gamma_level_history schema initialized")
