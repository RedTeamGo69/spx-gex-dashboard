"""
Historical GEX snapshot tracking using SQLite.

Saves daily snapshots of key GEX levels for trend analysis.
"""
from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "gex_history.db")
_lock = threading.Lock()


def _get_connection():
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS gex_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            minute_key TEXT NOT NULL UNIQUE,
            spot REAL NOT NULL,
            zero_gamma REAL NOT NULL,
            is_true_crossing INTEGER NOT NULL DEFAULT 1,
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
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_snapshots_date ON gex_snapshots(date)
    """)
    conn.commit()


# Initialize on import
with _lock:
    _conn = _get_connection()
    _ensure_table(_conn)


def save_snapshot(spot, levels, regime_info, stats, confidence_info, staleness_info, em_analysis=None):
    """Save a GEX snapshot. Deduplicates by minute-truncated timestamp."""
    now = datetime.now(NY_TZ)
    minute_key = now.strftime("%Y-%m-%d %H:%M")
    date_str = now.strftime("%Y-%m-%d")
    ts = now.isoformat()

    em_pts = None
    if em_analysis:
        em_data = em_analysis.get("expected_move", {})
        em_pts = em_data.get("expected_move_pts")

    with _lock:
        try:
            _conn.execute(
                """INSERT OR IGNORE INTO gex_snapshots
                   (timestamp, date, minute_key, spot, zero_gamma, is_true_crossing,
                    call_wall, put_wall, regime, net_gex, expected_move_pts,
                    confidence_score, freshness_score, coverage_ratio,
                    pc_ratio, gex_ratio, call_iv, put_iv)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ts, date_str, minute_key,
                    spot,
                    levels.get("zero_gamma", 0),
                    1 if levels.get("is_true_crossing", True) else 0,
                    levels.get("call_wall"),
                    levels.get("put_wall"),
                    regime_info.get("regime"),
                    stats.get("net_gex", 0),
                    em_pts,
                    confidence_info.get("score"),
                    staleness_info.get("freshness_score"),
                    stats.get("coverage_ratio"),
                    stats.get("pc_ratio"),
                    stats.get("gex_ratio"),
                    stats.get("call_iv"),
                    stats.get("put_iv"),
                ),
            )
            _conn.commit()
        except sqlite3.Error:
            pass  # silently skip on DB errors


def get_history(days=30):
    """Get historical snapshots as a list of dicts, most recent first."""
    with _lock:
        rows = _conn.execute(
            """SELECT * FROM gex_snapshots
               WHERE date >= date('now', ?)
               ORDER BY timestamp DESC""",
            (f"-{days} days",),
        ).fetchall()
    return [dict(r) for r in rows]


def get_zero_gamma_trend(days=14):
    """Get zero gamma values over time. Returns list of (date, zero_gamma, spot) tuples."""
    with _lock:
        rows = _conn.execute(
            """SELECT date, zero_gamma, spot
               FROM gex_snapshots
               WHERE date >= date('now', ?)
               ORDER BY timestamp ASC""",
            (f"-{days} days",),
        ).fetchall()
    return [(r["date"], r["zero_gamma"], r["spot"]) for r in rows]


def get_daily_summary(days=30):
    """Get one row per day (latest snapshot of each day). Returns list of dicts."""
    with _lock:
        rows = _conn.execute(
            """SELECT * FROM gex_snapshots
               WHERE id IN (
                   SELECT MAX(id) FROM gex_snapshots
                   WHERE date >= date('now', ?)
                   GROUP BY date
               )
               ORDER BY date DESC""",
            (f"-{days} days",),
        ).fetchall()
    return [dict(r) for r in rows]
