"""Backup-cron recovery for the required Monday snapshot workflow."""
from datetime import datetime
import sqlite3
from zoneinfo import ZoneInfo

import pytest

import scheduled_snapshot as snapshot


NY = ZoneInfo("America/New_York")
RUN_NOW = datetime(2026, 3, 16, 9, 45, tzinfo=NY)
WEEK_KEY = "2026-03-16"


def test_backup_retries_when_em_saved_but_weekly_setup_failed():
    """Run 1 saves EM then fails setup; Run 2 must not deduplicate."""
    state = {"daily_em": True, "weekly_em": True, "weekly_setup": False}

    def em_lookup(key, ticker, em_type):
        return {"expected_move_pts": 25.0} if state[f"{em_type}_em"] else None

    completion = snapshot._snapshot_completion_state(
        ticker="SPX",
        today_str=WEEK_KEY,
        run_now=RUN_NOW,
        is_week_first_trading_day=True,
        em_lookup=em_lookup,
        weekly_key_fn=lambda _: WEEK_KEY,
        weekly_setup_checker=lambda ticker, week: state["weekly_setup"],
    )

    assert completion == {
        "daily_em": True,
        "weekly_em": True,
        "weekly_setup": False,
        "complete": False,
    }

    # The backup retries the missing required stage; after that succeeds, a
    # later duplicate is allowed to exit green.
    state["weekly_setup"] = True
    completion = snapshot._snapshot_completion_state(
        ticker="SPX",
        today_str=WEEK_KEY,
        run_now=RUN_NOW,
        is_week_first_trading_day=True,
        em_lookup=em_lookup,
        weekly_key_fn=lambda _: WEEK_KEY,
        weekly_setup_checker=lambda ticker, week: state["weekly_setup"],
    )
    assert completion["complete"] is True


def test_weekly_setup_completion_requires_current_anchor_and_model(monkeypatch):
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE weekly_setup (week_start TEXT, ticker TEXT, "
        "monday_open REAL, monday_vix REAL, captured_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE saved_models (model_name TEXT, ticker TEXT, fitted_at TEXT)"
    )
    conn.execute(
        "INSERT INTO weekly_setup VALUES (?, ?, ?, ?, ?)",
        (WEEK_KEY, "SPX", 6700.0, 18.0, "2026-03-16T13:31:00+00:00"),
    )
    conn.execute(
        "INSERT INTO saved_models VALUES (?, ?, ?)",
        (snapshot._CALIBRATION_SPEC, "SPX", "2026-03-09T13:35:00+00:00"),
    )
    conn.commit()
    monkeypatch.setattr("range_finder.db.get_connection", lambda: conn)

    assert snapshot._weekly_setup_artifacts_complete("SPX", WEEK_KEY) is False

    conn.execute(
        "UPDATE saved_models SET fitted_at = ? WHERE model_name = ? AND ticker = ?",
        ("2026-03-16T13:35:00+00:00", snapshot._CALIBRATION_SPEC, "SPX"),
    )
    conn.commit()
    assert snapshot._weekly_setup_artifacts_complete("SPX", WEEK_KEY) is True


def test_required_weekly_setup_failure_is_not_a_successful_exit():
    def fail_setup():
        raise RuntimeError("feature rebuild failed")

    with pytest.raises(RuntimeError, match="feature rebuild failed"):
        snapshot._execute_required_weekly_setup(fail_setup)

    with pytest.raises(RuntimeError, match="did not complete"):
        snapshot._execute_required_weekly_setup(lambda: False)

    assert snapshot._execute_required_weekly_setup(lambda: True) is True
