"""
Expected-Move snapshot freeze/restore helpers.

These helpers freeze the daily / weekly / OpEx-cycle expected move at the
appropriate session start, persist the captured value to Postgres, and restore
it on subsequent renders so the chart's EM markers stay stable through each
session. Imported by `streamlit_app.py` (daily/weekly/monthly freeze) and
`ui_spread_finder.py` (Monday-open spot freeze via `_is_weekly_freeze_day`).
"""
from __future__ import annotations

from datetime import timedelta

import streamlit as st

from phase1.market_clock import now_ny
from phase1.gex_history import save_em_snapshot, get_em_snapshot


def _is_trading_day(dt_obj):
    """Check if a given date is a trading day (not weekend, not holiday)."""
    from phase1.market_clock import get_session_state
    from phase1.config import CASH_CALENDAR
    if dt_obj.weekday() >= 5:
        return False
    sess = get_session_state(CASH_CALENDAR, dt_obj if hasattr(dt_obj, 'hour') else None)
    # If market_open is None, the schedule was empty → holiday
    return sess.market_open is not None


def _is_weekly_freeze_day(now_et):
    """True if today is Monday (or Tuesday if Monday was a holiday)."""
    if now_et.weekday() == 0:  # Monday
        return True
    if now_et.weekday() == 1:  # Tuesday — check if Monday was a holiday
        monday = now_et - timedelta(days=1)
        if not _is_trading_day(monday):
            return True
    return False


def _is_monthly_freeze_day(now_et):
    """
    True if today is the first trading day of the current OpEx cycle — i.e.
    the first trading day strictly after the most recent standard monthly
    3rd-Friday OpEx. Normally that's the Monday after OpEx; if Monday is a
    holiday it slips to Tuesday, etc.
    """
    from phase1.gex_history import get_monthly_em_date_key
    from datetime import date as _date

    today = now_et.date() if hasattr(now_et, 'date') else now_et

    # get_monthly_em_date_key returns the *Monday* following the most recent
    # 3rd Friday. Walk forward from that Monday to find the actual first
    # trading day of the cycle (skipping holidays).
    cycle_open_mon = _date.fromisoformat(get_monthly_em_date_key(now_et))
    candidate = cycle_open_mon
    for _ in range(7):  # at most a week of consecutive holidays
        # _is_trading_day accepts a date or datetime; pass a datetime anchored
        # at noon in the same tz as now_et so the market-calendar lookup works.
        probe = now_et.replace(
            year=candidate.year, month=candidate.month, day=candidate.day,
            hour=12, minute=0, second=0, microsecond=0,
        )
        if _is_trading_day(probe):
            return candidate == today
        candidate = candidate + timedelta(days=1)
    return False


def _apply_typed_em_snapshot(em_live_data, is_market_open, spot, ticker, em_type, date_key, should_freeze):
    """
    Generic EM snapshot freeze/restore for any em_type (daily/weekly/monthly).

    em_live_data: dict with expected_move_pts, upper_level, etc. (or None)
    Returns the frozen snapshot dict, or None if unavailable.
    """
    sk_snap = f"em_snapshot_{em_type}_{ticker}"
    sk_date = f"em_snapshot_date_{em_type}_{ticker}"
    sk_time = f"em_snapshot_time_{em_type}_{ticker}"

    # Clear stale snapshot from a previous period
    if st.session_state.get(sk_date) != date_key:
        st.session_state.pop(sk_snap, None)
        st.session_state.pop(sk_date, None)
        st.session_state.pop(sk_time, None)

    # Try to restore from Postgres whenever session state is empty — this must
    # run regardless of market-open status, otherwise fresh sessions outside
    # 9:30–16:00 ET never pick up the frozen weekly/monthly snap that the
    # scheduled cron already persisted, and downstream code falls back to the
    # live (drifting) EM.
    if sk_snap not in st.session_state:
        try:
            db_snap = get_em_snapshot(date_key, ticker=ticker, em_type=em_type)
        except Exception:
            db_snap = None
        if db_snap and db_snap.get("expected_move_pts"):
            st.session_state[sk_snap] = db_snap
            st.session_state[sk_date] = date_key
            captured = db_snap.get("captured_at", "")
            if captured:
                try:
                    from datetime import datetime as dt_cls
                    cap_dt = dt_cls.fromisoformat(captured)
                    st.session_state[sk_time] = cap_dt.strftime("%I:%M:%S %p ET")
                except Exception:
                    st.session_state[sk_time] = "restored from DB"
            else:
                st.session_state[sk_time] = "restored from DB"

    if not is_market_open:
        # Outside market hours we never capture anything new — just return
        # whatever we restored (or None if nothing was ever saved).
        return st.session_state.get(sk_snap)

    # First capture of the period — only on the correct freeze day
    if sk_snap not in st.session_state and should_freeze and em_live_data and em_live_data.get("expected_move_pts"):
        snap = {
            "expected_move_pts": em_live_data.get("expected_move_pts"),
            "expected_move_pct": em_live_data.get("expected_move_pct"),
            "upper_level": em_live_data.get("upper_level"),
            "lower_level": em_live_data.get("lower_level"),
            "straddle": em_live_data.get("straddle"),
            "expiration": em_live_data.get("expiration"),
        }
        st.session_state[sk_snap] = snap
        st.session_state[sk_date] = date_key
        st.session_state[sk_time] = now_ny().strftime("%I:%M:%S %p ET")
        try:
            save_em_snapshot(em_live_data, date_key, ticker=ticker, em_type=em_type)
        except Exception:
            pass

    return st.session_state.get(sk_snap)


def _apply_em_snapshot(em_analysis, is_market_open, regime, levels, spot, ticker="SPX"):
    """Freeze expected move at first market-hours refresh; recompute classification with live data."""
    today_str = now_ny().strftime("%Y-%m-%d")
    em_live = em_analysis.get("expected_move", {})

    snap = _apply_typed_em_snapshot(em_live, is_market_open, spot, ticker, "daily", today_str, True)

    if snap and snap.get("expected_move_pts"):
        em_pct = snap.get("expected_move_pct") or em_analysis.get("expected_move", {}).get("expected_move_pct")
        em_analysis["expected_move"] = {
            **em_analysis.get("expected_move", {}),
            "expected_move_pts": snap["expected_move_pts"],
            "expected_move_pct": em_pct,
            "upper_level": snap["upper_level"],
            "lower_level": snap["lower_level"],
            "straddle": snap.get("straddle"),
        }
        on_pts = em_analysis.get("overnight_move", {}).get("overnight_move_pts")
        if on_pts is not None and snap["expected_move_pts"] > 0:
            from phase1.expected_move import classify_session
            em_analysis["classification"] = classify_session(
                expected_move_pts=snap["expected_move_pts"],
                overnight_move_pts=on_pts,
                gamma_regime=regime["regime"],
            )
            em_analysis["classification"]["move_source"] = ticker.lower()
        if snap.get("upper_level") is not None:
            em_analysis["level_context"] = {
                "em_upper": snap["upper_level"],
                "em_lower": snap["lower_level"],
                "zero_gamma": round(levels["zero_gamma"], 2),
                "zero_gamma_within_em": (
                    snap["lower_level"] <= levels["zero_gamma"] <= snap["upper_level"]
                ),
                "zero_gamma_distance_to_spot": round(spot - levels["zero_gamma"], 2),
            }

    return em_analysis
