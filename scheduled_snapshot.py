#!/usr/bin/env python3
"""
Scheduled GEX snapshot capture — runs WITHOUT Streamlit.

Designed to be triggered by GitHub Actions (or any cron scheduler) at:
  - 9:30 AM ET  (market open — freeze opening prices, EM levels)

Required env vars:
  TRADIER_TOKEN  — Tradier API bearer token
  DATABASE_URL   — Postgres connection string (e.g. Neon)
  FRED_API_KEY   — (optional) FRED API key for risk-free rate
"""
from __future__ import annotations

import os
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_logger = logging.getLogger(__name__)


def capture_snapshot():
    """Run the full GEX pipeline and save a snapshot + EM to Postgres."""

    # ── Validate env ──
    tradier_token = os.environ.get("TRADIER_TOKEN", "")
    if not tradier_token:
        _logger.error("TRADIER_TOKEN not set — aborting")
        sys.exit(1)

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        _logger.error("DATABASE_URL not set — nowhere to save snapshot")
        sys.exit(1)

    fred_key = os.environ.get("FRED_API_KEY", "")
    ticker = os.environ.get("TICKER", "SPX")

    # ── Imports (after env check so errors are clear) ──
    from phase1.market_clock import now_ny, get_calendar_snapshot
    from phase1.data_client import TradierDataClient
    from phase1.rates import fetch_risk_free_rate
    from phase1.parity import get_reference_spot_details
    import phase1.gex_engine as gex_engine
    from phase1.confidence import build_run_confidence
    from phase1.staleness import build_staleness_info
    from phase1.expected_move import build_expected_move_analysis
    from phase1.futures_data import fetch_es_from_yahoo, build_futures_context
    from phase1.gex_history import save_snapshot, save_em_snapshot

    # FORCE_WEEKLY_SETUP=1 bypasses the market-hours / weekend / holiday gates
    # so the workflow can be triggered manually (e.g. after a DB wipe) on any
    # day at any time to rebuild features and refit the HAR model. When set,
    # the GEX snapshot may be using stale Tradier quotes (API returns last
    # tick), which is fine — the weekly setup cares about SPX/VIX history
    # from yfinance, not the current GEX value.
    force_weekly_setup = os.environ.get("FORCE_WEEKLY_SETUP", "").strip() in ("1", "true", "yes")

    run_now = now_ny()
    today_str = run_now.strftime("%Y-%m-%d")
    _logger.info(f"Starting scheduled snapshot for {ticker} at {run_now.strftime('%I:%M:%S %p ET')} on {today_str}")
    if force_weekly_setup:
        _logger.info("FORCE_WEEKLY_SETUP=1 — bypassing market-hours / weekend / holiday gates")

    # ── Market hours sanity check ──
    # cron-job.org fires at exactly 9:30 AM ET, but guard against accidental
    # manual triggers outside market hours.
    hour, minute = run_now.hour, run_now.minute
    time_val = hour * 60 + minute  # minutes since midnight
    market_open  = 9 * 60 + 20     # 9:20 AM (small buffer)
    market_close = 10 * 60 + 15    # 10:15 AM (tolerate runner startup delay)
    if not force_weekly_setup and (time_val < market_open or time_val > market_close):
        _logger.info(f"Outside morning capture window ({run_now.strftime('%I:%M %p ET')}) — skipping")
        sys.exit(0)

    # Skip weekends (shouldn't happen with Mon-Fri cron, but just in case)
    if not force_weekly_setup and run_now.weekday() >= 5:
        _logger.info("Weekend — skipping")
        sys.exit(0)

    # ── Holiday check via pandas_market_calendars ──
    from phase1.market_clock import get_session_state
    from phase1.config import CASH_CALENDAR
    session = get_session_state(CASH_CALENDAR, run_now)
    if not force_weekly_setup and session.market_open is None:
        _logger.info(f"Market holiday ({today_str}) — skipping")
        sys.exit(0)

    # ── Wait for market open (9:30 AM ET) if triggered slightly early ──
    import time
    market_open_time = 9 * 60 + 30  # 9:30 AM in minutes
    if not force_weekly_setup and time_val < market_open_time:
        while True:
            wait_now = now_ny()
            current_minutes = wait_now.hour * 60 + wait_now.minute
            if current_minutes >= market_open_time:
                break
            wait_seconds = (market_open_time - current_minutes) * 60 - wait_now.second
            wait_seconds = max(wait_seconds, 1)
            _logger.info(f"Waiting {wait_seconds}s for market open (currently {wait_now.strftime('%I:%M:%S %p ET')})...")
            time.sleep(min(wait_seconds, 30))  # sleep in chunks to log progress

    # ── Ensure all Postgres tables exist (idempotent — cheap CREATE IF NOT
    # EXISTS). Runs every day so the range_finder tables are ready for any
    # code path that might touch them, including Tue–Fri runs after a fresh
    # database wipe. Phase1's own tables auto-init on module import. ──
    try:
        from range_finder.db import get_connection as _rf_get_connection
        from range_finder.db import init_all_tables as _rf_init_all_tables
        _rf_conn = _rf_get_connection()
        _rf_init_all_tables(_rf_conn)
        _logger.info("Range finder tables verified / created")
    except Exception as e:
        # Non-fatal: the GEX snapshot itself doesn't touch range_finder tables
        # on non-Monday runs. We log and keep going so the daily GEX capture
        # isn't held hostage by a range_finder init failure.
        _logger.warning(f"Range finder table init failed (non-fatal): {e}")

    # ── Fetch market data ──
    client = TradierDataClient(token=tradier_token)
    client.clear_cache()

    calendar_snapshot = get_calendar_snapshot(run_now)

    rfr_info = fetch_risk_free_rate(fred_key)
    rfr = rfr_info["rate"]
    _logger.info(f"Risk-free rate: {rfr:.4f} (source: {rfr_info['source']})")

    avail = client.get_expirations(ticker)
    if not avail:
        _logger.error(f"No expirations returned from Tradier API for {ticker}")
        sys.exit(1)
    nearest_exp = next((e for e in avail if e >= today_str), avail[0])

    spot_info = get_reference_spot_details(
        ticker=ticker,
        nearest_exp=nearest_exp,
        get_spot_price_func=client.get_spot_price,
        get_chain_cached_func=client.get_chain_cached,
        r=rfr,
        now=run_now,
    )
    spot = spot_info["spot"]
    _logger.info(f"{ticker} Spot: {spot:.2f} (source: {spot_info['source']})")

    # ── Select expirations: 0DTE + next 3 nearest ──
    target_exps = [e for e in avail if e >= today_str][:4]

    # ── Compute GEX ──
    gex_df, stats, all_options, _strike_sup, _exp_sup = (
        gex_engine.calculate_all(client, ticker, target_exps, spot, r=rfr, now=run_now)
    )

    if gex_df.empty:
        if force_weekly_setup:
            _logger.warning(
                "GEX calculation returned empty — skipping GEX snapshot save "
                "but continuing with weekly setup (force mode)"
            )
            levels = {"zero_gamma": spot, "call_wall": None, "put_wall": None,
                      "zero_gamma_is_true_crossing": False}
            regime_info = {"regime": "Unknown", "color": "#888888"}
        else:
            _logger.error("GEX calculation returned empty — no data to save")
            sys.exit(1)
    else:
        levels = gex_engine.find_key_levels(gex_df, spot, all_options=all_options, r=rfr)
        regime_info = gex_engine.get_gamma_regime_text(spot, levels["zero_gamma"])
    has_0dte = any(e == today_str for e in target_exps)
    staleness_info = build_staleness_info(calendar_snapshot, spot_info, stats, has_0dte=has_0dte)
    confidence_info = build_run_confidence(stats, spot_info, staleness_info=staleness_info)

    _logger.info(f"Zero Gamma: {levels['zero_gamma']:.2f} | Call Wall: {levels.get('call_wall')} | "
                 f"Put Wall: {levels.get('put_wall')} | Regime: {regime_info.get('regime')}")

    # ── Expected move ──
    index_quote = None
    prev_close = 0.0
    try:
        index_quote = client.get_full_quote(ticker)
        prev_close = index_quote.get("prevclose", 0.0)
    except Exception as e:
        _logger.warning(f"{ticker} quote fetch failed: {e}")

    dte0_exp = target_exps[0]
    dte0_entry = client.get_chain_cached(ticker, dte0_exp)
    dte0_calls = dte0_entry.get("calls", []) if dte0_entry.get("status") == "ok" else []
    dte0_puts = dte0_entry.get("puts", []) if dte0_entry.get("status") == "ok" else []

    futures_ctx = None
    try:
        yahoo_es = fetch_es_from_yahoo()
        if yahoo_es and yahoo_es.get("last") and prev_close > 0:
            futures_ctx = build_futures_context(
                es_last=yahoo_es["last"],
                es_high=yahoo_es.get("high"),
                es_low=yahoo_es.get("low"),
                spx_prevclose=prev_close,
                source=yahoo_es.get("source", "yahoo"),
            )
    except Exception as e:
        _logger.warning(f"ES futures fetch failed: {e}")

    em_analysis = build_expected_move_analysis(
        spot=spot,
        prev_close=prev_close,
        zero_gamma=levels["zero_gamma"],
        gamma_regime=regime_info.get("regime", "Unknown"),
        calls_0dte=dte0_calls,
        puts_0dte=dte0_puts,
        spy_quote=None,
        market_open=True,
        futures_context=futures_ctx,
    )

    # ── Save GEX snapshot ──
    try:
        save_snapshot(spot, levels, regime_info, stats, confidence_info, staleness_info, em_analysis, ticker=ticker)
        _logger.info(f"{ticker} GEX snapshot saved to Postgres")
    except Exception as e:
        _logger.error(f"Failed to save GEX snapshot: {e}")
        sys.exit(1)

    # ── Save EM snapshot (only first of day due to ON CONFLICT DO NOTHING) ──
    em_data = em_analysis.get("expected_move", {})
    if em_data.get("expected_move_pts"):
        try:
            save_em_snapshot(em_data, today_str, ticker=ticker, em_type="daily")
            _logger.info(f"Daily EM snapshot saved: {em_data['expected_move_pts']:.2f} pts")
        except Exception as e:
            _logger.warning(f"Daily EM snapshot save failed: {e}")

    # ── Weekly EM: capture on Mondays (or Tuesday if Monday was a holiday) ──
    # Intended freeze day is Monday's open, but if Monday's cron run failed
    # we still want some snap for the week rather than an indefinitely blank
    # marker. On Tue–Fri, capture only if no weekly snap exists yet for this
    # week's Monday key (save_em_snapshot uses ON CONFLICT DO NOTHING so a
    # successful Monday capture is never overwritten).
    from phase1.expected_move import find_weekly_expiration, find_monthly_expiration, compute_em_for_expiration
    from phase1.gex_history import get_weekly_em_date_key, get_monthly_em_date_key, get_em_snapshot

    is_monday = run_now.weekday() == 0
    is_tuesday_after_holiday = False
    if run_now.weekday() == 1:
        monday = run_now - __import__('datetime').timedelta(days=1)
        mon_session = get_session_state(CASH_CALENDAR, monday)
        is_tuesday_after_holiday = mon_session.market_open is None

    weekly_key = get_weekly_em_date_key(run_now)
    should_capture_weekly = is_monday or is_tuesday_after_holiday
    if not should_capture_weekly:
        # Backfill path: no snap yet for this week → capture today.
        try:
            existing_weekly = get_em_snapshot(weekly_key, ticker=ticker, em_type="weekly")
        except Exception:
            existing_weekly = None
        if not (existing_weekly and existing_weekly.get("expected_move_pts")):
            should_capture_weekly = True
            _logger.info(f"No weekly EM snap found for {weekly_key} — backfilling from today's open")

    if should_capture_weekly:
        weekly_exp = find_weekly_expiration(avail, run_now.date())
        if weekly_exp:
            weekly_em = compute_em_for_expiration(client, ticker, weekly_exp, spot)
            if weekly_em and weekly_em.get("expected_move_pts"):
                try:
                    save_em_snapshot(weekly_em, weekly_key, ticker=ticker, em_type="weekly")
                    _logger.info(f"Weekly EM saved (key={weekly_key}): ±{weekly_em['expected_move_pts']:.2f} pts (exp: {weekly_exp})")
                except Exception as e:
                    _logger.warning(f"Weekly EM save failed: {e}")

    # ── OpEx-cycle EM: capture on the first trading day of the new cycle ──
    # The OpEx cycle runs from the Monday following each 3rd-Friday standard
    # expiration through the next 3rd Friday. "First trading day of the cycle"
    # is normally that Monday, or the first non-holiday weekday after it.
    # Same backfill logic as weekly: if that run was missed, capture from a
    # later day rather than leaving the cycle blank.
    from datetime import date as _date, timedelta as _td

    monthly_key = get_monthly_em_date_key(run_now)  # = Monday after last OpEx
    cycle_open_mon = _date.fromisoformat(monthly_key)

    # Find the actual first trading day of this cycle (skip holidays).
    cycle_first_trading_day = None
    walker = cycle_open_mon
    for _ in range(7):
        probe = run_now.replace(
            year=walker.year, month=walker.month, day=walker.day,
            hour=12, minute=0, second=0, microsecond=0,
        )
        probe_session = get_session_state(CASH_CALENDAR, probe)
        if walker.weekday() < 5 and probe_session.market_open is not None:
            cycle_first_trading_day = walker
            break
        walker = walker + _td(days=1)

    is_cycle_open = (cycle_first_trading_day == run_now.date())
    should_capture_monthly = is_cycle_open
    if not should_capture_monthly:
        try:
            existing_monthly = get_em_snapshot(monthly_key, ticker=ticker, em_type="monthly")
        except Exception:
            existing_monthly = None
        if not (existing_monthly and existing_monthly.get("expected_move_pts")):
            should_capture_monthly = True
            _logger.info(f"No OpEx-cycle EM snap found for {monthly_key} — backfilling from today's open")

    if should_capture_monthly:
        monthly_exp = find_monthly_expiration(avail, run_now.date())
        if monthly_exp:
            monthly_em = compute_em_for_expiration(client, ticker, monthly_exp, spot)
            if monthly_em and monthly_em.get("expected_move_pts"):
                try:
                    save_em_snapshot(monthly_em, monthly_key, ticker=ticker, em_type="monthly")
                    _logger.info(f"OpEx-cycle EM saved (key={monthly_key}): ±{monthly_em['expected_move_pts']:.2f} pts (exp: {monthly_exp})")
                except Exception as e:
                    _logger.warning(f"OpEx-cycle EM save failed: {e}")

    # =========================================================================
    # WEEKLY SPREAD FINDER SETUP (Monday open, post-holiday Tuesday, or forced)
    # =========================================================================
    # FORCE_WEEKLY_SETUP (set at top of function) lets a manual
    # workflow_dispatch trigger a full rebuild on any day — useful for
    # bootstrapping a fresh Postgres without waiting for Monday's cron.
    if is_monday or is_tuesday_after_holiday or force_weekly_setup:
        if force_weekly_setup and not (is_monday or is_tuesday_after_holiday):
            _logger.info("FORCE_WEEKLY_SETUP=1 — running weekly spread finder setup on non-Monday")
        else:
            _logger.info("Running weekly spread finder setup...")
        try:
            _run_weekly_spread_setup(ticker, spot, run_now, fred_key, client, avail,
                                     levels, regime_info)
        except Exception as e:
            _logger.error(f"Weekly spread finder setup failed: {e}")

    _logger.info("Scheduled snapshot complete")


def _run_weekly_spread_setup(ticker, spot, run_now, fred_key, client, avail,
                              levels, regime_info):
    """Run the full spread finder pipeline: refresh data, rebuild features,
    save GEX, fit model, and persist Monday open + VIX."""
    import yfinance as yf
    from datetime import timedelta

    from range_finder.db import get_connection, init_all_tables
    from range_finder.data_collector import (
        fetch_spx_vix, save_spx_vix,
        fetch_fred_macro, save_fred_macro,
        build_event_flags,
    )
    from range_finder.feature_builder import build_features
    from range_finder.gex_bridge import (
        GEXContext, extract_gex_context, save_gex_to_range_finder,
    )
    from range_finder.har_model import (
        time_series_split, fit_model, evaluate_oos,
        save_model, MODEL_SPECS, forecast_next_week,
    )
    from range_finder.spread_levels import build_spread_plan, log_spread_plan

    conn = get_connection()
    init_all_tables(conn)

    # ── Step 1: Refresh market data ──
    _logger.info("  1/4 Refreshing SPX/VIX weekly data from yfinance...")
    try:
        df_spx = fetch_spx_vix(years=6)
        rows = save_spx_vix(conn, df_spx)
        _logger.info(f"  SPX/VIX: {len(df_spx)} weeks fetched, {rows} new")
    except Exception as e:
        _logger.warning(f"  SPX/VIX fetch failed: {e} (continuing with existing data)")

    if fred_key:
        try:
            df_macro = fetch_fred_macro(years=6)
            save_fred_macro(conn, df_macro)
            _logger.info(f"  FRED macro: {len(df_macro)} rows")
        except Exception as e:
            _logger.warning(f"  FRED fetch skipped: {e}")

    build_event_flags(conn)

    # ── Step 2: Rebuild features ──
    _logger.info("  2/4 Rebuilding feature matrix...")
    try:
        build_features(conn)
    except Exception as e:
        _logger.error(f"  Feature rebuild failed: {e}")
        return

    # ── Step 3: Save GEX to model ──
    _logger.info("  3/4 Saving GEX to range finder...")
    try:
        gex_ctx = extract_gex_context(levels, spot, regime_info)
        save_gex_to_range_finder(gex_ctx, conn)
    except Exception as e:
        _logger.warning(f"  GEX save failed: {e}")

    # ── Step 4: Fit model and forecast ──
    _logger.info("  4/4 Fitting model and generating forecast...")
    model_choice = "M3_extended"
    try:
        from range_finder.feature_builder import get_features
        df_feat = get_features(conn)
        if df_feat.empty:
            _logger.warning("  No features available — skipping forecast")
            return

        feat_cols = MODEL_SPECS.get(model_choice, [])
        avail_cols = [c for c in feat_cols if c in df_feat.columns and df_feat[c].notna().sum() > 20]

        X_train, X_test, y_train, y_test = time_series_split(df_feat, feature_cols=avail_cols)
        result = fit_model(X_train, y_train, model_name=model_choice)
        metrics = evaluate_oos(result, X_test, y_test, model_name=model_choice)
        save_model(result, avail_cols, model_choice, metrics, conn=conn)

        _logger.info(f"  Model fitted: OOS R² = {metrics['oos_r2']:.4f}, MAE = {metrics['mae_pct']*100:.2f}%")
    except Exception as e:
        _logger.error(f"  Model fitting failed: {e}")

    # ── Save Monday open + VIX to DB ──
    _logger.info("  Saving Monday open + VIX...")
    try:
        vix_close = 18.0
        try:
            vix_hist = yf.Ticker("^VIX").history(period="5d")
            if not vix_hist.empty:
                vix_close = round(float(vix_hist["Close"].dropna().iloc[-1]), 2)
        except Exception:
            pass

        days_since_monday = run_now.weekday()
        monday = run_now - timedelta(days=days_since_monday)
        week_start = monday.strftime("%Y-%m-%d")

        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()

        conn.execute("""
            INSERT INTO weekly_setup (week_start, ticker, monday_open, monday_vix, captured_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (week_start, ticker) DO UPDATE SET
                monday_open = excluded.monday_open,
                monday_vix  = excluded.monday_vix,
                captured_at = excluded.captured_at
        """, (week_start, ticker, round(spot, 2), vix_close, now_iso))
        conn.commit()

        _logger.info(f"  Monday open saved: {ticker} = {spot:.2f}, VIX = {vix_close}")
    except Exception as e:
        _logger.warning(f"  Monday open/VIX save failed: {e}")


if __name__ == "__main__":
    capture_snapshot()
