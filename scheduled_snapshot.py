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
    from phase1.market_clock import now_ny
    from phase1.data_client import TradierDataClient
    from phase1.rates import fetch_risk_free_rate
    from phase1.expected_move import build_expected_move_analysis
    from phase1.gamma_archive_capture import calculate_gamma_archive_observation
    from phase1.gex_history import save_em_snapshot
    from phase1.gamma_level_history import archive_computed_gamma_levels

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

    # ── Cadence gate: single-name tickers run weekly-only ──
    # Index/ETF tickers capture every market day (GEX/EM history + 0DTE for
    # SPX/XSP/SPY). Single stocks (NVDA/JPM/CAT/...) only need the weekly HAR
    # fit + the weekly/monthly EM capture, which fire on the week's first
    # trading day — exactly the SPX/XSP weekly-setup cadence. Skipping them on
    # the other weekdays avoids 4 no-op runs/week (and the Neon compute they'd
    # wake). FORCE_WEEKLY_SETUP overrides. Computed here (before the 9:30 wait,
    # the Tradier chain fetch, and GEX compute) so a skip is cheap. is_monday /
    # is_tuesday_after_holiday are reused by the weekly-setup block below.
    is_monday = run_now.weekday() == 0
    is_tuesday_after_holiday = False
    if run_now.weekday() == 1:
        monday = run_now - __import__('datetime').timedelta(days=1)
        mon_session = get_session_state(CASH_CALENDAR, monday)
        is_tuesday_after_holiday = mon_session.market_open is None
    is_week_first_trading_day = is_monday or is_tuesday_after_holiday

    from phase1.ticker_config import get_config as _get_cfg
    if (_get_cfg(ticker).get("category") == "stock"
            and not force_weekly_setup and not is_week_first_trading_day):
        _logger.info(
            f"{ticker}: single-name weekly-only cadence — today is not the week's "
            f"first trading day. Skipping (no daily GEX/EM run)."
        )
        sys.exit(0)

    # ── Already-captured-today guard (backup-cron dedup) ──
    # The workflow has TWO triggers: cron-job.org at 9:28 ET (primary, DST-
    # aware) and a GitHub `schedule:` backup pair (UTC, one lands ~9:45 ET in
    # each DST half — the other half's fire exits green at the window guard
    # above). When the primary already captured today, the backup finds the
    # em_snapshots row it wrote and exits green here, before any Tradier or
    # FRED call. When cron-job.org silently dies (expired PAT, service
    # outage), no row exists and the backup performs the day's capture —
    # that's the dead-man's switch. A DB hiccup during the check proceeds
    # with capture (idempotent ON CONFLICT writes make a duplicate harmless);
    # skipping on error would be the dangerous direction.
    if not force_weekly_setup:
        try:
            from phase1.gex_history import get_em_snapshot, get_weekly_em_date_key
            _is_stock = _get_cfg(ticker).get("category") == "stock"
            _daily_done = True if _is_stock else (
                get_em_snapshot(today_str, ticker=ticker, em_type="daily") is not None)
            _weekly_done = True
            if is_week_first_trading_day:
                _weekly_key = get_weekly_em_date_key(run_now)
                _weekly_done = get_em_snapshot(
                    _weekly_key, ticker=ticker, em_type="weekly") is not None
            if _daily_done and _weekly_done:
                _logger.info(
                    f"{ticker}: today's snapshot artifacts already exist "
                    f"(daily={_daily_done}, weekly_checked={is_week_first_trading_day}) "
                    f"— duplicate/backup fire, exiting green")
                sys.exit(0)
        except SystemExit:
            raise
        except Exception as _e:
            _logger.warning(f"Dedup guard check failed ({_e}) — proceeding with capture")

    # ── Pre-open warm-up ──
    # The cron fires slightly before 9:30 so the runner is already booted by
    # the bell. Anything that doesn't require the opening tick runs HERE,
    # during that idle window, so the post-9:30 critical path is only
    # spot/chain fetch + GEX compute + DB save.
    #
    # Safe to pre-fetch: Tradier client construction, FRED risk-free rate
    # (daily series, not 9:30-gated), and the Tradier expirations list
    # (set by CBOE in advance, doesn't change intraday).
    # NOT safe to pre-fetch: spot price, option chains — those need the open.
    #
    # Note: range_finder table init is intentionally NOT done here. The daily
    # GEX snapshot doesn't read range_finder tables, and _run_weekly_spread_setup
    # calls init_all_tables itself when it actually needs them. A previous
    # version inited every day "defensively" — that turned out to cost up to
    # 15+ min on Neon's free tier when DDL serialised against the parallel
    # XSP matrix job under compute throttling, which delayed the snapshot save
    # well past 9:30. Removed.

    client = TradierDataClient(token=tradier_token)
    client.clear_cache()

    rfr_info = fetch_risk_free_rate(fred_key)
    rfr = rfr_info["rate"]
    rfr_curve = rfr_info.get("curve")  # None → flat-rate fallback path
    _logger.info(f"Risk-free rate: {rfr:.4f} (source: {rfr_info['source']})")
    if rfr_curve:
        curve_summary = ", ".join(f"{int(k)}d={v*100:.2f}%" for k, v in sorted(rfr_curve.items()))
        _logger.info(f"Risk-free curve: {curve_summary}")

    avail = client.get_expirations(ticker)
    if not avail:
        _logger.error(f"No expirations returned from Tradier API for {ticker}")
        sys.exit(1)

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

    # Re-capture wall clock after the wait so the snapshot's time-to-expiry
    # math and logged "as of" timestamp reflect the actual post-open moment
    # rather than when the script first started.
    run_now = now_ny()

    # ── Canonical production Gamma calculation (shared with intraday) ──
    calculation = calculate_gamma_archive_observation(
        client=client,
        ticker=ticker,
        available_expirations=avail,
        captured_at=run_now,
        risk_free_rate=rfr,
        risk_free_curve=rfr_curve,
    )
    target_exps = list(calculation.target_expirations)
    spot = calculation.spot
    spot_info = calculation.spot_info
    gex_df = calculation.gex_df
    stats = calculation.stats
    levels = calculation.levels
    regime_info = calculation.regime_info
    staleness_info = calculation.staleness_info
    confidence_info = calculation.confidence_info
    _logger.info(f"{ticker} Spot: {spot:.2f} (source: {spot_info['source']})")

    if gex_df.empty:
        if force_weekly_setup:
            _logger.warning(
                "GEX calculation returned empty — skipping GEX snapshot save "
                "but continuing with weekly setup (force mode)"
            )
            # Use concrete spot-relative fallback values instead of None.
            # dict.get(key, default) returns None when the key EXISTS with
            # value None, so downstream code that does e.g.
            #     levels.get("call_wall", spot + 50)
            # would still see None and blow up on the next f"{...:.0f}"
            # format (TypeError: unsupported format string passed to
            # NoneType.__format__). Supplying a real fallback sidesteps
            # the gotcha entirely.
            levels = {
                "zero_gamma": spot,
                "zero_gamma_is_true_crossing": False,
                "zero_gamma_type": "Fallback node",
                "zero_gamma_method": "force_mode_empty_chain",
                "zero_gamma_abs_gex": None,
                "call_wall": round(spot * 1.01, 2),
                "call_wall_gex": 0.0,
                "call_wall_cluster": None,
                "put_wall": round(spot * 0.99, 2),
                "put_wall_gex": 0.0,
                "put_wall_cluster": None,
                "net_gex": 0.0,
                "per_exp_zero_gamma": {},
            }
            regime_info = {
                "regime": "Unknown",
                "color": "#888888",
                "distance_text": "n/a",
                "note": "Force-mode synthetic fallback — GEX chain was empty",
                "abs_distance": 0.0,
            }
        else:
            _logger.error("GEX calculation returned empty — no data to save")
            sys.exit(1)

    _logger.info(f"Zero Gamma: {levels['zero_gamma']:.2f} | Call Wall: {levels.get('call_wall')} | "
                 f"Put Wall: {levels.get('put_wall')} | Regime: {regime_info.get('regime')}")

    # Successful calculations already include EM from the cached nearest
    # chain. Force-mode's synthetic fallback is operational-only, but the
    # existing scheduled EM behavior remains intact for weekly rebuilding.
    em_analysis = calculation.em_analysis
    if em_analysis is None:
        dte0_exp = target_exps[0]
        dte0_entry = client.get_chain_cached(ticker, dte0_exp)
        dte0_calls = dte0_entry.get("calls", []) if dte0_entry.get("status") == "ok" else []
        dte0_puts = dte0_entry.get("puts", []) if dte0_entry.get("status") == "ok" else []
        em_analysis = build_expected_move_analysis(
            spot=spot,
            prev_close=float(calculation.quote.get("prevclose") or 0.0),
            zero_gamma=levels["zero_gamma"],
            gamma_regime=regime_info.get("regime", "Unknown"),
            calls_0dte=dte0_calls,
            puts_0dte=dte0_puts,
            market_open=True,
            expiration=dte0_exp,
            as_of=run_now.date(),
        )

    # ── Archive derived Gamma levels (first observation in the 09:30 bucket) ──
    # FORCE_WEEKLY_SETUP can proceed with synthetic spot-relative levels when
    # the option chain is empty; those are operational fallbacks, not research
    # observations, and must not enter the forward archive.
    if not gex_df.empty:
        archive_computed_gamma_levels(
            captured_at=run_now,
            ticker=ticker,
            spot=spot,
            levels=levels,
            regime_info=regime_info,
            stats=stats,
            confidence_info=confidence_info,
            staleness_info=staleness_info,
            em_analysis=em_analysis,
        )

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

    # is_monday / is_tuesday_after_holiday are computed once near the top of
    # capture_snapshot (the single-name cadence gate needs them before the
    # market-open wait), so they're reused here rather than recomputed.
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

    # The 0DTE spread finder setup (daily_spx refresh, daily HAR refit,
    # forecast_log_daily logging) was deliberately removed (2026-07): a
    # 1,036-session audit showed the VRP verdict had no predictive edge and
    # the feature was retired. DB tables keep their historical rows.

    # The Public.com fill-history sync ran here daily (gated to the SPX matrix
    # entry) to feed the Pre-Flight behavioral gate. Both the Monday Cockpit
    # and Pre-Flight tabs were removed (2026-08), leaving the sync with no
    # consumer, so the whole public_api package went with them — along with
    # its three tables and the PUBLIC_API_SECRET / PUBLIC_ACCOUNT_ID secrets.
    # Nothing here touches Public.com any more.

    _logger.info("Scheduled snapshot complete")


# The weekly spec whose Monday forecast gets logged to spread_log for the
# PI-coverage calibration audit (range_finder/calibration.py). M3_extended
# is the app's default production spec (bootstrap --model default, UI
# preferred_model default) — the calibration series should track what the
# user actually trades off.
_CALIBRATION_SPEC = "M3_extended"


def _safe_float(val):
    """float(val) or None — tolerates None/NaN/strings from feature rows."""
    try:
        f = float(val)
        return f if f == f else None   # NaN != NaN
    except (TypeError, ValueError):
        return None


def _run_weekly_spread_setup(ticker, spot, run_now, fred_key, client, avail,
                              levels, regime_info):
    """Run the full spread finder pipeline: refresh data, rebuild features,
    save GEX, fit model, log the Monday plan, and persist Monday open + VIX."""
    import yfinance as yf
    from datetime import timedelta

    from range_finder.db import get_connection, init_all_tables
    from range_finder.data_collector import (
        fetch_spx_vix, save_spx_vix,
        fetch_underlying_weekly, save_underlying_weekly,
        populate_earnings_flags,
        fetch_fred_macro, save_fred_macro,
        build_event_flags,
        fred_key_status, FRED_API_KEY,
    )
    from range_finder.feature_builder import build_features
    from range_finder.gex_bridge import (
        GEXContext, extract_gex_context, save_gex_to_range_finder,
    )
    from range_finder.har_model import (
        time_series_split, fit_model, evaluate_oos,
        save_model, MODEL_SPECS, forecast_next_week,
        feature_has_enough_data,
    )
    from range_finder.spread_levels import build_spread_plan, log_spread_plan
    from phase1.ticker_config import (
        get_config, uses_own_har, has_single_name_earnings,
        feature_source_ticker,
    )

    cfg = get_config(ticker)

    conn = get_connection()
    init_all_tables(conn)

    # ── Step 1: Refresh market data ──
    # SPX/XSP keep using fetch_spx_vix (their HAR features come from the
    # weekly_spx table — SPX-derived; XSP rides on SPX at 1/10 scale).
    # Own-HAR tickers (QQQ/AMZN/AMD) fetch their own OHLC + vol-proxy series
    # into weekly_underlying. SPX/VIX history is also refreshed when an
    # own-HAR job runs because the macro term-structure (^VIX9D, ^VIX3M) and
    # macro features still come from VIX-based feeds.
    _logger.info("  1/4 Refreshing SPX/VIX weekly data from yfinance...")
    try:
        df_spx = fetch_spx_vix(years=6)
        rows = save_spx_vix(conn, df_spx)
        _logger.info(f"  SPX/VIX: {len(df_spx)} weeks fetched, {rows} new")
    except Exception as e:
        _logger.warning(f"  SPX/VIX fetch failed: {e} (continuing with existing data)")

    if uses_own_har(ticker):
        _logger.info(f"  1/4 Refreshing per-ticker weekly data ({ticker} via {cfg['yf_symbol']}, "
                     f"vol proxy {cfg['vol_proxy_yf']})...")
        try:
            df_t = fetch_underlying_weekly(
                ticker=ticker,
                yf_symbol=cfg["yf_symbol"],
                vol_proxy_yf=cfg["vol_proxy_yf"],
                years=6,
            )
            rows = save_underlying_weekly(conn, ticker, df_t)
            _logger.info(f"  {ticker}: {len(df_t)} weeks fetched, {rows} upserted")
        except Exception as e:
            _logger.warning(f"  {ticker} weekly fetch failed: {e} (continuing with existing data)")

        if has_single_name_earnings(ticker):
            try:
                n_earn = populate_earnings_flags(conn, ticker)
                _logger.info(f"  {ticker} earnings flags: {n_earn} weeks populated")
            except Exception as e:
                _logger.warning(f"  {ticker} earnings flag population failed: {e}")

    if fred_key:
        _logger.info(f"  FRED key: {fred_key_status()}")
        try:
            df_macro = fetch_fred_macro(years=6)
            save_fred_macro(conn, df_macro)
            _logger.info(f"  FRED macro: {len(df_macro)} rows")
        except Exception as e:
            if not FRED_API_KEY:
                _logger.warning("  FRED fetch skipped: FRED_API_KEY not set")
            else:
                _logger.warning(
                    f"  FRED fetch failed ({e}) — existing DB macro data still "
                    "valid; pipeline continues"
                )

    build_event_flags(conn)

    # ── Step 1b: Score expired spread_log weeks (SPX only) ──
    # The calibration loop: every Monday, fill in actual high/low/range for
    # any logged plan whose week has expired and hasn't been scored yet.
    # Runs right after the weekly_spx refresh so update_expiration_outcome's
    # zero-network DB path (weekly_spx) has the expired week's bar. This is
    # what accumulates the data behind range_finder/calibration.py — the
    # write path was dormant for months, so scoring loops over ALL unscored
    # past weeks, not just last week.
    if ticker == "SPX":
        try:
            from range_finder.spread_levels import update_expiration_outcome
            this_monday = (run_now - timedelta(days=run_now.weekday())).strftime("%Y-%m-%d")
            cur = conn.cursor()
            cur.execute(
                "SELECT week_start FROM spread_log "
                "WHERE ticker = 'SPX' AND outcome IS NULL AND week_start < ? "
                "ORDER BY week_start DESC LIMIT 12",
                (this_monday,),
            )
            unscored = [r[0] for r in cur.fetchall()]
            for wk in unscored:
                try:
                    outcome = update_expiration_outcome(wk, conn, ticker="SPX")
                    _logger.info(f"  Scored expired week {wk}: {outcome}")
                except Exception as e:
                    _logger.warning(f"  Scoring {wk} failed: {e}")
            if not unscored:
                _logger.info("  No unscored expired weeks in spread_log")
        except Exception as e:
            _logger.warning(f"  Outcome scoring step failed: {e} (continuing)")

    # ── Step 2: Save GEX to model (BEFORE the feature rebuild) ──
    #
    # ORDER (corrected after the 2026-07 audit): save_gex_to_range_finder
    # runs BEFORE build_features. feature_builder joins gex_inputs
    # same-week with NO lag ("GEX (Monday open — same week, no lag
    # needed)") — every TRAINING row's gex_normalized is that week's own
    # Monday-9:31 capture, written by that week's cron. Rebuilding
    # features before saving this Monday's GEX therefore left the one row
    # being SERVED (the current week) with gex_normalized = NaN, which
    # forecast_next_week silently fills with the training mean — a
    # train/serve skew where the live forecast used a mean while every
    # training observation used a real Monday reading. Saving first gives
    # the current week's row the exact same information timestamp
    # (Monday 9:31) as the training rows. The lag-1 features (VIX, HV,
    # macro) are shifted inside build_features and are unaffected by this
    # ordering.
    _logger.info("  2/4 Saving GEX to range finder...")
    _gex_saved = False
    try:
        gex_ctx = extract_gex_context(levels, spot, regime_info)
        save_gex_to_range_finder(gex_ctx, conn, ticker=ticker)
        _gex_saved = True
    except Exception as e:
        _logger.warning(f"  GEX save failed ({e}) — features will rebuild "
                        "without this week's GEX row (train mean fills in)")

    # ── Step 3: Rebuild features ──
    _logger.info(f"  3/4 Rebuilding feature matrix ({ticker})...")
    try:
        # A scaled mini (XSP→SPX) and SPX itself feed off the parent's
        # feature rows; own-HAR tickers (QQQ/SPY/NDX/AMZN/AMD) get their own build.
        build_target = feature_source_ticker(ticker)
        build_features(conn, ticker=build_target)
    except Exception as e:
        _logger.error(f"  Feature rebuild failed: {e}")
        return

    # ── Step 4: Fit every model spec and save them all ──
    # Fitting every spec Monday (not just M3_extended) means a user who
    # switches the Spread Finder's model dropdown mid-week to compare
    # M2_vix vs M1_baseline etc. gets a clean load-from-Postgres of
    # a Monday-frozen fit — no "click Forecast to fit it for the first
    # time" prompt, no mid-week refit that reproduces Monday's result
    # unnecessarily. OLS with HC3 on ~few hundred weekly rows is
    # milliseconds per spec, so fitting all 5 adds ~1–2s total.
    _logger.info(f"  4/4 Fitting all {len(MODEL_SPECS)} model specs...")
    _cal_fit = None   # (result, avail_cols) of _CALIBRATION_SPEC — set in the loop
    try:
        from range_finder.feature_builder import get_features
        # A scaled mini (XSP→SPX) loads its parent's shared HAR
        # features; own-HAR tickers (QQQ/SPY/NDX/AMZN/AMD) load their own rows.
        _features_ticker = feature_source_ticker(ticker)
        # Window pinned to TRAIN_WINDOW_YEARS so deeper weekly_spx backfills
        # (the 10y history experiment) can't silently retrain production.
        from range_finder.har_model import train_window_min_date
        df_feat = get_features(conn, min_date=train_window_min_date(),
                               exclude_covid=True, ticker=_features_ticker)
        if df_feat.empty:
            _logger.warning("  No features available — skipping forecast")
            return

        fitted = 0
        failed = []
        for spec_name in MODEL_SPECS:
            try:
                feat_cols  = MODEL_SPECS[spec_name]
                avail_cols = [c for c in feat_cols if feature_has_enough_data(df_feat, c)]

                if len(avail_cols) < 2:
                    _logger.info(f"    {spec_name}: skipping — only {len(avail_cols)} usable features")
                    continue

                X_train, X_test, y_train, y_test = time_series_split(df_feat, feature_cols=avail_cols)
                result  = fit_model(X_train, y_train, model_name=spec_name)
                metrics = evaluate_oos(result, X_test, y_test, model_name=spec_name)
                save_model(result, avail_cols, spec_name, metrics, conn=conn, ticker=ticker)

                if spec_name == _CALIBRATION_SPEC:
                    _cal_fit = (result, avail_cols)

                _logger.info(
                    f"    {spec_name}: OOS R² = {metrics['oos_r2']:.4f}, "
                    f"MAE = {metrics['mae_pct']*100:.2f}%  (features: {len(avail_cols)})"
                )
                fitted += 1
            except Exception as e:
                failed.append(spec_name)
                _logger.warning(f"    {spec_name}: fit failed — {e}")

        _logger.info(f"  Fitted {fitted}/{len(MODEL_SPECS)} specs" + (f" (failed: {failed})" if failed else ""))
    except Exception as e:
        _logger.error(f"  Model fitting stage failed: {e}")

    # ── Save Monday open + VIX to DB ──
    # Prefer the true daily-candle Open over the live mid-move `spot` /
    # live-last VIX tick. Yahoo publishes today's Open within seconds of
    # 9:30 ET, so by the time this cron actually runs the daily bar is
    # almost always available. Fall back to `spot` / live VIX only if
    # the bar is still empty (which mostly happens when the workflow
    # fires before Yahoo's first tick). The capture + persist logic is shared
    # with the UI mid-week self-heal (capture_and_save_monday_anchor) so the
    # cron and the on-demand path can never drift apart.
    _logger.info("  Saving Monday open + VIX...")
    monday_open = None
    monday_vix = None
    try:
        from range_finder.data_collector import capture_and_save_monday_anchor

        # Live vol-proxy last close as the VIX fallback for the rare case where
        # the daily-Open bar isn't published yet at 9:30 (mirrors the prior
        # inline behavior; the underlying falls back to `spot`).
        _vix_fallback = None
        try:
            _vp_hist = yf.Ticker(cfg["vol_proxy_yf"]).history(period="5d")
            if not _vp_hist.empty:
                _vix_fallback = round(float(_vp_hist["Close"].dropna().iloc[-1]), 2)
        except Exception:
            pass

        days_since_monday = run_now.weekday()
        monday = run_now - timedelta(days=days_since_monday)
        week_start = monday.strftime("%Y-%m-%d")

        # Anchor to the week's FIRST TRADING SESSION, not run_now.date().
        # On a normal Monday (or Tuesday-after-holiday) cron those are the
        # same day, so behavior is unchanged — but a FORCE_WEEKLY_SETUP
        # dispatch is allowed on any weekday ("refresh after a code change"),
        # and passing run_now.date() there would persist e.g. Wednesday's open
        # as this week's frozen Monday anchor via save_weekly_setup's
        # ON CONFLICT DO UPDATE, silently re-snapping every ticker's strikes
        # mid-week. Resolving the first session off the exchange calendar keeps
        # the anchor pinned to Monday (or the true first trading day) regardless
        # of when the job runs.
        anchor_date = monday.date()
        try:
            from phase1.market_clock import get_schedule
            from phase1.config import CASH_CALENDAR
            _friday = (monday + timedelta(days=4)).strftime("%Y-%m-%d")
            _week_sched = get_schedule(CASH_CALENDAR, week_start, _friday)
            if not _week_sched.empty:
                anchor_date = _week_sched.index[0].date()
        except Exception as _e:
            _logger.warning(f"  Anchor-date calendar lookup failed ({_e}); "
                            f"using Monday {anchor_date}")

        monday_open, monday_vix, open_source = capture_and_save_monday_anchor(
            conn, ticker, week_start, anchor_date,
            spot_fallback=spot, live_vix_fallback=_vix_fallback, cfg=cfg,
        )
        _logger.info(
            f"  Monday open saved: {ticker}={monday_open:.2f} ({open_source}), "
            f"VIX={monday_vix:.2f}"
        )
    except Exception as e:
        _logger.warning(f"  Monday open/VIX save failed: {e}")

    # ── Step 5: Log this week's plan for the calibration audit (SPX only) ──
    # Forecast the week with the production spec, build the plan the way the
    # UI would (no chain snap — the logged strikes are the model's raw
    # placement), and persist forecast bounds + strikes to spread_log. Next
    # Monday's Step 1b scores it against the realized range. This is the
    # write half of the loop that range_finder/calibration.py reads.
    if ticker == "SPX" and _cal_fit is not None:
        try:
            from range_finder.feature_builder import get_feature_for_week

            week_start = (run_now - timedelta(days=run_now.weekday())).strftime("%Y-%m-%d")
            result, cal_cols = _cal_fit

            feature_row = get_feature_for_week(conn, week_start, ticker="SPX")
            if feature_row is None:
                feature_row = df_feat.iloc[-1]
                _logger.warning(f"  No feature row for {week_start} — "
                                "logging plan off the latest available row")

            spx_ref = monday_open or spot

            # Per-side band share from completed weekly history (falls back
            # to the legacy /2 when history is thin/unavailable). The logged
            # strikes therefore reflect the SAME placement the UI shows, and
            # the calibration audit scores the placement actually used.
            _side_q = None
            try:
                from range_finder.feature_builder import _load_weekly_for_ticker
                from range_finder.har_model import estimate_side_share_quantile
                _q_est = estimate_side_share_quantile(
                    _load_weekly_for_ticker(conn, "SPX"))
                _side_q = _q_est["q"]
                _logger.info(f"  Side-share quantile: q={_side_q} "
                             f"(n={_q_est['n']}, beta={_q_est['beta']})")
            except Exception as _e:
                _logger.warning(f"  Side-share estimate failed ({_e}) — "
                                f"using legacy /2 split")

            forecast = forecast_next_week(result, feature_row, cal_cols, spx_ref,
                                          side_share_q=_side_q)
            plan = build_spread_plan(
                forecast, feature_row,
                week_start=week_start,
                vix_level=monday_vix,
                spx_open=monday_open,
                ticker="SPX",
                side_share_q=_side_q,
            )
            log_spread_plan(
                conn, plan,
                ticker="SPX",
                model_name=_CALIBRATION_SPEC,
                lower_pct=forecast.get("lower_pct"),
            )
            _logger.info(
                f"  Plan logged for {week_start} ({_CALIBRATION_SPEC}): "
                f"point={forecast['point_pct']:.4f} "
                f"PI=[{forecast['lower_pct']:.4f}, {forecast['upper_pct']:.4f}]"
            )
        except Exception as e:
            _logger.warning(f"  Plan logging failed: {e} (calibration row skipped)")


if __name__ == "__main__":
    capture_snapshot()
