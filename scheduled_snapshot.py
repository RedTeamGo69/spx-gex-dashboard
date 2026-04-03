#!/usr/bin/env python3
"""
Scheduled GEX snapshot capture — runs WITHOUT Streamlit.

Designed to be triggered by GitHub Actions (or any cron scheduler) at:
  - 9:30 AM ET  (market open)
  - 3:59 PM ET  (just before close)

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

    run_now = now_ny()
    today_str = run_now.strftime("%Y-%m-%d")
    _logger.info(f"Starting scheduled snapshot for {ticker} at {run_now.strftime('%I:%M:%S %p ET')} on {today_str}")

    # ── Market hours guard ──
    # Both EDT and EST cron triggers fire; skip if outside 9:25 AM - 4:05 PM ET window.
    hour, minute = run_now.hour, run_now.minute
    time_val = hour * 60 + minute  # minutes since midnight
    market_open = 9 * 60 + 25      # 9:25 AM (5-min early buffer)
    market_close = 16 * 60 + 5     # 4:05 PM (5-min late buffer)
    if time_val < market_open or time_val > market_close:
        _logger.info(f"Outside market hours ({run_now.strftime('%I:%M %p ET')}) — skipping (likely wrong DST trigger)")
        sys.exit(0)

    # Skip weekends (shouldn't happen with Mon-Fri cron, but just in case)
    if run_now.weekday() >= 5:
        _logger.info("Weekend — skipping")
        sys.exit(0)

    # ── Holiday check via pandas_market_calendars ──
    from phase1.market_clock import get_session_state
    from phase1.config import CASH_CALENDAR
    session = get_session_state(CASH_CALENDAR, run_now)
    if session.market_open is None:
        _logger.info(f"Market holiday ({today_str}) — skipping")
        sys.exit(0)

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
    gex_df, _hm_gex, _hm_iv, stats, all_options, _strike_sup, _exp_sup = (
        gex_engine.calculate_all(client, ticker, target_exps, spot, target_exps, r=rfr, now=run_now)
    )

    if gex_df.empty:
        _logger.error("GEX calculation returned empty — no data to save")
        sys.exit(1)

    levels = gex_engine.find_key_levels(gex_df, spot, all_options=all_options, r=rfr)
    regime_info = gex_engine.get_gamma_regime_text(spot, levels["zero_gamma"])
    staleness_info = build_staleness_info(calendar_snapshot, spot_info, stats)
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
    from phase1.expected_move import find_weekly_expiration, find_monthly_expiration, compute_em_for_expiration
    from phase1.gex_history import get_weekly_em_date_key, get_monthly_em_date_key

    is_monday = run_now.weekday() == 0
    is_tuesday_after_holiday = False
    if run_now.weekday() == 1:
        monday = run_now - __import__('datetime').timedelta(days=1)
        mon_session = get_session_state(CASH_CALENDAR, monday)
        is_tuesday_after_holiday = mon_session.market_open is None

    if is_monday or is_tuesday_after_holiday:
        weekly_exp = find_weekly_expiration(avail, run_now.date())
        if weekly_exp:
            weekly_em = compute_em_for_expiration(client, ticker, weekly_exp, spot)
            if weekly_em and weekly_em.get("expected_move_pts"):
                try:
                    weekly_key = get_weekly_em_date_key(run_now)
                    save_em_snapshot(weekly_em, weekly_key, ticker=ticker, em_type="weekly")
                    _logger.info(f"Weekly EM saved: ±{weekly_em['expected_move_pts']:.2f} pts (exp: {weekly_exp})")
                except Exception as e:
                    _logger.warning(f"Weekly EM save failed: {e}")

    # ── Monthly EM: capture on the first trading day of the month ──
    is_first_trading_day = False
    from datetime import timedelta as _td
    first = run_now.replace(day=1, hour=12, minute=0, second=0, microsecond=0)
    for offset in range(10):
        candidate = first + _td(days=offset)
        if candidate.weekday() >= 5:
            continue
        cand_session = get_session_state(CASH_CALENDAR, candidate)
        if cand_session.market_open is None:
            continue
        is_first_trading_day = (candidate.date() == run_now.date())
        break

    if is_first_trading_day:
        monthly_exp = find_monthly_expiration(avail, run_now.date())
        if monthly_exp:
            monthly_em = compute_em_for_expiration(client, ticker, monthly_exp, spot)
            if monthly_em and monthly_em.get("expected_move_pts"):
                try:
                    monthly_key = get_monthly_em_date_key(run_now)
                    save_em_snapshot(monthly_em, monthly_key, ticker=ticker, em_type="monthly")
                    _logger.info(f"Monthly EM saved: ±{monthly_em['expected_move_pts']:.2f} pts (exp: {monthly_exp})")
                except Exception as e:
                    _logger.warning(f"Monthly EM save failed: {e}")

    _logger.info("Scheduled snapshot complete")


if __name__ == "__main__":
    capture_snapshot()
