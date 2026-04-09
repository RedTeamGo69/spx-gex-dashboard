"""
SPX Gamma Exposure (GEX) Dashboard — Streamlit Web App

Run locally:   streamlit run streamlit_app.py
Deploy:        Push to GitHub → connect at share.streamlit.io
"""
from __future__ import annotations

import os
import json
import logging
import calendar as cal_mod
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import COLORS, SF_BG, SF_BULL, SF_BEAR, SF_NEUT, SF_WARN, SF_CARD
from models import GEXData
from ui_charts import build_gex_bar_chart, build_profile_chart
from ui_sidebar import (
    render_expected_move_panel, render_key_levels, render_scenarios_table,
    render_wall_credibility, render_gex_stream, render_data_quality,
    _fmt_gex_short,
)
from ui_spread_finder import (
    _render_sf_spread_table, _render_gex_context_panel,
    _render_sf_range_gauge, _render_sf_strike_map, _render_sf_strike_map_tier,
)

# ── Phase1 engine imports ──
from phase1.config import HEATMAP_EXPS, NY_TZ, COMPUTATION_RANGE_PCT, build_config_snapshot
from phase1.market_clock import now_ny, get_calendar_snapshot
from phase1.data_client import PublicDataClient
from phase1.rates import fetch_risk_free_rate
from phase1.parity import get_reference_spot_details
import phase1.gex_engine as gex_engine
from phase1.confidence import build_run_confidence
from phase1.staleness import build_staleness_info
from phase1.wall_credibility import build_wall_credibility
from phase1.scenarios import run_scenario_engine
from phase1.expected_move import (
    build_expected_move_analysis, compute_em_for_expiration,
    find_weekly_expiration, find_monthly_expiration,
)
from phase1.futures_data import fetch_es_from_yahoo, build_futures_context
from phase1.ai_briefing import build_briefing_context, generate_briefing
from phase1.gex_history import (
    save_snapshot, get_daily_summary, get_zero_gamma_trend, get_history,
    get_backend as get_history_backend, check_db_connection,
    save_em_snapshot, get_em_snapshot,
    get_weekly_em_date_key, get_monthly_em_date_key,
)

# ── Range Finder imports ──
from range_finder.gex_bridge import (
    GEXContext, extract_gex_context, save_gex_to_range_finder,
    adjust_spread_with_gex, regime_to_gex_flag,
)
from range_finder.data_collector import (
    fetch_spx_vix as rf_fetch_spx_vix, save_spx_vix as rf_save_spx_vix,
    fetch_fred_macro as rf_fetch_fred_macro, save_fred_macro as rf_save_fred_macro,
    build_event_flags as rf_build_event_flags,
    get_weekly_spx as rf_get_weekly_spx,
)
from range_finder.feature_builder import (
    build_features as rf_build_features,
    get_features as rf_get_features,
    get_feature_for_week as rf_get_feature_for_week,
)
from range_finder.har_model import (
    MODEL_SPECS as RF_MODEL_SPECS, PI_ALPHA as RF_PI_ALPHA,
    time_series_split as rf_time_series_split,
    fit_model as rf_fit_model, evaluate_oos as rf_evaluate_oos,
    forecast_next_week as rf_forecast_next_week,
    save_model as rf_save_model, load_model as rf_load_model,
)
from range_finder.spread_levels import (
    build_spread_plan as rf_build_spread_plan,
    build_spread_tiers as rf_build_spread_tiers,
    log_spread_plan as rf_log_spread_plan,
    update_outcome as rf_update_outcome,
    update_expiration_outcome as rf_update_expiration_outcome,
    get_spread_log as rf_get_spread_log,
    STANDARD_WING_WIDTHS as RF_WING_WIDTHS,
    TICKER_CONFIG as RF_TICKER_CONFIG,
    SpreadPlan,
    SpreadTier,
)

_logger = logging.getLogger(__name__)

TOOL_VERSION = "v5-web"


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GEX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme + sidebar metric sizing overrides
st.markdown("""
<style>
    .stApp { background-color: #1a1a2e; }
    section[data-testid="stSidebar"] { background-color: #16213e; }

    /* ── Shrink sidebar metrics so they don't truncate ── */
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        background: #1a1a3e;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 8px 10px;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 11px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 16px !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricDelta"] {
        font-size: 10px !important;
    }

    /* ── Compact custom level cards ── */
    .level-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 6px;
        margin-bottom: 10px;
    }
    .level-card {
        background: #1a1a3e;
        border: 1px solid #333;
        border-radius: 6px;
        padding: 8px 8px 6px 8px;
        text-align: center;
    }
    .level-card .lbl {
        font-size: 10px;
        color: #888;
        margin-bottom: 2px;
    }
    .level-card .val {
        font-size: 15px;
        font-weight: bold;
        color: #fff;
        word-break: break-all;
    }

    /* ── Top EM bar ── */
    .em-bar {
        display: flex;
        justify-content: space-around;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
        padding: 8px 0;
    }
    .em-item {
        text-align: center;
        min-width: 100px;
    }
    .em-item .lbl {
        font-size: 11px;
        color: #888;
    }
    .em-item .val {
        font-size: 20px;
        font-weight: bold;
        color: #fff;
    }
    @media (max-width: 768px) {
        .em-bar { flex-direction: column; align-items: stretch; }
        .em-item .val { font-size: 16px; }
        .level-card .val { font-size: 13px; }
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Credentials
# ─────────────────────────────────────────────────────────────────────────────
def get_credentials():
    """Pull API keys from Streamlit secrets, env vars, or sidebar input."""
    public_secret_key = ""
    fred_key = ""
    gemini_key = ""

    # Try st.secrets first (for Streamlit Cloud deployment)
    try:
        public_secret_key = st.secrets.get("PUBLIC_SECRET_KEY", "")
        fred_key = st.secrets.get("FRED_API_KEY", "")
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    # Fall back to env vars
    if not public_secret_key:
        public_secret_key = os.environ.get("PUBLIC_SECRET_KEY", "")
    if not fred_key:
        fred_key = os.environ.get("FRED_API_KEY", "")
    if not gemini_key:
        gemini_key = os.environ.get("GEMINI_API_KEY", "")

    # Push Gemini key into env so the cached briefing function can read it
    # without the key participating in cache hashing.
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    return public_secret_key, fred_key, gemini_key


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(ttl=90, show_spinner=False)
def fetch_all_data(public_secret_key: str, fred_key: str, selected_exps: tuple, _run_id: str, ticker: str = "SPX"):
    """
    Run the full GEX engine pipeline. Cached for 90 seconds.
    _run_id forces a cache bust when the user clicks Refresh.
    """
    client = PublicDataClient(secret_key=public_secret_key)
    client.clear_cache()

    run_now = now_ny()
    calendar_snapshot = get_calendar_snapshot(run_now)

    rfr_info = fetch_risk_free_rate(fred_key)
    rfr = rfr_info["rate"]

    avail = client.get_expirations(ticker)
    if not avail:
        raise RuntimeError(f"No expirations returned from Public API for {ticker}")
    today_str = run_now.strftime("%Y-%m-%d")
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
    spot_source = spot_info["source"]

    target_exps = list(selected_exps)
    heatmap_exps = [e for e in avail if e >= today_str][:HEATMAP_EXPS]

    gex_df, hm_gex, hm_iv, stats, all_options, strike_support_df, exp_support_df = (
        gex_engine.calculate_all(client, ticker, target_exps, spot, heatmap_exps, r=rfr, now=run_now)
    )

    levels = gex_engine.find_key_levels(gex_df, spot, all_options=all_options, r=rfr)
    profile_df = gex_engine.compute_gex_profile_curve(all_options, spot, r=rfr)
    sensitivity_df = gex_engine.compute_zero_gamma_sensitivity(all_options, spot, r=rfr)
    scenarios_df = run_scenario_engine(all_options, base_spot=spot, base_r=rfr)
    has_0dte = any(e == today_str for e in target_exps)
    staleness_info = build_staleness_info(calendar_snapshot, spot_info, stats, has_0dte=has_0dte)
    confidence_info = build_run_confidence(stats, spot_info, staleness_info=staleness_info)
    wall_cred = build_wall_credibility(
        levels=levels,
        strike_support_df=strike_support_df,
        sensitivity_df=sensitivity_df,
        confidence_info=confidence_info,
        staleness_info=staleness_info,
    )
    regime_info = gex_engine.get_gamma_regime_text(spot, levels["zero_gamma"])

    # Expected move raw inputs (EM analysis happens in main() with futures data)
    index_quote = None
    spy_quote = None
    try:
        index_quote = client.get_full_quote(ticker)
    except Exception:
        pass
    try:
        spy_quote = client.get_full_quote("SPY")
    except Exception:
        pass

    prev_close = index_quote["prevclose"] if index_quote else 0.0
    dte0_exp = target_exps[0] if target_exps else nearest_exp
    dte0_entry = client.get_chain_cached(ticker, dte0_exp)
    dte0_calls = dte0_entry.get("calls", []) if dte0_entry.get("status") == "ok" else []
    dte0_puts = dte0_entry.get("puts", []) if dte0_entry.get("status") == "ok" else []

    # Try Yahoo ES futures (cached with the rest)
    yahoo_es = None
    try:
        yahoo_es = fetch_es_from_yahoo()
    except Exception:
        pass

    return GEXData(
        spot=spot,
        spot_source=spot_source,
        spot_info=spot_info,
        rfr=rfr,
        rfr_info=rfr_info,
        avail=avail,
        target_exps=target_exps,
        gex_df=gex_df,
        hm_gex=hm_gex,
        hm_iv=hm_iv,
        stats=stats,
        all_options=all_options,
        levels=levels,
        profile_df=profile_df,
        sensitivity_df=sensitivity_df,
        scenarios_df=scenarios_df,
        staleness_info=staleness_info,
        confidence_info=confidence_info,
        wall_cred=wall_cred,
        regime_info=regime_info,
        calendar_snapshot=calendar_snapshot,
        run_time=run_now.strftime("%I:%M:%S %p ET"),
        prev_close=prev_close,
        spy_quote=spy_quote,
        dte0_calls=dte0_calls,
        dte0_puts=dte0_puts,
        market_open=bool(spot_info.get("market_open")),
        yahoo_es=yahoo_es,
        chain_cache=dict(client.chain_cache),
    )


@st.cache_resource(ttl=90, show_spinner=False)
def fetch_multi_tf_gex(public_secret_key: str, avail_exps: tuple, spot: float, rfr: float, _run_id: str, ticker: str = "SPX"):
    """
    Compute GEX for 3 timeframe buckets: 0DTE, This Week, This Month.
    Returns dict of {label: gex_df}.
    """
    from phase1.market_clock import now_ny, compute_time_to_expiry_years
    from phase1.config import T_FLOOR

    client = PublicDataClient(secret_key=public_secret_key)
    run_now = now_ny()
    today_str = run_now.strftime("%Y-%m-%d")
    tomorrow_str = (run_now + timedelta(days=1)).strftime("%Y-%m-%d")

    # Build non-overlapping expiration buckets using DTE
    from datetime import date as _date
    import calendar as _cal
    last_day = run_now.replace(day=_cal.monthrange(run_now.year, run_now.month)[1]).strftime("%Y-%m-%d")
    ref_date = run_now.date() if hasattr(run_now, 'date') else run_now

    sorted_exps = sorted([e for e in avail_exps if e >= today_str])

    # Classify by days-to-expiration from today
    dte0_exps = []   # nearest single expiration
    week_exps = []   # 2–7 calendar days out (rest of this week)
    month_exps = []  # 8+ days out through end of month

    for exp_str in sorted_exps:
        exp_date = _date.fromisoformat(exp_str)
        days_out = (exp_date - ref_date).days
        if exp_str > last_day:
            continue
        if not dte0_exps and days_out <= 1:
            # Nearest expiration (today or tomorrow = 0DTE)
            dte0_exps.append(exp_str)
        elif days_out <= 7:
            week_exps.append(exp_str)
        else:
            month_exps.append(exp_str)

    # If no 0DTE found within 1 day, grab the very first available
    if not dte0_exps and sorted_exps:
        dte0_exps.append(sorted_exps[0])
        week_exps = [e for e in week_exps if e != sorted_exps[0]]

    buckets = {
        "0DTE": dte0_exps,
        "This Week": week_exps,
        "This Month": month_exps,
    }

    results = {}
    for label, exps in buckets.items():
        if not exps:
            continue
        all_opts = []
        client.prefetch_chains(ticker, exps)
        for exp in exps:
            T, _ = compute_time_to_expiry_years(exp, ts=run_now.astimezone(NY_TZ) if run_now.tzinfo else run_now, floor=T_FLOOR)
            entry = client.get_chain_cached(ticker, exp)
            if entry.get("status") != "ok":
                continue
            from phase1.model_inputs import prepare_option_for_model
            lower = spot * (1 - COMPUTATION_RANGE_PCT)
            upper = spot * (1 + COMPUTATION_RANGE_PCT)
            for raw_opt, sign in [(c, +1) for c in entry["calls"]] + [(p, -1) for p in entry["puts"]]:
                K = raw_opt["strike"]
                if K < lower or K > upper:
                    continue
                oi = raw_opt["openInterest"]
                if oi <= 0:
                    continue
                prep = prepare_option_for_model(raw_opt, sign, T, spot, rfr)
                if prep["accepted"]:
                    norm = prep["normalized"]
                    all_opts.append((K, oi, norm["iv"], sign, T))

        if all_opts:
            gex_df = gex_engine.compute_strike_gex_from_all_options(all_opts, spot, r=rfr)
            results[label] = gex_df

    return results


def _get_rf_conn():
    """Get or create the range finder database connection (Postgres or SQLite)."""
    from range_finder.db import get_connection, init_all_tables, get_backend
    conn = get_connection()
    init_all_tables(conn)
    return conn


def _render_trade_log_tab():
    """Display the spread_log table with color-coded outcomes and summary stats."""
    import pandas as pd
    from datetime import date as _date_cls

    conn = _get_rf_conn()
    rows = rf_get_spread_log(conn)

    if not rows:
        st.info("No spread plans logged yet. Save a spread plan from the Spread Finder tab to get started.")
        return

    df = pd.DataFrame(rows)

    # ── Summary stats (only rows with an outcome) ──
    has_outcome = df[df["outcome"].notna() & (df["outcome"] != "")]
    st.subheader("Summary")

    if has_outcome.empty:
        st.caption("No outcomes recorded yet — use the button below to update expired weeks.")
    else:
        n = len(has_outcome)
        wins = (has_outcome["outcome"] == "full_profit").sum()
        win_rate = wins / n * 100

        call_breaches = has_outcome["call_breached"].fillna(0).sum()
        put_breaches = has_outcome["put_breached"].fillna(0).sum()
        call_rate = call_breaches / n * 100
        put_rate = put_breaches / n * 100

        # Average range error: |model effective_range_pct - actual_range_pct|
        err_df = has_outcome.dropna(subset=["actual_range_pct", "effective_range_pct"])
        if not err_df.empty:
            avg_err = (err_df["actual_range_pct"].astype(float) - err_df["effective_range_pct"].astype(float)).abs().mean()
            avg_err_str = f"{avg_err * 100:.2f}%"
        else:
            avg_err_str = "N/A"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{wins}/{n} weeks")
        c2.metric("Avg Range Error", avg_err_str)
        c3.metric("Call Breach Rate", f"{call_rate:.1f}%")
        c4.metric("Put Breach Rate", f"{put_rate:.1f}%")

    # ── Update button ──
    st.divider()
    today = _date_cls.today()
    # Find rows where outcome is NULL and the week has expired (Friday = week_start + 4 days)
    updatable = [
        r for r in rows
        if (not r.get("outcome"))
        and r.get("call_short") is not None
        and (datetime.strptime(r["week_start"], "%Y-%m-%d").date() + timedelta(days=5)) <= today
    ]

    if updatable:
        latest = updatable[0]  # rows are already sorted DESC by week_start
        st.caption(f"Most recent expired week without outcome: **{latest['week_start']}**")
        if st.button("Update This Week's Outcome", key="tl_update_btn"):
            with st.spinner(f"Fetching OHLC for {latest['week_start']}..."):
                result = rf_update_expiration_outcome(latest["week_start"], conn)
            if result in ("full_profit", "call_loss", "put_loss", "max_loss"):
                st.success(f"Outcome for {latest['week_start']}: **{result}**")
                st.rerun()
            elif result == "no_data":
                st.error("yfinance returned no data for that week. Market may have been closed.")
            else:
                st.error(f"Unexpected result: {result}")
    else:
        st.caption("All expired weeks have outcomes recorded.")

    # ── Data table with color-coded outcome ──
    st.divider()
    st.subheader("Spread Log")

    display_cols = [
        "week_start", "spx_ref_close", "effective_range_pct",
        "call_short", "put_short", "wing_width_used",
        "actual_high", "actual_low", "actual_range_pct",
        "call_breached", "put_breached", "outcome",
    ]
    # Only include columns that actually exist
    display_cols = [c for c in display_cols if c in df.columns]
    display_df = df[display_cols].copy()

    # Format percentages for readability
    for pct_col in ["effective_range_pct", "actual_range_pct"]:
        if pct_col in display_df.columns:
            display_df[pct_col] = display_df[pct_col].apply(
                lambda x: f"{float(x)*100:.2f}%" if x is not None and x == x else ""
            )

    def _color_outcome(val):
        colors = {
            "full_profit": "background-color: #1b5e20; color: white",
            "call_loss": "background-color: #e65100; color: white",
            "put_loss": "background-color: #e65100; color: white",
            "max_loss": "background-color: #b71c1c; color: white",
        }
        return colors.get(val, "")

    styled = display_df.style.map(_color_outcome, subset=["outcome"] if "outcome" in display_df.columns else [])
    st.dataframe(styled, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Lazy imports to avoid circular dependency (ui_history imports from streamlit_app)
    from ui_history import (
        _render_history_tab, _render_em_tracker, _render_multi_timeframe,
        _render_pin_detection, _render_iv_surface, _check_level_crossings,
        _is_weekly_freeze_day, _is_monthly_freeze_day,
        _apply_typed_em_snapshot, _apply_em_snapshot,
    )
    from ui_spread_finder import _render_spread_finder_tab

    st.title("📊 Gamma Exposure Dashboard")
    st.caption(f"GEX Calculator {TOOL_VERSION} — Implied spot | Zero gamma sweep | Expected move | Hybrid IV")

    public_secret_key, fred_key, gemini_key = get_credentials()

    # ── Sidebar controls ──
    with st.sidebar:
        st.header("⚙️ Settings")

        if not public_secret_key:
            public_secret_key = st.text_input("Public.com API Key", type="password",
                                               help="Get yours at https://public.com/api/docs")
        if not fred_key:
            fred_key = st.text_input("FRED API Key (optional)", type="password",
                                      help="For live T-bill rates. Get at https://fred.stlouisfed.org/docs/api/api_key.html")

        if not public_secret_key:
            st.warning("Enter your Public.com API key to get started.")
            st.stop()

        st.divider()

        # Ticker selector
        ticker = st.selectbox("Index", ["SPX", "XSP"], index=0, key="ticker_select")

        # Expiration picker
        with st.spinner("Loading expirations..."):
            try:
                temp_client = PublicDataClient(secret_key=public_secret_key)
                avail = temp_client.get_expirations(ticker)
            except Exception as e:
                st.error(f"Could not fetch expirations: {e}")
                st.stop()

        run_now = now_ny()
        today_str = run_now.strftime("%Y-%m-%d")
        future_exps = [e for e in avail if e >= today_str]

        dte0 = future_exps[0] if future_exps else None
        tomorrow_str = (run_now + timedelta(days=1)).strftime("%Y-%m-%d")
        dte1 = next((e for e in avail if e >= tomorrow_str), None)

        mode = st.radio("Expiration", [
            "0DTE" + (f"  ({dte0})" if dte0 else "  (n/a)"),
            "Tomorrow" + (f"  ({dte1})" if dte1 else "  (n/a)"),
            "This week",
            "This month",
            "Custom",
        ], index=0)

        if "0DTE" in mode:
            selected = [dte0] if dte0 else []
        elif "Tomorrow" in mode:
            selected = [dte1] if dte1 else []
        elif "week" in mode:
            days_to_fri = (4 - run_now.weekday()) % 7
            # days_to_fri == 0 on Friday itself, which is correct (include today)
            fri = (run_now + timedelta(days=days_to_fri)).strftime("%Y-%m-%d")
            selected = [e for e in avail if today_str <= e <= fri]
        elif "month" in mode:
            ld = run_now.replace(day=cal_mod.monthrange(run_now.year, run_now.month)[1]).strftime("%Y-%m-%d")
            selected = [e for e in avail if today_str <= e <= ld]
        else:
            selected = st.multiselect("Pick expirations", future_exps, default=future_exps[:1])

        if not selected:
            st.warning("No expirations selected.")
            st.stop()

        st.divider()

        refresh_option = st.radio(
            "Auto-refresh",
            ["Off", "Every 5 min", "Every 30 min"],
            index=0,
            horizontal=True,
        )
        if st.button("🔄 Refresh Now", use_container_width=True, type="primary"):
            st.cache_resource.clear()

        # ── ES Futures Override (pre-market only) ──
        # We declare these with defaults; they'll be ignored during market hours.
        es_manual_last = 0.0
        es_manual_high = 0.0
        es_manual_low = 0.0

    # ── Refresh interval ──
    refresh_seconds = {"Off": 0, "Every 5 min": 300, "Every 30 min": 1800}.get(refresh_option, 0)

    # ── Run ID for cache busting ──
    run_id = f"{datetime.now(timezone.utc).isoformat()}" if refresh_seconds == 0 else "auto"

    # ── Fetch data ──
    with st.spinner("Crunching GEX..."):
        try:
            data = fetch_all_data(public_secret_key, fred_key or "", tuple(selected), run_id, ticker=ticker)
        except Exception as e:
            st.error(f"Engine error: {e}")
            st.stop()

    if data.gex_df.empty:
        st.warning("No GEX data returned. The selected expirations may have no usable contracts.")
        st.stop()

    # ── Build futures context: manual overrides > Yahoo auto ──
    spot = data.spot
    levels = data.levels
    regime = data.regime_info
    prev_close = data.prev_close
    yahoo_es = data.yahoo_es
    is_market_open = data.market_open

    # Show ES input fields only when market is closed
    if not is_market_open:
        with st.sidebar:
            st.divider()
            st.markdown("#### 📡 ES Futures (Pre-market)")
            st.caption("Auto-filled from Yahoo (~10m delay). Override with your own numbers.")
            es_manual_last = st.number_input(
                "ES Last Price", min_value=0.0, value=0.0,
                step=0.25, format="%.2f", key="es_last_input",
                help="Current ES futures price. Leave 0 to use Yahoo auto-fetch.",
            )
            es_col1, es_col2 = st.columns(2)
            es_manual_high = es_col1.number_input(
                "ES O/N High", min_value=0.0, value=0.0,
                step=0.25, format="%.2f", key="es_high_input",
            )
            es_manual_low = es_col2.number_input(
                "ES O/N Low", min_value=0.0, value=0.0,
                step=0.25, format="%.2f", key="es_low_input",
            )

    # Determine ES values: manual if user entered anything, else Yahoo
    has_manual = es_manual_last > 0

    if has_manual:
        es_last = es_manual_last
        es_high = es_manual_high if es_manual_high > 0 else None
        es_low = es_manual_low if es_manual_low > 0 else None
        es_source = "manual"
    elif yahoo_es is not None:
        es_last = yahoo_es["last"]
        es_high = yahoo_es.get("high")
        es_low = yahoo_es.get("low")
        es_source = "yahoo_es_f"
    else:
        es_last = None
        es_high = None
        es_low = None
        es_source = None

    futures_ctx = None
    if es_last and es_last > 0 and prev_close > 0:
        # XSP trades at 1/10 the scale of SPX. ES futures track SPX, so
        # scale ES down by 10 when the dashboard is set to XSP — otherwise
        # the overnight move math compares apples to oranges.
        if ticker == "XSP":
            es_last_scaled = es_last / 10.0
            es_high_scaled = (es_high / 10.0) if es_high else None
            es_low_scaled = (es_low / 10.0) if es_low else None
            futures_ctx = build_futures_context(
                es_last_scaled, es_high_scaled, es_low_scaled, prev_close, source=es_source
            )
        else:
            futures_ctx = build_futures_context(es_last, es_high, es_low, prev_close, source=es_source)

    # ── Build EM analysis (fresh each render, not cached) ──
    em_analysis = build_expected_move_analysis(
        spot=spot,
        prev_close=prev_close,
        zero_gamma=levels["zero_gamma"],
        gamma_regime=regime["regime"],
        calls_0dte=data.dte0_calls,
        puts_0dte=data.dte0_puts,
        spy_quote=data.spy_quote,
        market_open=data.market_open,
        futures_context=futures_ctx,
    )

    # ── Apply EM snapshot logic ──
    em_analysis = _apply_em_snapshot(em_analysis, is_market_open, regime, levels, spot, ticker=ticker)

    # ── Weekly & Monthly EM ──
    run_now = now_ny()
    temp_client = PublicDataClient(secret_key=public_secret_key)

    # Weekly EM: straddle from this Friday's expiration, frozen Monday at open
    weekly_exp = find_weekly_expiration(data.avail, run_now.date())
    weekly_em_live = compute_em_for_expiration(temp_client, ticker, weekly_exp, spot) if weekly_exp else None
    weekly_date_key = get_weekly_em_date_key(run_now)
    weekly_em_snap = _apply_typed_em_snapshot(
        weekly_em_live, is_market_open, spot, ticker,
        "weekly", weekly_date_key, _is_weekly_freeze_day(run_now),
    )

    # Monthly EM: straddle from monthly OpEx (3rd Friday), frozen 1st trading day
    monthly_exp = find_monthly_expiration(data.avail, run_now.date())
    monthly_em_live = compute_em_for_expiration(temp_client, ticker, monthly_exp, spot) if monthly_exp else None
    monthly_date_key = get_monthly_em_date_key(run_now)
    monthly_em_snap = _apply_typed_em_snapshot(
        monthly_em_live, is_market_open, spot, ticker,
        "monthly", monthly_date_key, _is_monthly_freeze_day(run_now),
    )

    # ── Save historical snapshot (market hours only, throttled to refresh interval) ──
    if is_market_open:
        now_utc = datetime.now(timezone.utc)
        last_snap_utc = st.session_state.get("_last_snapshot_utc")
        # Minimum seconds between saves: use refresh interval, or 300s (5 min) if auto-refresh is off
        min_gap = refresh_seconds if refresh_seconds > 0 else 300
        should_save = last_snap_utc is None or (now_utc - last_snap_utc).total_seconds() >= min_gap
        if should_save:
            try:
                save_snapshot(spot, levels, regime, data.stats, data.confidence_info, data.staleness_info, em_analysis, ticker=ticker)
                st.session_state["_last_snapshot_utc"] = now_utc
                st.session_state["last_save_ok"] = True
                st.session_state["last_save_time"] = now_utc.strftime("%H:%M:%S UTC")
                st.session_state["last_save_error"] = None
            except Exception as e:
                st.session_state["last_save_ok"] = False
                st.session_state["last_save_error"] = str(e)
                _logger.error(f"History save failed: {e}")
    else:
        st.session_state["last_save_ok"] = None
        st.session_state["last_save_time"] = "skipped (market closed)"

    # Show Yahoo ES status in sidebar (pre-market only)
    if not is_market_open:
        with st.sidebar:
            if yahoo_es and not has_manual:
                es_note = f"Yahoo ES: \\${yahoo_es['last']:.2f}"
                if yahoo_es.get("high"):
                    es_note += f" (H: \\${yahoo_es['high']:.2f} L: \\${yahoo_es['low']:.2f})"
                es_note += f" — {yahoo_es.get('note', '~10m delayed')}"
                st.caption(es_note)
            elif has_manual:
                st.caption(f"Using manual ES: \\${es_last:.2f}")
            else:
                st.caption("No ES data available — enter manually above.")

    # ── Header metrics ──
    regime_color = regime["color"]
    spot_c = COLORS["spot"]
    text_sec = COLORS["text_secondary"]
    text_mut = COLORS["text_muted"]
    st.markdown(
        f"<div style='text-align:center;padding:6px;'>"
        f"<span style='font-size:22px;font-weight:bold;color:{spot_c};'>{ticker} ${spot:.2f}</span>"
        f"&nbsp;&nbsp;&nbsp;"
        f"<span style='font-size:18px;color:{regime_color};font-weight:bold;'>{regime['regime']}</span>"
        f"&nbsp;&nbsp;"
        f"<span style='color:{text_sec};font-size:13px;'>({regime['distance_text']})</span>"
        f"&nbsp;&nbsp;&nbsp;"
        f"<span style='color:{text_mut};font-size:12px;'>Updated {data.run_time}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Market context banner ──
    market_ctx = em_analysis.get("market_context", "live")
    context_note = em_analysis.get("context_note")
    if market_ctx == "premarket":
        st.info("🌅 **Pre-market** — GEX levels and gamma regime are current. "
                "Expected move and session classification will be available after the 9:30 AM open.")
    elif market_ctx == "afterhours":
        st.warning(f"🌙 **After hours** — {context_note}")

    # ── Expected Move panel (top of page) ──
    em_data = em_analysis.get("expected_move", {})

    if market_ctx == "premarket":
        # ── PRE-MARKET: Only show ES overnight move + range, suppress stale straddle/classification ──
        futures_ctx_display = em_analysis.get("futures_context")
        overnite_range = em_analysis.get("overnight_range")

        if futures_ctx_display:
            on_color = COLORS["positive"] if futures_ctx_display["overnight_move_pts"] >= 0 else COLORS["negative"]
            on_arrow = "▲" if futures_ctx_display["overnight_move_pts"] > 0 else "▼" if futures_ctx_display["overnight_move_pts"] < 0 else "–"

            premarket_html = (
                '<div class="em-bar">'
                f'<div class="em-item"><div class="lbl">Overnight Move ({"ES÷10" if ticker == "XSP" else "ES"})</div>'
                f'<div class="val" style="color:{on_color};">{on_arrow} {futures_ctx_display["overnight_move_pts"]:+.1f} pts</div>'
                f'<div class="lbl" style="color:{on_color};">{futures_ctx_display["overnight_move_pct"]:+.2f}%</div></div>'
            )

            if overnite_range and overnite_range.get("es_high"):
                hi_move = overnite_range["high_move_from_close"]
                lo_move = overnite_range["low_move_from_close"]
                cw_c = COLORS["call_wall"]
                pw_c = COLORS["put_wall"]
                premarket_html += (
                    f'<div class="em-item"><div class="lbl">O/N High</div>'
                    f'<div class="val" style="font-size:16px;color:{cw_c};">${overnite_range["es_high"]:.0f}</div>'
                    f'<div class="lbl" style="color:{cw_c};">{hi_move:+.1f} pts</div></div>'
                    f'<div class="em-item"><div class="lbl">O/N Low</div>'
                    f'<div class="val" style="font-size:16px;color:{pw_c};">${overnite_range["es_low"]:.0f}</div>'
                    f'<div class="lbl" style="color:{pw_c};">{lo_move:+.1f} pts</div></div>'
                    f'<div class="em-item"><div class="lbl">O/N Range</div>'
                    f'<div class="val" style="font-size:16px;">{overnite_range["range_pts"]:.0f} pts</div></div>'
                )

            premarket_html += '</div>'
            st.markdown(premarket_html, unsafe_allow_html=True)

    elif em_data.get("expected_move_pts"):
        # ── MARKET HOURS / AFTER HOURS: Full EM framework ──
        classification = em_analysis.get("classification", {})
        overnight = em_analysis.get("overnight_move", {})
        spy = em_analysis.get("spy_proxy")
        futures_ctx_display = em_analysis.get("futures_context")
        overnite_range = em_analysis.get("overnight_range")
        move_source = classification.get("move_source", "spx")

        # Pick the right move numbers and label
        if market_ctx == "live":
            display_pts = overnight.get("overnight_move_pts", 0)
            display_pct = overnight.get("overnight_move_pct", 0)
            on_label = "Today's Move"
        elif move_source == "spx_realized":
            display_pts = overnight.get("overnight_move_pts", 0)
            display_pct = overnight.get("overnight_move_pct", 0)
            on_label = "Session Move"
        elif "es_futures" in move_source and futures_ctx_display:
            display_pts = futures_ctx_display["overnight_move_pts"]
            display_pct = futures_ctx_display["overnight_move_pct"]
            on_label = "Overnight (ES÷10)" if ticker == "XSP" else "Overnight (ES)"
        elif move_source == "spy_proxy" and spy:
            display_pts = spy["implied_spx_move_pts"]
            display_pct = spy["spy_move_pct"]
            on_label = "Overnight (SPY)"
        else:
            display_pts = overnight.get("overnight_move_pts", 0)
            display_pct = overnight.get("overnight_move_pct", 0)
            on_label = "Overnight"

        ratio = classification.get("move_ratio")

        display_pts = display_pts if display_pts is not None else 0.0
        display_pct = display_pct if display_pct is not None else 0.0

        on_color = COLORS["positive"] if (display_pts or 0) >= 0 else COLORS["negative"]
        on_arrow = "▲" if (display_pts or 0) > 0 else "▼" if (display_pts or 0) < 0 else "–"
        ratio_pct = f"{ratio*100:.0f}%" if ratio is not None else "–"

        if ratio is not None:
            ratio_color = COLORS["positive"] if ratio < 0.40 else COLORS["warning"] if ratio < 0.70 else COLORS["negative"]
        else:
            ratio_color = COLORS["text_secondary"]

        cls_name = classification.get("classification", "–")
        cls_bias = classification.get("bias", "")
        if cls_bias in ("range-bound", "mean-revert"):
            cls_color = COLORS["positive"]
        elif cls_bias in ("directional", "continued-trend"):
            cls_color = COLORS["negative"]
        else:
            cls_color = COLORS["warning"]

        em_bar_html = (
            '<div class="em-bar">'
            f'<div class="em-item"><div class="lbl">Expected Move</div><div class="val">&plusmn;{em_data.get("expected_move_pts", 0) or 0:.0f} pts</div></div>'
            f'<div class="em-item"><div class="lbl">EM Range</div><div class="val">${em_data.get("lower_level", 0) or 0:.0f} &ndash; ${em_data.get("upper_level", 0) or 0:.0f}</div></div>'
            f'<div class="em-item"><div class="lbl">{on_label}</div><div class="val" style="color:{on_color};">{on_arrow} {display_pts:+.1f} pts</div><div class="lbl" style="color:{on_color};">{display_pct:+.2f}%</div></div>'
            f'<div class="em-item"><div class="lbl">Vol Budget Used</div><div class="val" style="color:{ratio_color};">{ratio_pct}</div></div>'
            f'<div class="em-item"><div class="lbl">Session Type</div><div class="val" style="color:{cls_color};">{cls_name}</div></div>'
            '</div>'
        )
        st.markdown(em_bar_html, unsafe_allow_html=True)

        # Show when the straddle was captured
        snap_time = st.session_state.get(f"em_snapshot_time_daily_{ticker}")
        if market_ctx == "live" and snap_time:
            st.caption(f"📌 Expected move captured at {snap_time} — frozen for the session. Today's move and vol budget update live.")

        # Overnight range bar (when ES high/low available, after hours only)
        if market_ctx == "afterhours" and overnite_range and overnite_range.get("es_high"):
            hi_move = overnite_range["high_move_from_close"]
            lo_move = overnite_range["low_move_from_close"]
            rng = overnite_range["range_pts"]
            max_vs_em = overnite_range.get("max_move_vs_em")
            max_vs_em_str = f"{max_vs_em*100:.0f}% of EM" if max_vs_em else ""

            cw_c = COLORS["call_wall"]
            pw_c = COLORS["put_wall"]
            range_html = (
                '<div class="em-bar" style="padding:2px 0 4px 0;">'
                f'<div class="em-item"><div class="lbl">O/N High</div><div class="val" style="font-size:16px;color:{cw_c};">${overnite_range["es_high"]:.0f}</div><div class="lbl" style="color:{cw_c};">{hi_move:+.1f} pts</div></div>'
                f'<div class="em-item"><div class="lbl">O/N Low</div><div class="val" style="font-size:16px;color:{pw_c};">${overnite_range["es_low"]:.0f}</div><div class="lbl" style="color:{pw_c};">{lo_move:+.1f} pts</div></div>'
                f'<div class="em-item"><div class="lbl">O/N Range</div><div class="val" style="font-size:16px;">{rng:.0f} pts</div></div>'
                f'<div class="em-item"><div class="lbl">Max O/N Excursion</div><div class="val" style="font-size:16px;">{overnite_range["max_move_pts"]:.0f} pts</div><div class="lbl">{max_vs_em_str}</div></div>'
                '</div>'
            )
            st.markdown(range_html, unsafe_allow_html=True)

    # ── AI Trading Briefing ──
    # Only generates on first load or manual "Regenerate" click to conserve
    # Gemini API quota.  Auto-refresh cycles reuse the cached result.
    with st.expander("🧠 AI Briefing", expanded=True):
        if not gemini_key:
            st.caption(
                "Set `GEMINI_API_KEY` in Streamlit secrets or env var to enable "
                "the AI briefing. Falls back to a templated briefing without a key."
            )

        # Generate only when no cached briefing exists in session state
        if "_ai_briefing" not in st.session_state:
            try:
                _brief_ctx = build_briefing_context(data, em_analysis)
                _brief_text, _brief_source = generate_briefing(_brief_ctx)
                st.session_state["_ai_briefing"] = (_brief_text, _brief_source)
            except Exception as _brief_err:
                st.session_state["_ai_briefing"] = (
                    f"Briefing unavailable: {_brief_err}", "error",
                )

        _brief_text, _brief_source = st.session_state["_ai_briefing"]
        if _brief_source == "error":
            st.caption(_brief_text)
        else:
            st.markdown(_brief_text)
            _src_color = COLORS["text_muted"] if _brief_source == "gemini" else COLORS["warning"]
            st.markdown(
                f"<div style='font-size:10px;color:{_src_color};margin-top:6px;'>"
                f"source: {_brief_source} · model: gemini-2.5-flash-lite"
                f"</div>",
                unsafe_allow_html=True,
            )

        if st.button("🔄 Regenerate briefing", key="regen_briefing"):
            st.session_state.pop("_ai_briefing", None)
            st.session_state.pop("_gemini_backoff_until", None)  # clear quota backoff
            st.rerun()

    # ── Charts ──
    tab_gex, tab_profile, tab_multi, tab_history, tab_em_track, tab_iv_surface, tab_spread_finder, tab_trade_log = st.tabs(
        ["📊 Strike GEX", "📈 GEX Profile", "⏱️ Multi-TF", "📅 History", "🎯 EM Tracker", "🌊 IV Surface", "🎯 Spread Finder", "📋 Trade Log"]
    )

    with tab_gex:
        w_em_for_chart = weekly_em_snap or weekly_em_live or {}
        m_em_for_chart = monthly_em_snap or monthly_em_live or {}
        fig1 = build_gex_bar_chart(data.gex_df, levels, spot, em_analysis, weekly_em=w_em_for_chart, monthly_em=m_em_for_chart)
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    with tab_profile:
        with st.expander("How to read this chart"):
            st.markdown("""
**GEX Profile Curve — Total Gamma Exposure vs. Price**

This chart shows the **total dealer gamma exposure** at each price level, summed across all strikes and expirations.

- **X-axis** = Underlying price (SPX/XSP level)
- **Y-axis** = Total net GEX proxy (sum of dealer gamma at that price)
- **White dashed line** = Current spot price

**Key levels marked on the chart:**
- **Zero-Gamma (Zero-G)** — Where the curve crosses zero. This is the most important level:
  - **Above Zero-G:** Dealers are long gamma (positive GEX). They hedge by buying dips and selling rips, which *suppresses* volatility. Price tends to mean-revert.
  - **Below Zero-G:** Dealers are short gamma (negative GEX). They hedge by selling dips and buying rips, which *amplifies* moves. Price trends harder.
- **Call Wall** — Strike with the largest positive call gamma. Acts as a resistance/ceiling — dealer hedging pushes price back down as it approaches.
- **Put Wall** — Strike with the largest positive put gamma. Acts as a support/floor — dealer hedging pushes price back up as it approaches.

**How to use it:**
- **Tall positive peak near spot** = Strong mean-reversion zone. Good for selling credit spreads — price is "sticky" here.
- **Curve near zero or negative around spot** = Weak support. Price can move freely. Be cautious with tight spreads.
- **Steep slope near spot** = Small price moves cause large changes in dealer hedging. Expect choppy, range-bound action.
- **Flat curve** = Dealers have little gamma exposure. Price moves are driven by order flow, not hedging.

**Important caveat:** This model assumes dealers are uniformly net short all options (the standard convention). \
Actual dealer positioning varies by strike due to institutional overlays, collar programs, and retail put-selling. \
OI is end-of-day data — intraday 0DTE flow is not captured. Use these levels as probabilistic guides, not certainties.
""")
        fig2 = build_profile_chart(data.profile_df, levels, spot, regime, em_analysis, weekly_em=w_em_for_chart, monthly_em=m_em_for_chart)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    # ── C3: Multi-timeframe GEX comparison ──
    with tab_multi:
        _render_multi_timeframe(data.all_options, data.target_exps, data.avail, spot, levels, data.rfr, ticker=ticker)

    # ── C1: Historical GEX trend ──
    with tab_history:
        _render_history_tab(spot, ticker=ticker)

    # ── C5: Expected move consumption trackers (0DTE / Weekly / Monthly) ──
    with tab_em_track:
        # Daily (0DTE)
        daily_cap = st.session_state.get(f"em_snapshot_time_daily_{ticker}")
        daily_sub = f"Frozen at {daily_cap}" if daily_cap else None
        _render_em_tracker(em_analysis, spot, prev_close, market_ctx, label="0DTE", subtitle=daily_sub)

        st.divider()

        # Weekly
        weekly_is_frozen = weekly_em_snap is not None
        if not weekly_exp:
            weekly_sub = "No weekly expiration found"
        elif weekly_is_frozen:
            weekly_sub = f"Frozen Mon open | Exp: {weekly_exp}"
        else:
            weekly_sub = f"Live estimate (freezes Mon open) | Exp: {weekly_exp}"
        weekly_em_for_render = {"expected_move": weekly_em_snap} if weekly_em_snap else {"expected_move": weekly_em_live or {}}
        _render_em_tracker(weekly_em_for_render, spot, prev_close, market_ctx, label="Weekly", subtitle=weekly_sub, is_frozen=weekly_is_frozen)

        st.divider()

        # Monthly
        monthly_is_frozen = monthly_em_snap is not None
        if not monthly_exp:
            monthly_sub = "No monthly expiration found"
        elif monthly_is_frozen:
            monthly_sub = f"Frozen 1st trading day | Exp: {monthly_exp}"
        else:
            monthly_sub = f"Live estimate (freezes 1st trading day) | Exp: {monthly_exp}"
        monthly_em_for_render = {"expected_move": monthly_em_snap} if monthly_em_snap else {"expected_move": monthly_em_live or {}}
        _render_em_tracker(monthly_em_for_render, spot, prev_close, market_ctx, label="Monthly", subtitle=monthly_sub, is_frozen=monthly_is_frozen)

    # ── C4: IV surface visualization ──
    with tab_iv_surface:
        _render_iv_surface(data.hm_iv, data.hm_gex, spot)

    # ── C7: Spread Finder — Weekly credit spread placement ──
    with tab_spread_finder:
        _sf_weekly_em = weekly_em_snap or weekly_em_live or {}
        _render_spread_finder_tab(spot, levels, regime, data, ticker=ticker, weekly_em=_sf_weekly_em)

    with tab_trade_log:
        _render_trade_log_tab()

    # ── C2: Level crossing alerts ──
    alerts = _check_level_crossings(spot, levels, em_analysis)
    if alerts:
        for icon, msg in alerts:
            st.warning(f"{icon} {msg}")

    # ── C6: Pin point detection ──
    _render_pin_detection(data.stats, data.gex_df, spot)

    # ── Sidebar detail panels ──
    with st.sidebar:
        st.divider()
        render_gex_stream(data.stats, levels, spot)
        st.divider()
        if market_ctx != "premarket":
            render_expected_move_panel(em_analysis, ticker=ticker)
        render_key_levels(levels, spot, regime, data.confidence_info, data.staleness_info)
        st.divider()
        render_wall_credibility(data.wall_cred)
        st.divider()
        render_scenarios_table(data.scenarios_df)
        st.divider()
        render_data_quality(data.stats, data.staleness_info)

    # ── Auto-refresh ──
    # Uses st.rerun with a placeholder countdown so the app remains
    # responsive to user interaction during the wait.
    if refresh_seconds > 0:
        import time as _time
        _refresh_placeholder = st.empty()
        _elapsed = 0
        _tick = 5  # check every 5 seconds
        while _elapsed < refresh_seconds:
            remaining = refresh_seconds - _elapsed
            _refresh_placeholder.caption(f"Auto-refresh in {remaining}s")
            _time.sleep(min(_tick, remaining))
            _elapsed += _tick
        _refresh_placeholder.empty()
        st.cache_resource.clear()
        st.rerun()


if __name__ == "__main__":
    main()
