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
from phase1.data_client import TradierDataClient
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
    tradier_token = ""
    fred_key = ""
    gemini_key = ""

    # Try st.secrets first (for Streamlit Cloud deployment)
    try:
        tradier_token = st.secrets.get("TRADIER_TOKEN", "")
        fred_key = st.secrets.get("FRED_API_KEY", "")
        gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        pass

    # Fall back to env vars
    if not tradier_token:
        tradier_token = os.environ.get("TRADIER_TOKEN", "")
    if not fred_key:
        fred_key = os.environ.get("FRED_API_KEY", "")
    if not gemini_key:
        gemini_key = os.environ.get("GEMINI_API_KEY", "")

    # Push Gemini key into env so the cached briefing function can read it
    # without the key participating in cache hashing.
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key

    return tradier_token, fred_key, gemini_key


# ─────────────────────────────────────────────────────────────────────────────
# Data fetching (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(ttl=90, show_spinner=False)
def fetch_all_data(tradier_token: str, fred_key: str, selected_exps: tuple, _run_id: str, ticker: str = "SPX"):
    """
    Run the full GEX engine pipeline. Cached for 90 seconds.
    _run_id forces a cache bust when the user clicks Refresh.
    """
    client = TradierDataClient(token=tradier_token)
    client.clear_cache()

    run_now = now_ny()
    calendar_snapshot = get_calendar_snapshot(run_now)

    rfr_info = fetch_risk_free_rate(fred_key)
    rfr = rfr_info["rate"]

    avail = client.get_expirations(ticker)
    if not avail:
        raise RuntimeError(f"No expirations returned from Tradier API for {ticker}")
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
def fetch_multi_tf_gex(tradier_token: str, avail_exps: tuple, spot: float, rfr: float, _run_id: str, ticker: str = "SPX"):
    """
    Compute GEX for 3 timeframe buckets: 0DTE, This Week, This Month.
    Returns dict of {label: gex_df}.
    """
    from phase1.market_clock import now_ny, compute_time_to_expiry_years
    from phase1.config import T_FLOOR

    client = TradierDataClient(token=tradier_token)
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


def _render_history_tab(current_spot, ticker="SPX"):
    """C1: Render historical GEX trend chart."""
    backend = get_history_backend()

    # ── Save status diagnostic ──
    save_ok = st.session_state.get("last_save_ok")
    save_time = st.session_state.get("last_save_time", "–")
    if save_ok is True:
        status_icon = "✅"
        status_text = f"Last save: {save_time}"
    elif save_ok is False:
        status_icon = "❌"
        err = st.session_state.get("last_save_error", "unknown")
        status_text = f"Save failed: {err}"
    else:
        status_icon = "⏳"
        status_text = "No save attempted yet"

    if backend == "postgres":
        st.caption(f"💾 Neon Postgres — history persists across sessions &nbsp;|&nbsp; {status_icon} {status_text}")
        if save_ok is False:
            st.error(f"⚠️ Save error: {st.session_state.get('last_save_error', 'unknown')}")
    else:
        st.warning(
            "⚡ **Session-only storage** — history is lost on page refresh. "
            "To persist history across sessions, add `DATABASE_URL` to your Streamlit secrets "
            "(Settings → Secrets on Streamlit Cloud) with your Neon Postgres connection string."
        )
        st.caption(f"{status_icon} {status_text}")

    # ── DB Diagnostic ──
    if backend == "postgres":
        with st.expander("🔧 Database Diagnostic"):
            if st.button("Check DB Connection", key="check_db"):
                diag = check_db_connection()
                if diag["ok"]:
                    st.success(f"Connected! **{diag['total_rows']}** rows stored. Date range: {diag['date_range']}")
                    if diag["recent"]:
                        st.markdown("**Most recent snapshots:**")
                        for ts, s, zg in diag["recent"]:
                            st.caption(f"  {ts} — Spot: {s}, ZG: {zg}")
                    st.caption(f"Connection: `{diag['conn_str_prefix']}`")
                else:
                    st.error(f"DB Error: {diag.get('error', 'unknown')}")

    # ── View toggle + charts (fragment — switching views doesn't rerun the page) ──
    @st.fragment
    def _history_fragment():
        view = st.radio("View", ["Daily Summary", "Today's Snapshots"], horizontal=True, key="hist_view")

        if view == "Today's Snapshots":
            all_snaps = get_history(days=1, ticker=ticker)
            if not all_snaps:
                st.info("No snapshots recorded yet this session. Each refresh saves a snapshot automatically.")
                return

            snap_df = pd.DataFrame(all_snaps)
            snap_df["time"] = pd.to_datetime(snap_df["timestamp"]).dt.strftime("%H:%M:%S")
            display_cols = ["time", "spot", "zero_gamma", "call_wall", "put_wall", "regime",
                            "net_gex", "gex_ratio", "call_iv", "put_iv"]
            avail_cols = [c for c in display_cols if c in snap_df.columns]
            st.markdown(f"**{len(snap_df)} snapshot(s)** recorded today")
            st.dataframe(snap_df[avail_cols], use_container_width=True, hide_index=True)

            if len(snap_df) >= 2:
                snap_df["ts"] = pd.to_datetime(snap_df["timestamp"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=snap_df["ts"], y=snap_df["spot"], mode="lines+markers",
                    name="Spot", line=dict(color=COLORS["spot"], width=2), marker=dict(size=4),
                ))
                fig.add_trace(go.Scatter(
                    x=snap_df["ts"], y=snap_df["zero_gamma"], mode="lines+markers",
                    name="Zero Gamma", line=dict(color=COLORS["zero_gamma"], width=2), marker=dict(size=4),
                ))
                fig.update_layout(
                    paper_bgcolor=COLORS["bg_primary"], plot_bgcolor=COLORS["bg_primary"],
                    font_color="white", font_size=10,
                    margin=dict(l=60, r=10, t=60, b=40),
                    title="Intraday Spot vs Zero Gamma",
                    xaxis=dict(gridcolor=COLORS["grid_major"]),
                    yaxis=dict(title="Price", gridcolor=COLORS["grid_minor"]),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=400, dragmode=False,
                )
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})
            return

        # ── Daily Summary view ──
        days = st.selectbox("History range", [7, 14, 30, 60], index=1, key="hist_days")
        history = get_daily_summary(days=days, ticker=ticker)

        if not history:
            st.info("No historical data yet. Snapshots are saved automatically on each refresh.")
            return

        hist_df = pd.DataFrame(history)
        hist_df["date"] = pd.to_datetime(hist_df["date"])
        if "scan_type" not in hist_df.columns:
            hist_df["scan_type"] = "close"

        open_df = hist_df[hist_df["scan_type"] == "open"].copy()
        close_df = hist_df[hist_df["scan_type"] == "close"].copy()
        if close_df.empty and not open_df.empty:
            close_df = open_df.copy()

        fig = go.Figure()

        if not open_df.empty:
            fig.add_trace(go.Scatter(
                x=open_df["date"], y=open_df["spot"],
                mode="markers", name="Spot (Open)",
                marker=dict(size=8, color=COLORS["spot"], symbol="circle-open", line=dict(width=2)),
                hovertemplate="Open %{x|%b %d}<br>Spot: $%{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=open_df["date"], y=open_df["zero_gamma"],
                mode="markers", name="ZG (Open)",
                marker=dict(size=8, color=COLORS["zero_gamma"], symbol="circle-open", line=dict(width=2)),
                hovertemplate="Open %{x|%b %d}<br>ZG: $%{y:,.0f}<extra></extra>",
            ))

        if not close_df.empty:
            fig.add_trace(go.Scatter(
                x=close_df["date"], y=close_df["spot"],
                mode="lines+markers", name="Spot (Close)",
                line=dict(color=COLORS["spot"], width=2),
                marker=dict(size=5),
                hovertemplate="Close %{x|%b %d}<br>Spot: $%{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=close_df["date"], y=close_df["zero_gamma"],
                mode="lines+markers", name="ZG (Close)",
                line=dict(color=COLORS["zero_gamma"], width=2),
                marker=dict(size=5),
                hovertemplate="Close %{x|%b %d}<br>ZG: $%{y:,.0f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=close_df["date"], y=close_df["call_wall"],
                mode="lines", name="Call Wall",
                line=dict(color=COLORS["call_wall"], width=1, dash="dash"),
            ))
            fig.add_trace(go.Scatter(
                x=close_df["date"], y=close_df["put_wall"],
                mode="lines", name="Put Wall",
                line=dict(color=COLORS["put_wall"], width=1, dash="dash"),
            ))

        for _, o_row in open_df.iterrows():
            c_match = close_df[close_df["date"] == o_row["date"]]
            if c_match.empty:
                continue
            c_row = c_match.iloc[0]
            spot_open, spot_close = o_row["spot"], c_row["spot"]
            if spot_open and spot_close and spot_open != spot_close:
                color = "rgba(105,240,174,0.10)" if spot_close >= spot_open else "rgba(255,107,107,0.10)"
                fig.add_shape(
                    type="rect", x0=o_row["date"], x1=o_row["date"],
                    y0=min(spot_open, spot_close), y1=max(spot_open, spot_close),
                    fillcolor=color, line_width=0, layer="below",
                )

        fig.update_layout(
            paper_bgcolor=COLORS["bg_primary"], plot_bgcolor=COLORS["bg_primary"],
            font_color="white", font_size=10,
            margin=dict(l=60, r=10, t=60, b=40),
            title=f"GEX Key Levels — Last {days} Days (Open + Close)",
            xaxis=dict(gridcolor=COLORS["grid_major"]),
            yaxis=dict(title="Price", gridcolor=COLORS["grid_minor"]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=500, dragmode=False,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

        with st.expander("📋 Daily Summary"):
            display_cols = ["date", "scan_type", "spot", "zero_gamma", "call_wall", "put_wall",
                            "regime", "confidence_score", "coverage_ratio"]
            avail_cols = [c for c in display_cols if c in hist_df.columns]
            st.dataframe(hist_df[avail_cols].head(60), use_container_width=True, hide_index=True)

    _history_fragment()


def _render_em_tracker(em_analysis, spot, prev_close, market_ctx, label="0DTE", subtitle=None, is_frozen=True):
    """C5: Show how much of the expected move has been consumed."""
    em_data = em_analysis.get("expected_move", {}) if isinstance(em_analysis, dict) and "expected_move" in em_analysis else em_analysis
    if em_data is None:
        em_data = {}
    em_pts = em_data.get("expected_move_pts")

    if not em_pts or em_pts <= 0:
        st.info(f"{label} expected move data not available for tracking.")
        return

    upper = em_data.get("upper_level")
    lower = em_data.get("lower_level")

    # The EM range is anchored to spot at capture time (upper = anchor + em, lower = anchor - em).
    # Measure consumption from that same anchor so the % matches the visual range.
    em_anchor = (upper + lower) / 2 if (upper is not None and lower is not None) else prev_close
    if is_frozen and em_anchor > 0:
        current_move = abs(spot - em_anchor)
        move_pct_of_em = (current_move / em_pts) * 100
        direction = "up" if spot >= em_anchor else "down"
    else:
        # Live (unfrozen) data: anchor = current spot, so tracking is meaningless
        current_move = None
        move_pct_of_em = None
        direction = None

    # Gauge display
    if move_pct_of_em is not None:
        if move_pct_of_em < 40:
            gauge_color = COLORS["positive"]
            status = "Plenty of room"
        elif move_pct_of_em < 70:
            gauge_color = COLORS["warning"]
            status = "Moderate — watch for reversal"
        elif move_pct_of_em < 100:
            gauge_color = COLORS["negative"]
            status = "Extended — mean reversion likely"
        else:
            gauge_color = "#ff1744"
            status = "Beyond EM — trend day or breakout"

    if subtitle:
        st.caption(subtitle)

    col1, col2, col3 = st.columns(3)
    col1.metric(f"{label} Expected Move", f"±{em_pts:.0f} pts")
    if current_move is not None:
        col2.metric("Current Move", f"{current_move:.1f} pts {direction}")
        col3.metric("EM Consumed", f"{move_pct_of_em:.0f}%")
    else:
        col2.metric("Current Move", "—")
        col3.metric("EM Consumed", "—")

    if move_pct_of_em is not None:
        st.progress(min(move_pct_of_em / 100, 1.0))
        st.markdown(f"**Status:** <span style='color:{gauge_color};'>{status}</span>", unsafe_allow_html=True)

    # Visual range display
    if upper is None or lower is None:
        return
    padding = em_pts * 0.3
    fig = go.Figure()
    fig.add_shape(type="rect", x0=lower, x1=upper, y0=0, y1=1,
                  fillcolor="rgba(179,136,255,0.15)", line=dict(color=COLORS["em_level"], width=1))
    fig.add_vline(x=em_anchor, line_color=COLORS["text_muted"], line_dash="dash", line_width=1,
                  annotation_text="EM Anchor", annotation_font_color=COLORS["text_muted"], annotation_font_size=9)
    fig.add_vline(x=spot, line_color=COLORS["spot"], line_width=2,
                  annotation_text=f"Spot ${spot:.0f}", annotation_font_color=COLORS["spot"], annotation_font_size=10)
    fig.add_vline(x=lower, line_color=COLORS["em_level"], line_dash="dot", line_width=1,
                  annotation_text=f"EM− ${lower:.0f}", annotation_font_color=COLORS["em_level"], annotation_font_size=9)
    fig.add_vline(x=upper, line_color=COLORS["em_level"], line_dash="dot", line_width=1,
                  annotation_text=f"EM+ ${upper:.0f}", annotation_font_color=COLORS["em_level"], annotation_font_size=9)

    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"], plot_bgcolor=COLORS["bg_primary"],
        font_color="white", height=150, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(
            gridcolor=COLORS["grid_major"], title="Price",
            range=[lower - padding, upper + padding],
            tickformat="$,.0f",
        ),
        yaxis=dict(visible=False), showlegend=False,
        title=f"{label} Expected Move Range",
        dragmode=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})


def _render_multi_timeframe(all_options, target_exps, avail_exps, spot, levels, rfr, ticker="SPX"):
    """C3: Multi-timeframe GEX comparison — 0DTE vs Weekly vs Monthly."""
    tradier_token, _, _ = get_credentials()
    if not tradier_token:
        st.info("API token required for multi-timeframe analysis.")
        return

    run_id = st.session_state.get(f"em_snapshot_date_daily_{ticker}", "default")
    with st.spinner("Computing multi-timeframe GEX..."):
        tf_data = fetch_multi_tf_gex(tradier_token, tuple(avail_exps), spot, rfr, run_id, ticker=ticker)

    if not tf_data:
        st.info("No multi-timeframe data available.")
        return

    # Color mapping and opacity for timeframes (render order: back to front)
    # Longest timeframe in back (most transparent), 0DTE on top (most opaque)
    tf_style = {
        "This Month": {"color": "#69f0ae", "opacity": 0.35},
        "This Week":  {"color": "#ffd600", "opacity": 0.55},
        "0DTE":       {"color": "#ff6b6b", "opacity": 0.85},
    }
    # Render in back-to-front order so 0DTE is always visible on top
    render_order = ["This Month", "This Week", "0DTE"]

    fig = go.Figure()
    for label in render_order:
        gex_df = tf_data.get(label)
        if gex_df is None or gex_df.empty:
            continue
        df = gex_df.sort_values("strike")
        style = tf_style.get(label, {"color": "#9c88ff", "opacity": 0.6})
        fig.add_trace(go.Bar(
            y=df["strike"], x=df["net_gex"], orientation="h",
            name=label, marker_color=style["color"], marker_opacity=style["opacity"],
            hovertemplate=f"{label}<br>Strike: $%{{y:.0f}}<br>GEX: %{{x:,.0f}}<extra></extra>",
        ))

    # Key level lines
    fig.add_hline(y=spot, line_color=COLORS["spot"], line_dash="dash", line_width=1.5,
                  annotation_text=f"Spot ${spot:.0f}", annotation_font_color=COLORS["spot"],
                  annotation_font_size=9, annotation_position="top left")
    fig.add_hline(y=levels["zero_gamma"], line_color=COLORS["zero_gamma"], line_dash="dot", line_width=1.5,
                  annotation_text=f"ZG ${levels['zero_gamma']:.0f}", annotation_font_color=COLORS["zero_gamma"],
                  annotation_font_size=9, annotation_position="top left")

    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"], plot_bgcolor=COLORS["bg_primary"],
        font_color="white", font_size=10,
        margin=dict(l=80, r=10, t=35, b=35),
        title="Multi-Timeframe GEX Comparison",
        xaxis=dict(title="Net GEX proxy", gridcolor=COLORS["grid_major"], zerolinecolor=COLORS["zeroline"]),
        yaxis=dict(title="Strike", gridcolor=COLORS["grid_minor"], tickfont_size=8),
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=2000, dragmode=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    # Summary metrics
    st.markdown("##### Timeframe Summary")
    cols = st.columns(len(tf_data))
    for i, (label, gex_df) in enumerate(tf_data.items()):
        net = gex_df["net_gex"].sum()
        pos = gex_df[gex_df["net_gex"] > 0]["net_gex"].sum()
        neg = gex_df[gex_df["net_gex"] < 0]["net_gex"].sum()
        with cols[i]:
            st.markdown(f"**{label}**")
            gex_color = COLORS["positive"] if net > 0 else COLORS["negative"]
            st.markdown(f"Net GEX: <span style='color:{gex_color};'>{_fmt_gex_short(net)}</span>", unsafe_allow_html=True)
            st.caption(f"+{_fmt_gex_short(pos)} / {_fmt_gex_short(neg)}")


def _render_pin_detection(stats, gex_df, spot):
    """C6: Detect potential pin strikes where call+put OI concentrates."""
    if gex_df.empty:
        return

    # Find strikes with highest combined OI near spot (within 1%)
    near_spot = gex_df[abs(gex_df["strike"] - spot) / spot < 0.01].copy()
    if near_spot.empty:
        return

    # Pin candidates: strikes where call+put OI are both significant
    if "call_oi" in near_spot.columns and "put_oi" in near_spot.columns:
        near_spot["total_oi"] = near_spot["call_oi"] + near_spot["put_oi"]
        near_spot["oi_balance"] = 1 - abs(near_spot["call_oi"] - near_spot["put_oi"]) / near_spot["total_oi"].clip(lower=1)
        # Pin candidates have high OI and balanced call/put ratio
        pins = near_spot[near_spot["oi_balance"] > 0.3].nlargest(3, "total_oi")
        if not pins.empty:
            pin_strikes = ", ".join([f"{s:.0f}" for s in pins["strike"]])
            st.markdown(
                f'<div style="font-size:12px;color:#ccc;">📌 Pin candidates: <b>{pin_strikes}</b></div>',
                unsafe_allow_html=True,
            )


def _check_level_crossings(spot, levels, em_analysis):
    """C2: Check if spot has crossed key levels and return alerts."""
    alerts = []
    zg = levels.get("zero_gamma", 0)
    cw = levels.get("call_wall", 0)
    pw = levels.get("put_wall", 0)

    dist_zg = abs(spot - zg)
    dist_cw = abs(spot - cw)
    dist_pw = abs(spot - pw)

    # Alert when within 0.3% of a key level
    threshold = spot * 0.003

    if dist_zg < threshold:
        alerts.append(("⚡", f"Spot near Zero Gamma ({zg:.0f}) — regime flip zone"))
    if dist_cw < threshold:
        alerts.append(("🟢", f"Spot near Call Wall ({cw:.0f}) — resistance"))
    if dist_pw < threshold:
        alerts.append(("🔴", f"Spot near Put Wall ({pw:.0f}) — support"))

    em_data = em_analysis.get("expected_move", {})
    upper = em_data.get("upper_level")
    lower = em_data.get("lower_level")
    if upper and spot > upper:
        alerts.append(("🚀", f"ABOVE expected move upper ({upper:.0f})"))
    elif lower and spot < lower:
        alerts.append(("💥", f"BELOW expected move lower ({lower:.0f})"))

    return alerts


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
    """True if today is the first trading day of the month."""
    first = now_et.replace(day=1, hour=12, minute=0, second=0, microsecond=0)
    # Walk forward from the 1st to find the first trading weekday
    for offset in range(10):
        candidate = first + timedelta(days=offset)
        if candidate.weekday() >= 5:
            continue
        if not _is_trading_day(candidate):
            continue
        # This is the first trading day
        today = now_et.date() if hasattr(now_et, 'date') else now_et
        return candidate.date() == today
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

    if not is_market_open:
        return st.session_state.get(sk_snap)

    # Try to restore from Postgres
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


# ─────────────────────────────────────────────────────────────────────────────
# Spread Finder Tab — Weekly credit spread placement powered by HAR model + GEX
# ─────────────────────────────────────────────────────────────────────────────



@st.cache_resource
def _get_rf_conn():
    """Get or create the range finder database connection (Postgres or SQLite)."""
    from range_finder.db import get_connection, init_all_tables, get_backend
    conn = get_connection()
    init_all_tables(conn)
    return conn


def _build_chain_quotes_for_spreads(data: GEXData, ticker: str) -> tuple[dict, str | None]:
    """Build a strike→{call_bid, call_ask, put_bid, put_ask} lookup from cached chain data.

    Targets the next Friday expiration specifically, since the spread finder
    builds weekly credit spreads that expire on Fridays.

    Returns (quotes_dict, selected_expiration_str_or_None).
    """
    if not data.chain_cache:
        return {}, None

    from datetime import date as date_cls, timedelta

    # Find all cached expirations for this ticker, sorted by date
    cached_exps = sorted(
        exp for (t, exp) in data.chain_cache if t == ticker
    )
    if not cached_exps:
        return {}, None

    today = date_cls.today()
    today_str = today.isoformat()
    future_exps = [e for e in cached_exps if e >= today_str]

    # Find the next Friday from today
    days_until_friday = (4 - today.weekday()) % 7
    if days_until_friday == 0 and today.weekday() == 4:
        # Today is Friday — target next Friday for new trades
        days_until_friday = 7
    next_friday = today + timedelta(days=days_until_friday or 7)

    # Look for a Friday expiration in the cached chains
    target_exp = None
    friday_exps = [
        e for e in future_exps
        if date_cls.fromisoformat(e).weekday() == 4  # Friday
    ]
    if friday_exps:
        # Pick the nearest Friday (usually this coming Friday)
        target_exp = min(friday_exps, key=lambda e: abs((date_cls.fromisoformat(e) - next_friday).days))

    # Fallback: if no Friday expiration is cached, pick the closest to next Friday
    if target_exp is None and future_exps:
        target_exp = min(friday_exps if friday_exps else future_exps,
                         key=lambda e: abs((date_cls.fromisoformat(e) - next_friday).days))

    if target_exp is None:
        return {}, None

    entry = data.chain_cache.get((ticker, target_exp))
    if not entry or entry.get("status") != "ok":
        return {}, None

    quotes = {}  # strike -> {call_bid, call_ask, put_bid, put_ask}

    for opt in entry.get("calls", []):
        K = opt["strike"]
        if K not in quotes:
            quotes[K] = {}
        quotes[K]["call_bid"] = opt.get("bid", 0.0) or 0.0
        quotes[K]["call_ask"] = opt.get("ask", 0.0) or 0.0

    for opt in entry.get("puts", []):
        K = opt["strike"]
        if K not in quotes:
            quotes[K] = {}
        quotes[K]["put_bid"] = opt.get("bid", 0.0) or 0.0
        quotes[K]["put_ask"] = opt.get("ask", 0.0) or 0.0

    return quotes, target_exp


def _render_spread_finder_tab(spot: float, levels: dict, regime: dict, data, ticker: str = "SPX", weekly_em: dict = None):
    """Render the Spread Finder tab — HAR model forecast + GEX-enhanced spread placement."""
    import sqlite3
    import yfinance as yf
    from phase1.market_clock import now_ny

    ticker_cfg = RF_TICKER_CONFIG.get(ticker, RF_TICKER_CONFIG["SPX"])

    # ── Fetch latest VIX close (cached for the session) ──
    if "_sf_live_vix" not in st.session_state:
        try:
            vix_hist = yf.Ticker("^VIX").history(period="5d")
            if not vix_hist.empty:
                st.session_state["_sf_live_vix"] = round(float(vix_hist["Close"].dropna().iloc[-1]), 2)
            else:
                st.session_state["_sf_live_vix"] = 18.0
        except Exception:
            st.session_state["_sf_live_vix"] = 18.0
    live_vix = st.session_state["_sf_live_vix"]

    # ── Monday open freeze logic ──
    # On the weekly freeze day (Monday or Tue after holiday) at market open,
    # capture spot as the weekly reference. Rest of the week uses that frozen value.
    # Before Monday open (weekends), use the live spot (Friday close).
    run_now = now_ny()
    is_freeze_day = _is_weekly_freeze_day(run_now)
    is_market_open = data.market_open

    mon_open_key = f"sf_monday_open_{ticker}"
    mon_vix_key = f"sf_monday_vix_{ticker}"
    mon_open_week_key = f"sf_monday_open_week_{ticker}"

    # Determine which week we're in (use ISO week number)
    current_week = run_now.isocalendar()[1]

    # Freeze Monday's open on the freeze day when market is open
    if is_freeze_day and is_market_open:
        stored_week = st.session_state.get(mon_open_week_key)
        if stored_week != current_week:
            # First market-hours refresh on the freeze day — lock the open price + VIX
            st.session_state[mon_open_key] = round(spot, 2)
            st.session_state[mon_vix_key] = live_vix
            st.session_state[mon_open_week_key] = current_week

    # Determine the reference price/VIX and their source label
    frozen_open = st.session_state.get(mon_open_key)
    frozen_vix = st.session_state.get(mon_vix_key)
    frozen_week = st.session_state.get(mon_open_week_key)

    if frozen_week == current_week and frozen_open:
        default_ref = frozen_open
        default_vix = frozen_vix or live_vix
        ref_source = "Mon open (frozen)"
    else:
        # Try to restore Monday open + VIX from weekly_setup table
        restored_open = None
        restored_vix = None
        if run_now.weekday() < 5:  # weekday — might have a saved Monday open
            try:
                from datetime import timedelta as _td
                days_since_monday = run_now.weekday()
                monday = run_now - _td(days=days_since_monday)
                week_start_str = monday.strftime("%Y-%m-%d")
                rf_conn = _get_rf_conn()
                cur = rf_conn.cursor()
                cur.execute(
                    "SELECT monday_open, monday_vix FROM weekly_setup WHERE week_start = ? AND ticker = ?",
                    (week_start_str, ticker),
                )
                row = cur.fetchone()
                if row and row[0]:
                    restored_open = row[0]
                    restored_vix = row[1]
                    st.session_state[mon_open_key] = restored_open
                    if restored_vix:
                        st.session_state[mon_vix_key] = restored_vix
                    st.session_state[mon_open_week_key] = current_week
            except Exception:
                pass

        if restored_open:
            default_ref = restored_open
            default_vix = restored_vix or live_vix
            ref_source = "Mon open (from DB)"
        else:
            default_ref = round(spot, 2)
            default_vix = live_vix
            ref_source = "Fri close" if run_now.weekday() >= 5 else "live spot"

    # ── Auto-update reference price and VIX when ticker changes ──
    ref_key = f"sf_ref_price_{ticker}"
    vix_key = f"sf_vix_level_{ticker}"
    prev_ticker = st.session_state.get("_sf_prev_ticker")
    if prev_ticker != ticker:
        st.session_state[ref_key] = default_ref
        st.session_state[vix_key] = default_vix
        st.session_state["_sf_prev_ticker"] = ticker

    # Also update the defaults on first render if not yet set
    if ref_key not in st.session_state:
        st.session_state[ref_key] = default_ref
    if vix_key not in st.session_state:
        st.session_state[vix_key] = default_vix

    st.markdown(f"### {ticker} Weekly Credit Spread Finder")
    from range_finder.db import get_backend as rf_get_backend
    _rf_be = rf_get_backend()
    _rf_be_icon = "💾" if _rf_be == "postgres" else "⚡"
    st.caption(f"HAR regression range forecast + live GEX adjustment for optimal strike placement &nbsp;|&nbsp; {_rf_be_icon} {'Neon Postgres' if _rf_be == 'postgres' else 'Session-only (no DATABASE_URL)'}")

    # ── Extract GEX context from current dashboard data ──
    gex_ctx = extract_gex_context(levels, spot, regime)

    # ── Sidebar-like controls within the tab ──
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        step_size = ticker_cfg["strike_increment"]
        spx_close_input = st.number_input(
            f"{ticker} Reference ({ref_source})",
            min_value=100.0, max_value=15000.0, value=default_ref, step=float(step_size),
            help=f"Reference price for range calculation. Source: {ref_source}. "
                 "Frozen at Monday's open on the first market-hours refresh of the week.",
            key=ref_key,
        )

    with col_ctrl2:
        vix_source = "Mon open" if (frozen_week == current_week and frozen_vix) else "last close"
        vix_input = st.number_input(
            f"VIX Level ({vix_source})",
            min_value=5.0, max_value=100.0, value=default_vix, step=0.5,
            help=f"VIX level for BSM credit estimation. Source: {vix_source}. "
                 "Frozen at Monday's open alongside the reference price.",
            key=vix_key,
        )

    with col_ctrl3:
        model_choice = st.selectbox(
            "Model Spec",
            options=list(RF_MODEL_SPECS.keys()),
            index=2,
            help="M3_extended recommended; M4_full when GEX data is populated",
            key=f"sf_model_choice_{ticker}",
        )

    # Action buttons
    col_btn_main, col_btn1, col_btn2, col_btn3, col_btn4 = st.columns([1.3, 1, 1, 1, 1])

    with col_btn_main:
        do_weekly = st.button("Weekly Setup", key=f"sf_weekly_{ticker}", type="primary", use_container_width=True,
                              help="Run all steps: Refresh → Rebuild → Save GEX → Forecast")
    with col_btn1:
        do_refresh = st.button("Refresh Data", key=f"sf_refresh_{ticker}", use_container_width=True)
    with col_btn2:
        do_rebuild = st.button("Rebuild Features", key=f"sf_rebuild_{ticker}", use_container_width=True)
    with col_btn3:
        do_save_gex = st.button("Save GEX", key=f"sf_save_gex_{ticker}", use_container_width=True)
    with col_btn4:
        do_forecast = st.button("Forecast", key=f"sf_forecast_{ticker}", use_container_width=True)

    # Weekly Setup runs all four steps in sequence
    if do_weekly:
        do_refresh = do_rebuild = do_save_gex = do_forecast = True

    conn = _get_rf_conn()

    # ── Step 1: Refresh market data ──
    if do_refresh:
        with st.spinner("1/4 — Fetching SPX / VIX weekly data..."):
            try:
                df_spx = rf_fetch_spx_vix(years=6)
                rows_written = rf_save_spx_vix(conn, df_spx)
                if len(df_spx) == 0:
                    st.success("SPX/VIX data already up to date")
                else:
                    st.success(f"SPX/VIX data refreshed — {len(df_spx)} weeks fetched, {rows_written} new")
            except Exception as e:
                if "empty" in str(e).lower() and datetime.today().weekday() >= 4:
                    st.warning(f"SPX/VIX fetch returned empty data (expected on weekends/holidays). Existing data is still valid.")
                else:
                    st.error(f"SPX/VIX fetch failed: {e}")

        with st.spinner("1/4 — Fetching FRED macro data..."):
            try:
                df_macro = rf_fetch_fred_macro(years=6)
                rf_save_fred_macro(conn, df_macro)
                st.success(f"FRED macro data refreshed — {len(df_macro)} rows")
            except Exception as e:
                st.warning(f"FRED fetch skipped: {e} (set FRED_API_KEY in secrets to enable)")

        rf_build_event_flags(conn)

    # ── Step 2: Rebuild features ──
    if do_rebuild:
        with st.spinner("2/4 — Computing feature matrix..."):
            try:
                rf_build_features(conn)
                st.success("Features rebuilt")
            except Exception as e:
                st.error(f"Feature rebuild failed: {e}")

    # ── Step 3: Save live GEX ──
    if do_save_gex:
        try:
            gex_flag = save_gex_to_range_finder(gex_ctx, conn)
            regime_label = {1: "positive", 0: "neutral", -1: "negative"}.get(gex_flag, "unknown")
            st.success(f"GEX saved: regime={regime_label}, flag={gex_flag}")
        except Exception as e:
            st.error(f"GEX save failed: {e}")

    st.markdown("---")

    # ── Check data availability (reload after rebuild if needed) ──
    try:
        df_feat = rf_get_features(conn)
    except Exception:
        df_feat = pd.DataFrame()

    if df_feat.empty:
        st.info(
            "No feature data found. Click **Refresh Market Data** then **Rebuild Features** "
            "to initialize the range prediction model (requires FRED API key in environment)."
        )
        # Still show GEX context even without model data
        _render_gex_context_panel(gex_ctx, spot)
        return

    # ── Fit model or load from cache ──
    # ── Step 4: Fit model and forecast ──
    if do_forecast:
        with st.spinner(f"4/4 — Fitting {model_choice}..."):
            try:
                feat_cols = RF_MODEL_SPECS[model_choice]
                avail_cols = [c for c in feat_cols if c in df_feat.columns and df_feat[c].notna().sum() > 20]

                X_train, X_test, y_train, y_test = rf_time_series_split(
                    df_feat, feature_cols=avail_cols
                )
                result = rf_fit_model(X_train, y_train, model_name=model_choice)
                metrics = rf_evaluate_oos(result, X_test, y_test, model_name=model_choice)
                rf_save_model(result, avail_cols, model_choice, metrics, conn=conn)

                st.session_state[f"sf_model_result_{ticker}"]   = result
                st.session_state[f"sf_model_features_{ticker}"] = avail_cols
                st.session_state[f"sf_model_metrics_{ticker}"]  = metrics
            except Exception as e:
                st.error(f"Model fitting failed: {e}")
                return
        st.success(f"Model fitted | OOS R² = {metrics['oos_r2']:.4f}")

    # Try to load model from session or disk
    if f"sf_model_result_{ticker}" not in st.session_state:
        try:
            payload = rf_load_model(model_choice, conn=conn)
            st.session_state[f"sf_model_result_{ticker}"]   = payload["result"]
            st.session_state[f"sf_model_features_{ticker}"] = payload["feature_cols"]
            st.session_state[f"sf_model_metrics_{ticker}"]  = payload["metrics"]
        except FileNotFoundError:
            st.info("Click **Generate Forecast** to fit the model for the first time.")
            _render_gex_context_panel(gex_ctx, spot)
            return
        except Exception as e:
            st.warning(f"Saved model incompatible: {e}. Click **Generate Forecast** to refit.")
            _render_gex_context_panel(gex_ctx, spot)
            return

    result    = st.session_state[f"sf_model_result_{ticker}"]
    feat_cols = st.session_state[f"sf_model_features_{ticker}"]
    metrics   = st.session_state[f"sf_model_metrics_{ticker}"]

    # ── Determine week start ──
    from datetime import date as date_type
    today = datetime.today()
    days_ahead = (7 - today.weekday()) % 7 or 7
    next_monday = today + timedelta(days=days_ahead)
    week_start = next_monday.strftime("%Y-%m-%d")

    # ── Get feature row ──
    feature_row = rf_get_feature_for_week(conn, week_start)
    if feature_row is None:
        feature_row = df_feat.iloc[-1]

    # ── Cache key for spread computations ──
    # These depend on reference price, VIX, ticker, and week — NOT on which
    # risk tier is selected. Cache in session_state so switching tiers skips
    # the expensive forecast → plan → tiers pipeline entirely.
    _em_cache_sig = (
        round((weekly_em or {}).get("upper_level", 0) or 0, 2),
        round((weekly_em or {}).get("lower_level", 0) or 0, 2),
    )
    _sf_cache_key = (ticker, round(spx_close_input, 2), round(vix_input, 2), week_start, _em_cache_sig)
    _sf_prev_key = st.session_state.get(f"_sf_cache_key_{ticker}")

    if _sf_prev_key == _sf_cache_key and f"_sf_spread_tiers_{ticker}" in st.session_state:
        # Inputs unchanged — reuse cached results
        forecast     = st.session_state[f"_sf_forecast_{ticker}"]
        chain_quotes = st.session_state[f"_sf_chain_quotes_{ticker}"]
        chain_exp    = st.session_state[f"_sf_chain_exp_{ticker}"]
        plan         = st.session_state[f"_sf_plan_{ticker}"]
        spread_tiers = st.session_state[f"_sf_spread_tiers_{ticker}"]
        gex_adj      = st.session_state[f"_sf_gex_adj_{ticker}"]
    else:
        # ── Generate forecast ──
        forecast = rf_forecast_next_week(
            result, feature_row, feat_cols,
            spx_close_input, alpha=RF_PI_ALPHA,
        )

        # ── Extract chain quotes for market-based credit estimation ──
        chain_quotes, chain_exp = _build_chain_quotes_for_spreads(data, ticker)

        # ── Build spread plan ──
        plan = rf_build_spread_plan(
            forecast    = forecast,
            feature_row = feature_row,
            week_start  = week_start,
            vix_level   = vix_input,
            ticker      = ticker,
            chain_quotes= chain_quotes,
        )

        # ── Build risk tiers ──
        spread_tiers = rf_build_spread_tiers(
            forecast     = forecast,
            plan         = plan,
            spx_ref      = spx_close_input,
            vix_level    = vix_input,
            chain_quotes = chain_quotes,
            ticker       = ticker,
            weekly_em    = weekly_em,
        )

        # ── GEX enhancement ──
        gex_adj = adjust_spread_with_gex(plan, gex_ctx)

        # Cache results for tier switching
        st.session_state[f"_sf_cache_key_{ticker}"]    = _sf_cache_key
        st.session_state[f"_sf_forecast_{ticker}"]     = forecast
        st.session_state[f"_sf_chain_quotes_{ticker}"] = chain_quotes
        st.session_state[f"_sf_chain_exp_{ticker}"]    = chain_exp
        st.session_state[f"_sf_plan_{ticker}"]         = plan
        st.session_state[f"_sf_spread_tiers_{ticker}"] = spread_tiers
        st.session_state[f"_sf_gex_adj_{ticker}"]      = gex_adj

    # =========================================================================
    # METRIC CARDS
    # =========================================================================

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Point Estimate",
        f"{forecast['point_pct']*100:.2f}%",
        f"vs VIX: {forecast['model_vs_vix']*100:+.2f}%",
    )
    c2.metric(
        f"{forecast['confidence_level']}% PI Upper",
        f"{forecast['upper_pct']*100:.2f}%",
        "used for strike selection",
    )
    c3.metric(
        "Effective Range",
        f"{plan.effective_range_pct*100:.2f}%",
        f"buffer: +{plan.buffer_pct*100:.2f}%",
    )
    c4.metric(
        "GEX Regime",
        gex_ctx.gamma_regime.title(),
        f"flag: {gex_adj['gex_regime_flag']:+d}",
    )
    c5.metric(
        "OOS R²",
        f"{metrics['oos_r2']:.4f}",
        f"MAE: {metrics['mae_pct']*100:.2f}%",
    )

    st.markdown("---")

    # =========================================================================
    # RISK TIER SELECTOR + DEPENDENT UI (wrapped in fragment for fast switching)
    # =========================================================================

    # Store everything the fragment needs in session_state so it doesn't
    # rely on closure variables that become stale across fragment reruns.
    st.session_state["_rtf_spread_tiers"] = spread_tiers
    st.session_state["_rtf_forecast"]     = forecast
    st.session_state["_rtf_plan"]         = plan
    st.session_state["_rtf_spx_close"]    = spx_close_input
    st.session_state["_rtf_gex_ctx"]      = gex_ctx
    st.session_state["_rtf_ticker"]       = ticker
    st.session_state["_rtf_weekly_em"]    = weekly_em
    st.session_state["_rtf_chain_exp"]    = chain_exp
    st.session_state["_rtf_spot"]         = spot
    st.session_state["_rtf_gex_adj"]      = gex_adj

    @st.fragment
    def _risk_tier_fragment():
        _TIER_COLORS = {
            "aggressive":   "#ff4b4b",
            "moderate":     "#ffa726",
            "conservative": "#66bb6a",
        }

        # Read from session_state to avoid stale closure references
        _spread_tiers  = st.session_state["_rtf_spread_tiers"]
        _forecast      = st.session_state["_rtf_forecast"]
        _plan          = st.session_state["_rtf_plan"]
        _spx_close_inp = st.session_state["_rtf_spx_close"]
        _gex_ctx       = st.session_state["_rtf_gex_ctx"]
        _ticker        = st.session_state["_rtf_ticker"]
        _weekly_em     = st.session_state["_rtf_weekly_em"]
        _chain_exp     = st.session_state["_rtf_chain_exp"]
        _spot          = st.session_state["_rtf_spot"]
        _gex_adj       = st.session_state["_rtf_gex_adj"]

        tier_labels = [t.label for t in _spread_tiers]
        default_idx = len(tier_labels) - 1
        selected_tier_idx = st.radio(
            "Risk Tier",
            range(len(tier_labels)),
            format_func=lambda i: f"{tier_labels[i]}  ({_spread_tiers[i].range_pct*100:.1f}%)",
            index=default_idx,
            key=f"sf_risk_tier_{_ticker}",
            horizontal=True,
        )

        selected_tier = _spread_tiers[selected_tier_idx]
        tier_color = _TIER_COLORS.get(selected_tier.risk_level, "#888")
        st.markdown(
            f"<span style='color:{tier_color};font-size:18px;font-weight:bold;'>"
            f"{selected_tier.risk_level.upper()}</span>"
            f" &nbsp;—&nbsp; Range: {selected_tier.range_pct*100:.2f}%"
            f" &nbsp;|&nbsp; Calls above `{selected_tier.call_short:,.0f}`"
            f" &nbsp;|&nbsp; Puts below `{selected_tier.put_short:,.0f}`",
            unsafe_allow_html=True,
        )

        # =====================================================================
        # RANGE GAUGE + STRIKE MAP (side by side)
        # =====================================================================

        col_gauge, col_strikes = st.columns([1, 1])

        with col_gauge:
            st.markdown("**Range Distribution**")
            _render_sf_range_gauge(_forecast, _plan, _spx_close_inp)

        with col_strikes:
            st.markdown("**Strike Map with GEX Walls**")
            _render_sf_strike_map_tier(
                selected_tier, _plan, _spx_close_inp, _gex_ctx,
                _plan.recommended_width, ticker=_ticker,
                weekly_em=_weekly_em,
            )

        st.markdown("---")

        st.markdown(f"**Spread Parameters — {selected_tier.label}**")

        col_call, col_put = st.columns(2)

        with col_call:
            st.markdown(f"Call Spreads — short above `{selected_tier.call_short:,.0f}`")
            _render_sf_spread_table(selected_tier.call_spreads, _plan.recommended_width)

        with col_put:
            st.markdown(f"Put Spreads — short below `{selected_tier.put_short:,.0f}`")
            _render_sf_spread_table(selected_tier.put_spreads, _plan.recommended_width)

        # Show credit source note with chain expiration
        all_tier_spreads = selected_tier.call_spreads + selected_tier.put_spreads
        has_market = any(getattr(s, "credit_source", "bsm") == "market" for s in all_tier_spreads)
        has_bsm = any(getattr(s, "credit_source", "bsm") == "bsm" for s in all_tier_spreads)
        exp_note = f" Chain: {_chain_exp}" if _chain_exp else ""
        if has_market and has_bsm:
            st.caption(f"Credits from Friday chain bid/ask.{exp_note} &nbsp;|&nbsp; * = BSM estimate (strike not in chain).")
        elif has_market:
            st.caption(f"Credits from Friday chain bid/ask (short bid - long ask).{exp_note}")
        else:
            st.caption("Credits are BSM estimates (no Friday chain data available). Verify with broker before trading.")

        # =====================================================================
        # MODEL STRIKES (before EM floor) — shown only when EM adjusted
        # =====================================================================

        _has_model_call = selected_tier.model_call_short is not None and selected_tier.model_call_spreads
        _has_model_put  = selected_tier.model_put_short  is not None and selected_tier.model_put_spreads
        if _has_model_call or _has_model_put:
            st.markdown(
                f"**Model Strikes (before EM floor) — {selected_tier.label}**"
                " &nbsp; <span style='color:#ffa726;font-size:12px;'>"
                "These are the HAR model's original strikes, inside the weekly expected move.</span>",
                unsafe_allow_html=True,
            )
            col_mc, col_mp = st.columns(2)
            with col_mc:
                if _has_model_call:
                    st.markdown(f"Call Spreads — short above `{selected_tier.model_call_short:,.0f}`")
                    _render_sf_spread_table(selected_tier.model_call_spreads, _plan.recommended_width)
                else:
                    st.caption("Call short not adjusted by EM floor")
            with col_mp:
                if _has_model_put:
                    st.markdown(f"Put Spreads — short below `{selected_tier.model_put_short:,.0f}`")
                    _render_sf_spread_table(selected_tier.model_put_spreads, _plan.recommended_width)
                else:
                    st.caption("Put short not adjusted by EM floor")

        # =====================================================================
        # GEX CONTEXT + WARNINGS
        # =====================================================================

        st.markdown("---")

        col_gex, col_warn = st.columns([1, 1])

        with col_gex:
            _render_gex_context_panel(_gex_ctx, _spot)

        with col_warn:
            st.markdown("**Warnings & GEX Notes**")

            all_warnings = list(_plan.warnings) + _gex_adj.get("gex_adjustment_notes", [])

            # Per-tier weekly EM floor warnings
            _em_upper = (_weekly_em or {}).get("upper_level", 0) or 0
            _em_lower = (_weekly_em or {}).get("lower_level", 0) or 0
            if _em_upper and _em_lower:
                _half = selected_tier.range_pct / 2
                _raw_call = _spx_close_inp * (1 + _half)
                _raw_put  = _spx_close_inp * (1 - _half)
                if _raw_call < _em_upper:
                    all_warnings.append(
                        f"Weekly EM floor applied: call short widened from ~{_raw_call:,.0f} "
                        f"to {selected_tier.call_short:,.0f} (EM upper = {_em_upper:,.0f})"
                    )
                if _raw_put > _em_lower:
                    all_warnings.append(
                        f"Weekly EM floor applied: put short widened from ~{_raw_put:,.0f} "
                        f"to {selected_tier.put_short:,.0f} (EM lower = {_em_lower:,.0f})"
                    )

            if all_warnings:
                for w in all_warnings:
                    st.warning(w)
            else:
                st.success("No warnings for this week.")

            # Event flags
            events = {"FOMC": _plan.has_fomc, "CPI": _plan.has_cpi, "NFP": _plan.has_nfp, "OPEX": _plan.has_opex}
            active = [k for k, v in events.items() if v]
            if active:
                st.markdown(f"**Events this week:** {', '.join(active)}")
            else:
                st.caption("No major events this week")

    _risk_tier_fragment()

    # =========================================================================
    # LOG PLAN BUTTON
    # =========================================================================

    st.markdown("---")
    if st.button("Save Spread Plan to Database", key=f"sf_log_plan_{ticker}"):
        try:
            rf_log_spread_plan(conn, plan, wing_width_used=plan.recommended_width)
            st.success(f"Plan for {week_start} logged")
        except Exception as e:
            st.error(f"Failed to log plan: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Trade Log tab — spread outcome history + breach tracker
# ─────────────────────────────────────────────────────────────────────────────

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
    st.title("📊 Gamma Exposure Dashboard")
    st.caption(f"GEX Calculator {TOOL_VERSION} — Implied spot | Zero gamma sweep | Expected move | Hybrid IV")

    tradier_token, fred_key, gemini_key = get_credentials()

    # ── Sidebar controls ──
    with st.sidebar:
        st.header("⚙️ Settings")

        if not tradier_token:
            tradier_token = st.text_input("Tradier API Token", type="password",
                                           help="Get yours at https://web.tradier.com/user/api")
        if not fred_key:
            fred_key = st.text_input("FRED API Key (optional)", type="password",
                                      help="For live T-bill rates. Get at https://fred.stlouisfed.org/docs/api/api_key.html")

        if not tradier_token:
            st.warning("Enter your Tradier API token to get started.")
            st.stop()

        st.divider()

        # Ticker selector
        ticker = st.selectbox("Index", ["SPX", "XSP"], index=0, key="ticker_select")

        # Expiration picker
        with st.spinner("Loading expirations..."):
            try:
                temp_client = TradierDataClient(token=tradier_token)
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
            data = fetch_all_data(tradier_token, fred_key or "", tuple(selected), run_id, ticker=ticker)
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
    temp_client = TradierDataClient(token=tradier_token)

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
        from ui_history import _render_iv_surface
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
