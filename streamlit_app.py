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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Phase1 engine imports ──
from phase1.config import HEATMAP_EXPS, NY_TZ, build_config_snapshot
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
    STANDARD_WING_WIDTHS as RF_WING_WIDTHS,
    TICKER_CONFIG as RF_TICKER_CONFIG,
    SpreadPlan,
    SpreadTier,
)

_logger = logging.getLogger(__name__)

TOOL_VERSION = "v5-web"

# ── Color palette ──
COLORS = {
    "bg_primary": "#1a1a2e",
    "bg_sidebar": "#16213e",
    "bg_card": "#1a1a3e",
    "spot": "#ffffff",
    "zero_gamma": "#00e5ff",
    "call_wall": "#69f0ae",
    "put_wall": "#ff8a80",
    "positive": "#00c853",
    "negative": "#ff5252",
    "bar_green": "#00c853",
    "bar_red": "#ff1744",
    "em_level": "#b388ff",
    "em_weekly": "#ffd740",
    "em_monthly": "#4dd0e1",
    "profile_line": "#9c88ff",
    "warning": "#ffd600",
    "text_muted": "#888",
    "text_secondary": "#aaa",
    "text_light": "#cfd3ff",
    "text_white": "#fff",
    "grid_major": "#333",
    "grid_minor": "#222",
    "zeroline": "#555",
}


@dataclass
class GEXData:
    spot: float
    spot_source: str
    spot_info: dict
    rfr: float
    rfr_info: dict
    avail: list
    target_exps: list
    gex_df: Any  # pd.DataFrame
    hm_gex: Any
    hm_iv: Any
    stats: dict
    all_options: list
    levels: dict
    profile_df: Any
    sensitivity_df: Any
    scenarios_df: Any
    staleness_info: dict
    confidence_info: dict
    wall_cred: dict
    regime_info: dict
    calendar_snapshot: dict
    run_time: str
    prev_close: float
    spy_quote: dict | None
    dte0_calls: list
    dte0_puts: list
    market_open: bool
    yahoo_es: dict | None
    chain_cache: dict | None


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

    # Try st.secrets first (for Streamlit Cloud deployment)
    try:
        tradier_token = st.secrets.get("TRADIER_TOKEN", "")
        fred_key = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        pass

    # Fall back to env vars
    if not tradier_token:
        tradier_token = os.environ.get("TRADIER_TOKEN", "")
    if not fred_key:
        fred_key = os.environ.get("FRED_API_KEY", "")

    return tradier_token, fred_key


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
            lower = spot * 0.95
            upper = spot * 1.05
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


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────
def build_gex_bar_chart(gex_df, levels, spot, em_analysis, weekly_em=None, monthly_em=None):
    df = gex_df.copy().sort_values("strike").reset_index(drop=True)
    strikes = df["strike"].values
    net_gex = df["net_gex"].values
    colors = [COLORS["bar_green"] if g >= 0 else COLORS["bar_red"] for g in net_gex]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=strikes, x=net_gex, orientation="h",
        marker_color=colors, marker_opacity=0.85,
        hovertemplate="Strike: $%{y:.0f}<br>Net GEX: %{x:,.0f}<extra></extra>",
    ))

    # Level lines
    for val, color, dash, name in [
        (spot, COLORS["spot"], "dash", "Spot"),
        (levels["zero_gamma"], COLORS["zero_gamma"], "dot", "Zero Γ"),
        (levels["call_wall"], COLORS["call_wall"], "dashdot", "Call Wall"),
        (levels["put_wall"], COLORS["put_wall"], "dashdot", "Put Wall"),
    ]:
        fig.add_hline(y=val, line_color=color, line_dash=dash, line_width=1.5,
                       annotation_text=f"{name} ${val:.0f}",
                       annotation_font_color=color, annotation_font_size=9,
                       annotation_position="top left")

    # EM levels (0DTE — purple dotted)
    em = em_analysis.get("expected_move", {})
    if em.get("upper_level"):
        for val, label in [(em["upper_level"], "EM+"), (em["lower_level"], "EM−")]:
            fig.add_hline(y=val, line_color=COLORS["em_level"], line_dash="dot", line_width=1.2,
                           annotation_text=f"{label} ${val:.0f}",
                           annotation_font_color=COLORS["em_level"], annotation_font_size=8,
                           annotation_position="bottom right")

    # Weekly EM levels (amber dashed)
    w_em = weekly_em or {}
    if w_em.get("upper_level"):
        for val, label in [(w_em["upper_level"], "wEM+"), (w_em["lower_level"], "wEM−")]:
            fig.add_hline(y=val, line_color=COLORS["em_weekly"], line_dash="dash", line_width=1,
                           annotation_text=f"{label} ${val:.0f}",
                           annotation_font_color=COLORS["em_weekly"], annotation_font_size=7,
                           annotation_position="top right")

    # Monthly EM levels (cyan longdash)
    m_em = monthly_em or {}
    if m_em.get("upper_level"):
        for val, label in [(m_em["upper_level"], "mEM+"), (m_em["lower_level"], "mEM−")]:
            fig.add_hline(y=val, line_color=COLORS["em_monthly"], line_dash="longdash", line_width=1,
                           annotation_text=f"{label} ${val:.0f}",
                           annotation_font_color=COLORS["em_monthly"], annotation_font_size=7,
                           annotation_position="top right")

    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"], plot_bgcolor=COLORS["bg_primary"],
        font_color="white", font_size=10,
        margin=dict(l=80, r=10, t=35, b=35),
        title="Strike-by-Strike Net GEX Proxy",
        xaxis=dict(title="Net GEX proxy", gridcolor=COLORS["grid_major"], zerolinecolor=COLORS["zeroline"]),
        yaxis=dict(title="Strike", gridcolor=COLORS["grid_minor"], tickfont_size=8),
        showlegend=False, height=700, dragmode=False,
    )
    return fig


def build_profile_chart(profile_df, levels, spot, regime_info, em_analysis, weekly_em=None, monthly_em=None):
    if profile_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=profile_df["price"], y=profile_df["total_gex"],
        mode="lines", line=dict(color=COLORS["profile_line"], width=2),
        hovertemplate="Price: $%{x:.2f}<br>GEX: %{y:,.0f}<extra></extra>",
    ))

    # Level lines
    fig.add_vline(x=spot, line_color=COLORS["spot"], line_dash="dash", line_width=1.5,
                   annotation_text="Spot", annotation_font_color=COLORS["spot"])
    fig.add_vline(x=levels["zero_gamma"], line_color=COLORS["zero_gamma"], line_dash="dot", line_width=1.5,
                   annotation_text="Zero Γ", annotation_font_color=COLORS["zero_gamma"])
    fig.add_hline(y=0, line_color=COLORS["zeroline"], line_width=1)

    # EM levels (0DTE — purple dotted)
    em = em_analysis.get("expected_move", {})
    if em.get("upper_level"):
        for val, label in [(em["upper_level"], "EM+"), (em["lower_level"], "EM−")]:
            fig.add_vline(x=val, line_color=COLORS["em_level"], line_dash="dot", line_width=1.2,
                           annotation_text=label, annotation_font_color=COLORS["em_level"],
                           annotation_font_size=9)

    # Weekly EM levels (amber dashed)
    w_em = weekly_em or {}
    if w_em.get("upper_level"):
        for val, label in [(w_em["upper_level"], "wEM+"), (w_em["lower_level"], "wEM−")]:
            fig.add_vline(x=val, line_color=COLORS["em_weekly"], line_dash="dash", line_width=1,
                           annotation_text=label, annotation_font_color=COLORS["em_weekly"],
                           annotation_font_size=8)

    # Monthly EM levels (cyan longdash)
    m_em = monthly_em or {}
    if m_em.get("upper_level"):
        for val, label in [(m_em["upper_level"], "mEM+"), (m_em["lower_level"], "mEM−")]:
            fig.add_vline(x=val, line_color=COLORS["em_monthly"], line_dash="longdash", line_width=1,
                           annotation_text=label, annotation_font_color=COLORS["em_monthly"],
                           annotation_font_size=8)

    # Regime badge
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98,
        text=f"<b>{regime_info['regime']}</b><br>{regime_info['distance_text']}",
        font=dict(color=regime_info["color"], size=11),
        bgcolor="rgba(20,20,30,0.7)", bordercolor=regime_info["color"],
        borderwidth=1, borderpad=5, showarrow=False, align="left",
    )

    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"], plot_bgcolor=COLORS["bg_primary"],
        font_color="white", font_size=10,
        margin=dict(l=60, r=10, t=35, b=40),
        title="GEX Profile Curve",
        xaxis=dict(title="Underlying Price", gridcolor=COLORS["grid_major"]),
        yaxis=dict(title="Total GEX proxy", gridcolor=COLORS["grid_minor"], zerolinecolor=COLORS["zeroline"]),
        showlegend=False, height=500, dragmode=False,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar rendering
# ─────────────────────────────────────────────────────────────────────────────
def _render_move_display(overnight, classification, futures_ctx, market_ctx):
    """Render the overnight/today's move section."""
    move_source = classification.get("move_source", "spx")

    if market_ctx == "live":
        on_pts = overnight.get("overnight_move_pts")
        if on_pts is not None:
            arrow = "🟢 ▲" if on_pts >= 0 else "🔴 ▼"
            st.markdown(
                f"**Today's Move:** {arrow} **{on_pts:+.1f} pts** ({overnight['overnight_move_pct']:+.2f}%)"
            )
    elif "es_futures" in move_source and futures_ctx:
        arrow = "🟢 ▲" if futures_ctx["overnight_move_pts"] >= 0 else "🔴 ▼"
        st.markdown(
            f"**Overnight (ES):** {arrow} **{futures_ctx['overnight_move_pts']:+.1f} pts** ({futures_ctx['overnight_move_pct']:+.2f}%)"
        )
        src_label = "manual" if futures_ctx["source"] == "manual" else "Yahoo ~10m delayed"
        st.caption(f"ES: \\${futures_ctx['es_last']:.2f} vs SPX prevclose \\${futures_ctx['spx_prevclose']:.2f} ({src_label})")
    else:
        on_pts = overnight.get("overnight_move_pts")
        if on_pts is not None:
            label = "Session Move" if market_ctx == "afterhours" else "Overnight Move"
            arrow = "🟢 ▲" if on_pts >= 0 else "🔴 ▼"
            st.markdown(
                f"**{label}:** {arrow} **{on_pts:+.1f} pts** ({overnight['overnight_move_pct']:+.2f}%)"
            )
    return move_source


def _render_overnight_range(overnite_range, move_source, spy):
    """Render overnight range and SPY proxy."""
    if overnite_range and overnite_range.get("es_high"):
        hi = overnite_range["high_move_from_close"]
        lo = overnite_range["low_move_from_close"]
        max_em = overnite_range.get("max_move_vs_em")
        max_em_str = f" ({max_em*100:.0f}% of EM)" if max_em else ""
        st.markdown(
            f"**O/N Range:** :green[${overnite_range['es_high']:.0f}] ({hi:+.0f}) "
            f"— :red[${overnite_range['es_low']:.0f}] ({lo:+.0f}) "
            f"= {overnite_range['range_pts']:.0f} pts"
        )
        st.caption(f"Max overnight excursion: {overnite_range['max_move_pts']:.0f} pts{max_em_str}")

    if spy and "es_futures" not in move_source:
        st.caption(
            f"SPY Pre-mkt: ${spy['spy_price']:.2f} ({spy['spy_move_pct']:+.2f}%) "
            f"→ ~{spy['implied_spx_move_pts']:+.1f} SPX pts"
        )


def _render_classification(classification, level_ctx):
    """Render session classification and zero gamma context."""
    ratio = classification.get("move_ratio")
    if ratio is not None:
        pct = min(ratio * 100, 100)
        label = classification.get("move_ratio_label", "")
        st.markdown(f"**Vol Budget Used:** {pct:.0f}% ({label})")
        st.progress(min(ratio, 1.0))

    if classification.get("classification"):
        bias = classification.get("bias", "")
        if bias in ("range-bound", "mean-revert"):
            cls_icon = "🟢"
        elif bias in ("directional", "continued-trend"):
            cls_icon = "🔴"
        else:
            cls_icon = "🟡"

        st.markdown(f"### {cls_icon} {classification['classification']}")
        if classification.get("description"):
            st.caption(classification["description"])
        if classification.get("historical_tendencies"):
            st.markdown(f"**Tendencies:** {classification['historical_tendencies'][0]}")
        if classification.get("confidence_note"):
            st.caption(classification["confidence_note"])

    if level_ctx and level_ctx.get("zero_gamma_within_em") is not None:
        inside = "✅ inside" if level_ctx["zero_gamma_within_em"] else "⚠️ outside"
        st.markdown(
            f"**Zero Γ:** {inside} expected range "
            f"({level_ctx['zero_gamma_distance_to_spot']:+.1f} pts from spot)"
        )


def render_expected_move_panel(em_analysis):
    em_data = em_analysis.get("expected_move", {})
    overnight = em_analysis.get("overnight_move", {})
    classification = em_analysis.get("classification", {})
    spy = em_analysis.get("spy_proxy")
    futures_ctx = em_analysis.get("futures_context")
    overnite_range = em_analysis.get("overnight_range")
    level_ctx = em_analysis.get("level_context")
    market_ctx = em_analysis.get("market_context", "live")

    if em_data.get("expected_move_pts") is None:
        st.caption("Expected move data not available.")
        return

    st.markdown("#### ⚡ Expected Move — 0DTE")

    # Straddle & range
    straddle = em_data.get("straddle", {})
    em_pts_val = em_data.get("expected_move_pts", 0) or 0
    em_pct_val = em_data.get("expected_move_pct", 0) or 0
    em_lower = em_data.get("lower_level")
    em_upper = em_data.get("upper_level")
    straddle_html = (
        '<div class="level-grid" style="grid-template-columns: 1fr 1fr;">'
        f'<div class="level-card"><div class="lbl">ATM Straddle</div><div class="val">{em_pts_val:.1f} pts</div><div class="lbl">{em_pct_val:.2f}%</div></div>'
        f'<div class="level-card"><div class="lbl">Strike</div><div class="val">${straddle.get("strike", "?")}</div></div>'
        '</div>'
    )
    st.markdown(straddle_html, unsafe_allow_html=True)

    if em_lower is not None and em_upper is not None:
        st.markdown(
            f"**Expected Range:** "
            f":red[${em_lower:.0f}] — :green[${em_upper:.0f}]"
        )

    move_source = _render_move_display(overnight, classification, futures_ctx, market_ctx)
    _render_overnight_range(overnite_range, move_source, spy)
    _render_classification(classification, level_ctx)

    st.divider()


def render_key_levels(levels, spot, regime_info, confidence_info, staleness_info):
    conf_score = confidence_info.get("score", 0)
    conf_label = confidence_info.get("label", "?")
    conf_color = COLORS["positive"] if conf_label == "High" else COLORS["warning"] if conf_label == "Moderate" else COLORS["negative"]

    spot_c = COLORS["spot"]
    cw_c = COLORS["call_wall"]
    pw_c = COLORS["put_wall"]
    zg_c = COLORS["zero_gamma"]
    html = (
        '<div class="level-grid">'
        f'<div class="level-card"><div class="lbl">Spot</div><div class="val" style="color:{spot_c};">${spot:.2f}</div></div>'
        f'<div class="level-card"><div class="lbl">Call Wall</div><div class="val" style="color:{cw_c};">${levels["call_wall"]:.0f}</div></div>'
        f'<div class="level-card"><div class="lbl">Put Wall</div><div class="val" style="color:{pw_c};">${levels["put_wall"]:.0f}</div></div>'
        f'<div class="level-card"><div class="lbl">Zero Gamma</div><div class="val" style="color:{zg_c};">${levels["zero_gamma"]:.2f}</div></div>'
        f'<div class="level-card"><div class="lbl">Regime</div><div class="val" style="color:{regime_info["color"]};font-size:12px;">{regime_info["regime"]}</div></div>'
        f'<div class="level-card"><div class="lbl">Confidence</div><div class="val" style="color:{conf_color};">{conf_score:.0f}</div><div class="lbl" style="color:{conf_color};">{conf_label}</div></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    if not levels.get("is_true_crossing", True):
        st.warning("⚠️ Zero gamma is a fallback estimate — no true sign-change crossing was found in the sweep range. Use this level with caution.")

    st.caption(
        "**Dealer positioning assumption:** GEX models assume dealers are net short calls and net short puts "
        "(the standard retail convention). In reality, dealer positioning varies by strike — institutional "
        "overlays (collars, risk reversals) and retail put-selling can invert the sign at specific strikes. "
        "Open interest updates once daily (EOD), so intraday flow is not reflected. "
        "Treat GEX levels as probabilistic zones, not hard barriers."
    )


def _fmt_delta(val, base_val):
    """Format a value as 'value (delta)' with color coding."""
    delta = val - base_val
    if abs(delta) < 0.5:
        return f"${val:.0f}"
    color = COLORS["positive"] if delta > 0 else COLORS["negative"]
    return f"${val:.0f} <span style='color:{color};font-size:9px;'>({delta:+.0f})</span>"


def _fmt_gex_short(v):
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.1f}M"
    if abs(v) >= 1000:
        return f"{v/1000:.0f}K"
    return f"{v:.0f}"


def render_scenarios_table(scenarios_df):
    if scenarios_df is None or scenarios_df.empty:
        return
    st.markdown("#### Scenario Analysis")
    st.caption("How key levels shift under spot shocks. Deltas shown vs Base.")

    base = scenarios_df.iloc[0]
    base_cw = float(base["call_wall"])
    base_pw = float(base["put_wall"])
    base_zg = float(base["zero_gamma"])
    base_gex = float(base.get("net_gex", 0))

    rows_html = ""
    for _, row in scenarios_df.iterrows():
        regime = row.get("gamma_regime", "")
        is_base = row["scenario"] == "Base"
        r_color = COLORS["positive"] if "Pos" in regime else COLORS["negative"] if "Neg" in regime else COLORS["zero_gamma"]
        tl_c = COLORS["text_light"]
        cw_c = COLORS["call_wall"]
        pw_c = COLORS["put_wall"]
        zg_c = COLORS["zero_gamma"]
        bg = f"background:{COLORS['bg_card']};" if is_base else ""

        net_gex = row.get("net_gex", 0)
        gex_delta = net_gex - base_gex
        gex_color = COLORS["positive"] if net_gex > 0 else COLORS["negative"]

        cw_val = f"${row['call_wall']:.0f}" if is_base else _fmt_delta(row["call_wall"], base_cw)
        pw_val = f"${row['put_wall']:.0f}" if is_base else _fmt_delta(row["put_wall"], base_pw)
        zg_val = f"${row['zero_gamma']:.0f}" if is_base else _fmt_delta(row["zero_gamma"], base_zg)

        rows_html += (
            f'<tr style="{bg}">'
            f'<td style="color:{tl_c};font-weight:bold;">{row["scenario"]}</td>'
            f'<td>${row["spot"]:.0f}</td>'
            f'<td style="color:{cw_c};">{cw_val}</td>'
            f'<td style="color:{pw_c};">{pw_val}</td>'
            f'<td style="color:{zg_c};">{zg_val}</td>'
            f'<td style="color:{gex_color};font-size:9px;">{_fmt_gex_short(net_gex)}</td>'
            f'<td style="color:{r_color};font-size:9px;">{regime}</td></tr>'
        )

    bg_card = COLORS["bg_card"]
    text_m = COLORS["text_muted"]
    table_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:11px;margin-bottom:10px;">'
        f'<thead><tr style="background:{bg_card};color:{text_m};font-size:10px;">'
        '<th style="padding:4px;text-align:left;">Scenario</th>'
        '<th style="padding:4px;">Spot</th>'
        '<th style="padding:4px;">CW</th>'
        '<th style="padding:4px;">PW</th>'
        '<th style="padding:4px;">ZG</th>'
        '<th style="padding:4px;">Net GEX</th>'
        '<th style="padding:4px;">Regime</th>'
        '</tr></thead>'
        f'<tbody style="color:#ddd;text-align:center;">{rows_html}</tbody>'
        '</table>'
    )
    st.markdown(table_html, unsafe_allow_html=True)


def render_wall_credibility(wall_cred):
    if not wall_cred:
        return
    st.markdown("#### Wall Credibility")
    for key, label in [("call_wall", "Call Wall"), ("put_wall", "Put Wall"), ("zero_gamma", "Zero Γ")]:
        info = wall_cred.get(key, {})
        if not info:
            continue
        score = info.get("score", 0)
        lbl = info.get("label", "?")
        color = "🟢" if lbl == "High" else "🟡" if lbl == "Moderate" else "🔴"
        st.markdown(f"{color} **{label}:** {score:.0f}/100 ({lbl})")
        reasons = info.get("reasons", [])
        for r in reasons[:2]:
            st.caption(f"  • {r}")


def render_gex_stream(stats, levels, spot):
    """Sidebar GEX Stream panel — key metrics at a glance."""
    st.markdown("#### 📡 GEX Stream")

    gex_ratio = stats.get("gex_ratio", 0)
    net_gex = stats.get("net_gex", 0)
    pc_ratio = stats.get("pc_ratio", 0)
    call_iv = stats.get("call_iv", 0)
    put_iv = stats.get("put_iv", 0)

    # Color coding
    gr_color = COLORS["positive"] if gex_ratio > 1 else COLORS["negative"]
    ng_color = COLORS["positive"] if net_gex > 0 else COLORS["negative"]
    cw_c = COLORS["call_wall"]
    pw_c = COLORS["put_wall"]
    zg_c = COLORS["zero_gamma"]
    text_w = COLORS["text_white"]
    text_m = COLORS["text_muted"]

    # Format net GEX
    ng_fmt = stats.get("net_gex_fmt", f"{net_gex:.0f}")

    # GEX Ratio sigma (rough heuristic: 1.0 = neutral)
    gr_sigma = abs(gex_ratio - 1.0) / 0.5
    gr_sigma_str = f"{gr_sigma:.1f}σ"

    stream_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:8px;">
      <tr>
        <td style="color:{text_m};padding:3px 6px;">GEX Ratio</td>
        <td style="color:{gr_color};font-weight:bold;text-align:right;padding:3px 6px;">{gex_ratio:.2f}</td>
        <td style="color:{text_m};font-size:10px;text-align:right;padding:3px 6px;">{gr_sigma_str}</td>
        <td style="color:{text_m};padding:3px 6px;">Net GEX</td>
        <td style="color:{ng_color};font-weight:bold;text-align:right;padding:3px 6px;">{ng_fmt}</td>
      </tr>
      <tr style="border-top:1px solid #333;">
        <td style="color:{text_m};padding:3px 6px;">Call OI</td>
        <td colspan="2" style="color:{cw_c};font-weight:bold;text-align:right;padding:3px 6px;">{stats.get("call_oi", "0")} @ {stats.get("call_oi_strike", 0):.0f}</td>
        <td colspan="2" style="color:{text_m};"></td>
      </tr>
      <tr>
        <td style="color:{text_m};padding:3px 6px;">Pos GEX</td>
        <td colspan="2" style="color:{cw_c};font-weight:bold;text-align:right;padding:3px 6px;">{stats.get("pos_gex", "0")} @ {stats.get("pos_gex_strike", 0):.0f}</td>
        <td colspan="2" style="color:{text_m};"></td>
      </tr>
      <tr>
        <td style="color:{text_m};padding:3px 6px;">Zero Gamma</td>
        <td colspan="4" style="color:{zg_c};font-weight:bold;text-align:right;padding:3px 6px;">{levels.get("zero_gamma", 0):,.2f}</td>
      </tr>
      <tr>
        <td style="color:{text_m};padding:3px 6px;">Neg GEX</td>
        <td colspan="2" style="color:{pw_c};font-weight:bold;text-align:right;padding:3px 6px;">{stats.get("neg_gex", "0")} @ {stats.get("neg_gex_strike", 0):.0f}</td>
        <td colspan="2" style="color:{text_m};"></td>
      </tr>
      <tr>
        <td style="color:{text_m};padding:3px 6px;">Put OI</td>
        <td colspan="2" style="color:{pw_c};font-weight:bold;text-align:right;padding:3px 6px;">{stats.get("put_oi", "0")} @ {stats.get("put_oi_strike", 0):.0f}</td>
        <td colspan="2" style="color:{text_m};"></td>
      </tr>
      <tr style="border-top:1px solid #333;">
        <td style="color:{text_m};padding:3px 6px;">Call IV</td>
        <td style="color:{text_w};font-weight:bold;text-align:right;padding:3px 6px;">{call_iv:.1f}%</td>
        <td style="color:{text_m};padding:3px 6px;"></td>
        <td style="color:{text_m};padding:3px 6px;">Put IV</td>
        <td style="color:{text_w};font-weight:bold;text-align:right;padding:3px 6px;">{put_iv:.1f}%</td>
      </tr>
      <tr style="border-top:1px solid #333;">
        <td style="color:{text_m};padding:3px 6px;">P/C OI Ratio</td>
        <td colspan="4" style="color:{text_w};font-weight:bold;text-align:right;padding:3px 6px;">{pc_ratio:.2f}</td>
      </tr>
    </table>
    """
    st.markdown(stream_html, unsafe_allow_html=True)


def render_data_quality(stats, staleness_info):
    st.markdown("#### Data Quality")

    fresh = staleness_info.get("freshness_score", 0)
    fresh_lbl = staleness_info.get("freshness_label", "?")
    fresh_color = COLORS["positive"] if fresh_lbl == "High" else COLORS["warning"] if fresh_lbl == "Moderate" else COLORS["negative"]

    coverage = stats.get("coverage_ratio", 0)
    cov_color = COLORS["positive"] if coverage >= 0.95 else COLORS["warning"] if coverage >= 0.85 else COLORS["negative"]

    html = (
        '<div class="level-grid">'
        f'<div class="level-card"><div class="lbl">Used Options</div><div class="val">{stats.get("used_option_count", 0):,}</div></div>'
        f'<div class="level-card"><div class="lbl">Coverage</div><div class="val" style="color:{cov_color};">{coverage*100:.1f}%</div></div>'
        f'<div class="level-card"><div class="lbl">Freshness</div><div class="val" style="color:{fresh_color};">{fresh:.0f}</div><div class="lbl" style="color:{fresh_color};">{fresh_lbl}</div></div>'
        f'<div class="level-card"><div class="lbl">Direct IV</div><div class="val">{stats.get("direct_iv_count", 0):,}</div></div>'
        f'<div class="level-card"><div class="lbl">Synthetic IV</div><div class="val">{stats.get("synthetic_iv_count", 0):,}</div></div>'
        f'<div class="level-card"><div class="lbl">Skipped (quality)</div><div class="val">{stats.get("skipped_count", 0):,}</div></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    # Detailed filter breakdown
    range_filt = stats.get("range_filtered_count", 0)
    zero_oi = stats.get("zero_oi_filtered_count", 0)
    if range_filt or zero_oi:
        st.caption(f"Filtered: {range_filt:,} out-of-range strikes, {zero_oi:,} zero-OI contracts")


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

    # ── View toggle: daily summary vs intraday snapshots ──
    view = st.radio("View", ["Daily Summary", "Today's Snapshots"], horizontal=True, key="hist_view")

    if view == "Today's Snapshots":
        # Show all snapshots from the current session (useful for debugging)
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

        # Intraday chart
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
    # Backward compat: tag rows if scan_type not present (old data)
    if "scan_type" not in hist_df.columns:
        hist_df["scan_type"] = "close"

    open_df = hist_df[hist_df["scan_type"] == "open"].copy()
    close_df = hist_df[hist_df["scan_type"] == "close"].copy()
    # For days with only an open scan and no close yet, also include in close for line continuity
    if close_df.empty and not open_df.empty:
        close_df = open_df.copy()

    fig = go.Figure()

    # -- Open scan (9:30 AM) --
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

    # -- Close scan (3:59 PM) — connected lines --
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

    # -- Open-to-Close range shading per day --
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

    # Summary table — show open and close side by side
    with st.expander("📋 Daily Summary"):
        display_cols = ["date", "scan_type", "spot", "zero_gamma", "call_wall", "put_wall",
                        "regime", "confidence_score", "coverage_ratio"]
        avail_cols = [c for c in display_cols if c in hist_df.columns]
        st.dataframe(hist_df[avail_cols].head(60), use_container_width=True, hide_index=True)


def _render_em_tracker(em_analysis, spot, prev_close, market_ctx, label="0DTE", subtitle=None):
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
    if em_anchor > 0:
        current_move = abs(spot - em_anchor)
        move_pct_of_em = (current_move / em_pts) * 100
        direction = "up" if spot >= em_anchor else "down"
    else:
        current_move = 0
        move_pct_of_em = 0
        direction = "flat"

    # Gauge display
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
    col2.metric("Current Move", f"{current_move:.1f} pts {direction}")
    col3.metric("EM Consumed", f"{move_pct_of_em:.0f}%")

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
    tradier_token, _ = get_credentials()
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
        height=700, dragmode=False,
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


def _render_iv_surface(hm_iv, hm_gex, spot):
    """C4: Render IV surface heatmap (strike × expiration)."""
    if hm_iv is None or hm_iv.empty:
        st.info("IV surface data not available. Requires multiple expirations.")
        return

    view = st.radio("View", ["IV Surface", "GEX Heatmap"], horizontal=True, key="iv_view")

    with st.expander("How to read this chart"):
        if view == "IV Surface":
            st.markdown("""
**IV Surface — Implied Volatility across strikes and expirations**

- **X-axis** = Expiration date (left = near-term, right = further out)
- **Y-axis** = Strike price (white dashed line = current spot)
- **Color** = Implied volatility at that strike/expiration
  - **Dark/purple** = low IV (cheap options)
  - **Bright/yellow** = high IV (expensive options)

**What to look for:**
- **Volatility smile/skew:** OTM puts (below spot) typically have higher IV than OTM calls — this shows as brighter colors below the spot line. A steep skew means the market is pricing in more downside risk.
- **Term structure:** Compare left to right. If near-term IV is higher than far-term (brighter on the left), the market expects a move soon (event-driven). If far-term is higher, the market is calm short-term.
- **Hot spots:** Bright patches at specific strikes/expirations can indicate unusual activity or event pricing (e.g., FOMC, earnings).
""")
        else:
            st.markdown("""
**GEX Heatmap — Gamma Exposure across strikes and expirations**

- **X-axis** = Expiration date
- **Y-axis** = Strike price (white dashed line = current spot)
- **Color** = Net gamma exposure at that strike/expiration
  - **Green** = positive gamma (dealers are long gamma — they buy dips, sell rips, suppressing moves)
  - **Red** = negative gamma (dealers are short gamma — they sell dips, buy rips, amplifying moves)
  - **Yellow** = neutral / near zero

**What to look for:**
- **Green cluster around spot:** Dealers are positioned to suppress volatility. Price tends to stay range-bound. Good for selling credit spreads.
- **Red zone around spot:** Dealers amplify moves. Expect larger swings. Widen your spreads or reduce size.
- **Bright green at a specific strike:** That strike acts as a "magnet" — price may pin there near expiration as dealer hedging creates support/resistance.
- **Comparison across expirations:** GEX from near-term expirations has the strongest effect on price action (gamma decays as expiration approaches).
""")

    hm_data = hm_iv if view == "IV Surface" else hm_gex

    if hm_data.empty:
        st.info(f"{view} data not available.")
        return

    # Clean data
    strikes = hm_data.index.tolist()
    expirations = hm_data.columns.tolist()
    z_values = hm_data.values

    colorscale = "Viridis" if view == "IV Surface" else "RdYlGn"
    title = "Implied Volatility Surface" if view == "IV Surface" else "GEX Heatmap"

    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=expirations,
        y=strikes,
        colorscale=colorscale,
        hovertemplate="Exp: %{x}<br>Strike: %{y}<br>Value: %{z:.4f}<extra></extra>",
    ))

    # Mark spot on y-axis — use white with dark background for contrast against heatmap
    fig.add_hline(
        y=spot, line_color="white", line_dash="dash", line_width=2,
        annotation_text=f"  Spot ${spot:.0f}  ",
        annotation_font_color="white", annotation_font_size=11,
        annotation_bgcolor="rgba(0,0,0,0.7)",
        annotation_bordercolor="white", annotation_borderwidth=1,
        annotation_borderpad=3,
        annotation_position="right",
    )

    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"], plot_bgcolor=COLORS["bg_primary"],
        font_color="white", font_size=10,
        margin=dict(l=60, r=10, t=35, b=60),
        title=title,
        xaxis=dict(title="Expiration", tickangle=45),
        yaxis=dict(title="Strike"),
        height=600, dragmode=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})


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

SF_BG   = "#0e1117"
SF_BULL = "#26a69a"
SF_BEAR = "#ef5350"
SF_NEUT = "#90a4ae"
SF_WARN = "#ffa726"
SF_CARD = "#1e2130"


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
                df_spx = rf_fetch_spx_vix(years=3)
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
                df_macro = rf_fetch_fred_macro(years=3)
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
    )

    # ── GEX enhancement ──
    gex_adj = adjust_spread_with_gex(plan, gex_ctx)

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
    # RISK TIER SELECTOR + RANGE GAUGE
    # =========================================================================

    _TIER_COLORS = {
        "aggressive":   "#ff4b4b",
        "moderate":     "#ffa726",
        "conservative": "#66bb6a",
    }

    col_gauge, col_tier_select = st.columns([1, 1])

    with col_gauge:
        st.markdown("**Range Distribution**")
        _render_sf_range_gauge(forecast, plan, spx_close_input)

    with col_tier_select:
        st.markdown("**Risk Tier**")
        tier_labels = [t.label for t in spread_tiers]
        # Default to the conservative tier (last one)
        default_idx = len(tier_labels) - 1
        selected_tier_idx = st.radio(
            "Select risk level",
            range(len(tier_labels)),
            format_func=lambda i: f"{tier_labels[i]}  ({spread_tiers[i].range_pct*100:.1f}%)",
            index=default_idx,
            key=f"sf_risk_tier_{ticker}",
            horizontal=True,
            label_visibility="collapsed",
        )

        selected_tier = spread_tiers[selected_tier_idx]
        tier_color = _TIER_COLORS.get(selected_tier.risk_level, "#888")
        st.markdown(
            f"<span style='color:{tier_color};font-size:18px;font-weight:bold;'>"
            f"{selected_tier.risk_level.upper()}</span>"
            f" &nbsp;—&nbsp; Range: {selected_tier.range_pct*100:.2f}%"
            f" &nbsp;|&nbsp; Calls above `{selected_tier.call_short:,.0f}`"
            f" &nbsp;|&nbsp; Puts below `{selected_tier.put_short:,.0f}`",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =========================================================================
    # STRIKE MAP + SPREAD TABLES (for selected tier)
    # =========================================================================

    col_strikes, col_spacer = st.columns([1, 1])

    with col_strikes:
        st.markdown("**Strike Map with GEX Walls**")
        _render_sf_strike_map_tier(
            selected_tier, plan, spx_close_input, gex_ctx,
            plan.recommended_width, ticker=ticker,
            weekly_em=weekly_em,
        )

    st.markdown("---")

    st.markdown(f"**Spread Parameters — {selected_tier.label}**")

    col_call, col_put = st.columns(2)

    with col_call:
        st.markdown(f"Call Spreads — short above `{selected_tier.call_short:,.0f}`")
        _render_sf_spread_table(selected_tier.call_spreads, plan.recommended_width)

    with col_put:
        st.markdown(f"Put Spreads — short below `{selected_tier.put_short:,.0f}`")
        _render_sf_spread_table(selected_tier.put_spreads, plan.recommended_width)

    # Show credit source note with chain expiration
    all_tier_spreads = selected_tier.call_spreads + selected_tier.put_spreads
    has_market = any(getattr(s, "credit_source", "bsm") == "market" for s in all_tier_spreads)
    has_bsm = any(getattr(s, "credit_source", "bsm") == "bsm" for s in all_tier_spreads)
    exp_note = f" Chain: {chain_exp}" if chain_exp else ""
    if has_market and has_bsm:
        st.caption(f"Credits from Friday chain bid/ask.{exp_note} &nbsp;|&nbsp; * = BSM estimate (strike not in chain).")
    elif has_market:
        st.caption(f"Credits from Friday chain bid/ask (short bid - long ask).{exp_note}")
    else:
        st.caption("Credits are BSM estimates (no Friday chain data available). Verify with broker before trading.")

    # =========================================================================
    # GEX CONTEXT + WARNINGS
    # =========================================================================

    st.markdown("---")

    col_gex, col_warn = st.columns([1, 1])

    with col_gex:
        _render_gex_context_panel(gex_ctx, spot)

    with col_warn:
        st.markdown("**Warnings & GEX Notes**")

        all_warnings = list(plan.warnings) + gex_adj.get("gex_adjustment_notes", [])
        if all_warnings:
            for w in all_warnings:
                st.warning(w)
        else:
            st.success("No warnings for this week.")

        # Event flags
        events = {"FOMC": plan.has_fomc, "CPI": plan.has_cpi, "NFP": plan.has_nfp, "OPEX": plan.has_opex}
        active = [k for k, v in events.items() if v]
        if active:
            st.markdown(f"**Events this week:** {', '.join(active)}")
        else:
            st.caption("No major events this week")

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


def _render_gex_context_panel(gex_ctx: GEXContext, spot: float):
    """Render the GEX context panel showing live gamma levels."""
    st.markdown("**Live GEX Context**")

    gex_flag = regime_to_gex_flag(gex_ctx.gamma_regime)
    regime_color = SF_BULL if gex_flag == 1 else SF_BEAR if gex_flag == -1 else SF_NEUT

    zg_dist = abs(spot - gex_ctx.zero_gamma)
    zg_pct = zg_dist / spot * 100

    st.markdown(
        f"<div style='background:{SF_CARD};padding:12px;border-radius:8px;border:1px solid #333;'>"
        f"<div style='font-size:13px;color:#888;'>Gamma Regime</div>"
        f"<div style='font-size:20px;font-weight:bold;color:{regime_color};'>{gex_ctx.gamma_regime.title()}</div>"
        f"<div style='margin-top:8px;font-size:12px;color:#aaa;'>"
        f"Zero-Gamma: <b>${gex_ctx.zero_gamma:,.0f}</b> ({zg_pct:.2f}% from spot)<br>"
        f"Call Wall: <b>${gex_ctx.call_wall:,.0f}</b><br>"
        f"Put Wall: <b>${gex_ctx.put_wall:,.0f}</b><br>"
        f"Spot: <b>${spot:,.2f}</b>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_sf_range_gauge(forecast: dict, plan: SpreadPlan, spx_ref: float):
    """Bar chart showing point estimate, PI bounds, and effective range."""
    import plotly.graph_objects as go

    categories = ["Lower PI", "Point Est", "Upper PI", "Effective\n(+buffer)"]
    values = [
        forecast["lower_pct"] * 100,
        forecast["point_pct"] * 100,
        forecast["upper_pct"] * 100,
        plan.effective_range_pct * 100,
    ]
    colors = [SF_NEUT, SF_BULL, SF_WARN, SF_BEAR]

    fig = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=colors,
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
    ))

    vix_line = forecast["vix_implied_pct"] * 100
    fig.add_hline(
        y=vix_line, line_dash="dash", line_color=SF_NEUT,
        annotation_text=f"VIX implied: {vix_line:.2f}%",
        annotation_position="top right",
    )

    fig.update_layout(
        plot_bgcolor=SF_BG, paper_bgcolor=SF_BG, font_color="#e0e0e0",
        yaxis_title="Range % (High - Low / Open)",
        showlegend=False,
        margin=dict(t=30, b=10, l=10, r=10),
        height=320, dragmode=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})


def _render_sf_strike_map(plan: SpreadPlan, spx_ref: float, gex_ctx: GEXContext, selected_width: float = 25, ticker: str = "SPX"):
    """Horizontal price map showing reference, effective range, strikes, and GEX walls."""
    import plotly.graph_objects as go

    call_short = plan.call_spreads[0].short_strike if plan.call_spreads else plan.effective_upper_px + 10
    call_long  = plan.call_spreads[0].long_strike  if plan.call_spreads else call_short + 25
    put_short  = plan.put_spreads[0].short_strike  if plan.put_spreads  else plan.effective_lower_px - 10
    put_long   = plan.put_spreads[0].long_strike   if plan.put_spreads  else put_short - 25

    # Use user-selected width spreads
    for s in plan.call_spreads:
        if s.wing_width == selected_width:
            call_short, call_long = s.short_strike, s.long_strike
    for s in plan.put_spreads:
        if s.wing_width == selected_width:
            put_short, put_long = s.short_strike, s.long_strike

    fig = go.Figure()

    # ── Horizontal layout: each level gets its own Y row ──
    # Sort all levels and assign Y positions to avoid overlap
    levels = [
        (put_long,               "Put Long",    SF_BEAR,              "triangle-left",  8),
        (put_short,              "Put Short",   SF_BEAR,              "diamond",        10),
        (plan.effective_lower_px, "Eff Lower",  SF_WARN,              "triangle-up",     9),
        (gex_ctx.put_wall,       "Put Wall",    COLORS["put_wall"],   "square",          9),
        (gex_ctx.zero_gamma,     "Zero-G",      COLORS["zero_gamma"], "x",              10),
        (spx_ref,                f"{ticker} Ref", COLORS["spot"],     "star",           12),
        (gex_ctx.call_wall,      "Call Wall",   COLORS["call_wall"],  "square",          9),
        (plan.effective_upper_px, "Eff Upper",  SF_WARN,              "triangle-up",     9),
        (call_short,             "Call Short",  SF_BEAR,              "diamond",        10),
        (call_long,              "Call Long",   SF_BEAR,              "triangle-right",  8),
    ]

    # Sort by price for clean left-to-right layout
    levels.sort(key=lambda x: x[0])

    # Effective range band (horizontal)
    fig.add_shape(type="rect",
        x0=plan.effective_lower_px, x1=plan.effective_upper_px,
        y0=-0.5, y1=len(levels) - 0.5,
        fillcolor=SF_BULL, opacity=0.10, line_width=0,
    )

    # Call spread zone
    fig.add_shape(type="rect",
        x0=min(call_short, call_long), x1=max(call_short, call_long),
        y0=-0.5, y1=len(levels) - 0.5,
        fillcolor=SF_BEAR, opacity=0.15, line_width=0,
    )

    # Put spread zone
    fig.add_shape(type="rect",
        x0=min(put_long, put_short), x1=max(put_long, put_short),
        y0=-0.5, y1=len(levels) - 0.5,
        fillcolor=SF_BEAR, opacity=0.15, line_width=0,
    )

    # Plot each level as a scatter point on its own row
    for i, (price, label, color, symbol, size) in enumerate(levels):
        fig.add_trace(go.Scatter(
            x=[price], y=[i],
            mode="markers+text",
            marker=dict(color=color, size=size, symbol=symbol, line=dict(width=1, color="#fff")),
            text=[f"{label}  {price:,.0f}"],
            textposition="middle right" if price <= spx_ref else "middle left",
            textfont=dict(size=11, color=color),
            showlegend=False,
            hovertemplate=f"{label}: {price:,.0f}<extra></extra>",
        ))

    # Reference price vertical line
    fig.add_vline(
        x=spx_ref, line_dash="solid", line_color=COLORS["spot"],
        line_width=2, opacity=0.4,
    )

    all_prices = [l[0] for l in levels]
    margin_px = (max(all_prices) - min(all_prices)) * 0.15

    fig.update_layout(
        plot_bgcolor=SF_BG, paper_bgcolor=SF_BG, font_color="#e0e0e0",
        xaxis_title=f"{ticker} Price Level",
        xaxis_range=[min(all_prices) - margin_px, max(all_prices) + margin_px],
        yaxis_visible=False,
        showlegend=False,
        margin=dict(t=10, b=30, l=10, r=10),
        height=380, dragmode=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})


def _render_sf_strike_map_tier(
    tier: SpreadTier, plan: SpreadPlan, spx_ref: float,
    gex_ctx: GEXContext, selected_width: float = 25, ticker: str = "SPX",
    weekly_em: dict = None,
):
    """Strike map that updates based on the selected risk tier."""
    import plotly.graph_objects as go

    _TIER_COLORS = {
        "aggressive":   "#ff4b4b",
        "moderate":     "#ffa726",
        "conservative": "#66bb6a",
    }
    tier_color = _TIER_COLORS.get(tier.risk_level, "#888")

    # Get strikes from the selected tier
    call_short = tier.call_short
    put_short  = tier.put_short

    # Default long strikes from first spread
    call_long = tier.call_spreads[0].long_strike if tier.call_spreads else call_short + 25
    put_long  = tier.put_spreads[0].long_strike  if tier.put_spreads  else put_short - 25

    # Use recommended wing width if available
    for s in tier.call_spreads:
        if s.wing_width == selected_width:
            call_long = s.long_strike
    for s in tier.put_spreads:
        if s.wing_width == selected_width:
            put_long = s.long_strike

    # Weekly expected move from Friday straddle (already computed by GEX engine)
    em_upper = (weekly_em or {}).get("upper_level", 0)
    em_lower = (weekly_em or {}).get("lower_level", 0)
    has_em = em_upper > 0 and em_lower > 0

    fig = go.Figure()

    # Build level markers
    levels = [
        (put_long,   "Put Long",   SF_BEAR,   "triangle-left",  8),
        (put_short,  "Put Short",  tier_color, "diamond",       10),
        (gex_ctx.put_wall,   "Put Wall",  COLORS["put_wall"],  "square",  9),
        (gex_ctx.zero_gamma, "Zero-G",    COLORS["zero_gamma"], "x",     10),
        (spx_ref,    f"{ticker} Ref", COLORS["spot"], "star",           12),
        (gex_ctx.call_wall,  "Call Wall", COLORS["call_wall"],  "square",  9),
        (call_short, "Call Short", tier_color, "diamond",       10),
        (call_long,  "Call Long",  SF_BEAR,   "triangle-right",  8),
    ]

    # Add EM levels if weekly straddle data is available
    em_color = "#29b6f6"  # light blue
    if has_em:
        levels.append((em_upper, "EM Upper", em_color, "line-ew", 9))
        levels.append((em_lower, "EM Lower", em_color, "line-ew", 9))

    levels.sort(key=lambda x: x[0])

    # Weekly expected move band
    if has_em:
        fig.add_shape(type="rect",
            x0=em_lower, x1=em_upper,
            y0=-0.5, y1=len(levels) - 0.5,
            fillcolor=em_color, opacity=0.06, line_width=1,
            line_color=em_color, line_dash="dash",
        )

    # Tier range band
    half = tier.range_pct / 2
    tier_lower = spx_ref * (1 - half)
    tier_upper = spx_ref * (1 + half)
    fig.add_shape(type="rect",
        x0=tier_lower, x1=tier_upper,
        y0=-0.5, y1=len(levels) - 0.5,
        fillcolor=tier_color, opacity=0.08, line_width=1,
        line_color=tier_color, line_dash="dot",
    )

    # Call spread zone
    fig.add_shape(type="rect",
        x0=min(call_short, call_long), x1=max(call_short, call_long),
        y0=-0.5, y1=len(levels) - 0.5,
        fillcolor=SF_BEAR, opacity=0.15, line_width=0,
    )

    # Put spread zone
    fig.add_shape(type="rect",
        x0=min(put_long, put_short), x1=max(put_long, put_short),
        y0=-0.5, y1=len(levels) - 0.5,
        fillcolor=SF_BEAR, opacity=0.15, line_width=0,
    )

    for i, (price, label, color, symbol, size) in enumerate(levels):
        fig.add_trace(go.Scatter(
            x=[price], y=[i],
            mode="markers+text",
            marker=dict(color=color, size=size, symbol=symbol, line=dict(width=1, color="#fff")),
            text=[f"{label}  {price:,.0f}"],
            textposition="middle right" if price <= spx_ref else "middle left",
            textfont=dict(size=11, color=color),
            showlegend=False,
            hovertemplate=f"{label}: {price:,.0f}<extra></extra>",
        ))

    fig.add_vline(x=spx_ref, line_dash="solid", line_color=COLORS["spot"], line_width=2, opacity=0.4)

    all_prices = [l[0] for l in levels]
    margin_px = (max(all_prices) - min(all_prices)) * 0.15

    fig.update_layout(
        plot_bgcolor=SF_BG, paper_bgcolor=SF_BG, font_color="#e0e0e0",
        xaxis_title=f"{ticker} Price Level",
        xaxis_range=[min(all_prices) - margin_px, max(all_prices) + margin_px],
        yaxis_visible=False, showlegend=False,
        margin=dict(t=10, b=30, l=10, r=10),
        height=380, dragmode=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})


def _render_sf_spread_table(spreads, recommended_width: int):
    """Render spread parameters as a styled dataframe."""
    if not spreads:
        st.info("No spreads available.")
        return

    rows = []
    for s in spreads:
        width_label = f"{int(s.wing_width)}pt" if s.wing_width == int(s.wing_width) else f"{s.wing_width}pt"
        if getattr(s, "below_min_width", False):
            width_label += "*"
        rows.append({
            "Width": width_label,
            "Short": f"{s.short_strike:,.0f}",
            "Long": f"{s.long_strike:,.0f}",
            "Est Credit": f"{s.estimated_credit:.2f}" + (" *" if getattr(s, "credit_source", "bsm") == "bsm" else ""),
            "Max Loss $": f"${s.max_loss:,.0f}",
            "Breakeven": f"{s.breakeven:,.0f}",
            "Ratio": f"{s.credit_ratio:.1%}",
            "OK": "Y" if s.meets_min_credit else "N",
            "Rec": s.wing_width == recommended_width,
        })

    df = pd.DataFrame(rows)

    # BUG FIX: apply styling before dropping the helper column
    def highlight_rec(row):
        if row["Rec"]:
            return ["background-color: #1a3a2a"] * len(row)
        return [""] * len(row)

    display_df = df.drop(columns=["Rec"])
    # Apply row highlighting using the original df's Rec column
    styles = []
    for _, row in df.iterrows():
        if row["Rec"]:
            styles.append(["background-color: #1a3a2a"] * len(display_df.columns))
        else:
            styles.append([""] * len(display_df.columns))

    styled = display_df.style.apply(lambda x: styles[x.name], axis=1)
    styled = styled.map(
        lambda v: f"color: {SF_BULL}" if v == "Y" else f"color: {SF_BEAR}" if v == "N" else "",
        subset=["OK"]
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Gamma Exposure Dashboard")
    st.caption(f"GEX Calculator {TOOL_VERSION} — Implied spot | Zero gamma sweep | Expected move | Hybrid IV")

    tradier_token, fred_key = get_credentials()

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
                f'<div class="em-item"><div class="lbl">Overnight Move (ES)</div>'
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
            on_label = "Overnight (ES)"
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

    # ── Charts ──
    tab_gex, tab_profile, tab_multi, tab_history, tab_em_track, tab_iv_surface, tab_spread_finder = st.tabs(
        ["📊 Strike GEX", "📈 GEX Profile", "⏱️ Multi-TF", "📅 History", "🎯 EM Tracker", "🌊 IV Surface", "🎯 Spread Finder"]
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
        weekly_sub = f"Frozen Mon open | Exp: {weekly_exp}" if weekly_exp else "No weekly expiration found"
        weekly_em_for_render = {"expected_move": weekly_em_snap} if weekly_em_snap else {"expected_move": weekly_em_live or {}}
        _render_em_tracker(weekly_em_for_render, spot, prev_close, market_ctx, label="Weekly", subtitle=weekly_sub)

        st.divider()

        # Monthly
        monthly_sub = f"Frozen 1st trading day | Exp: {monthly_exp}" if monthly_exp else "No monthly expiration found"
        monthly_em_for_render = {"expected_move": monthly_em_snap} if monthly_em_snap else {"expected_move": monthly_em_live or {}}
        _render_em_tracker(monthly_em_for_render, spot, prev_close, market_ctx, label="Monthly", subtitle=monthly_sub)

    # ── C4: IV surface visualization ──
    with tab_iv_surface:
        _render_iv_surface(data.hm_iv, data.hm_gex, spot)

    # ── C7: Spread Finder — Weekly credit spread placement ──
    with tab_spread_finder:
        _sf_weekly_em = weekly_em_snap or weekly_em_live or {}
        _render_spread_finder_tab(spot, levels, regime, data, ticker=ticker, weekly_em=_sf_weekly_em)

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
            render_expected_move_panel(em_analysis)
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
