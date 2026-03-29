"""
SPX Gamma Exposure (GEX) Dashboard — Streamlit Web App

Run locally:   streamlit run streamlit_app.py
Deploy:        Push to GitHub → connect at share.streamlit.io
"""
from __future__ import annotations

import os
import json
import calendar as cal_mod
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Phase1 engine imports ──
from phase1.config import HEATMAP_EXPS, build_config_snapshot
from phase1.market_clock import now_ny, get_calendar_snapshot
from phase1.data_client import TradierDataClient
from phase1.rates import fetch_risk_free_rate
from phase1.parity import get_reference_spot_details
import phase1.gex_engine as gex_engine
from phase1.confidence import build_run_confidence
from phase1.staleness import build_staleness_info
from phase1.wall_credibility import build_wall_credibility
from phase1.scenarios import run_scenario_engine
from phase1.expected_move import build_expected_move_analysis
from phase1.futures_data import fetch_es_from_yahoo, build_futures_context

TOOL_VERSION = "v5-web"

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SPX GEX Dashboard",
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
@st.cache_data(ttl=90, show_spinner=False)
def fetch_all_data(tradier_token: str, fred_key: str, selected_exps: tuple, _run_id: str):
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

    avail = client.get_expirations("SPX")
    today_str = run_now.strftime("%Y-%m-%d")
    nearest_exp = next((e for e in avail if e >= today_str), avail[0])

    spot_info = get_reference_spot_details(
        ticker="SPX",
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
        gex_engine.calculate_all(client, "SPX", target_exps, spot, heatmap_exps, r=rfr, now=run_now)
    )

    levels = gex_engine.find_key_levels(gex_df, spot, all_options=all_options, r=rfr)
    profile_df = gex_engine.compute_gex_profile_curve(all_options, spot, r=rfr)
    sensitivity_df = gex_engine.compute_zero_gamma_sensitivity(all_options, spot, r=rfr)
    scenarios_df = run_scenario_engine(all_options, base_spot=spot, base_r=rfr)
    staleness_info = build_staleness_info(calendar_snapshot, spot_info, stats)
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
    spx_quote = None
    spy_quote = None
    try:
        spx_quote = client.get_full_quote("SPX")
    except Exception:
        pass
    try:
        spy_quote = client.get_full_quote("SPY")
    except Exception:
        pass

    prev_close = spx_quote["prevclose"] if spx_quote else 0.0
    dte0_exp = target_exps[0] if target_exps else nearest_exp
    dte0_entry = client.get_chain_cached("SPX", dte0_exp)
    dte0_calls = dte0_entry.get("calls", []) if dte0_entry.get("status") == "ok" else []
    dte0_puts = dte0_entry.get("puts", []) if dte0_entry.get("status") == "ok" else []

    # Try Yahoo ES futures (cached with the rest)
    yahoo_es = None
    try:
        yahoo_es = fetch_es_from_yahoo()
    except Exception:
        pass

    return {
        "spot": spot,
        "spot_source": spot_source,
        "spot_info": spot_info,
        "rfr": rfr,
        "rfr_info": rfr_info,
        "avail": avail,
        "target_exps": target_exps,
        "gex_df": gex_df,
        "hm_gex": hm_gex,
        "hm_iv": hm_iv,
        "stats": stats,
        "all_options": all_options,
        "levels": levels,
        "profile_df": profile_df,
        "sensitivity_df": sensitivity_df,
        "scenarios_df": scenarios_df,
        "staleness_info": staleness_info,
        "confidence_info": confidence_info,
        "wall_cred": wall_cred,
        "regime_info": regime_info,
        "calendar_snapshot": calendar_snapshot,
        "run_time": run_now.strftime("%I:%M:%S %p ET"),
        # Raw EM inputs for main() to assemble with futures data
        "prev_close": prev_close,
        "spy_quote": spy_quote,
        "dte0_calls": dte0_calls,
        "dte0_puts": dte0_puts,
        "market_open": bool(spot_info.get("market_open")),
        "yahoo_es": yahoo_es,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders
# ─────────────────────────────────────────────────────────────────────────────
def build_gex_bar_chart(gex_df, levels, spot, em_analysis):
    df = gex_df.copy().sort_values("strike").reset_index(drop=True)
    strikes = df["strike"].values
    net_gex = df["net_gex"].values
    colors = ["#00c853" if g >= 0 else "#ff1744" for g in net_gex]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=strikes, x=net_gex, orientation="h",
        marker_color=colors, marker_opacity=0.85,
        hovertemplate="Strike: $%{y:.0f}<br>Net GEX: %{x:,.0f}<extra></extra>",
    ))

    # Level lines
    for val, color, dash, name in [
        (spot, "#ffd600", "dash", "Spot"),
        (levels["zero_gamma"], "#00e5ff", "dot", "Zero Γ"),
        (levels["call_wall"], "#69f0ae", "dashdot", "Call Wall"),
        (levels["put_wall"], "#ff8a80", "dashdot", "Put Wall"),
    ]:
        fig.add_hline(y=val, line_color=color, line_dash=dash, line_width=1.5,
                       annotation_text=f"{name} ${val:.0f}",
                       annotation_font_color=color, annotation_font_size=9,
                       annotation_position="top left")

    # EM levels
    em = em_analysis.get("expected_move", {})
    if em.get("upper_level"):
        for val, label in [(em["upper_level"], "EM+"), (em["lower_level"], "EM−")]:
            fig.add_hline(y=val, line_color="#b388ff", line_dash="dot", line_width=1.2,
                           annotation_text=f"{label} ${val:.0f}",
                           annotation_font_color="#b388ff", annotation_font_size=8,
                           annotation_position="bottom right")

    fig.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font_color="white", font_size=10,
        margin=dict(l=80, r=10, t=35, b=35),
        title="Strike-by-Strike Net GEX Proxy",
        xaxis=dict(title="Net GEX proxy", gridcolor="#333", zerolinecolor="#555"),
        yaxis=dict(title="Strike", gridcolor="#222", tickfont_size=8),
        showlegend=False, height=700,
    )
    return fig


def build_profile_chart(profile_df, levels, spot, regime_info, em_analysis):
    if profile_df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=profile_df["price"], y=profile_df["total_gex"],
        mode="lines", line=dict(color="#9c88ff", width=2),
        hovertemplate="Price: $%{x:.2f}<br>GEX: %{y:,.0f}<extra></extra>",
    ))

    # Level lines
    fig.add_vline(x=spot, line_color="#ffd600", line_dash="dash", line_width=1.5,
                   annotation_text="Spot", annotation_font_color="#ffd600")
    fig.add_vline(x=levels["zero_gamma"], line_color="#00e5ff", line_dash="dot", line_width=1.5,
                   annotation_text="Zero Γ", annotation_font_color="#00e5ff")
    fig.add_hline(y=0, line_color="#555", line_width=1)

    # EM levels
    em = em_analysis.get("expected_move", {})
    if em.get("upper_level"):
        for val, label in [(em["upper_level"], "EM+"), (em["lower_level"], "EM−")]:
            fig.add_vline(x=val, line_color="#b388ff", line_dash="dot", line_width=1.2,
                           annotation_text=label, annotation_font_color="#b388ff",
                           annotation_font_size=9)

    # Regime badge
    fig.add_annotation(
        xref="paper", yref="paper", x=0.01, y=0.98,
        text=f"<b>{regime_info['regime']}</b><br>{regime_info['distance_text']}",
        font=dict(color=regime_info["color"], size=11),
        bgcolor="rgba(20,20,30,0.7)", bordercolor=regime_info["color"],
        borderwidth=1, borderpad=5, showarrow=False, align="left",
    )

    fig.update_layout(
        paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e",
        font_color="white", font_size=10,
        margin=dict(l=60, r=10, t=35, b=40),
        title="GEX Profile Curve",
        xaxis=dict(title="Underlying Price", gridcolor="#333"),
        yaxis=dict(title="Total GEX proxy", gridcolor="#222", zerolinecolor="#555"),
        showlegend=False, height=500,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar rendering
# ─────────────────────────────────────────────────────────────────────────────
def render_expected_move_panel(em_analysis):
    em = em_analysis.get("expected_move", {})
    on = em_analysis.get("overnight_move", {})
    cl = em_analysis.get("classification", {})
    spy = em_analysis.get("spy_proxy")
    fc = em_analysis.get("futures_context")
    overnite_range = em_analysis.get("overnight_range")
    lctx = em_analysis.get("level_context")
    market_ctx = em_analysis.get("market_context", "live")

    if em.get("expected_move_pts") is None:
        st.caption("Expected move data not available.")
        return

    st.markdown("#### ⚡ Expected Move — 0DTE")

    # Straddle & range
    straddle = em.get("straddle", {})
    straddle_html = (
        '<div class="level-grid" style="grid-template-columns: 1fr 1fr;">'
        f'<div class="level-card"><div class="lbl">ATM Straddle</div><div class="val">{em["expected_move_pts"]:.1f} pts</div><div class="lbl">{em["expected_move_pct"]:.2f}%</div></div>'
        f'<div class="level-card"><div class="lbl">Strike</div><div class="val">${straddle.get("strike", "?")}</div></div>'
        '</div>'
    )
    st.markdown(straddle_html, unsafe_allow_html=True)

    st.markdown(
        f"**Expected Range:** "
        f":red[${em['lower_level']:.0f}] — :green[${em['upper_level']:.0f}]"
    )

    # Move display — label depends on market context
    move_source = cl.get("move_source", "spx")

    if market_ctx == "live":
        # During market hours: show "Today's Move" from live SPX
        on_pts = on.get("overnight_move_pts")
        if on_pts is not None:
            arrow = "🟢 ▲" if on_pts >= 0 else "🔴 ▼"
            st.markdown(
                f"**Today's Move:** {arrow} **{on_pts:+.1f} pts** ({on['overnight_move_pct']:+.2f}%)"
            )
    elif "es_futures" in move_source and fc:
        arrow = "🟢 ▲" if fc["overnight_move_pts"] >= 0 else "🔴 ▼"
        st.markdown(
            f"**Overnight (ES):** {arrow} **{fc['overnight_move_pts']:+.1f} pts** ({fc['overnight_move_pct']:+.2f}%)"
        )
        src_label = "manual" if fc["source"] == "manual" else "Yahoo ~10m delayed"
        st.caption(f"ES: \\${fc['es_last']:.2f} vs SPX prevclose \\${fc['spx_prevclose']:.2f} ({src_label})")
    else:
        on_pts = on.get("overnight_move_pts")
        if on_pts is not None:
            label = "Session Move" if market_ctx == "afterhours" else "Overnight Move"
            arrow = "🟢 ▲" if on_pts >= 0 else "🔴 ▼"
            st.markdown(
                f"**{label}:** {arrow} **{on_pts:+.1f} pts** ({on['overnight_move_pct']:+.2f}%)"
            )

    # Overnight range (from ES high/low)
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

    # SPY proxy (show if available but ES not used)
    if spy and "es_futures" not in move_source:
        st.caption(
            f"SPY Pre-mkt: ${spy['spy_price']:.2f} ({spy['spy_move_pct']:+.2f}%) "
            f"→ ~{spy['implied_spx_move_pts']:+.1f} SPX pts"
        )

    # Move ratio bar
    ratio = cl.get("move_ratio")
    if ratio is not None:
        pct = min(ratio * 100, 100)
        label = cl.get("move_ratio_label", "")
        st.markdown(f"**Vol Budget Used:** {pct:.0f}% ({label})")
        st.progress(min(ratio, 1.0))

    # Session classification
    if cl.get("classification"):
        bias = cl.get("bias", "")
        if bias in ("range-bound", "mean-revert"):
            cls_icon = "🟢"
        elif bias in ("directional", "continued-trend"):
            cls_icon = "🔴"
        else:
            cls_icon = "🟡"

        st.markdown(f"### {cls_icon} {cl['classification']}")
        if cl.get("description"):
            st.caption(cl["description"])
        if cl.get("favored_strategies"):
            st.markdown(f"**Favored:** {', '.join(cl['favored_strategies'])}")

    # Zero gamma context
    if lctx and lctx.get("zero_gamma_within_em") is not None:
        inside = "✅ inside" if lctx["zero_gamma_within_em"] else "⚠️ outside"
        st.markdown(
            f"**Zero Γ:** {inside} expected range "
            f"({lctx['zero_gamma_distance_to_spot']:+.1f} pts from spot)"
        )

    st.divider()


def render_key_levels(levels, spot, regime_info, confidence_info, staleness_info):
    conf_score = confidence_info.get("score", 0)
    conf_label = confidence_info.get("label", "?")
    conf_color = "#00c853" if conf_label == "High" else "#ffd600" if conf_label == "Moderate" else "#ff5252"

    html = (
        '<div class="level-grid">'
        f'<div class="level-card"><div class="lbl">Spot</div><div class="val" style="color:#ffd600;">${spot:.2f}</div></div>'
        f'<div class="level-card"><div class="lbl">Call Wall</div><div class="val" style="color:#69f0ae;">${levels["call_wall"]:.0f}</div></div>'
        f'<div class="level-card"><div class="lbl">Put Wall</div><div class="val" style="color:#ff8a80;">${levels["put_wall"]:.0f}</div></div>'
        f'<div class="level-card"><div class="lbl">Zero Gamma</div><div class="val" style="color:#00e5ff;">${levels["zero_gamma"]:.2f}</div></div>'
        f'<div class="level-card"><div class="lbl">Regime</div><div class="val" style="color:{regime_info["color"]};font-size:12px;">{regime_info["regime"]}</div></div>'
        f'<div class="level-card"><div class="lbl">Confidence</div><div class="val" style="color:{conf_color};">{conf_score:.0f}</div><div class="lbl" style="color:{conf_color};">{conf_label}</div></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_scenarios_table(scenarios_df):
    if scenarios_df is None or scenarios_df.empty:
        return
    st.markdown("#### Scenario Analysis")

    rows_html = ""
    for _, row in scenarios_df.iterrows():
        regime = row.get("gamma_regime", "")
        r_color = "#00c853" if "Pos" in regime else "#ff5252" if "Neg" in regime else "#00e5ff"
        rows_html += (
            f'<tr><td style="color:#cfd3ff;font-weight:bold;">{row["scenario"]}</td>'
            f'<td>${row["spot"]:.0f}</td>'
            f'<td style="color:#69f0ae;">${row["call_wall"]:.0f}</td>'
            f'<td style="color:#ff8a80;">${row["put_wall"]:.0f}</td>'
            f'<td style="color:#00e5ff;">${row["zero_gamma"]:.0f}</td>'
            f'<td style="color:{r_color};font-size:9px;">{regime}</td></tr>'
        )

    table_html = (
        '<table style="width:100%;border-collapse:collapse;font-size:11px;margin-bottom:10px;">'
        '<thead><tr style="background:#1a1a3e;color:#888;font-size:10px;">'
        '<th style="padding:4px;text-align:left;">Scenario</th>'
        '<th style="padding:4px;">Spot</th>'
        '<th style="padding:4px;">CW</th>'
        '<th style="padding:4px;">PW</th>'
        '<th style="padding:4px;">ZG</th>'
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


def render_data_quality(stats, staleness_info):
    st.markdown("#### Data Quality")

    fresh = staleness_info.get("freshness_score", 0)
    fresh_lbl = staleness_info.get("freshness_label", "?")
    fresh_color = "#00c853" if fresh_lbl == "High" else "#ffd600" if fresh_lbl == "Moderate" else "#ff5252"

    coverage = stats.get("coverage_ratio", 0)
    cov_color = "#00c853" if coverage >= 0.95 else "#ffd600" if coverage >= 0.85 else "#ff5252"

    html = (
        '<div class="level-grid">'
        f'<div class="level-card"><div class="lbl">Used Options</div><div class="val">{stats.get("used_option_count", 0):,}</div></div>'
        f'<div class="level-card"><div class="lbl">Coverage</div><div class="val" style="color:{cov_color};">{coverage*100:.1f}%</div></div>'
        f'<div class="level-card"><div class="lbl">Freshness</div><div class="val" style="color:{fresh_color};">{fresh:.0f}</div><div class="lbl" style="color:{fresh_color};">{fresh_lbl}</div></div>'
        f'<div class="level-card"><div class="lbl">Direct IV</div><div class="val">{stats.get("direct_iv_count", 0):,}</div></div>'
        f'<div class="level-card"><div class="lbl">Synthetic IV</div><div class="val">{stats.get("synthetic_iv_count", 0):,}</div></div>'
        f'<div class="level-card"><div class="lbl">Skipped</div><div class="val">{stats.get("skipped_count", 0):,}</div></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 SPX Gamma Exposure")
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

        # Expiration picker
        with st.spinner("Loading expirations..."):
            try:
                temp_client = TradierDataClient(token=tradier_token)
                avail = temp_client.get_expirations("SPX")
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
            st.cache_data.clear()

        # ── ES Futures Override (pre-market only) ──
        # We declare these with defaults; they'll be ignored during market hours.
        es_manual_last = 0.0
        es_manual_high = 0.0
        es_manual_low = 0.0

    # ── Refresh interval ──
    refresh_seconds = {"Off": 0, "Every 5 min": 300, "Every 30 min": 1800}.get(refresh_option, 0)

    # ── Run ID for cache busting ──
    run_id = f"{datetime.utcnow().isoformat()}" if refresh_seconds == 0 else "auto"

    # ── Fetch data ──
    with st.spinner("Crunching GEX..."):
        try:
            data = fetch_all_data(tradier_token, fred_key or "", tuple(selected), run_id)
        except Exception as e:
            st.error(f"Engine error: {e}")
            st.stop()

    if data["gex_df"].empty:
        st.warning("No GEX data returned. The selected expirations may have no usable contracts.")
        st.stop()

    # ── Build futures context: manual overrides > Yahoo auto ──
    spot = data["spot"]
    levels = data["levels"]
    regime = data["regime_info"]
    prev_close = data["prev_close"]
    yahoo_es = data.get("yahoo_es")
    is_market_open = data["market_open"]

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
    em = build_expected_move_analysis(
        spot=spot,
        prev_close=prev_close,
        zero_gamma=levels["zero_gamma"],
        gamma_regime=regime["regime"],
        calls_0dte=data["dte0_calls"],
        puts_0dte=data["dte0_puts"],
        spy_quote=data["spy_quote"],
        market_open=data["market_open"],
        futures_context=futures_ctx,
    )

    # ── Straddle snapshot: freeze EM at first market-hours refresh ──
    today_str_snap = now_ny().strftime("%Y-%m-%d")

    # Clear stale snapshot from a previous day
    if st.session_state.get("em_snapshot_date") != today_str_snap:
        st.session_state.pop("em_snapshot", None)
        st.session_state.pop("em_snapshot_date", None)
        st.session_state.pop("em_snapshot_time", None)

    if is_market_open:
        em_live = em.get("expected_move", {})
        if "em_snapshot" not in st.session_state and em_live.get("expected_move_pts"):
            # First market-hours refresh — capture the straddle
            st.session_state["em_snapshot"] = {
                "expected_move_pts": em_live["expected_move_pts"],
                "expected_move_pct": em_live["expected_move_pct"],
                "upper_level": em_live["upper_level"],
                "lower_level": em_live["lower_level"],
                "straddle": em_live.get("straddle"),
            }
            st.session_state["em_snapshot_date"] = today_str_snap
            st.session_state["em_snapshot_time"] = now_ny().strftime("%I:%M:%S %p ET")

        # Replace the live EM with the frozen snapshot for display
        if "em_snapshot" in st.session_state:
            snap = st.session_state["em_snapshot"]
            em["expected_move"] = {
                **em.get("expected_move", {}),
                "expected_move_pts": snap["expected_move_pts"],
                "expected_move_pct": snap["expected_move_pct"],
                "upper_level": snap["upper_level"],
                "lower_level": snap["lower_level"],
                "straddle": snap["straddle"],
            }
            # Recompute vol budget with frozen EM vs live move
            on_pts = em.get("overnight_move", {}).get("overnight_move_pts")
            if on_pts is not None and snap["expected_move_pts"] > 0:
                from phase1.expected_move import classify_session
                em["classification"] = classify_session(
                    expected_move_pts=snap["expected_move_pts"],
                    overnight_move_pts=on_pts,
                    gamma_regime=regime["regime"],
                )
                em["classification"]["move_source"] = "spx"
            # Update level context with frozen EM
            if snap["upper_level"] is not None:
                em["level_context"] = {
                    "em_upper": snap["upper_level"],
                    "em_lower": snap["lower_level"],
                    "zero_gamma": round(levels["zero_gamma"], 2),
                    "zero_gamma_within_em": (
                        snap["lower_level"] <= levels["zero_gamma"] <= snap["upper_level"]
                    ),
                    "zero_gamma_distance_to_spot": round(spot - levels["zero_gamma"], 2),
                }

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
    st.markdown(
        f"<div style='text-align:center;padding:6px;'>"
        f"<span style='font-size:22px;font-weight:bold;color:#ffd600;'>SPX ${spot:.2f}</span>"
        f"&nbsp;&nbsp;&nbsp;"
        f"<span style='font-size:18px;color:{regime_color};font-weight:bold;'>{regime['regime']}</span>"
        f"&nbsp;&nbsp;"
        f"<span style='color:#aaa;font-size:13px;'>({regime['distance_text']})</span>"
        f"&nbsp;&nbsp;&nbsp;"
        f"<span style='color:#888;font-size:12px;'>Updated {data['run_time']}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Market context banner ──
    market_ctx = em.get("market_context", "live")
    context_note = em.get("context_note")
    if market_ctx == "premarket":
        st.info("🌅 **Pre-market** — GEX levels and gamma regime are current. "
                "Expected move and session classification will be available after the 9:30 AM open.")
    elif market_ctx == "afterhours":
        st.warning(f"🌙 **After hours** — {context_note}")

    # ── Expected Move panel (top of page) ──
    em_data = em.get("expected_move", {})

    if market_ctx == "premarket":
        # ── PRE-MARKET: Only show ES overnight move + range, suppress stale straddle/classification ──
        fc = em.get("futures_context")
        overnite_range = em.get("overnight_range")

        if fc:
            on_color = "#00c853" if fc["overnight_move_pts"] >= 0 else "#ff5252"
            on_arrow = "▲" if fc["overnight_move_pts"] > 0 else "▼" if fc["overnight_move_pts"] < 0 else "–"

            premarket_html = (
                '<div class="em-bar">'
                f'<div class="em-item"><div class="lbl">Overnight Move (ES)</div>'
                f'<div class="val" style="color:{on_color};">{on_arrow} {fc["overnight_move_pts"]:+.1f} pts</div>'
                f'<div class="lbl" style="color:{on_color};">{fc["overnight_move_pct"]:+.2f}%</div></div>'
            )

            if overnite_range and overnite_range.get("es_high"):
                hi_move = overnite_range["high_move_from_close"]
                lo_move = overnite_range["low_move_from_close"]
                premarket_html += (
                    f'<div class="em-item"><div class="lbl">O/N High</div>'
                    f'<div class="val" style="font-size:16px;color:#69f0ae;">${overnite_range["es_high"]:.0f}</div>'
                    f'<div class="lbl" style="color:#69f0ae;">{hi_move:+.1f} pts</div></div>'
                    f'<div class="em-item"><div class="lbl">O/N Low</div>'
                    f'<div class="val" style="font-size:16px;color:#ff8a80;">${overnite_range["es_low"]:.0f}</div>'
                    f'<div class="lbl" style="color:#ff8a80;">{lo_move:+.1f} pts</div></div>'
                    f'<div class="em-item"><div class="lbl">O/N Range</div>'
                    f'<div class="val" style="font-size:16px;">{overnite_range["range_pts"]:.0f} pts</div></div>'
                )

            premarket_html += '</div>'
            st.markdown(premarket_html, unsafe_allow_html=True)

    elif em_data.get("expected_move_pts"):
        # ── MARKET HOURS / AFTER HOURS: Full EM framework ──
        cl = em.get("classification", {})
        on = em.get("overnight_move", {})
        spy = em.get("spy_proxy")
        fc = em.get("futures_context")
        overnite_range = em.get("overnight_range")
        move_source = cl.get("move_source", "spx")

        # Pick the right move numbers and label
        if market_ctx == "live":
            # During market hours: SPX is live, show "Today's Move"
            display_pts = on.get("overnight_move_pts", 0)
            display_pct = on.get("overnight_move_pct", 0)
            on_label = "Today's Move"
        elif move_source == "spx_realized":
            display_pts = on.get("overnight_move_pts", 0)
            display_pct = on.get("overnight_move_pct", 0)
            on_label = "Session Move"
        elif "es_futures" in move_source and fc:
            display_pts = fc["overnight_move_pts"]
            display_pct = fc["overnight_move_pct"]
            on_label = "Overnight (ES)"
        elif move_source == "spy_proxy" and spy:
            display_pts = spy["implied_spx_move_pts"]
            display_pct = spy["spy_move_pct"]
            on_label = "Overnight (SPY)"
        else:
            display_pts = on.get("overnight_move_pts", 0)
            display_pct = on.get("overnight_move_pct", 0)
            on_label = "Overnight"

        ratio = cl.get("move_ratio")

        on_color = "#00c853" if (display_pts or 0) >= 0 else "#ff5252"
        on_arrow = "▲" if (display_pts or 0) > 0 else "▼" if (display_pts or 0) < 0 else "–"
        ratio_pct = f"{ratio*100:.0f}%" if ratio is not None else "–"

        if ratio is not None:
            ratio_color = "#00c853" if ratio < 0.40 else "#ffd600" if ratio < 0.70 else "#ff5252"
        else:
            ratio_color = "#aaa"

        cls_name = cl.get("classification", "–")
        cls_bias = cl.get("bias", "")
        if cls_bias in ("range-bound", "mean-revert"):
            cls_color = "#00c853"
        elif cls_bias in ("directional", "continued-trend"):
            cls_color = "#ff5252"
        else:
            cls_color = "#ffd600"

        em_bar_html = (
            '<div class="em-bar">'
            f'<div class="em-item"><div class="lbl">Expected Move</div><div class="val">&plusmn;{em_data["expected_move_pts"]:.0f} pts</div></div>'
            f'<div class="em-item"><div class="lbl">EM Range</div><div class="val">${em_data["lower_level"]:.0f} &ndash; ${em_data["upper_level"]:.0f}</div></div>'
            f'<div class="em-item"><div class="lbl">{on_label}</div><div class="val" style="color:{on_color};">{on_arrow} {display_pts:+.1f} pts</div><div class="lbl" style="color:{on_color};">{display_pct:+.2f}%</div></div>'
            f'<div class="em-item"><div class="lbl">Vol Budget Used</div><div class="val" style="color:{ratio_color};">{ratio_pct}</div></div>'
            f'<div class="em-item"><div class="lbl">Session Type</div><div class="val" style="color:{cls_color};">{cls_name}</div></div>'
            '</div>'
        )
        st.markdown(em_bar_html, unsafe_allow_html=True)

        # Show when the straddle was captured
        snap_time = st.session_state.get("em_snapshot_time")
        if market_ctx == "live" and snap_time:
            st.caption(f"📌 Expected move captured at {snap_time} — frozen for the session. Today's move and vol budget update live.")

        # Overnight range bar (when ES high/low available, after hours only)
        if market_ctx == "afterhours" and overnite_range and overnite_range.get("es_high"):
            hi_move = overnite_range["high_move_from_close"]
            lo_move = overnite_range["low_move_from_close"]
            rng = overnite_range["range_pts"]
            max_vs_em = overnite_range.get("max_move_vs_em")
            max_vs_em_str = f"{max_vs_em*100:.0f}% of EM" if max_vs_em else ""

            range_html = (
                '<div class="em-bar" style="padding:2px 0 4px 0;">'
                f'<div class="em-item"><div class="lbl">O/N High</div><div class="val" style="font-size:16px;color:#69f0ae;">${overnite_range["es_high"]:.0f}</div><div class="lbl" style="color:#69f0ae;">{hi_move:+.1f} pts</div></div>'
                f'<div class="em-item"><div class="lbl">O/N Low</div><div class="val" style="font-size:16px;color:#ff8a80;">${overnite_range["es_low"]:.0f}</div><div class="lbl" style="color:#ff8a80;">{lo_move:+.1f} pts</div></div>'
                f'<div class="em-item"><div class="lbl">O/N Range</div><div class="val" style="font-size:16px;">{rng:.0f} pts</div></div>'
                f'<div class="em-item"><div class="lbl">Max O/N Excursion</div><div class="val" style="font-size:16px;">{overnite_range["max_move_pts"]:.0f} pts</div><div class="lbl">{max_vs_em_str}</div></div>'
                '</div>'
            )
            st.markdown(range_html, unsafe_allow_html=True)

    # ── Charts ──
    tab_gex, tab_profile = st.tabs(["📊 Strike GEX", "📈 GEX Profile"])

    with tab_gex:
        fig1 = build_gex_bar_chart(data["gex_df"], levels, spot, em)
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False})

    with tab_profile:
        fig2 = build_profile_chart(data["profile_df"], levels, spot, regime, em)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # ── Sidebar detail panels ──
    with st.sidebar:
        st.divider()
        if market_ctx != "premarket":
            render_expected_move_panel(em)
        render_key_levels(levels, spot, regime, data["confidence_info"], data["staleness_info"])
        st.divider()
        render_wall_credibility(data["wall_cred"])
        st.divider()
        render_scenarios_table(data["scenarios_df"])
        st.divider()
        render_data_quality(data["stats"], data["staleness_info"])

    # ── Auto-refresh ──
    if refresh_seconds > 0:
        import time
        time.sleep(refresh_seconds)
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
