"""
SPX Gamma Exposure (GEX) Dashboard — Streamlit Web App

Run locally:   streamlit run streamlit_app.py
Deploy:        Push to GitHub → connect at share.streamlit.io
"""
from __future__ import annotations

import os
import json
import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import COLORS
from models import GEXData
from ui_charts import build_gex_bar_chart
from ui_sidebar import (
    render_expected_move_panel, render_key_levels,
    render_wall_credibility, render_gex_stream, render_data_quality,
)

# ── Phase1 engine imports ──
from phase1.market_clock import now_ny, get_calendar_snapshot
from phase1.data_client import TradierDataClient
from phase1.rates import fetch_risk_free_rate
from phase1.parity import get_reference_spot_details
import phase1.gex_engine as gex_engine
from phase1.confidence import build_run_confidence
from phase1.staleness import build_staleness_info
from phase1.wall_credibility import build_wall_credibility
from phase1.expected_move import (
    build_expected_move_analysis, compute_em_for_expiration,
    find_weekly_expiration, find_monthly_expiration,
)
from phase1.futures_data import fetch_es_from_yahoo, build_futures_context
from phase1.gex_history import (
    save_snapshot, get_weekly_em_date_key, get_monthly_em_date_key,
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
@st.cache_data(ttl=600, show_spinner=False)
def get_expirations_cached(tradier_token: str, ticker: str) -> list[str]:
    """Tradier expirations change at most once per day; cache for 10 minutes
    so the sidebar render doesn't hit the API on every widget rerun."""
    return TradierDataClient(token=tradier_token).get_expirations(ticker)


@st.cache_resource(ttl=90, show_spinner=False)
def fetch_all_data(tradier_token: str, fred_key: str, selected_exps: tuple, _run_id: str, ticker: str = "SPX"):
    """
    Run the full GEX engine pipeline. Cached for 90 seconds.
    _run_id is kept stable; cache freshness is driven by the TTL and the
    "Refresh Now" button (which calls st.cache_resource.clear()).
    """
    client = TradierDataClient(token=tradier_token)
    client.clear_cache()

    run_now = now_ny()
    calendar_snapshot = get_calendar_snapshot(run_now)

    rfr_info = fetch_risk_free_rate(fred_key)
    rfr = rfr_info["rate"]
    rfr_curve = rfr_info.get("curve")  # None → flat-rate fallback path

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
        r_curve=rfr_curve,
    )
    spot = spot_info["spot"]
    spot_source = spot_info["source"]

    target_exps = list(selected_exps)

    gex_df, stats, all_options, strike_support_df, exp_support_df = (
        gex_engine.calculate_all(client, ticker, target_exps, spot, r=rfr, now=run_now,
                                 r_curve=rfr_curve)
    )

    levels = gex_engine.find_key_levels(gex_df, spot, all_options=all_options, r=rfr,
                                         r_curve=rfr_curve)
    has_0dte = any(e == today_str for e in target_exps)
    staleness_info = build_staleness_info(calendar_snapshot, spot_info, stats, has_0dte=has_0dte)
    confidence_info = build_run_confidence(stats, spot_info, staleness_info=staleness_info)
    wall_cred = build_wall_credibility(
        levels=levels,
        strike_support_df=strike_support_df,
        sensitivity_df=None,
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

    # Pre-fetch the Friday weekly chain the Spread Finder needs.
    # The spread finder builds weekly credit spreads for a specific Friday
    # expiration (the Friday of the week starting "next Monday").  If we
    # rely only on whatever the user picked in the sidebar, users who have
    # "0DTE"/"Tomorrow" selected end up with the spread finder silently
    # pricing weekly spreads off today's 0-DTE chain — producing $0.00
    # credits for far-OTM strikes because the options are effectively
    # worthless at 0 DTE.  Fetching the target Friday unconditionally here
    # guarantees live bid/ask for the expiration the model is actually
    # forecasting, regardless of sidebar state.
    try:
        from ui_spread_finder import find_spread_finder_friday_exp
        sf_friday_exp = find_spread_finder_friday_exp(avail, run_now.date())
        if sf_friday_exp and (ticker, sf_friday_exp) not in client.chain_cache:
            client.get_chain_cached(ticker, sf_friday_exp)
    except Exception as _sf_err:
        _logger.warning(f"Spread finder weekly chain pre-fetch failed: {_sf_err}")

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
        stats=stats,
        all_options=all_options,
        levels=levels,
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
        dte0_exp=dte0_exp,
        market_open=bool(spot_info.get("market_open")),
        yahoo_es=yahoo_es,
        chain_cache=dict(client.chain_cache),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Lazy imports to avoid circular dependency (ui_spread_finder imports from streamlit_app)
    from ui_history import (
        _is_weekly_freeze_day, _is_monthly_freeze_day,
        _apply_typed_em_snapshot, _apply_em_snapshot,
    )
    from ui_spread_finder import _render_spread_finder_tab

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
                avail = get_expirations_cached(tradier_token, ticker)
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
            "OpEx Cycle",
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
            # Friday-after-close edge case: Tradier drops today's expired 0DTE
            # from the expirations list once it's settled, so every entry in
            # future_exps is already past this Friday. In that case the naive
            # [today_str .. fri] window is empty — roll forward a week so the
            # user sees next week's expirations instead of "No expirations
            # selected". Same handling for any day where every future exp is
            # already past this Friday.
            if future_exps and future_exps[0] > fri:
                fri = (run_now + timedelta(days=days_to_fri + 7)).strftime("%Y-%m-%d")
            selected = [e for e in future_exps if e <= fri]
        elif "OpEx" in mode:
            # Select expirations from today through the next standard 3rd-Friday
            # OpEx. find_monthly_expiration rolls to next month's OpEx once the
            # current month's is behind us, so the bucket stays populated across
            # the whole cycle instead of silently shrinking to empty in the
            # second half of a calendar month (the old "This month" behavior).
            cycle_end = find_monthly_expiration(future_exps, run_now.date())
            if cycle_end:
                selected = [e for e in future_exps if e <= cycle_end]
            else:
                selected = []
        else:
            # Custom → date-range picker. Tradier only lists expirations on
            # specific days (weekdays + whatever 0DTE the product offers), so
            # we let the user pick any [from, to] span and then filter the
            # available expirations down to whatever falls inside it.
            _today = run_now.date()
            _min_d = date.fromisoformat(future_exps[0])  if future_exps else _today
            _max_d = date.fromisoformat(future_exps[-1]) if future_exps else _today
            # Clamp the default "from" into the allowed window.  On weekends
            # or Friday-after-close, `today` is already behind future_exps[0]
            # (which is next Monday), and Streamlit's date_input raises a
            # StreamlitAPIException when value[0] < min_value.
            _default_from = max(_today, _min_d)
            _default_end  = min(_default_from + timedelta(days=7), _max_d)
            if _default_end < _default_from:
                _default_end = _default_from
            _range = st.date_input(
                "Pick date range",
                value=(_default_from, _default_end),
                min_value=_min_d,
                max_value=_max_d,
                format="YYYY-MM-DD",
                help="Pick a single date (click twice on the same day) or a range.",
            )
            # st.date_input returns a date while the user is mid-selection
            # and a tuple of dates once they've picked both endpoints.
            if isinstance(_range, tuple):
                _from, _to = _range if len(_range) == 2 else (_range[0], _range[0])
            else:
                _from = _to = _range

            _from_s = _from.strftime("%Y-%m-%d")
            _to_s   = _to.strftime("%Y-%m-%d")
            selected = [e for e in future_exps if _from_s <= e <= _to_s]

            if selected:
                st.caption(
                    f"{len(selected)} expiration{'s' if len(selected) > 1 else ''} "
                    f"in {_from_s} → {_to_s}"
                )
            else:
                st.caption(
                    f"No Tradier expirations fall inside {_from_s} → {_to_s}. "
                    "Widen the range or pick a weekday the product trades on."
                )

        # Final safety net: if the chosen mode somehow produced an empty list
        # but there ARE future expirations available, fall back to the nearest
        # one rather than killing the page with st.stop(). Covers any residual
        # edge case (e.g. OpEx day post-close where the 3rd-Friday exp has been
        # removed from Tradier's list).
        if not selected and future_exps:
            selected = [future_exps[0]]
            st.caption(f"(No exps matched the selected window — falling back to {future_exps[0]})")

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
    # Stable key: cache freshness is driven by the @st.cache_resource TTL and
    # the "Refresh Now" button (which calls st.cache_resource.clear()). Using a
    # per-rerun timestamp here previously busted the cache on every widget
    # interaction, forcing a full Tradier/FRED/Yahoo/GEX repipeline each time.
    run_id = "stable"

    # ── Fetch data ──
    with st.spinner("Crunching GEX..."):
        try:
            data = fetch_all_data(tradier_token, fred_key or "", tuple(selected), run_id, ticker=ticker)
        except Exception as e:
            st.error(f"Engine error: {e}")
            st.stop()

    if data.gex_df.empty:
        # After hours, selecting "0DTE" when today's expiration has already
        # closed produces an empty frame — the gex engine now drops expired
        # expirations rather than pretend their T_FLOOR gamma is real.  Give
        # a friendlier nudge toward "Tomorrow" / "This week" in that case.
        expired_cnt = (data.stats or {}).get("expired_exp_count", 0)
        if expired_cnt and expired_cnt == len(data.target_exps):
            st.info(
                "All selected expirations have already settled for the day. "
                "Switch to **Tomorrow** or **This week** for forward-looking GEX."
            )
        else:
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

    # es_prevclose is only available on the Yahoo path (manual-entry users
    # don't enter a prior ES close). When present it removes the ES-SPX basis
    # from the overnight move calculation. See phase1/futures_data.py.
    es_prevclose = yahoo_es.get("prevclose") if (yahoo_es and not has_manual) else None

    futures_ctx = None
    if es_last and es_last > 0 and prev_close > 0:
        # XSP trades at 1/10 the scale of SPX. ES futures track SPX, so
        # scale ES down by 10 when the dashboard is set to XSP — otherwise
        # the overnight move math compares apples to oranges.
        if ticker == "XSP":
            es_last_scaled = es_last / 10.0
            es_high_scaled = (es_high / 10.0) if es_high else None
            es_low_scaled = (es_low / 10.0) if es_low else None
            es_prevclose_scaled = (es_prevclose / 10.0) if es_prevclose else None
            futures_ctx = build_futures_context(
                es_last_scaled, es_high_scaled, es_low_scaled, prev_close,
                source=es_source, es_prevclose=es_prevclose_scaled,
            )
        else:
            futures_ctx = build_futures_context(
                es_last, es_high, es_low, prev_close,
                source=es_source, es_prevclose=es_prevclose,
            )

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
        expiration=data.dte0_exp,
    )

    # ── Apply EM snapshot logic ──
    em_analysis = _apply_em_snapshot(em_analysis, is_market_open, regime, levels, spot, ticker=ticker)

    # ── Weekly & Monthly EM ──
    run_now = now_ny()
    temp_client = TradierDataClient(token=tradier_token)
    # Seed the temp client's chain cache with the snapshot we captured
    # inside fetch_all_data so weekly/monthly EM lookups reuse the chains
    # we already paid for instead of hitting Tradier again. Without this,
    # every rerun pays for 2+ extra option-chain fetches (one per
    # compute_em_for_expiration call) even when fetch_all_data has already
    # pre-warmed the target expirations (see the "pre-fetch" block near
    # the top of fetch_all_data).
    if data.chain_cache:
        temp_client.chain_cache.update(data.chain_cache)

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

    # ── Pick the EM that matches the expiration view the user chose ──
    # The header "Expected Move" card used to always show the 0DTE EM even
    # when the user selected "This week" or "OpEx Cycle", which was
    # misleading. Map the sidebar mode to the matching frozen snapshot
    # (or a live compute for Custom/Tomorrow where there's no natural
    # session-start anchor).
    daily_em = em_analysis.get("expected_move", {}) or {}

    def _em_for_exp(exp_str):
        if not exp_str:
            return {}
        try:
            return compute_em_for_expiration(temp_client, ticker, exp_str, spot) or {}
        except Exception:
            return {}

    if "0DTE" in mode:
        display_em = daily_em
        display_em_label = "0DTE EM"
    elif "Tomorrow" in mode:
        tmr_em = _em_for_exp(selected[0] if selected else None)
        display_em = tmr_em or daily_em
        display_em_label = "Tomorrow EM"
    elif "week" in mode:
        display_em = weekly_em_snap or weekly_em_live or {}
        display_em_label = "Weekly EM"
    elif "OpEx" in mode:
        display_em = monthly_em_snap or monthly_em_live or {}
        display_em_label = "OpEx-Cycle EM"
    else:  # Custom
        farthest = selected[-1] if selected else None
        display_em = _em_for_exp(farthest) or daily_em
        display_em_label = f"Custom EM ({farthest})" if farthest else "Custom EM"

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
    # Flag fallback zero-gamma so the banner tells the truth: if the sweep
    # couldn't find a true sign-change crossing, the displayed "Positive
    # Gamma" / "Negative Gamma" label is derived from a min-abs-GEX node
    # that can be dozens of points away from the real zero-gamma. The
    # sidebar already warns about this but users acting on the header
    # regime label never saw the qualification.
    zg_is_true = bool(levels.get("zero_gamma_is_true_crossing", True))
    regime_suffix = "" if zg_is_true else " <span style='color:#ffa726;font-size:13px;'>(fallback ZG)</span>"
    st.markdown(
        f"<div style='text-align:center;padding:6px;'>"
        f"<span style='font-size:22px;font-weight:bold;color:{spot_c};'>{ticker} ${spot:.2f}</span>"
        f"&nbsp;&nbsp;&nbsp;"
        f"<span style='font-size:18px;color:{regime_color};font-weight:bold;'>{regime['regime']}</span>{regime_suffix}"
        f"&nbsp;&nbsp;"
        f"<span style='color:{text_sec};font-size:13px;'>({regime['distance_text']})</span>"
        f"&nbsp;&nbsp;&nbsp;"
        f"<span style='color:{text_mut};font-size:12px;'>Updated {data.run_time}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if not zg_is_true:
        zg_type = levels.get("zero_gamma_type", "Fallback node")
        if zg_type == "Rescued crossing":
            st.warning(
                "⚠️ **Zero gamma is a rescued crossing** — the coarse GEX "
                "sweep didn't find a sign change, and the level above was "
                "only caught by refining around the nearest-to-zero node. "
                "The crossing is real but fragile: a single-strike IV or "
                "OI move can swing it 10+ points. Trade with tighter "
                "stops or skip."
            )
        else:
            st.warning(
                "⚠️ **Zero gamma is a fallback estimate** — the GEX sweep didn't "
                "find a true sign-change crossing in the window, so the regime "
                "label above is derived from the nearest-to-zero GEX node. In "
                "high-vol weeks this can be 10+ points off the true flip. Trade "
                "with tighter stops or skip."
            )

    # ── Market context banner ──
    market_ctx = em_analysis.get("market_context", "live")
    context_note = em_analysis.get("context_note")
    if market_ctx == "premarket":
        st.info("🌅 **Pre-market** — GEX levels and gamma regime are current. "
                "Expected move and session classification will be available after the 9:30 AM open.")
    elif market_ctx == "afterhours":
        st.warning(f"🌙 **After hours** — {context_note}")

    # ── Sidebar detail panels ──
    # Rendered here (before tabs/spread finder) so the sidebar refreshes
    # immediately on every rerun instead of waiting for the slow HAR model
    # to finish.
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
        render_data_quality(data.stats, data.staleness_info)

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
        cls_signal = classification.get("signal_strength", "weak")
        cls_acc = classification.get("bucket_accuracy")

        # Color is gated on calibrated signal strength so weak buckets
        # (55% accuracy — barely above random) stop rendering with a
        # decisive green/red that implies a reliable read.
        if cls_signal == "strong":
            if cls_bias in ("range-bound", "mean-revert"):
                cls_color = COLORS["positive"]
            elif cls_bias in ("directional", "continued-trend"):
                cls_color = COLORS["negative"]
            else:
                cls_color = COLORS["warning"]
        elif cls_signal == "moderate":
            cls_color = COLORS["warning"]
        else:
            cls_color = COLORS["text_muted"]

        cls_acc_label = f"hist {cls_acc*100:.0f}%" if cls_acc is not None else ""

        # The EM card shown here tracks the expiration view the user picked
        # in the sidebar (see display_em / display_em_label above) — it's no
        # longer hard-wired to the 0DTE daily EM. The vol-budget ratio still
        # uses the 0DTE EM denominator because today's overnight move can't
        # sensibly be normalized by a weekly or monthly straddle.
        _em_pts = display_em.get("expected_move_pts", 0) or 0
        _em_lo  = display_em.get("lower_level", 0) or 0
        _em_hi  = display_em.get("upper_level", 0) or 0
        em_bar_html = (
            '<div class="em-bar">'
            f'<div class="em-item"><div class="lbl">{display_em_label}</div><div class="val">&plusmn;{_em_pts:.0f} pts</div></div>'
            f'<div class="em-item"><div class="lbl">EM Range</div><div class="val">${_em_lo:.0f} &ndash; ${_em_hi:.0f}</div></div>'
            f'<div class="em-item"><div class="lbl">{on_label}</div><div class="val" style="color:{on_color};">{on_arrow} {display_pts:+.1f} pts</div><div class="lbl" style="color:{on_color};">{display_pct:+.2f}%</div></div>'
            f'<div class="em-item"><div class="lbl">Vol Budget Used</div><div class="val" style="color:{ratio_color};">{ratio_pct}</div></div>'
            f'<div class="em-item"><div class="lbl">Session Type</div><div class="val" style="color:{cls_color};">{cls_name}</div><div class="lbl" style="color:{cls_color};">{cls_acc_label}</div></div>'
            '</div>'
        )
        st.markdown(em_bar_html, unsafe_allow_html=True)

        # Show when the relevant straddle was captured. Each horizon has its
        # own freeze timestamp key (daily / weekly / monthly); Custom and
        # Tomorrow are live recomputes so they don't have one.
        _snap_key_for_view = {
            "0DTE EM": f"em_snapshot_time_daily_{ticker}",
            "Weekly EM": f"em_snapshot_time_weekly_{ticker}",
            "OpEx-Cycle EM": f"em_snapshot_time_monthly_{ticker}",
        }.get(display_em_label)
        snap_time = st.session_state.get(_snap_key_for_view) if _snap_key_for_view else None
        if market_ctx == "live" and snap_time:
            st.caption(f"📌 {display_em_label} captured at {snap_time} — frozen for the horizon. Today's move and vol budget update live.")

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
    tab_gex, tab_spread_finder = st.tabs(["📊 Strike GEX", "🎯 Spread Finder"])

    with tab_gex:
        # Weekly/monthly markers ONLY use the frozen snap — never the live EM.
        # Falling back to live data makes the markers drift every refresh; if
        # the snap is missing we'd rather draw nothing than a moving target.
        #
        # We also hide each EM band whenever the user's expiration view has
        # zoomed PAST that EM's coverage window, because an EM measured from
        # a Friday straddle is meaningless once you're looking at strikes
        # that expire next month — the underlying can (and usually does)
        # move far beyond the weekly ±EM by then. Rules:
        #   daily   EM → show only if every selected exp is today
        #   weekly  EM → show only if the farthest selected exp ≤ this Friday
        #   monthly EM → show only if the farthest selected exp ≤ this OpEx
        _today_str = run_now.strftime("%Y-%m-%d")
        _farthest  = max(selected) if selected else _today_str

        _show_daily_em   = all(e == _today_str for e in selected) if selected else False
        _show_weekly_em  = bool(weekly_exp)  and _farthest <= weekly_exp
        _show_monthly_em = bool(monthly_exp) and _farthest <= monthly_exp

        w_em_for_chart = (weekly_em_snap or {}) if _show_weekly_em else {}
        m_em_for_chart = (monthly_em_snap or {}) if _show_monthly_em else {}
        fig1 = build_gex_bar_chart(
            data.gex_df, levels, spot, em_analysis,
            weekly_em=w_em_for_chart, monthly_em=m_em_for_chart,
            show_daily_em=_show_daily_em,
        )
        st.plotly_chart(fig1, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

    # ── Spread Finder — Weekly credit spread placement ──
    with tab_spread_finder:
        # Spread finder also uses the frozen weekly snap only — a drifting
        # live EM would invalidate the spread plan intra-week.
        _sf_weekly_em = weekly_em_snap or {}
        _render_spread_finder_tab(spot, levels, regime, data, ticker=ticker, weekly_em=_sf_weekly_em)

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
