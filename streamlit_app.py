"""
Gamma Lens — SPX Gamma Exposure (GEX) Dashboard — Streamlit Web App

Run locally:   streamlit run streamlit_app.py
Deploy:        Push to GitHub → connect at share.streamlit.io

UI layer = the "pro terminal" redesign. Interactive controls are custom HTML
driven by st.query_params (see ui_theme); all data/model logic is unchanged.
"""
from __future__ import annotations

import os
import logging
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import COLORS
from models import GEXData
from ui_charts import render_gex_html
from ui_sidebar import (
    render_expected_move_panel, render_key_levels,
    render_gex_stream, render_quality_card,
)
from ui_theme import (
    inject_global_css, map_exp_mode, map_refresh,
    render_header, fmt_commas, esc, QUICK_TICKERS, LOGO_PATH,
)
from ui_controls import (
    render_settings_controls, render_tab_control, render_refresh_button,
)
from ui_url_state import rehydrate_from_url, sync_to_url
from ui_pwa import inject_pwa_head
from ui_mobile import inject_mobile_shell

# ── Phase1 engine imports ──
from phase1.market_clock import now_ny, get_calendar_snapshot
from phase1.data_client import TradierDataClient, resolve_quote_spot
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
from phase1.gex_history import (
    get_weekly_em_date_key, get_monthly_em_date_key,
)
from phase1.gamma_level_history import archive_computed_gamma_levels

_logger = logging.getLogger(__name__)

TOOL_VERSION = "v5-web"


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Gamma Lens",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)


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


@st.cache_data(ttl=600, show_spinner=False)
def validate_ticker_cached(tradier_token: str, symbol: str):
    """Validate a typed symbol against Tradier (must be optionable).

    Returns the instrument dict ({symbol, type, name, has_options}) or None.
    Cached 10 minutes so re-typing a symbol doesn't re-hit the API."""
    return TradierDataClient(token=tradier_token).validate_ticker(symbol)


@st.cache_data(ttl=3600, show_spinner=False)
def get_risk_free_rate_cached(fred_key: str) -> dict:
    """FRED publishes these series once per business day; cache for an hour
    so switching expiration modes (which busts the fetch_all_data cache key)
    doesn't re-fire the 5-request FRED scalar+curve fetch every time."""
    return fetch_risk_free_rate(fred_key)


@st.cache_resource(ttl=90, show_spinner=False)
def fetch_all_data(tradier_token: str, fred_key: str, selected_exps: tuple, _run_id: str, ticker: str = "SPX"):
    """
    Run the full GEX engine pipeline. Cached for 90 seconds.
    _run_id is kept stable; cache freshness is driven by the TTL and the
    "Refresh Now" button (which calls st.cache_resource.clear()).
    """
    client = TradierDataClient(token=tradier_token)

    run_now = now_ny()
    calendar_snapshot = get_calendar_snapshot(run_now)

    rfr_info = get_risk_free_rate_cached(fred_key)
    rfr = rfr_info["rate"]
    rfr_curve = rfr_info.get("curve")  # None → flat-rate fallback path

    # Reuse the sidebar's cached expirations list instead of issuing a
    # second Tradier /expirations request inside this pipeline.
    avail = get_expirations_cached(tradier_token, ticker)
    if not avail:
        raise RuntimeError(f"No expirations returned from Tradier API for {ticker}")
    today_str = run_now.strftime("%Y-%m-%d")
    nearest_exp = next((e for e in avail if e >= today_str), avail[0])

    # Fetch the index's full quote up front so parity's spot lookup and
    # prev_close share one /markets/quotes response instead of each issuing
    # their own request.
    index_quote = None
    try:
        index_quote = client.get_full_quote(ticker)
    except Exception:
        pass

    def _spot_price_for(t):
        if t == ticker and index_quote:
            return resolve_quote_spot(index_quote)
        return client.get_spot_price(t)

    spot_info = get_reference_spot_details(
        ticker=ticker,
        nearest_exp=nearest_exp,
        get_spot_price_func=_spot_price_for,
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

    from phase1.ticker_config import get_config as _tk_cfg
    levels = gex_engine.find_key_levels(gex_df, spot, all_options=all_options, r=rfr,
                                         r_curve=rfr_curve,
                                         q=float(_tk_cfg(ticker).get("dividend_yield", 0.0) or 0.0))
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

    # Pre-fetch the weekly / monthly EM chains as well. main() recomputes
    # the weekly and OpEx-cycle expected-move straddles on EVERY rerun via
    # a throwaway client seeded from this snapshot's chain_cache — if those
    # expirations aren't in the snapshot (e.g. the monthly 3rd-Friday chain
    # while in 0DTE mode), every widget interaction pays for a full Tradier
    # chain fetch that is then thrown away with the temp client.
    try:
        for _em_exp in {
            find_weekly_expiration(avail, run_now.date()),
            find_monthly_expiration(avail, run_now.date()),
        }:
            if _em_exp and (ticker, _em_exp) not in client.chain_cache:
                client.get_chain_cached(ticker, _em_exp)
    except Exception as _em_err:
        _logger.warning(f"Weekly/monthly EM chain pre-fetch failed: {_em_err}")

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
        dte0_calls=dte0_calls,
        dte0_puts=dte0_puts,
        dte0_exp=dte0_exp,
        market_open=bool(spot_info.get("market_open")),
        chain_cache=dict(client.chain_cache),
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI helpers (presentation only)
# ─────────────────────────────────────────────────────────────────────────────
def _regime_color(regime_text: str) -> str:
    t = (regime_text or "").lower()
    if "positive" in t:
        return COLORS["positive"]
    if "negative" in t:
        return COLORS["negative"]
    return COLORS["warning"]


def _banner(text: str, accent: str, bg: str) -> str:
    return (f'<div class="term-banner" style="background:{bg};border-left:4px solid {accent};'
            f'color:var(--text-secondary);">{text}</div>')


def _next_clock_boundary_ms(period_seconds: int, now: datetime) -> int:
    """Epoch-milliseconds of the next wall-clock multiple of ``period_seconds``.

    Aligns refreshes to the clock instead of to page-load time: a 300s period
    lands on :00/:05/…/:35/:40 and an 1800s period on :00/:30. Both divide an
    hour evenly and NY is a whole-hour UTC offset, so epoch alignment equals
    wall-clock alignment. Drives both the refresh interval and a per-boundary
    component key (see the auto-refresh block in ``main``).
    """
    period_ms = period_seconds * 1000
    now_ms = now.timestamp() * 1000
    return (int(now_ms // period_ms) + 1) * period_ms


def _build_em_strip(display_em, display_em_label, on_label, on_pts, on_pct,
                    ratio, classification) -> str:
    """5-cell EM strip across the top of the main column."""
    em_pts = display_em.get("expected_move_pts", 0) or 0
    lo = display_em.get("lower_level", 0) or 0
    hi = display_em.get("upper_level", 0) or 0

    on_color = COLORS["positive"] if (on_pts or 0) >= 0 else COLORS["negative"]
    on_arrow = "▲" if (on_pts or 0) > 0 else "▼" if (on_pts or 0) < 0 else "–"

    if ratio is not None:
        ratio_pct = f"{ratio*100:.0f}%"
        ratio_color = (COLORS["positive"] if ratio < 0.40
                       else COLORS["warning"] if ratio < 0.70 else COLORS["negative"])
    else:
        ratio_pct = "—"
        ratio_color = COLORS["text_muted"]

    cls_name = classification.get("classification", "–")
    cls_bias = classification.get("bias", "")
    cls_signal = classification.get("signal_strength", "weak")
    if cls_signal == "strong":
        cls_color = (COLORS["positive"] if cls_bias in ("range-bound", "mean-revert")
                     else COLORS["negative"] if cls_bias in ("directional", "continued-trend")
                     else COLORS["warning"])
    elif cls_signal == "moderate":
        cls_color = COLORS["warning"]
    else:
        cls_color = COLORS["text_muted"]

    def cell(label, value, color):
        return (f'<div class="cell"><div class="cl">{esc(label)}</div>'
                f'<div class="cv" style="color:{color};">{value}</div></div>')

    # When the expected move is unavailable the straddle inputs come through
    # as 0 — rendering "±0 pts" / "EM RANGE 0–0" reads as a real, extremely
    # tight forecast (the most dangerous possible misread for strike
    # placement). Show an explicit unavailable dash instead.
    _em_available = em_pts and em_pts > 0 and lo > 0 and hi > 0
    em_cell_val = f"±{em_pts:.0f} pts" if _em_available else "—"
    range_cell_val = f"{fmt_commas(lo,0)}–{fmt_commas(hi,0)}" if _em_available else "unavailable"
    em_color = COLORS["em_level"] if _em_available else COLORS["text_muted"]
    range_color = COLORS["text_white"] if _em_available else COLORS["text_muted"]

    on_val = (f'{on_arrow} {on_pts:+.1f} <span style="font-size:11px;color:var(--text-dim);">{(on_pct or 0):+.2f}%</span>'
              if on_pts is not None else "—")
    return (
        '<div class="em-strip">'
        + cell(display_em_label.upper(), em_cell_val, em_color)
        + cell("EM RANGE", range_cell_val, range_color)
        + cell("TODAY'S MOVE", on_val, on_color)
        + cell("VOL BUDGET", ratio_pct, COLORS["warning"])
        + cell("SESSION", esc(cls_name), cls_color)
        + '</div>'
    )


def _build_em_stamp(display_em_label, ticker, run_time, market_ctx) -> str:
    """Timestamp caption beneath the EM strip.

    Shows when the displayed EM straddle was frozen (daily / weekly / OpEx
    horizons each have their own freeze key; Tomorrow / Custom are live
    recomputes and have none) and always shows the last data-refresh time.
    """
    snap_key = {
        "0DTE EM": f"em_snapshot_time_daily_{ticker}",
        "Weekly EM": f"em_snapshot_time_weekly_{ticker}",
        "OpEx-Cycle EM": f"em_snapshot_time_monthly_{ticker}",
    }.get(display_em_label)
    snap_time = st.session_state.get(snap_key) if snap_key else None

    parts = []
    if market_ctx == "live" and snap_time:
        parts.append(
            f'<span>📌 {esc(display_em_label)} captured at <b>{esc(snap_time)}</b> '
            f'— frozen for the horizon; today’s move &amp; vol budget update live</span>'
        )
    parts.append(f'<span>⟳ Updated <b>{esc(run_time)}</b></span>')
    inner = '<span class="sep">·</span>'.join(parts)
    return f'<div class="em-stamp">{inner}</div>'


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
    from ui_tv_export import render_tv_export_section

    inject_global_css()
    inject_pwa_head()
    inject_mobile_shell()

    tradier_token, fred_key = get_credentials()

    # ── Credentials gate (only if no token configured) ──
    if not tradier_token:
        st.markdown("### ⚙️ Connect Tradier to start")
        tradier_token = st.text_input("Tradier API Token", type="password",
                                       help="Get yours at https://web.tradier.com/user/api")
        if not fred_key:
            fred_key = st.text_input("FRED API Key (optional)", type="password",
                                     help="For live T-bill rates.")
        if not tradier_token:
            st.warning("Enter your Tradier API token to get started.")
            st.stop()

    # ── Active ticker (resolved from st.session_state; widget callbacks set it).
    # State lives in session — native-widget reruns patch the page over the
    # websocket instead of the old full-page reload, so session_state survives.
    from phase1.ticker_config import all_tickers, get_config
    _curated = all_tickers()
    # Seed nav state (ticker / tab / exp / recents / export extras) from the URL
    # once per session BEFORE the setdefaults below, so a Streamlit Cloud session
    # reset (which wipes session_state) restores where the user was instead of
    # snapping back to SPX / GEX / no-recents. See ui_url_state.
    rehydrate_from_url()
    st.session_state.setdefault("ticker_meta", {})
    st.session_state.setdefault("recent_tickers", [])
    st.session_state.setdefault("active_ticker", "SPX")

    ticker = (st.session_state.get("active_ticker") or "SPX").strip().upper()
    ticker_error = None
    if ticker not in _curated:
        meta = st.session_state["ticker_meta"].get(ticker)
        if meta is None:
            with st.spinner(f"Validating {ticker}…"):
                meta = validate_ticker_cached(tradier_token, ticker)
        if meta:
            st.session_state["ticker_meta"][ticker] = meta
        else:
            ticker_error = (f"'{ticker}' isn't a recognized optionable symbol on Tradier — "
                            "showing SPX instead.")
            ticker = "SPX"
            st.session_state["active_ticker"] = "SPX"

    # Track recently-viewed tickers (most-recent-first), excluding the static
    # quick picks which already have their own chip row. Keyed off QUICK_TICKERS
    # (not the live all_tickers() set) so a custom symbol the pipeline later
    # registers into TICKER_CONFIG isn't evicted from the recents list.
    RECENTS_MAX = 6
    if ticker not in QUICK_TICKERS:  # a valid custom symbol (SPX-fallback is curated)
        st.session_state["recent_tickers"] = (
            [ticker] + [t for t in st.session_state["recent_tickers"] if t != ticker]
        )[:RECENTS_MAX]
    recents = [t for t in st.session_state["recent_tickers"] if t not in QUICK_TICKERS]

    _meta_active = st.session_state["ticker_meta"].get(ticker)
    _cat = (_meta_active or {}).get("type") or get_config(ticker).get("category", "")
    ticker_type = {"index": "Index", "etf": "ETF", "stock": "Stock"}.get(_cat, _cat or "—")

    # ── Layout scaffolding: a full-width header/banner area on top, then the
    # aside | main columns below — everything on one page (no separate sidebar).
    # Built before the data fetch so the native controls in the aside column can
    # resolve ticker/expiration first; the post-fetch header/cards/chart then
    # fill their reserved slots. ──
    header_box = st.container()
    banner_box = st.container()
    em_box = st.container()  # full-width EM strip row, above the aside|main columns
    aside_col, main_col = st.columns([1, 3], gap="small")

    # ── Aside controls (native widgets → websocket rerun, no page reload/flash) ──
    with aside_col:
        with st.container(border=True, key="settings_card"):
            exp_token, refresh_token, cal_start, cal_end = render_settings_controls(
                ticker, ticker_type, recents)
            render_refresh_button()

    mode = map_exp_mode(exp_token)
    mode_short = mode
    refresh_option = map_refresh(refresh_token)

    # ── Expirations ──
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

    # ── Resolve the selected expirations from the mode (data logic unchanged) ──
    if "0DTE" in mode:
        selected = [dte0] if dte0 else []
    elif "Tomorrow" in mode:
        selected = [dte1] if dte1 else []
    elif "week" in mode:
        days_to_fri = (4 - run_now.weekday()) % 7
        fri = (run_now + timedelta(days=days_to_fri)).strftime("%Y-%m-%d")
        if future_exps and future_exps[0] > fri:
            fri = (run_now + timedelta(days=days_to_fri + 7)).strftime("%Y-%m-%d")
        selected = [e for e in future_exps if e <= fri]
    elif "OpEx" in mode:
        cycle_end = find_monthly_expiration(future_exps, run_now.date())
        selected = [e for e in future_exps if e <= cycle_end] if cycle_end else []
    else:  # Custom — date range from the sidebar date picker (cal_start/cal_end)
        _today = run_now.date()
        _min_d = date.fromisoformat(future_exps[0]) if future_exps else _today
        _max_d = date.fromisoformat(future_exps[-1]) if future_exps else _today
        if cal_start:
            try:
                _from = date.fromisoformat(cal_start)
            except ValueError:
                _from = max(_today, _min_d)
        else:
            _from = max(_today, _min_d)
        if cal_end:
            try:
                _to = date.fromisoformat(cal_end)
            except ValueError:
                _to = _from
        elif cal_start:
            _to = _from
        else:
            _to = min(_from + timedelta(days=7), _max_d)
        if future_exps:
            _from = max(_from, _min_d)
        _from_s = _from.strftime("%Y-%m-%d")
        _to_s = _to.strftime("%Y-%m-%d")
        selected = [e for e in future_exps if _from_s <= e <= _to_s]

    if not selected and future_exps:
        selected = [future_exps[0]]
    if not selected:
        st.html(_banner("No expirations available for this ticker / window.",
                        COLORS["warning"], "rgba(255,180,84,.10)"))
        st.stop()

    refresh_seconds = {"Off": 0, "Every 5 min": 300, "Every 30 min": 1800}.get(refresh_option, 0)
    run_id = "stable"

    # ── Fetch data ──
    with st.spinner("Crunching GEX..."):
        try:
            data = fetch_all_data(tradier_token, fred_key or "", tuple(selected), run_id, ticker=ticker)
        except Exception as e:
            st.error(f"Engine error: {e}")
            st.stop()

    if data.gex_df.empty:
        expired_cnt = (data.stats or {}).get("expired_exp_count", 0)
        if expired_cnt and expired_cnt == len(data.target_exps):
            msg = ("All selected expirations have already settled for the day. "
                   "Switch to <b>Tomorrow</b> or <b>This week</b> for forward-looking GEX.")
        else:
            msg = "No GEX data returned. The selected expirations may have no usable contracts."
        st.html(_banner(msg, COLORS["warning"], "rgba(255,180,84,.10)"))
        st.stop()

    spot = data.spot
    levels = data.levels
    regime = data.regime_info
    prev_close = data.prev_close
    is_market_open = data.market_open

    # Register the per-ticker config derived from the live chain (no-op for curated five).
    try:
        from phase1.ticker_config import register_dynamic_config
        _meta_t = st.session_state.get("ticker_meta", {}).get(ticker)
        register_dynamic_config(
            ticker,
            strikes=data.gex_df["strike"].tolist(),
            spot=float(spot or 0.0),
            instrument_type=(_meta_t or {}).get("type"),
        )
    except Exception as _cfg_err:
        _logger.warning(f"dynamic config registration failed for {ticker}: {_cfg_err}")

    # ── Build EM analysis (fresh each render, not cached) ──
    em_analysis = build_expected_move_analysis(
        spot=spot,
        prev_close=prev_close,
        zero_gamma=levels["zero_gamma"],
        gamma_regime=regime["regime"],
        calls_0dte=data.dte0_calls,
        puts_0dte=data.dte0_puts,
        market_open=data.market_open,
        expiration=data.dte0_exp,
    )

    # Opportunistic research archive: the observation time comes from the
    # cached engine result, not this rerun, so a stale cached computation can
    # never be filed into a later 30-minute bucket. The archive helper skips
    # non-RTH times, deduplicates by ticker/session/bucket, and never raises.
    archive_computed_gamma_levels(
        captured_at=data.calendar_snapshot.get("now_ny", ""),
        ticker=ticker,
        spot=spot,
        levels=levels,
        regime_info=regime,
        stats=data.stats,
        confidence_info=data.confidence_info,
        staleness_info=data.staleness_info,
        em_analysis=em_analysis,
    )

    # EM freezing is UI/application state. Archive the live contemporaneous EM
    # above before this may substitute the first observation saved for the day.
    em_analysis = _apply_em_snapshot(em_analysis, is_market_open, regime, levels, spot, ticker=ticker)

    # ── Weekly & Monthly EM ──
    run_now = now_ny()
    temp_client = TradierDataClient(token=tradier_token)
    if data.chain_cache:
        temp_client.chain_cache.update(data.chain_cache)

    weekly_exp = find_weekly_expiration(data.avail, run_now.date())
    weekly_em_live = compute_em_for_expiration(temp_client, ticker, weekly_exp, spot) if weekly_exp else None
    weekly_date_key = get_weekly_em_date_key(run_now)
    weekly_em_snap = _apply_typed_em_snapshot(
        weekly_em_live, is_market_open, spot, ticker,
        "weekly", weekly_date_key, _is_weekly_freeze_day(run_now),
    )

    monthly_exp = find_monthly_expiration(data.avail, run_now.date())
    monthly_em_live = compute_em_for_expiration(temp_client, ticker, monthly_exp, spot) if monthly_exp else None
    monthly_date_key = get_monthly_em_date_key(run_now)
    monthly_em_snap = _apply_typed_em_snapshot(
        monthly_em_live, is_market_open, spot, ticker,
        "monthly", monthly_date_key, _is_monthly_freeze_day(run_now),
    )

    # ── Pick the EM that matches the expiration view the user chose ──
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

    # ─────────────────────────────────────────────────────────────────────────
    # RENDER
    # ─────────────────────────────────────────────────────────────────────────
    regime_color = _regime_color(regime.get("regime", ""))
    change_pts = (spot - prev_close) if prev_close else None
    change_pct = (change_pts / prev_close * 100) if (prev_close and change_pts is not None) else None
    zg_is_true = bool(levels.get("zero_gamma_is_true_crossing", True))
    regime_note = regime.get("distance_text", "")
    if not zg_is_true:
        regime_note = (regime_note + " · fallback ZG").strip(" ·")

    header_html = render_header(
        ticker=ticker, spot=spot,
        day_change_pts=change_pts, day_change_pct=change_pct,
        regime_label=(regime.get("regime", "") or "").upper(),
        regime_color=regime_color, regime_note=regime_note,
        live=is_market_open,
    )
    with header_box:
        st.html(header_html)

    # ── Banners (ticker error / ZG fallback / afterhours) ──
    banners = []
    if ticker_error:
        banners.append(_banner(esc(ticker_error), COLORS["warning"], "rgba(255,180,84,.10)"))
    if not zg_is_true:
        zg_type = levels.get("zero_gamma_type", "Fallback node")
        if zg_type == "Rescued crossing":
            banners.append(_banner(
                "⚠️ <b>Zero gamma is a rescued crossing</b> — the coarse sweep didn't find a sign change; "
                "only caught by refining the nearest-to-zero node. Real but fragile (a single-strike move "
                "can swing it 10+ pts). Trade tighter or skip.",
                COLORS["warning"], "rgba(255,180,84,.08)"))
        else:
            banners.append(_banner(
                "⚠️ <b>Zero gamma is a fallback estimate</b> — no true sign-change crossing was found in the "
                "window, so the regime label is from the nearest-to-zero GEX node. In high-vol weeks this can "
                "be 10+ pts off the true flip. Trade tighter or skip.",
                COLORS["warning"], "rgba(255,180,84,.08)"))
    market_ctx = em_analysis.get("market_context", "live")
    if market_ctx == "afterhours":
        ctx_note = em_analysis.get("context_note") or "Cash market is closed."
        banners.append(_banner(f"🌙 <b>Market closed</b> — {esc(ctx_note)}", COLORS["accent_blue"],
                               "rgba(110,168,255,.08)"))
    if banners:
        with banner_box:
            st.html("".join(banners))

    # ── Aside display cards → a swipeable pager on mobile, a vertical stack on
    # desktop (same .term-card markup; ui_theme CSS reflows it per viewport, the
    # dots are wired up by ui_mobile). Drop any card that renders empty (e.g.
    # wall credibility) so the dot count matches the real slides. ──
    cards = [
        render_key_levels(
            levels, spot, regime, data.confidence_info, ticker, mode_short,
            daily_em=daily_em,
            weekly_em=weekly_em_snap or weekly_em_live or {},
            monthly_em=monthly_em_snap or monthly_em_live or {},
        ),
        render_gex_stream(data.stats, levels, spot),
        render_expected_move_panel(em_analysis, spot, ticker=ticker),
        render_quality_card(data.wall_cred, data.stats, data.staleness_info),
    ]
    cards = [c for c in cards if c and c.strip()]
    slides = "".join(f'<div class="gl-slide">{c}</div>' for c in cards)
    dots = "".join(
        f'<button class="gl-dot" data-i="{i}" aria-label="Card {i + 1}"></button>'
        for i in range(len(cards))
    )
    pager = f'<div class="gl-pager"><div class="gl-track">{slides}</div><div class="gl-dots">{dots}</div></div>'
    with aside_col:
        st.html(pager)

    # ── EM strip → full-width row above the columns (top of page on mobile). ──
    overnight = em_analysis.get("overnight_move", {}) or {}
    classification = em_analysis.get("classification", {}) or {}
    on_label = "Today's Move" if market_ctx == "live" else "Session Move"
    em_strip = _build_em_strip(
        display_em or {}, display_em_label, on_label,
        overnight.get("overnight_move_pts"), overnight.get("overnight_move_pct"),
        classification.get("move_ratio"), classification,
    )
    with em_box:
        st.html(em_strip + _build_em_stamp(display_em_label, ticker, data.run_time, market_ctx))

    # ── Main column: tab control + tab content ──
    with main_col:
        # Tabs (native segmented control → websocket rerun, no reload) + content
        tab = render_tab_control()
        if tab == "gex":
            _today_str = run_now.strftime("%Y-%m-%d")
            _farthest = max(selected) if selected else _today_str
            _show_daily_em = all(e == _today_str for e in selected) if selected else False
            _show_weekly_em = bool(weekly_exp) and _farthest <= weekly_exp
            _show_monthly_em = bool(monthly_exp) and _farthest <= monthly_exp
            w_em_for_chart = (weekly_em_snap or {}) if _show_weekly_em else {}
            m_em_for_chart = (monthly_em_snap or {}) if _show_monthly_em else {}
            st.html(render_gex_html(
                data.gex_df, levels, spot, em_analysis,
                weekly_em=w_em_for_chart, monthly_em=m_em_for_chart,
                show_daily_em=_show_daily_em, ticker=ticker,
            ))
            # EM inputs mirror the Key Levels aside card (snap-or-live), not
            # the chart's expiration gating — the Pine indicator has its own
            # per-band visibility toggles, so export everything available.
            render_tv_export_section(
                ticker=ticker, spot=spot, levels=levels,
                gex_df_empty=data.gex_df.empty,
                daily_em=daily_em,
                weekly_em=(weekly_em_snap or weekly_em_live or {}),
                monthly_em=(monthly_em_snap or monthly_em_live or {}),
                run_now=run_now,
            )
        elif tab == "spread":
            _render_spread_finder_tab(spot, levels, regime, data, ticker=ticker, weekly_em=(weekly_em_snap or {}))

    # ── Auto-refresh (aligned to the wall clock, not to page-load time) ──
    if refresh_seconds > 0:
        from streamlit_autorefresh import st_autorefresh
        _now_ms = now_ny().timestamp() * 1000
        _period_ms = refresh_seconds * 1000
        _boundary_ms = _next_clock_boundary_ms(refresh_seconds, now_ny())
        _interval_ms = max(int(_boundary_ms - _now_ms), 1000)
        # Encode the TARGET boundary in the key so the component remounts every
        # tick. st_autorefresh uses a repeating window.setInterval and only
        # re-arms when the interval *value* changes; in steady state consecutive
        # ms-to-boundary values are ~equal, so without this the timer free-runs
        # and drifts off the clock by the render latency each cycle. A
        # per-boundary key forces a fresh, exactly-timed interval every tick.
        _bucket = _boundary_ms // _period_ms
        st_autorefresh(interval=_interval_ms,
                       key=f"auto_refresh_{refresh_seconds}_{_bucket}")

    # Mirror the resolved nav state into the URL so it survives a session reset
    # (Cloud process recycle / websocket reconnect). Runs last, after the tab
    # content has resolved active_ticker / _tab_last / _exp_last / recents /
    # export extras. Change-gated, so it's a no-op on the steady-state reruns.
    sync_to_url()


if __name__ == "__main__":
    main()
