"""
Spread Finder tab UI — weekly credit spread planning with forecast,
GEX context, and interactive strike maps.
Extracted from streamlit_app.py.
"""
from __future__ import annotations

from datetime import date as date_cls, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import COLORS
from models import GEXData
from ui_history import _is_weekly_freeze_day

from range_finder.gex_bridge import (
    GEXContext, extract_gex_context, save_gex_to_range_finder,
    adjust_spread_with_gex, regime_to_gex_flag,
)
from range_finder.data_collector import (
    fetch_spx_vix as rf_fetch_spx_vix, save_spx_vix as rf_save_spx_vix,
    fetch_fred_macro as rf_fetch_fred_macro, save_fred_macro as rf_save_fred_macro,
    build_event_flags as rf_build_event_flags,
    get_weekly_spx as rf_get_weekly_spx,
    fred_key_status as rf_fred_key_status,
    FRED_API_KEY as RF_FRED_API_KEY,
)
from range_finder.feature_builder import (
    build_features as rf_build_features,
    get_features as rf_get_features,
    get_feature_for_week as rf_get_feature_for_week,
)
from range_finder.har_model import (
    MODEL_SPECS as RF_MODEL_SPECS, PI_ALPHA as RF_PI_ALPHA,
    GEX_MIN_WEEKS_FOR_FIT as RF_GEX_MIN_WEEKS,
    feature_has_enough_data as rf_feature_has_enough_data,
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


# ─────────────────────────────────────────────────────────────────────────────
# Spread Finder Tab — Weekly credit spread placement powered by HAR model + GEX
# ─────────────────────────────────────────────────────────────────────────────

from theme import SF_BG, SF_BULL, SF_BEAR, SF_NEUT, SF_WARN, SF_CARD


@st.cache_resource
def _get_rf_conn():
    """Get or create the range finder Postgres connection."""
    from range_finder.db import get_connection, init_all_tables
    conn = get_connection()
    init_all_tables(conn)
    return conn


def _spread_finder_target_friday(ref_date: "date_cls | None" = None) -> "date_cls":
    """Return the calendar Friday of the week the Spread Finder is planning for.

    On Mon-Thu we're inside a live trading week — traders entering new
    credit spreads want *this* week's Friday (the one that's 0-4 days
    away).  On Fri-Sun the current week is effectively done, so we roll
    forward to next Monday's week and pick its Friday.  The same rule is
    applied in ``_render_spread_finder_tab`` when deriving ``week_start``
    so both stay in sync.
    """
    today = ref_date or date_cls.today()
    wd = today.weekday()
    if wd <= 3:  # Mon-Thu → this week's Monday
        monday = today - timedelta(days=wd)
    else:        # Fri-Sun → next Monday
        monday = today + timedelta(days=(7 - wd))
    return monday + timedelta(days=4)


def find_spread_finder_friday_exp(
    avail: "list[str]",
    ref_date: "date_cls | None" = None,
) -> "str | None":
    """Return the Tradier-listed expiration that matches the target Friday.

    Prefers an exact ISO-date match against ``avail``; falls back to the
    nearest listed expiration within 3 calendar days of the target (handles
    holiday-shifted weeklies such as Good Friday).  Returns ``None`` when
    no candidate is available.
    """
    target = _spread_finder_target_friday(ref_date)
    target_iso = target.strftime("%Y-%m-%d")
    if target_iso in avail:
        return target_iso

    window_start = (target - timedelta(days=3)).strftime("%Y-%m-%d")
    window_end = (target + timedelta(days=3)).strftime("%Y-%m-%d")
    candidates = [e for e in avail if window_start <= e <= window_end]
    if not candidates:
        return None
    candidates.sort(key=lambda e: abs((date_cls.fromisoformat(e) - target).days))
    return candidates[0]


def _build_chain_quotes_for_spreads(
    data: GEXData,
    ticker: str,
    ref_date: "date_cls | None" = None,
) -> tuple[dict, str | None]:
    """Build a strike→{call_bid, call_ask, put_bid, put_ask} lookup from the
    Friday chain that matches the Spread Finder's planned week.

    The target expiration is anchored to *the week the spread finder is
    forecasting* (see ``_spread_finder_target_friday``), not to "whichever
    expiration the user happened to pick in the sidebar".  Before this was
    added, a user who had ``0DTE`` or ``Tomorrow`` selected would see the
    spread finder silently fall back to today's chain — producing $0.00
    credits for far-OTM weekly strikes because it was pricing 0-DTE puts
    instead of Friday weeklies.  The pre-fetch in ``fetch_all_data`` makes
    sure the right chain is always in ``data.chain_cache`` regardless of
    sidebar state, and this function just looks up that exact Friday.

    Returns (quotes_dict, selected_expiration_str_or_None).  When the
    correct Friday isn't available we return empty so ``build_spread_side``
    falls back cleanly to its BSM estimator (the UI caption tells the user
    we're on BSM rather than market quotes).
    """
    if not data.chain_cache:
        return {}, None

    # Resolve the expiration we SHOULD be looking at. Prefer the full
    # expiration universe from data.avail so holiday-shifted Fridays can
    # still match; fall back to whatever's already in the chain cache.
    avail = list(getattr(data, "avail", None) or [])
    if not avail:
        avail = sorted({exp for (t, exp) in data.chain_cache if t == ticker})

    target_exp = find_spread_finder_friday_exp(avail, ref_date=ref_date)
    if target_exp is None:
        return {}, None

    entry = data.chain_cache.get((ticker, target_exp))
    if not entry or entry.get("status") != "ok":
        # The right Friday isn't cached — don't silently substitute another
        # expiration (that's exactly how we used to end up pricing weekly
        # spreads off today's 0DTE chain).  Let the caller fall back to BSM.
        return {}, None

    quotes: dict = {}  # strike -> {call_bid, call_ask, put_bid, put_ask}

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


def _build_spread_finder_excel(
    *,
    forecast: dict,
    plan,
    spread_tiers: list,
    gex_ctx,
    gex_adj: dict,
    metrics: dict,
    spx_ref: float,
    vix: float,
    ticker: str,
    week_start: str,
    chain_exp: str | None,
) -> bytes:
    """Build an .xlsx workbook of the current spread-finder forecast and tiers.

    Only call this after the forecast/plan/tiers have actually been generated
    (the render function already gates on that via an early return when no
    model is loaded), so every field referenced here is guaranteed populated.
    """
    from io import BytesIO

    # ── Summary sheet ──
    summary_rows = [
        ("Ticker",                    ticker),
        ("Week start (Mon)",          week_start),
        ("Chain expiration",          chain_exp or "n/a"),
        ("Generated at",              getattr(plan, "generated_at", "")),
        ("SPX reference close",       round(spx_ref, 2)),
        ("VIX level",                 round(vix, 2)),
        ("", ""),
        ("Point estimate %",          round(forecast["point_pct"] * 100, 4)),
        (f"PI upper % ({forecast['confidence_level']}% CI)",
                                      round(forecast["upper_pct"] * 100, 4)),
        (f"PI lower % ({forecast['confidence_level']}% CI)",
                                      round(forecast["lower_pct"] * 100, 4)),
        ("VIX-implied weekly %",      round(forecast["vix_implied_pct"] * 100, 4)),
        ("Model vs VIX %",            round(forecast["model_vs_vix"] * 100, 4)),
        ("", ""),
        ("Effective range %",         round(plan.effective_range_pct * 100, 4)),
        ("Effective upper px",        round(plan.effective_upper_px, 2)),
        ("Effective lower px",        round(plan.effective_lower_px, 2)),
        ("Buffer %",                  round(plan.buffer_pct * 100, 4)),
        ("Buffer pts",                round(plan.buffer_pts, 2)),
        ("Buffer reason",             plan.buffer_reason),
        ("Recommended wing width",    plan.recommended_width),
        ("", ""),
        ("GEX regime",                gex_ctx.gamma_regime),
        ("GEX flag",                  gex_adj.get("gex_regime_flag")),
        ("Zero gamma",                round(gex_ctx.zero_gamma, 2)),
        ("Call wall",                 round(gex_ctx.call_wall, 2)),
        ("Put wall",                  round(gex_ctx.put_wall, 2)),
        ("Net GEX ($)",               gex_ctx.net_gex),
        ("", ""),
        ("Model OOS R²",              round(metrics["oos_r2"], 6)),
        ("Model MAE %",               round(metrics["mae_pct"] * 100, 4)),
        ("", ""),
        ("Event: FOMC",               plan.has_fomc),
        ("Event: CPI",                plan.has_cpi),
        ("Event: NFP",                plan.has_nfp),
        ("Event: OPEX",               plan.has_opex),
        ("Event count",               plan.event_count),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Field", "Value"])

    # ── Spread tiers sheet: one row per (tier, side, wing width) ──
    tier_rows = []
    for tier in spread_tiers:
        for side in list(tier.call_spreads or []) + list(tier.put_spreads or []):
            tier_rows.append({
                "Tier":             tier.label,
                "Risk level":       tier.risk_level,
                "Range %":          round(tier.range_pct * 100, 4),
                "Side":             side.side,
                "Wing width":       side.wing_width,
                "Short strike":     side.short_strike,
                "Long strike":      side.long_strike,
                "Short % OTM":      round(side.short_pct * 100, 4),
                "Est credit":       round(side.estimated_credit, 2),
                "Credit source":    side.credit_source,
                "Max profit":       round(side.max_profit, 2),
                "Max loss":         round(side.max_loss, 2),
                "Breakeven":        round(side.breakeven, 2),
                "Credit ratio":     round(side.credit_ratio, 4),
                "Meets min credit": side.meets_min_credit,
                "Below min width":  getattr(side, "below_min_width", False),
            })
    tiers_df = pd.DataFrame(tier_rows)

    # ── Warnings sheet (only if the plan produced any) ──
    warnings_df = pd.DataFrame(
        {"Warning": list(getattr(plan, "warnings", []) or [])}
    )

    # ── Write workbook ──
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        if not tiers_df.empty:
            tiers_df.to_excel(writer, sheet_name="Spread Tiers", index=False)
        if not warnings_df.empty:
            warnings_df.to_excel(writer, sheet_name="Warnings", index=False)

        # Auto-size columns for readability
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            for col_cells in ws.columns:
                max_len = max(
                    (len(str(cell.value)) for cell in col_cells if cell.value is not None),
                    default=10,
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max_len + 2, 40)

    buffer.seek(0)
    return buffer.getvalue()


@st.fragment
def _render_spread_finder_tab(spot: float, levels: dict, regime: dict, data, ticker: str = "SPX", weekly_em: dict = None):
    """Render the Spread Finder tab — HAR model forecast + GEX-enhanced spread placement.

    Wrapped in @st.fragment so widget interactions inside the tab (horizon
    slider, model-spec dropdown, credit width, etc.) only rerun this tab
    instead of triggering a full-page rerun that rebuilds the GEX chart,
    the sidebar, and re-fetches weekly/monthly EM."""
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
    # capture the true daily-candle Open as the weekly reference. Rest of
    # the week uses that frozen value. Before Monday open (weekends), use
    # the live spot (Friday close).
    run_now = now_ny()
    is_freeze_day = _is_weekly_freeze_day(run_now)
    is_market_open = data.market_open

    mon_open_key = f"sf_monday_open_{ticker}"
    mon_vix_key = f"sf_monday_vix_{ticker}"
    mon_open_week_key = f"sf_monday_open_week_{ticker}"

    # Determine which week we're in (use ISO week number)
    current_week = run_now.isocalendar()[1]

    def _daily_open_today(symbol: str):
        """Today's daily-candle Open, or None if Yahoo hasn't published
        it yet (happens briefly after 9:30 while the first tick settles)."""
        try:
            hist = yf.Ticker(symbol).history(period="5d")
        except Exception:
            return None
        if hist is None or hist.empty or "Open" not in hist.columns:
            return None
        today = run_now.date()
        for ts, row in hist.iterrows():
            if hasattr(ts, "date") and ts.date() == today:
                op = row.get("Open")
                if op is not None and not (isinstance(op, float) and op != op):
                    return float(op)
        return None

    # Freeze Monday's open on the freeze day when market is open
    if is_freeze_day and is_market_open:
        stored_week = st.session_state.get(mon_open_week_key)
        if stored_week != current_week:
            # First market-hours refresh on the freeze day — lock the
            # TRUE daily-candle Open, not whatever tick `spot` happens to
            # land on while this refresh is running. Fall back to spot
            # only if yfinance hasn't returned today's bar yet.
            spx_daily_open = _daily_open_today("^SPX")
            if spx_daily_open is not None:
                frozen_spot = round(spx_daily_open / 10.0, 2) if ticker == "XSP" else round(spx_daily_open, 2)
            else:
                frozen_spot = round(spot, 2)

            vix_daily_open = _daily_open_today("^VIX")
            frozen_vix_val = round(vix_daily_open, 2) if vix_daily_open is not None else live_vix

            st.session_state[mon_open_key] = frozen_spot
            st.session_state[mon_vix_key] = frozen_vix_val
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
    # Also evict the ticker we're *leaving*'s cached HAR fit so the
    # incoming ticker lands on a clean load-from-Postgres path. Without
    # this, switching SPX→XSP→(GEX tab)→SPX could leave the outgoing
    # ticker's `_mdl_result_{ticker}` / `_mdl_name_{ticker}` session
    # entries in place; the name check at the bottom of this function
    # then sees cached_name == current dropdown choice and skips the
    # reload, freezing the tab on the last-displayed model.
    ref_key = f"sf_ref_price_{ticker}"
    vix_key = f"sf_vix_level_{ticker}"
    prev_ticker = st.session_state.get("_sf_prev_ticker")
    if prev_ticker != ticker:
        st.session_state[ref_key] = default_ref
        st.session_state[vix_key] = default_vix
        if prev_ticker:
            for _suffix in ("sf_model_result_", "sf_model_features_",
                            "sf_model_metrics_", "sf_model_name_"):
                st.session_state.pop(f"{_suffix}{prev_ticker}", None)
        st.session_state["_sf_prev_ticker"] = ticker

    # Also update the defaults on first render if not yet set
    if ref_key not in st.session_state:
        st.session_state[ref_key] = default_ref
    if vix_key not in st.session_state:
        st.session_state[vix_key] = default_vix

    st.markdown(f"### {ticker} Weekly Credit Spread Finder")
    st.caption("HAR regression range forecast + live GEX adjustment for optimal strike placement &nbsp;|&nbsp; 💾 Neon Postgres")

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
                # Distinguish "no key" from "FRED returned an error" — the
                # old message lumped them together and blamed the user for
                # a missing key whenever FRED itself had a 500. Also
                # surface the key status so you can eyeball whether
                # Streamlit actually picked up the secret.
                if not RF_FRED_API_KEY:
                    st.warning(
                        "FRED fetch skipped: FRED_API_KEY is not set. "
                        "Add it under Streamlit Cloud → Manage app → Secrets, "
                        "or export FRED_API_KEY in your local env."
                    )
                else:
                    st.warning(
                        f"FRED fetch failed — {e}. "
                        f"Key status: {rf_fred_key_status()}. "
                        "Existing macro data in the DB is still valid; "
                        "the rest of the pipeline will keep running. "
                        "Try again in a minute — FRED's API occasionally "
                        "returns 500s during their maintenance windows."
                    )

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
            gex_flag = save_gex_to_range_finder(gex_ctx, conn, ticker=ticker)
            regime_label = {1: "positive", 0: "neutral", -1: "negative"}.get(gex_flag, "unknown")
            st.success(f"GEX saved: regime={regime_label}, flag={gex_flag} (ticker={ticker})")
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
    # Session-state layout: we keep the fit result, features, metrics and
    # the *name* of the spec that produced them.  Tracking the name lets
    # the dropdown actually work — without it, switching M3→M4 would
    # silently keep using the old M3 fit (metric cards and strikes both)
    # because the rest of the code just reads `sf_model_result_{ticker}`
    # regardless of what the selectbox currently says.
    _mdl_result_key  = f"sf_model_result_{ticker}"
    _mdl_feat_key    = f"sf_model_features_{ticker}"
    _mdl_metrics_key = f"sf_model_metrics_{ticker}"
    _mdl_name_key    = f"sf_model_name_{ticker}"

    if do_forecast:
        # Weekly Setup fits every spec (same as the Monday cron) so a
        # user can switch the model dropdown afterwards without tripping
        # the "click Forecast to fit this spec" prompt. A standalone
        # Forecast click still fits only the currently-selected spec —
        # that's the fast path for iterating on one model.
        _specs_to_fit = list(RF_MODEL_SPECS.keys()) if do_weekly else [model_choice]
        _spinner_label = (
            f"4/4 — Fitting {len(_specs_to_fit)} specs..."
            if len(_specs_to_fit) > 1 else f"4/4 — Fitting {model_choice}..."
        )

        _selected_result = None
        _selected_avail  = None
        _selected_metrics = None

        with st.spinner(_spinner_label):
            for _spec in _specs_to_fit:
                try:
                    # Start from the static feature list but COPY it — we may
                    # append `gex_normalized` below and we don't want to mutate
                    # the module-level MODEL_SPECS dict.
                    feat_cols = list(RF_MODEL_SPECS[_spec])

                    # Mirror run_full_pipeline's dynamic GEX injection: when the
                    # user has built up enough weekly GEX history via the Save
                    # GEX button (>RF_GEX_MIN_WEEKS non-null rows of gex_normalized),
                    # fold it into M4_full as a real training feature.  This is
                    # the whole reason M4_full is called "full" — without this
                    # the UI-fitted M4 is just M3 + term structure + yield
                    # spread, ignoring all the GEX snapshots accumulated.
                    if _spec == "M4_full":
                        gex_col = "gex_normalized"
                        if rf_feature_has_enough_data(df_feat, gex_col):
                            if gex_col not in feat_cols:
                                feat_cols.append(gex_col)
                                # Only surface the "using N weeks of GEX" note
                                # when fitting the spec the user is looking at
                                # — otherwise the Weekly Setup run spams the
                                # UI with notes about every background spec.
                                if _spec == model_choice:
                                    st.caption(
                                        f"ℹ️ M4_full: using {int(df_feat[gex_col].notna().sum())} "
                                        f"weeks of stored GEX history as a training feature."
                                    )
                        else:
                            _weeks = int(df_feat[gex_col].notna().sum()) if gex_col in df_feat.columns else 0
                            if _spec == model_choice:
                                st.caption(
                                    f"ℹ️ M4_full: only {_weeks} weeks of GEX history — need >{RF_GEX_MIN_WEEKS} to "
                                    f"fold `gex_normalized` into the fit. Keep clicking **Save GEX** "
                                    f"each week; in the meantime M4 runs without the GEX feature."
                                )

                    avail_cols = [c for c in feat_cols if rf_feature_has_enough_data(df_feat, c)]
                    if len(avail_cols) < 2:
                        if _spec == model_choice:
                            st.warning(f"{_spec}: only {len(avail_cols)} usable features — skipped")
                        continue

                    X_train, X_test, y_train, y_test = rf_time_series_split(
                        df_feat, feature_cols=avail_cols
                    )
                    _result  = rf_fit_model(X_train, y_train, model_name=_spec)
                    _metrics = rf_evaluate_oos(_result, X_test, y_test, model_name=_spec)
                    rf_save_model(_result, avail_cols, _spec, _metrics, conn=conn, ticker=ticker)

                    if _spec == model_choice:
                        _selected_result  = _result
                        _selected_avail   = avail_cols
                        _selected_metrics = _metrics
                except Exception as e:
                    st.error(f"{_spec} fitting failed: {e}")

        # Summary line for the Weekly Setup path so the user can see which
        # specs landed in Postgres at a glance.
        if do_weekly:
            st.success(f"Fitted {len(_specs_to_fit)} specs (all saved to Postgres)")

        # Prime session state with the currently-selected spec's fit so
        # the rest of this render uses it without falling through to the
        # load-from-Postgres path (same behavior as the previous single-
        # spec code).
        if _selected_result is not None:
            st.session_state[_mdl_result_key]  = _selected_result
            st.session_state[_mdl_feat_key]    = _selected_avail
            st.session_state[_mdl_metrics_key] = _selected_metrics
            st.session_state[_mdl_name_key]    = model_choice
            st.success(
                f"Model fitted | {model_choice} | "
                f"OOS R² = {_selected_metrics['oos_r2']:.4f}"
            )

    # If the user toggled the model dropdown, the cached fit in session
    # state belongs to a different spec — evict it so the load block
    # below pulls the right saved fit for the newly-selected spec from
    # Postgres (or shows the "click Forecast" nudge if that spec has
    # never been fitted yet).
    _cached_mdl_name = st.session_state.get(_mdl_name_key)
    if _cached_mdl_name is not None and _cached_mdl_name != model_choice:
        for _k in (_mdl_result_key, _mdl_feat_key, _mdl_metrics_key, _mdl_name_key):
            st.session_state.pop(_k, None)

    # Try to load model from session or disk
    if _mdl_result_key not in st.session_state:
        try:
            payload = rf_load_model(model_choice, conn=conn, ticker=ticker)
            st.session_state[_mdl_result_key]  = payload["result"]
            st.session_state[_mdl_feat_key]    = payload["feature_cols"]
            st.session_state[_mdl_metrics_key] = payload["metrics"]
            st.session_state[_mdl_name_key]    = model_choice
        except FileNotFoundError:
            st.info(f"No saved fit for **{model_choice}** yet. Click **Forecast** to fit it for the first time.")
            _render_gex_context_panel(gex_ctx, spot)
            return
        except Exception as e:
            st.warning(f"Saved {model_choice} model incompatible: {e}. Click **Forecast** to refit.")
            _render_gex_context_panel(gex_ctx, spot)
            return

    result       = st.session_state[_mdl_result_key]
    feat_cols    = st.session_state[_mdl_feat_key]
    metrics      = st.session_state[_mdl_metrics_key]
    active_model = st.session_state.get(_mdl_name_key, model_choice)

    # ── Determine week start ──
    # Anchored to NY wall clock (run_now = now_ny() above) so that the
    # week_start convention here matches the Friday chain pre-fetch in
    # streamlit_app.fetch_all_data — otherwise a UTC-hosted server could
    # roll into "tomorrow" a few hours before NY does and end up looking
    # at a different expiration than the one the pre-fetch cached.
    #
    # Mon-Thu: plan THIS week's spreads (week_start = this Monday, expiring
    # this Friday).  Fri-Sun: this week is done, so roll forward to next
    # Monday's week.  This has to match _spread_finder_target_friday above.
    _wd = run_now.weekday()
    if _wd <= 3:                           # Mon-Thu
        monday_dt = run_now - timedelta(days=_wd)
    else:                                  # Fri-Sun
        monday_dt = run_now + timedelta(days=(7 - _wd))
    week_start = monday_dt.strftime("%Y-%m-%d")
    sf_ref_date = run_now.date()

    # ── Get feature row ──
    feature_row = rf_get_feature_for_week(conn, week_start)
    feature_row_is_stale = False
    if feature_row is None:
        feature_row = df_feat.iloc[-1]
        feature_row_is_stale = True

    # ── Regime-shift circuit breaker ──────────────────────────────────────
    # HAR features are lagged by one week — vix_close in feature_row is
    # last Friday's close. When IV spikes overnight (e.g., VIX 15 → 40 on
    # a news shock), the model's input features still reflect the pre-spike
    # world for a full week, so the forecast's PI is anchored to the wrong
    # vol regime and the Spread Finder will place strikes dangerously
    # close to spot. The weekly EM floor partially mitigates this (it
    # uses the live straddle, so shorts get pushed to the market-implied
    # range), but it doesn't stop the user from trusting the "model says
    # range will be 2%" read.
    #
    # Detect the shift by comparing the live VIX to the trailing VIX
    # already in the feature row. Ratio > 1.5 is the threshold — that's
    # roughly a 2σ move on the weekly VIX change distribution.
    _trailing_vix = None
    try:
        _trailing_vix_raw = feature_row.get("vix_close")
        if _trailing_vix_raw is not None:
            _trailing_vix = float(_trailing_vix_raw)
    except Exception:
        pass

    regime_shift = None
    if _trailing_vix and _trailing_vix > 0 and live_vix and live_vix > 0:
        _vix_ratio = live_vix / _trailing_vix
        if _vix_ratio >= 1.5:
            regime_shift = {
                "severity": "extreme" if _vix_ratio >= 2.0 else "elevated",
                "live_vix": live_vix,
                "trailing_vix": _trailing_vix,
                "ratio": _vix_ratio,
            }

    if regime_shift is not None:
        _sev_word = regime_shift["severity"]
        st.error(
            f"⚠️ **VIX regime shift detected ({_sev_word})** — "
            f"live VIX **{regime_shift['live_vix']:.1f}** vs trailing "
            f"feature VIX **{regime_shift['trailing_vix']:.1f}** "
            f"(**{regime_shift['ratio']:.2f}×**).\n\n"
            f"The HAR model's features lag by one week, so the forecast below "
            f"is anchored to the pre-spike vol regime. Short strikes sized "
            f"against this forecast are likely **too narrow**. Rely on the "
            f"live weekly EM floor (which reflects the current straddle) — "
            f"or, better, skip the trade until features catch up."
        )
    if feature_row_is_stale:
        # On Fri-Sun the spread finder forecasts NEXT week, so it asks for
        # next-Monday's feature row — which legitimately can't exist until
        # next week's market data lands. Telling the user to click Weekly
        # Setup in that case is actively wrong (the click can't materialize
        # a row for a future week). Distinguish the two cases honestly.
        _is_weekend_or_friday = run_now.weekday() >= 4
        if _is_weekend_or_friday:
            _fallback_idx = df_feat.index[-1]
            _fallback_label = (
                _fallback_idx.strftime("%Y-%m-%d")
                if hasattr(_fallback_idx, "strftime") else str(_fallback_idx)
            )
            st.info(
                f"ℹ️ Forecasting next week ({week_start}) using the most recent "
                f"completed week's features ({_fallback_label}). Next Monday's "
                "feature row will only exist once next week's market data lands; "
                "until then this fallback is the best available input. **Weekly "
                "Setup will not help right now** — it can't build a row for a "
                "week that hasn't started."
            )
        else:
            st.warning(
                "⚠️ This week's features have not been rebuilt yet — "
                "using the most recent available feature row. Forecast may be "
                "stale. Click **Weekly Setup** (or **Rebuild Features**) to refresh."
            )

    # ── Build forecast → plan → tiers from the latest GEX refresh ──
    # We intentionally DO NOT cache these on a session-state key any more.
    # Everything below is cheap arithmetic on top of the already-loaded HAR
    # model (the only expensive step — rf_fit_model — is gated behind the
    # "Forecast" button and cached separately via rf_load_model), so
    # recomputing on every page rerun lets the spread finder pick up fresh
    # chain bid/ask as soon as fetch_all_data refreshes data.chain_cache
    # (i.e. on auto-refresh, "Refresh Now", or any normal rerun — no need
    # to click "Forecast" again to get updated credits).
    #
    # Risk-tier switching stays snappy because the _risk_tier_fragment
    # below is wrapped in @st.fragment and only re-reads the spread_tiers
    # we stash in session_state — the outer recompute doesn't happen on
    # tier toggles.
    forecast = rf_forecast_next_week(
        result, feature_row, feat_cols,
        spx_close_input, alpha=RF_PI_ALPHA,
    )

    # Live chain quotes for the Spread Finder's planned Friday, read from
    # data.chain_cache on every rerun (fetch_all_data repopulates that
    # snapshot with fresh Tradier bid/ask on each refresh — see the
    # pre-fetch block in streamlit_app.fetch_all_data).
    chain_quotes, chain_exp = _build_chain_quotes_for_spreads(
        data, ticker, ref_date=sf_ref_date,
    )

    plan = rf_build_spread_plan(
        forecast    = forecast,
        feature_row = feature_row,
        week_start  = week_start,
        vix_level   = vix_input,
        ticker      = ticker,
        chain_quotes= chain_quotes,
    )

    spread_tiers = rf_build_spread_tiers(
        forecast     = forecast,
        plan         = plan,
        spx_ref      = spx_close_input,
        vix_level    = vix_input,
        chain_quotes = chain_quotes,
        ticker       = ticker,
        weekly_em    = weekly_em,
    )

    gex_adj = adjust_spread_with_gex(plan, gex_ctx)

    # =========================================================================
    # METRIC CARDS
    # =========================================================================

    # Five metric cards. The PI card holds both tier bounds (Lower ↔ Upper)
    # so all four risk tiers named in the spec — Lower PI, Point, Upper PI,
    # Effective — stay visible without cramping the row into 6 columns
    # (which clips labels on typical laptop widths).
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Point Estimate",
        f"{forecast['point_pct']*100:.2f}%",
        f"vs VIX: {forecast['model_vs_vix']*100:+.2f}%",
    )
    c2.metric(
        f"{forecast['confidence_level']}% PI Range",
        f"{forecast['lower_pct']*100:.2f}% — {forecast['upper_pct']*100:.2f}%",
        "aggressive ↔ moderate",
    )
    c3.metric(
        "Effective Range",
        f"{plan.effective_range_pct*100:.2f}%",
        f"conservative · buffer: +{plan.buffer_pct*100:.2f}%",
    )
    c4.metric(
        "GEX Regime",
        gex_ctx.gamma_regime.title(),
        f"flag: {gex_adj['gex_regime_flag']:+d}",
    )
    # Flag it when the dropdown's selection and the fit we're actually
    # rendering don't agree — shouldn't happen in normal flow since the
    # eviction block above reloads the saved fit for the new spec, but
    # if that load fell through (e.g. user is still on the first render
    # after toggling) we'd rather the card tell the truth than lie.
    _mdl_label = active_model if active_model == model_choice else f"{active_model} ⚠"
    c5.metric(
        f"OOS R² · {_mdl_label}",
        f"{metrics['oos_r2']:.4f}",
        f"MAE: {metrics['mae_pct']*100:.2f}%",
    )

    # ── Excel export ──
    # Reaching this point guarantees the forecast/plan/tiers have all been
    # generated (the no-model branch above returns early), so the workbook
    # is always populated with real data — never a blank template.
    _xlsx_col, _ = st.columns([1, 4])
    with _xlsx_col:
        try:
            _xlsx_bytes = _build_spread_finder_excel(
                forecast    = forecast,
                plan        = plan,
                spread_tiers= spread_tiers,
                gex_ctx     = gex_ctx,
                gex_adj     = gex_adj,
                metrics     = metrics,
                spx_ref     = spx_close_input,
                vix         = vix_input,
                ticker      = ticker,
                week_start  = week_start,
                chain_exp   = chain_exp,
            )
            st.download_button(
                label       = "Export to Excel",
                data        = _xlsx_bytes,
                file_name   = f"spread_finder_{ticker}_{week_start}.xlsx",
                mime        = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
                key         = f"_sf_xlsx_{ticker}_{week_start}",
            )
        except Exception as _xlsx_err:
            st.caption(f"Excel export unavailable: {_xlsx_err}")

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
            st.markdown("**Forecast Range**")
            _render_sf_range_gauge(
                _forecast, _plan, _spx_close_inp,
                tier_label=selected_tier.label,
                tier_range_pct=selected_tier.range_pct,
                tier_risk_level=selected_tier.risk_level,
            )

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

            # Per-tier weekly EM floor warnings — use the tier's stored
            # pre-floor strikes so the message matches the "Model Strikes
            # (before EM floor)" section exactly (chain-snap rounding aware).
            _em_upper = (_weekly_em or {}).get("upper_level", 0) or 0
            _em_lower = (_weekly_em or {}).get("lower_level", 0) or 0
            if _em_upper > 0 and _em_lower > 0:
                if selected_tier.model_call_short is not None:
                    all_warnings.append(
                        f"Weekly EM floor applied: call short widened from "
                        f"{selected_tier.model_call_short:,.0f} to "
                        f"{selected_tier.call_short:,.0f} (EM upper = {_em_upper:,.0f})"
                    )
                if selected_tier.model_put_short is not None:
                    all_warnings.append(
                        f"Weekly EM floor applied: put short widened from "
                        f"{selected_tier.model_put_short:,.0f} to "
                        f"{selected_tier.put_short:,.0f} (EM lower = {_em_lower:,.0f})"
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


def _render_sf_range_gauge(
    forecast: dict,
    plan: SpreadPlan,
    spx_ref: float,
    tier_label: str = None,
    tier_range_pct: float = None,
    tier_risk_level: str = None,
):
    """Horizontal number-line gauge of the weekly range forecast.

    The old view stacked four ascending bars (Lower PI, Point, Upper PI,
    Effective) which read like four independent predictions climbing in
    magnitude. They aren't — they're all positions on the same weekly
    range-% axis: Point is the model's central forecast, Lower/Upper PI
    are the 10th/90th percentiles of its predictive distribution, and
    Effective is Upper PI plus a fixed safety buffer. This view lays
    them out on one horizontal axis:

      - Amber band  = 80% prediction interval (Lower PI → Upper PI)
      - Red band    = buffer extension (Upper PI → Effective)
      - Bull dot    = Point Estimate (the center of the distribution)
      - Dashed tick = VIX-implied range for reference
      - White caret = the Risk Tier currently driving strike placement

    Because the forecast is fit on log(range), the back-transformed
    distribution is asymmetric — Point will usually sit below the
    midpoint of the PI band. That's statistically correct, not a bug,
    and the horizontal layout actually makes it visible (whereas the
    old four-bar chart hid it entirely).
    """
    import plotly.graph_objects as go

    lower_pct     = forecast["lower_pct"]         * 100
    point_pct     = forecast["point_pct"]         * 100
    upper_pct     = forecast["upper_pct"]         * 100
    effective_pct = plan.effective_range_pct      * 100
    vix_pct       = forecast["vix_implied_pct"]   * 100
    confidence    = forecast["confidence_level"]

    # Axis extends a touch past the biggest value on screen so the
    # right-most label doesn't collide with the plot edge.
    axis_max = max(effective_pct, vix_pct, point_pct) * 1.18 + 0.4

    _TIER_HIGHLIGHT = {
        "aggressive":   "#ff4b4b",
        "moderate":     "#ffa726",
        "conservative": "#66bb6a",
    }
    tier_color = _TIER_HIGHLIGHT.get((tier_risk_level or "").lower(), "#ffffff")

    fig = go.Figure()

    # 80% PI band (Lower → Upper)
    fig.add_shape(type="rect",
        x0=lower_pct, x1=upper_pct, y0=0.38, y1=0.62,
        fillcolor=SF_WARN, opacity=0.28, line_width=0, layer="below",
    )
    # Buffer extension (Upper → Effective)
    fig.add_shape(type="rect",
        x0=upper_pct, x1=effective_pct, y0=0.38, y1=0.62,
        fillcolor=SF_BEAR, opacity=0.22, line_width=0, layer="below",
    )

    # End-caps for the PI band and Effective marker
    for x, color in [
        (lower_pct,     SF_WARN),
        (upper_pct,     SF_WARN),
        (effective_pct, SF_BEAR),
    ]:
        fig.add_shape(type="line",
            x0=x, x1=x, y0=0.34, y1=0.66,
            line=dict(color=color, width=2),
        )

    # Bottom labels for the three bounds
    for x, label, color in [
        (lower_pct,     f"Lower PI<br><b>{lower_pct:.2f}%</b>",         SF_WARN),
        (upper_pct,     f"Upper PI<br><b>{upper_pct:.2f}%</b>",         SF_WARN),
        (effective_pct, f"Effective<br><b>{effective_pct:.2f}%</b>",    SF_BEAR),
    ]:
        fig.add_annotation(
            x=x, y=0.22, text=label, showarrow=False,
            font=dict(size=10, color=color), align="center",
        )

    # "80% PI" text centered in the amber band
    pi_mid = (lower_pct + upper_pct) / 2
    fig.add_annotation(
        x=pi_mid, y=0.50,
        text=f"<b>{confidence}% PI</b>",
        showarrow=False,
        font=dict(size=11, color="#0e1117"),
    )

    # Point Estimate marker
    fig.add_trace(go.Scatter(
        x=[point_pct], y=[0.5], mode="markers",
        marker=dict(
            size=18, color=SF_BULL, symbol="circle",
            line=dict(width=2, color=SF_BG),
        ),
        hovertemplate=f"Point Estimate: {point_pct:.2f}%<extra></extra>",
        showlegend=False,
    ))
    fig.add_annotation(
        x=point_pct, y=0.80,
        text=f"Point<br><b>{point_pct:.2f}%</b>",
        showarrow=False, font=dict(size=10, color=SF_BULL), align="center",
    )

    # VIX-implied range (dashed reference line, label tucked under axis
    # so it doesn't collide with the Point/Upper labels when VIX sits
    # near them — this is the common case on quiet weeks).
    fig.add_shape(type="line",
        x0=vix_pct, x1=vix_pct, y0=0.05, y1=0.95,
        line=dict(color=SF_NEUT, width=1, dash="dash"),
    )
    fig.add_annotation(
        x=vix_pct, y=0.02,
        text=f"VIX implied {vix_pct:.2f}%",
        showarrow=False, font=dict(size=9, color=SF_NEUT),
    )

    # Currently-selected Risk Tier — white caret above the band so the
    # user can see exactly which % is driving the strikes they're looking
    # at. Skipped when the caller didn't pass tier info.
    if tier_range_pct is not None:
        tier_pct = tier_range_pct * 100
        fig.add_shape(type="line",
            x0=tier_pct, x1=tier_pct, y0=0.30, y1=0.70,
            line=dict(color=tier_color, width=3),
        )
        fig.add_annotation(
            x=tier_pct, y=1.02,
            text=f"▼ <b>{tier_label or 'Selected'}</b>",
            showarrow=False, font=dict(size=11, color=tier_color),
            align="center",
        )

    fig.update_xaxes(
        range=[0, axis_max], showgrid=True, gridcolor="#1f2530",
        ticksuffix="%", zeroline=True, zerolinecolor="#333",
        title="Weekly range % (High − Low / Open)",
        title_font=dict(size=11, color="#9aa4b2"),
    )
    fig.update_yaxes(
        range=[-0.05, 1.15], visible=False, fixedrange=True,
    )

    fig.update_layout(
        plot_bgcolor=SF_BG, paper_bgcolor=SF_BG, font_color="#e0e0e0",
        showlegend=False,
        margin=dict(t=40, b=50, l=20, r=20),
        height=260, dragmode=False,
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
    """Swim-lane strike map grouped by semantic role.

    The old layout sorted every level (walls, shorts, spot, EM bounds,
    etc.) onto its own Y-row — clean for avoiding label overlap but
    forced the reader to mentally re-group "which of these is my trade
    vs structural GEX vs a forecast marker?" every time they scanned.

    This version keeps one shared price axis on X but splits the Y into
    three labeled lanes:

      TRADE   — Put Long / Put Short / Ref / Spot / Call Short / Call Long
      RANGE   — Effective bounds + EM bounds
      GEX     — Put Wall / Zero-Gamma / Call Wall

    Background shading:
      - tier-colored band between Put Short and Call Short (the "safe"
        window for the selected risk tier — where the spread profits)
      - translucent blue band across the weekly EM envelope

    Label collisions within a lane are avoided by alternating marker
    text positions top/bottom after sorting each lane's markers by
    price.
    """
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
    em_color = "#29b6f6"  # light blue

    # ── Lane layout ──
    # TRADE on top because the shorts are the decision the reader is
    # optimizing; GEX at the bottom as the structural backdrop.
    LANE_TRADE, LANE_RANGE, LANE_GEX = 2.0, 1.0, 0.0
    Y_BOT, Y_TOP = -0.6, 2.85

    fig = go.Figure()

    # ── Background bands (span all lanes) ──
    # The tier-colored "safe window" between the shorts — price that
    # stays in this window keeps the credit spread profitable.
    fig.add_shape(type="rect",
        x0=put_short, x1=call_short,
        y0=Y_BOT, y1=Y_TOP,
        fillcolor=tier_color, opacity=0.08, line_width=0, layer="below",
    )
    # Weekly EM envelope — provides a sanity check on where the straddle
    # thinks price can move, independent of the HAR forecast.
    if has_em:
        fig.add_shape(type="rect",
            x0=em_lower, x1=em_upper,
            y0=Y_BOT, y1=Y_TOP,
            fillcolor=em_color, opacity=0.05,
            line=dict(color=em_color, width=1, dash="dot"),
            layer="below",
        )

    # Lane dividers — faint so they separate without distracting.
    for y_div in (0.5, 1.5):
        fig.add_shape(type="line",
            xref="paper", x0=0, x1=1,
            y0=y_div, y1=y_div,
            line=dict(color="#262b36", width=1, dash="dot"),
            layer="below",
        )

    # ── Vertical reference lines ──
    fig.add_vline(x=spx_ref, line_dash="solid", line_color=COLORS["spot"],
                  line_width=2, opacity=0.35)
    fig.add_vline(x=gex_ctx.spot, line_dash="dash", line_color="#ffc107",
                  line_width=1.5, opacity=0.5)

    # ── Wing connectors within the TRADE lane ──
    # Dotted line from short to long so the wing width reads visually.
    fig.add_shape(type="line",
        x0=put_short, x1=put_long,
        y0=LANE_TRADE, y1=LANE_TRADE,
        line=dict(color=SF_BEAR, width=2, dash="dot"),
    )
    fig.add_shape(type="line",
        x0=call_short, x1=call_long,
        y0=LANE_TRADE, y1=LANE_TRADE,
        line=dict(color=SF_BEAR, width=2, dash="dot"),
    )

    # ── Lane content ──
    trade_lane = [
        (put_long,      "Put Long",        SF_BEAR,             "triangle-left",   9),
        (put_short,     "Put Short",       tier_color,          "diamond",        12),
        (spx_ref,       f"{ticker} Ref",   COLORS["spot"],      "star",           14),
        (gex_ctx.spot,  "Spot",            "#ffc107",           "circle",         11),
        (call_short,    "Call Short",      tier_color,          "diamond",        12),
        (call_long,     "Call Long",       SF_BEAR,             "triangle-right",  9),
    ]
    range_lane = [
        (plan.effective_lower_px, "Eff Lo", SF_WARN, "triangle-up", 9),
        (plan.effective_upper_px, "Eff Hi", SF_WARN, "triangle-up", 9),
    ]
    if has_em:
        range_lane.extend([
            (em_lower, "EM Lo", em_color, "line-ew", 11),
            (em_upper, "EM Hi", em_color, "line-ew", 11),
        ])
    gex_lane = [
        (gex_ctx.put_wall,   "Put Wall",  COLORS["put_wall"],   "square", 11),
        (gex_ctx.zero_gamma, "Zero-G",    COLORS["zero_gamma"], "x",      12),
        (gex_ctx.call_wall,  "Call Wall", COLORS["call_wall"],  "square", 11),
    ]

    def _plot_lane(markers, y: float):
        """Place markers on a single lane and alternate labels above/below
        so adjacent strikes don't stack their text."""
        markers_sorted = sorted(markers, key=lambda m: m[0])
        for idx, (x, label, color, symbol, size) in enumerate(markers_sorted):
            text_pos = "top center" if idx % 2 == 0 else "bottom center"
            fig.add_trace(go.Scatter(
                x=[x], y=[y], mode="markers+text",
                marker=dict(color=color, size=size, symbol=symbol,
                            line=dict(width=1, color="#ffffff")),
                text=[f"{label} {x:,.0f}"],
                textposition=text_pos,
                textfont=dict(size=10, color=color),
                showlegend=False,
                hovertemplate=f"{label}: {x:,.0f}<extra></extra>",
            ))

    _plot_lane(trade_lane, LANE_TRADE)
    _plot_lane(range_lane, LANE_RANGE)
    _plot_lane(gex_lane,   LANE_GEX)

    # ── Lane labels (left gutter, tied to the plot area) ──
    for text, y in (("TRADE", LANE_TRADE), ("RANGE", LANE_RANGE), ("GEX", LANE_GEX)):
        fig.add_annotation(
            xref="paper", x=0.003, y=y,
            text=f"<b>{text}</b>",
            showarrow=False,
            font=dict(size=10, color="#6b7380"),
            xanchor="left",
        )

    # ── Axis ──
    all_prices = [m[0] for m in (trade_lane + range_lane + gex_lane)]
    margin_px = (max(all_prices) - min(all_prices)) * 0.12

    fig.update_xaxes(
        range=[min(all_prices) - margin_px, max(all_prices) + margin_px],
        title_text=f"{ticker} Price Level",
        title_font=dict(size=11, color="#9aa4b2"),
        showgrid=True, gridcolor="#1f2530",
    )
    fig.update_yaxes(
        range=[Y_BOT, Y_TOP],
        visible=False, fixedrange=True,
    )

    fig.update_layout(
        plot_bgcolor=SF_BG, paper_bgcolor=SF_BG, font_color="#e0e0e0",
        showlegend=False,
        margin=dict(t=10, b=40, l=10, r=10),
        height=320, dragmode=False,
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

