"""TradingView export section for the Strike GEX tab.

Thin Streamlit glue over phase1.tv_export: packs the levels already
computed for the page into a GL1 paste string, resolves the HAR PI band
from persisted state only (never a live fetch or refit), and renders the
copy / one-time-setup UI. Pine Script cannot pull external data, so the
string is the transport — see phase1/tv_export.py for the format.
"""

from __future__ import annotations

import streamlit as st

from phase1.tv_export import (
    PINE_INDICATOR_SOURCE,
    build_tv_levels,
    build_tv_string,
    missing_tokens,
)

_SETUP_INSTRUCTIONS = """
**One-time setup (~1 minute)**

1. On TradingView, open any chart and open the **Pine Editor** (bottom panel).
2. Delete the starter code, paste the script below, and click **Add to chart**.
3. Open the indicator's **Settings → Inputs** and paste the level string from
   the *Level string* tab into **"Gamma Lens level string"**.

After that, updating levels is only step 3 — copy a fresh string here, paste
it into the indicator's settings. The script itself never changes. The
indicator warns on-chart if the string's ticker doesn't match the chart
symbol or if the levels look stale.
"""


@st.cache_data(ttl=600, show_spinner=False)
def _cached_tv_har_pi(week_start: str, model_choice: str,
                      ticker: str) -> "tuple[float, float] | None":
    """(pi_lower_px, pi_upper_px) for the planning week, or None.

    Persisted-state only — the saved HAR fit plus the cron-captured Monday
    anchor (prior weekly close as fallback), mirroring the persisted path of
    ui_spread_finder._collect_week_bands_for_ticker minus plan/tiers/chain,
    with the same conformal hook the Spread Finder applies so both surfaces
    quote identical PI prices. Returns None on ANY failure: the pi token is
    then simply omitted from the export string (graceful degrade for
    tickers without a saved fit / weekly setup).
    """
    try:
        import pandas as pd

        from phase1.ticker_config import (feature_source_ticker,
                                          is_spread_finder_eligible)
        from range_finder.conformal import maybe_apply_conformal
        from range_finder.gex_policy import uses_disabled_gex_feature
        from range_finder.har_model import PI_ALPHA, forecast_next_week
        from ui_spread_finder import (_cached_prior_week_close,
                                      _cached_rf_get_features,
                                      _cached_rf_load_model,
                                      _cached_side_share_q,
                                      _cached_weekly_setup, _get_rf_conn)

        if not is_spread_finder_eligible(ticker):
            return None

        conn = _get_rf_conn()
        src = feature_source_ticker(ticker)
        df_feat = _cached_rf_get_features(conn, ticker=src)
        if df_feat.empty:
            return None

        # A scaled mini (XSP→SPX) may have its fit saved under either key.
        try:
            payload = _cached_rf_load_model(model_choice, ticker)
        except Exception:
            if src == ticker:
                return None
            payload = _cached_rf_load_model(model_choice, src)

        # A pre-mitigation M4 fit may still be present in Postgres. Omitting
        # the HAR band is safer than exporting a recommendation influenced by
        # the disabled GEX calibration.
        if uses_disabled_gex_feature(payload["feature_cols"]):
            return None

        wk_ts = pd.Timestamp(week_start)
        feature_row = (df_feat.loc[wk_ts] if wk_ts in df_feat.index
                       else df_feat.iloc[-1])

        ref = None
        setup = _cached_weekly_setup(conn, week_start, ticker)
        if setup and setup[0]:
            ref = float(setup[0])
        if ref is None:
            ref = _cached_prior_week_close(week_start, ticker)
        if ref is None:
            return None

        forecast = forecast_next_week(
            payload["result"], feature_row, payload["feature_cols"],
            ref, alpha=PI_ALPHA, side_share_q=_cached_side_share_q(ticker),
        )
        forecast = maybe_apply_conformal(forecast, conn, ticker, model_choice)
        lo, hi = forecast.get("pi_lower_px"), forecast.get("pi_upper_px")
        if lo is None or hi is None:
            return None
        return (float(lo), float(hi))
    except Exception:
        return None


def _resolve_har_pi(ticker: str, run_now) -> "tuple[float, float] | None":
    """Planning week + active model spec → cached PI band (or None).

    Session-state and week resolution stay OUT of the cached function so the
    cache key is exactly (week_start, model_choice, ticker).
    """
    try:
        from datetime import timedelta

        from ui_spread_finder import (_default_model_for_ticker,
                                      _spread_finder_target_friday)

        # Planning week = the Spread Finder's target Friday, back to its
        # Monday. Mon-Thu that's this week; Fri-Sun it rolls forward. Reading
        # it off the Spread Finder's helper (rather than a second copy of the
        # rule) is what keeps the export and the tab on the same week.
        week_start = (_spread_finder_target_friday(run_now.date())
                      - timedelta(days=4)).strftime("%Y-%m-%d")
        model_choice = (st.session_state.get(f"sf_model_choice_{ticker}")
                        or _default_model_for_ticker(ticker))
        return _cached_tv_har_pi(week_start, model_choice, ticker)
    except Exception:
        return None


def render_tv_export_section(*, ticker: str, spot, levels: dict,
                             gex_df_empty: bool, daily_em: dict | None,
                             weekly_em: dict | None, monthly_em: dict | None,
                             run_now) -> None:
    """Expander under the Strike GEX chart with the GL1 string + setup kit."""
    if gex_df_empty:
        # Empty-chain levels are all spot fallbacks — exporting them would
        # paint fake walls on the user's chart.
        return

    date_str = run_now.strftime("%Y-%m-%d")
    tv = build_tv_levels(
        ticker=ticker, date=date_str, spot=spot, levels=levels,
        daily_em=daily_em, weekly_em=weekly_em, monthly_em=monthly_em,
        har_pi=_resolve_har_pi(ticker, run_now),
    )
    paste_str = build_tv_string(tv)
    omitted = missing_tokens(tv)

    with st.expander("📡 TradingView export", expanded=False):
        tab_str, tab_setup = st.tabs(["Level string", "One-time indicator setup"])
        with tab_str:
            st.code(paste_str, language=None)
            caption = (f"Levels as of {date_str} · paste into the GL Levels "
                       f"indicator settings on your {tv.ticker} chart.")
            if omitted:
                caption += f" Not available right now: {', '.join(omitted)}."
            st.caption(caption)
        with tab_setup:
            st.markdown(_SETUP_INSTRUCTIONS)
            st.code(PINE_INDICATOR_SOURCE, language=None)
            st.download_button(
                "Download gamma_lens_levels.pine",
                data=PINE_INDICATOR_SOURCE,
                file_name="gamma_lens_levels.pine",
                mime="text/plain",
                key=f"tv_pine_dl_{ticker}",
                use_container_width=True,
            )
