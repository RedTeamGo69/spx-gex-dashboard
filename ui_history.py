"""
History, EM tracking, IV surface, pin detection, level crossings,
and EM snapshot management UI components.
Extracted from streamlit_app.py.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from phase1.market_clock import now_ny
from phase1.gex_history import (
    get_daily_summary, get_history,
    get_backend as get_history_backend, check_db_connection,
    save_em_snapshot, get_em_snapshot,
)

from theme import COLORS
from ui_sidebar import _fmt_gex_short


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
    # Lazy import to avoid circular dependency:
    # streamlit_app -> ui_spread_finder -> ui_history -> streamlit_app
    from streamlit_app import get_credentials, fetch_multi_tf_gex

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


def _render_iv_surface(hm_iv, hm_gex, spot):
    """C4: Render IV surface heatmap (strike × expiration)."""
    if hm_iv is None or hm_iv.empty:
        st.info("IV surface data not available. Requires multiple expirations.")
        return

    @st.fragment
    def _iv_fragment():
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

    _iv_fragment()


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

    # Try to restore from Postgres whenever session state is empty — this must
    # run regardless of market-open status, otherwise fresh sessions outside
    # 9:30–16:00 ET never pick up the frozen weekly/monthly snap that the
    # scheduled cron already persisted, and downstream code falls back to the
    # live (drifting) EM.
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

    if not is_market_open:
        # Outside market hours we never capture anything new — just return
        # whatever we restored (or None if nothing was ever saved).
        return st.session_state.get(sk_snap)

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


