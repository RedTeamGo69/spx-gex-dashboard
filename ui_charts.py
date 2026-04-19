"""Chart-building helpers extracted from streamlit_app.py."""

import plotly.graph_objects as go

from theme import COLORS
from phase1.config import TRADING_HOURS_PER_YEAR


def build_gex_bar_chart(gex_df, levels, spot, em_analysis,
                         weekly_em=None, monthly_em=None, show_daily_em=True):
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

    # Charm overlay — per-strike $-delta drift per *trading hour* traced
    # on a secondary X axis at the top of the chart. Charm stored in
    # gex_df is annualized against trading-time T (252 × 6.5 hours), so
    # dividing by TRADING_HOURS_PER_YEAR yields per-hour pressure that
    # mirrors how 0DTE dealer hedging actually accelerates through the
    # session. Positive values sit to the right of zero on xaxis2 (dealer
    # book gaining delta as time passes → mechanical buying pressure at
    # that strike); negative values sit to the left (mechanical selling).
    # A single accent color is used instead of green/red segmenting so it
    # doesn't compete visually with the GEX bars underneath — the line's
    # position relative to the zero line carries the direction.
    if "net_charm" in df.columns:
        charm_per_hr = df["net_charm"].values / float(TRADING_HOURS_PER_YEAR)
        fig.add_trace(go.Scatter(
            y=strikes, x=charm_per_hr,
            mode="lines",
            line=dict(color=COLORS["charm_line"], width=1.5),
            xaxis="x2",
            name="Charm/hr",
            hovertemplate="Strike: $%{y:.0f}<br>Charm/hr: %{x:,.0f}<extra></extra>",
            opacity=0.85,
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

    # EM levels (0DTE — purple dotted).  Suppressed when the caller says
    # the selected expiration view has extended beyond today — the 0DTE
    # straddle's ±EM is only meaningful for today's session.
    em = em_analysis.get("expected_move", {})
    if show_daily_em and em.get("upper_level"):
        for val, label in [(em["upper_level"], "EM+"), (em["lower_level"], "EM−")]:
            fig.add_hline(y=val, line_color=COLORS["em_level"], line_dash="dot", line_width=1.2,
                           annotation_text=f"{label} ${val:.0f}",
                           annotation_font_color=COLORS["em_level"], annotation_font_size=8,
                           annotation_position="bottom right")

    # Weekly EM levels — amber dashed lines + translucent shaded band so the
    # zone the market is pricing for the week is visually obvious (not just
    # two thin lines).  Band is frozen at Monday's open price, so it stays
    # put as spot moves through the week.
    w_em = weekly_em or {}
    if w_em.get("upper_level") and w_em.get("lower_level"):
        fig.add_hrect(
            y0=w_em["lower_level"], y1=w_em["upper_level"],
            fillcolor=COLORS["em_weekly"], opacity=0.06,
            line_width=0, layer="below",
        )
        for val, label in [(w_em["upper_level"], "wEM+"), (w_em["lower_level"], "wEM−")]:
            fig.add_hline(y=val, line_color=COLORS["em_weekly"], line_dash="dash", line_width=1,
                           annotation_text=f"{label} ${val:.0f}",
                           annotation_font_color=COLORS["em_weekly"], annotation_font_size=7,
                           annotation_position="top right")

    # OpEx-cycle EM levels (cyan longdash + band) — frozen on the first
    # trading day of the cycle (Monday after each 3rd-Friday expiration),
    # using the next 3rd Friday's straddle.
    m_em = monthly_em or {}
    if m_em.get("upper_level") and m_em.get("lower_level"):
        fig.add_hrect(
            y0=m_em["lower_level"], y1=m_em["upper_level"],
            fillcolor=COLORS["em_monthly"], opacity=0.04,
            line_width=0, layer="below",
        )
        for val, label in [(m_em["upper_level"], "OpEx+"), (m_em["lower_level"], "OpEx−")]:
            fig.add_hline(y=val, line_color=COLORS["em_monthly"], line_dash="longdash", line_width=1,
                           annotation_text=f"{label} ${val:.0f}",
                           annotation_font_color=COLORS["em_monthly"], annotation_font_size=7,
                           annotation_position="top right")

    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"], plot_bgcolor=COLORS["bg_primary"],
        font_color="white", font_size=10,
        margin=dict(l=80, r=10, t=55, b=35),
        title="Strike-by-Strike Net GEX Proxy",
        xaxis=dict(title="Net GEX proxy", gridcolor=COLORS["grid_major"], zerolinecolor=COLORS["zeroline"]),
        xaxis2=dict(
            title=dict(text="Charm/hr ($Δ drift)", font=dict(color=COLORS["charm_line"], size=9)),
            overlaying="x", side="top",
            showgrid=False,
            zeroline=True, zerolinecolor=COLORS["charm_line"], zerolinewidth=0.5,
            tickfont=dict(color=COLORS["charm_line"], size=8),
        ),
        yaxis=dict(title="Strike", gridcolor=COLORS["grid_minor"], tickfont_size=8),
        showlegend=False, height=2000, dragmode=False,
    )
    return fig
