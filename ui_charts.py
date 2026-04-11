"""Chart-building helpers extracted from streamlit_app.py."""

import plotly.graph_objects as go

from theme import COLORS


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

    # OpEx-cycle EM levels (cyan longdash) — frozen on the Monday after each
    # standard 3rd-Friday expiration, using the next 3rd Friday's straddle.
    m_em = monthly_em or {}
    if m_em.get("upper_level"):
        for val, label in [(m_em["upper_level"], "OpEx+"), (m_em["lower_level"], "OpEx−")]:
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
        showlegend=False, height=2000, dragmode=False,
    )
    return fig
