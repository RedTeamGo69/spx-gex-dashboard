"""Chart-building helpers extracted from streamlit_app.py."""

import numpy as np
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

    # Charm overlay — per-strike $-delta drift per *trading hour* on a
    # secondary X axis. Per-strike net_charm is the dealer option-book
    # delta drift / year under SqueezeMetrics convention (dealer long
    # calls, short puts):
    #
    #   CEX > 0  →  book gains delta as time passes  →  dealer SELLS to
    #               re-hedge to delta-neutral  →  REPELLING (red)
    #   CEX < 0  →  book loses delta as time passes  →  dealer BUYS to
    #               re-hedge  →  SUPPORTIVE / pinning (green)
    #
    # Visual layout choices:
    #   * The X2 axis is forced symmetric around zero so direction is
    #     readable even when the entire chain is one-signed (which is
    #     structurally common — OTM-heavy SPX chains contribute negative
    #     CEX on both call and put legs).
    #   * Translucent green / red fills to zero make the sign read in a
    #     single glance without needing to inspect axis ticks.
    #   * The amber outline line stays on top so micro-shape (peaks,
    #     zero-crossings near walls) remains visible above the fills.
    x2_range = None
    if "net_charm" in df.columns and len(df) > 0:
        charm_per_hr = df["net_charm"].values / float(TRADING_HOURS_PER_YEAR)
        charm_per_hr = np.where(np.isfinite(charm_per_hr), charm_per_hr, 0.0)

        max_abs_charm = float(np.max(np.abs(charm_per_hr)))
        if max_abs_charm > 0:
            x2_range = [-max_abs_charm * 1.05, max_abs_charm * 1.05]

        # Sign-clipped fill series. Using 0 (not NaN) so adjacent strikes
        # of opposite sign meet cleanly at the zero axis instead of
        # leaving a visual gap at the crossing.
        supportive = np.minimum(charm_per_hr, 0.0)  # green fill
        repelling  = np.maximum(charm_per_hr, 0.0)  # red fill

        fig.add_trace(go.Scatter(
            y=strikes, x=supportive, mode="lines",
            line=dict(color=COLORS["bar_green"], width=0.4),
            fill="tozerox",
            fillcolor="rgba(0, 200, 83, 0.18)",
            xaxis="x2",
            name="Supportive",
            hoverinfo="skip",
            showlegend=False,
        ))
        fig.add_trace(go.Scatter(
            y=strikes, x=repelling, mode="lines",
            line=dict(color=COLORS["bar_red"], width=0.4),
            fill="tozerox",
            fillcolor="rgba(255, 23, 68, 0.18)",
            xaxis="x2",
            name="Repelling",
            hoverinfo="skip",
            showlegend=False,
        ))

        # Direction labels for hover. Built once as customdata so users
        # can hover on any strike and see the corrected sign meaning
        # without having to memorize the convention.
        direction_labels = np.where(
            charm_per_hr > 0, "Repelling (dealer sells)",
            np.where(charm_per_hr < 0, "Supportive (dealer buys)", "Neutral"),
        )
        fig.add_trace(go.Scatter(
            y=strikes, x=charm_per_hr, mode="lines",
            line=dict(color=COLORS["charm_line"], width=1.5),
            xaxis="x2",
            name="Charm/hr",
            customdata=direction_labels,
            hovertemplate=(
                "Strike: $%{y:.0f}"
                "<br>Charm/hr: %{x:,.0f} $Δ"
                "<br>%{customdata}"
                "<extra></extra>"
            ),
            opacity=0.9,
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
            title=dict(
                text="Charm/hr ($Δ drift) — ← supportive (dealer buys) | repelling (dealer sells) →",
                font=dict(color=COLORS["charm_line"], size=9),
            ),
            overlaying="x", side="top",
            showgrid=False,
            range=x2_range,
            zeroline=True, zerolinecolor=COLORS["charm_line"], zerolinewidth=1.2,
            tickfont=dict(color=COLORS["charm_line"], size=8),
        ),
        yaxis=dict(title="Strike", gridcolor=COLORS["grid_minor"], tickfont_size=8),
        showlegend=False, height=2000, dragmode=False,
    )
    return fig
