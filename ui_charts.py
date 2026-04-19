"""Chart-building helpers extracted from streamlit_app.py."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from theme import COLORS
from phase1.config import TRADING_HOURS_PER_YEAR


def build_gex_bar_chart(gex_df, levels, spot, em_analysis,
                         weekly_em=None, monthly_em=None, show_daily_em=True):
    """
    Two-panel GEX + Charm chart.

    Left panel (~78% width):  strike-by-strike net GEX horizontal bars
    Right panel (~22% width): per-strike charm ($Δ drift per trading
                              hour) as a line with sign-coded fills

    Both panels share the strike Y axis, so the reader can scan
    horizontally across any strike level and read GEX (left) and charm
    (right) with a single visual sweep. Each panel owns its own X axis
    and zero, which fixes the zero-misalignment that plagued the earlier
    single-chart dual-X-axis overlay — the two zeros no longer need to
    be geometrically aligned because they're in separate plots.

    Reference lines (spot, zero-gamma, walls, EM levels) are duplicated
    across both panels so the level context carries through; annotation
    labels are only placed on the left panel to avoid double-labeling.
    """
    df = gex_df.copy().sort_values("strike").reset_index(drop=True)
    strikes = df["strike"].values
    net_gex = df["net_gex"].values
    colors = [COLORS["bar_green"] if g >= 0 else COLORS["bar_red"] for g in net_gex]

    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.01,
        column_widths=[0.78, 0.22],
    )

    # ── LEFT PANEL: GEX bars ──────────────────────────────────────────
    fig.add_trace(go.Bar(
        y=strikes, x=net_gex, orientation="h",
        marker_color=colors, marker_opacity=0.85,
        hovertemplate="Strike: $%{y:.0f}<br>Net GEX: %{x:,.0f}<extra></extra>",
    ), row=1, col=1)

    # ── RIGHT PANEL: Charm/hr with sign-coded fills ───────────────────
    # CEX sign under SqueezeMetrics convention:
    #   CEX > 0 → dealer book GAINS delta → dealer SELLS → repelling (red)
    #   CEX < 0 → dealer book LOSES delta → dealer BUYS  → supportive (green)
    # No competing bars on this side so fill opacity is bumped up from
    # the old overlay (0.18 → 0.30) for a crisper direction read.
    charm_range = None
    if "net_charm" in df.columns and len(df) > 0:
        charm_per_hr = df["net_charm"].values / float(TRADING_HOURS_PER_YEAR)
        charm_per_hr = np.where(np.isfinite(charm_per_hr), charm_per_hr, 0.0)

        max_abs_charm = float(np.max(np.abs(charm_per_hr)))
        if max_abs_charm > 0:
            charm_range = [-max_abs_charm * 1.05, max_abs_charm * 1.05]

        supportive = np.minimum(charm_per_hr, 0.0)
        repelling  = np.maximum(charm_per_hr, 0.0)

        fig.add_trace(go.Scatter(
            y=strikes, x=supportive, mode="lines",
            line=dict(color=COLORS["bar_green"], width=0.4),
            fill="tozerox",
            fillcolor="rgba(0, 200, 83, 0.30)",
            name="Supportive",
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=2)
        fig.add_trace(go.Scatter(
            y=strikes, x=repelling, mode="lines",
            line=dict(color=COLORS["bar_red"], width=0.4),
            fill="tozerox",
            fillcolor="rgba(255, 23, 68, 0.30)",
            name="Repelling",
            hoverinfo="skip",
            showlegend=False,
        ), row=1, col=2)

        direction_labels = np.where(
            charm_per_hr > 0, "Repelling (dealer sells)",
            np.where(charm_per_hr < 0, "Supportive (dealer buys)", "Neutral"),
        )
        fig.add_trace(go.Scatter(
            y=strikes, x=charm_per_hr, mode="lines",
            line=dict(color=COLORS["charm_line"], width=1.5),
            name="Charm/hr",
            customdata=direction_labels,
            hovertemplate=(
                "<b>Strike: $%{y:.0f}</b>"
                "<br>Charm/hr: %{x:,.0f} $Δ"
                "<br><b>%{customdata}</b>"
                "<br><i>(left = supportive, right = repelling)</i>"
                "<extra></extra>"
            ),
            opacity=0.95,
        ), row=1, col=2)

    # ── Reference lines (duplicated across both panels) ───────────────
    # Helper: draw a horizontal line on both panels, with the label
    # annotation only on the left (col=1) to avoid double-labeling.
    def _span_hline(val, color, dash, width, label=None, font_size=9,
                    position="top left"):
        fig.add_hline(
            y=val, line_color=color, line_dash=dash, line_width=width,
            annotation_text=(f"{label} ${val:.0f}" if label else None),
            annotation_font_color=color, annotation_font_size=font_size,
            annotation_position=position,
            row=1, col=1,
        )
        fig.add_hline(
            y=val, line_color=color, line_dash=dash, line_width=width,
            row=1, col=2,
        )

    def _span_hrect(y0, y1, fillcolor, opacity):
        for col in (1, 2):
            fig.add_hrect(
                y0=y0, y1=y1,
                fillcolor=fillcolor, opacity=opacity,
                line_width=0, layer="below",
                row=1, col=col,
            )

    for val, color, dash, name in [
        (spot, COLORS["spot"], "dash", "Spot"),
        (levels["zero_gamma"], COLORS["zero_gamma"], "dot", "Zero Γ"),
        (levels["call_wall"], COLORS["call_wall"], "dashdot", "Call Wall"),
        (levels["put_wall"], COLORS["put_wall"], "dashdot", "Put Wall"),
    ]:
        _span_hline(val, color, dash, width=1.5, label=name, font_size=9,
                    position="top left")

    em = em_analysis.get("expected_move", {})
    if show_daily_em and em.get("upper_level"):
        for val, label in [(em["upper_level"], "EM+"), (em["lower_level"], "EM−")]:
            _span_hline(val, COLORS["em_level"], "dot", width=1.2,
                        label=label, font_size=8, position="bottom right")

    w_em = weekly_em or {}
    if w_em.get("upper_level") and w_em.get("lower_level"):
        _span_hrect(w_em["lower_level"], w_em["upper_level"],
                    COLORS["em_weekly"], 0.06)
        for val, label in [(w_em["upper_level"], "wEM+"), (w_em["lower_level"], "wEM−")]:
            _span_hline(val, COLORS["em_weekly"], "dash", width=1,
                        label=label, font_size=7, position="top right")

    m_em = monthly_em or {}
    if m_em.get("upper_level") and m_em.get("lower_level"):
        _span_hrect(m_em["lower_level"], m_em["upper_level"],
                    COLORS["em_monthly"], 0.04)
        for val, label in [(m_em["upper_level"], "OpEx+"), (m_em["lower_level"], "OpEx−")]:
            _span_hline(val, COLORS["em_monthly"], "longdash", width=1,
                        label=label, font_size=7, position="top right")

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(
        paper_bgcolor=COLORS["bg_primary"], plot_bgcolor=COLORS["bg_primary"],
        font_color="white", font_size=10,
        margin=dict(l=80, r=10, t=55, b=35),
        title="Strike-by-Strike Net GEX Proxy  +  Charm/hr",
        showlegend=False, height=2000, dragmode=False,
    )

    # Left panel X axis (GEX)
    fig.update_xaxes(
        title_text="Net GEX proxy",
        gridcolor=COLORS["grid_major"],
        zerolinecolor=COLORS["zeroline"],
        row=1, col=1,
    )

    # Right panel X axis (Charm/hr) — amber-themed, symmetric around zero
    fig.update_xaxes(
        title=dict(
            text="Charm/hr — ← supportive | repelling →",
            font=dict(color=COLORS["charm_line"], size=9),
        ),
        showgrid=False,
        range=charm_range,
        zeroline=True,
        zerolinecolor=COLORS["charm_line"],
        zerolinewidth=1.2,
        tickfont=dict(color=COLORS["charm_line"], size=8),
        row=1, col=2,
    )

    # Shared Y axis
    fig.update_yaxes(
        title_text="Strike",
        gridcolor=COLORS["grid_minor"],
        tickfont_size=8,
        row=1, col=1,
    )

    return fig
