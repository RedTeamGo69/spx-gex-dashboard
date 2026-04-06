"""
JSON chart data builders for the GEX dashboard.

Builds Plotly-compatible JSON for the bar chart and profile curve chart.
"""
from __future__ import annotations

import json

import numpy as np

from phase1.dashboard import fmt_gex_val


def build_bar_chart_json(df, strikes, net_gex, levels, spot):
    bar_colors = ["#00c853" if g >= 0 else "#ff1744" for g in net_gex]
    hover = [f"Strike: ${s:.0f}<br>Net GEX proxy: {fmt_gex_val(g)}" for s, g in zip(strikes, net_gex)]

    cw = levels["call_wall"]
    pw = levels["put_wall"]

    tick_vals = []
    tick_texts = []

    for s in strikes:
        if s % 5 != 0:
            continue
        lbl = f"${s:.0f}"
        tick_vals.append(float(s))
        if s == round(cw):
            tick_texts.append(f"<span style='color:#69f0ae'><b>{lbl}</b></span>")
        elif s == round(pw):
            tick_texts.append(f"<span style='color:#ff8a80'><b>{lbl}</b></span>")
        else:
            tick_texts.append(lbl)

    shapes = []
    annotations = []

    if not levels.get("zero_gamma_is_true_crossing", False):
        annotations.append(
            {
                "xref": "paper",
                "yref": "paper",
                "x": 0.99,
                "y": 0.02,
                "text": "* no sign-change crossing found in sweep window",
                "showarrow": False,
                "xanchor": "right",
                "font": {"size": 9, "color": "#ffd600"},
                "bgcolor": "rgba(20,20,30,0.55)",
                "bordercolor": "#666",
                "borderwidth": 1,
                "borderpad": 4,
            }
        )

    for val, color, dash in [
        (spot, "#ffd600", "dash"),
        (levels["zero_gamma"], "#00e5ff", "dot"),
        (cw, "#69f0ae", "dashdot"),
        (pw, "#ff8a80", "dashdot"),
    ]:
        shapes.append(
            {
                "type": "line",
                "xref": "paper",
                "yref": "y",
                "x0": 0,
                "x1": 1,
                "y0": float(val),
                "y1": float(val),
                "line": {"color": color, "width": 1.5, "dash": dash},
                "layer": "above",
            }
        )

    label_items = [(spot, "#ffd600"), (levels["zero_gamma"], "#00e5ff")]
    label_items.sort(key=lambda x: x[0])

    for i, (val, color) in enumerate(label_items):
        yanchor = "bottom"
        yshift = 3
        if i > 0 and abs(val - label_items[i - 1][0]) < 8:
            yanchor = "top"
            yshift = -3
        annotations.append(
            {
                "x": 0.01,
                "xref": "paper",
                "y": float(val),
                "yref": "y",
                "text": f"<b>{val:.2f}</b>",
                "font": {"color": color, "size": 8},
                "showarrow": False,
                "xanchor": "left",
                "yanchor": yanchor,
                "yshift": yshift,
            }
        )

    fig_data = [
        {
            "type": "bar",
            "orientation": "h",
            "y": [float(s) for s in strikes],
            "x": net_gex.tolist(),
            "marker": {"color": bar_colors, "opacity": 0.85},
            "hovertext": hover,
            "hoverinfo": "text",
            "width": [4] * len(strikes),
        }
    ]

    fig_layout = {
        "paper_bgcolor": "#1a1a2e",
        "plot_bgcolor": "#1a1a2e",
        "font": {"color": "white", "size": 10},
        "margin": {"l": 84, "r": 12, "t": 28, "b": 38},
        "title": {"text": "Strike-by-Strike Net GEX Proxy", "font": {"size": 13}},
        "bargap": 0.05,
        "showlegend": False,
        "dragmode": False,
        "xaxis": {
            "title": "Net GEX proxy (OI × Γ × 100)",
            "gridcolor": "#333",
            "zerolinecolor": "#555",
            "zerolinewidth": 1,
            "tickfont": {"size": 9},
            "fixedrange": True,
        },
        "yaxis": {
            "title": "Strike",
            "tickvals": tick_vals,
            "ticktext": tick_texts,
            "tickfont": {"size": 8},
            "gridcolor": "#222",
            "fixedrange": True,
            "automargin": True,
        },
        "shapes": shapes,
        "annotations": annotations,
    }

    return json.dumps({"data": fig_data, "layout": fig_layout})


def build_profile_chart_json(profile_df, levels, spot, regime_info):
    if profile_df.empty:
        return json.dumps({"data": [], "layout": {}})

    prices = profile_df["price"].tolist()
    total_gex = profile_df["total_gex"].tolist()

    hover = [f"Price: ${p:.2f}<br>Total GEX proxy: {fmt_gex_val(g)}" for p, g in zip(prices, total_gex)]

    spot_gex = np.interp(spot, prices, total_gex) if len(prices) > 1 else (total_gex[0] if total_gex else 0.0)

    shapes = [
        {
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": float(spot),
            "x1": float(spot),
            "y0": 0,
            "y1": 1,
            "line": {"color": "#ffd600", "width": 1.5, "dash": "dash"},
        },
        {
            "type": "line",
            "xref": "x",
            "yref": "paper",
            "x0": float(levels["zero_gamma"]),
            "x1": float(levels["zero_gamma"]),
            "y0": 0,
            "y1": 1,
            "line": {"color": "#00e5ff", "width": 1.5, "dash": "dot"},
        },
        {
            "type": "line",
            "xref": "paper",
            "yref": "y",
            "x0": 0,
            "x1": 1,
            "y0": 0,
            "y1": 0,
            "line": {"color": "#777", "width": 1, "dash": "solid"},
        },
    ]

    annotations = [
        {
            "x": float(spot),
            "y": float(spot_gex),
            "xref": "x",
            "yref": "y",
            "text": "<b>Spot</b>",
            "showarrow": True,
            "arrowhead": 0,
            "ax": 0,
            "ay": -22,
            "font": {"color": "#ffd600", "size": 10},
        },
        {
            "x": float(levels["zero_gamma"]),
            "y": 0,
            "xref": "x",
            "yref": "y",
            "text": "<b>Zero Γ</b>" if levels.get("zero_gamma_is_true_crossing", False) else "<b>Gamma Pivot*</b>",
            "showarrow": True,
            "arrowhead": 0,
            "ax": 0,
            "ay": 26,
            "font": {"color": "#00e5ff", "size": 10},
        },
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0.01,
            "y": 0.98,
            "text": f"<b style='color:{regime_info['color']}'>{regime_info['regime']}</b><br>{regime_info['distance_text']}",
            "showarrow": False,
            "align": "left",
            "font": {"size": 11, "color": "white"},
            "bgcolor": "rgba(20,20,30,0.65)",
            "bordercolor": regime_info["color"],
            "borderwidth": 1,
            "borderpad": 5,
        },
        {
            "xref": "paper",
            "yref": "paper",
            "x": 0.01,
            "y": 0.84,
            "text": regime_info["note"],
            "showarrow": False,
            "align": "left",
            "font": {"size": 10, "color": "#dddddd"},
            "bgcolor": "rgba(20,20,30,0.55)",
            "bordercolor": "#444",
            "borderwidth": 1,
            "borderpad": 5,
        },
    ]

    fig_data = [
        {
            "type": "scatter",
            "mode": "lines",
            "x": prices,
            "y": total_gex,
            "line": {"color": "#9c88ff", "width": 2},
            "hovertext": hover,
            "hoverinfo": "text",
            "name": "GEX Profile",
        }
    ]

    fig_layout = {
        "paper_bgcolor": "#1a1a2e",
        "plot_bgcolor": "#1a1a2e",
        "font": {"color": "white", "size": 10},
        "margin": {"l": 62, "r": 12, "t": 28, "b": 40},
        "title": {"text": "GEX Profile Curve", "font": {"size": 13}},
        "showlegend": False,
        "dragmode": False,
        "xaxis": {
            "title": "Underlying price",
            "gridcolor": "#333",
            "tickfont": {"size": 9},
            "fixedrange": True,
        },
        "yaxis": {
            "title": "Total GEX proxy",
            "gridcolor": "#222",
            "zerolinecolor": "#555",
            "tickfont": {"size": 9},
            "fixedrange": True,
        },
        "shapes": shapes,
        "annotations": annotations,
    }

    return json.dumps({"data": fig_data, "layout": fig_layout})
