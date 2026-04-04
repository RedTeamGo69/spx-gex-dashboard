from __future__ import annotations

import json
import os
import webbrowser
import numpy as np
import pandas as pd

import phase1.gex_engine as gex_engine
from phase1.run_metadata import write_run_metadata_json


def fmt_gex_val(v):
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1000:
        return f"{v/1000:.1f}K"
    if abs(v) >= 1:
        return f"{v:.0f}"
    if v == 0:
        return ""
    return f"{v:.1f}"

def fmt_gex(v):
    if v is None:
        return "n/a"
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1000:
        return f"{v/1000:.1f}K"
    if abs(v) >= 1:
        return f"{v:.0f}"
    return f"{v:.4f}"

def fmt_oi(v):
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1000:
        return f"{v/1000:.1f}K"
    return f"{v:.0f}"


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


def build_status_html(stats, spot_info=None, run_metadata=None):
    failed = stats.get("failed_expirations", [])
    skipped_count = stats.get("skipped_count", 0)
    skipped_oi = stats.get("skipped_oi", 0.0)
    hybrid = stats.get("hybrid_iv_mode", False)

    parts = []

    if run_metadata:
        cal = run_metadata.get("calendar_snapshot", {})
        rf = run_metadata.get("risk_free", {})
        sel = run_metadata.get("selection", {})
        cfg = run_metadata.get("config", {})

        parts.append("<div class='status-line'><b>Run provenance</b></div>")
        parts.append(
            f"<div class='status-line'>Run NY time: {cal.get('now_ny', 'n/a')}</div>"
        )
        parts.append(
            f"<div class='status-line'>Cash open: {cal.get('cash_market_open')} "
            f"| Options open: {cal.get('options_market_open')}</div>"
        )
        parts.append(
            f"<div class='status-line'>Rate source: {rf.get('label', 'n/a')}</div>"
        )
        spot_ref = run_metadata.get("spot_reference", {})
        if spot_ref.get("expiration_close_ny"):
            parts.append(
                f"<div class='status-line'>Nearest exp close NY: {spot_ref.get('expiration_close_ny')}</div>"
            )
        parts.append(
            f"<div class='status-line'>Selected exps: {sel.get('selected_expirations_count', 0)} "
            f"| Heatmap exps: {sel.get('heatmap_expirations_count', 0)}</div>"
        )
        parts.append(
            f"<div class='status-line'>Strike range: ±{cfg.get('strike_range_pct', 0)*100:.1f}% "
            f"| Profile step: {cfg.get('profile_step', 'n/a')} "
            f"| Hybrid IV: {cfg.get('hybrid_iv_mode')}</div>"
        )
        conf = run_metadata.get("confidence", {})
        if conf:
            conf_color = "#00c853" if conf.get("label") == "High" else "#ffd600" if conf.get("label") == "Moderate" else "#ff5252"
            parts.append(
                f"<div class='status-line'><b>Run confidence:</b> "
                f"<span style='color:{conf_color}'>{conf.get('score')} / 100 ({conf.get('label')})</span></div>"
            )
            for reason in conf.get("reasons", [])[:3]:
                parts.append(f"<div class='status-line'>• {reason}</div>")        

        stale = run_metadata.get("staleness", {})
        if stale:
            stale_label = stale.get("freshness_label")
            stale_color = "#00c853" if stale_label == "High" else "#ffd600" if stale_label == "Moderate" else "#ff5252"
            parts.append(
                f"<div class='status-line'><b>Market-data freshness:</b> "
                f"<span style='color:{stale_color}'>{stale.get('freshness_score')} / 100 ({stale_label})</span></div>"
            )
            guidance = stale.get("trading_guidance")
            if guidance:
                parts.append(f"<div class='status-line'>• {guidance}</div>")
            for reason in stale.get("reasons", [])[:2]:
                parts.append(f"<div class='status-line'>• {reason}</div>")

    if hybrid:
        parts.append(
            "<div class='status-line'><b>Method:</b> hybrid IV mode "
            "(direct IV first, then synthetic IV from vendor gamma, used consistently across bars, zero gamma, and profile curve).</div>"
        )

    parts.append(
        f"<div class='status-line'><b>Direct IV contracts:</b> {stats.get('direct_iv_count', 0):,} "
        f"| <b>Synthetic IV contracts:</b> {stats.get('synthetic_iv_count', 0):,}</div>"
    )

    parts.append(
        f"<div class='status-line'><b>Synthetic accepted:</b> {stats.get('synthetic_fit_accept_count', 0):,} "
        f"| <b>Synthetic rejected:</b> {stats.get('synthetic_fit_reject_count', 0):,} "
        f"| <b>No input:</b> {stats.get('no_model_input_count', 0):,}</div>"
    )

    if stats.get("synthetic_fit_avg_rel_error") is not None:
        parts.append(
            f"<div class='status-line'>Synthetic fit avg err: {stats.get('synthetic_fit_avg_rel_error')*100:.2f}% "
            f"| max err: {stats.get('synthetic_fit_max_rel_error')*100:.2f}%</div>"
        )

    if stats.get("strike_support_avg") is not None:
        parts.append(
            f"<div class='status-line'><b>Avg strike support:</b> {stats.get('strike_support_avg'):.1f} "
            f"| <b>Fragile strikes:</b> {stats.get('fragile_strike_count', 0)}</div>"
        )

    if stats.get("expiration_support_avg") is not None:
        parts.append(
            f"<div class='status-line'><b>Avg expiration support:</b> {stats.get('expiration_support_avg'):.1f}</div>"
        )        

    if skipped_count > 0:
        parts.append(
            f"<div class='status-line'><b>Skipped contracts:</b> "
            f"{skipped_count:,}  |  OI skipped: {fmt_oi(skipped_oi)}</div>"
        )

    if failed:
        exp_text = ", ".join(failed)
        parts.append(
            f"<div class='status-line warn'><b>Failed expirations:</b> {exp_text}</div>"
        )

    if spot_info and spot_info.get("parity_diagnostics"):
        pdiag = spot_info["parity_diagnostics"]
        cq = pdiag["call_quality"]
        pq = pdiag["put_quality"]

        parts.append("<div class='status-line'><b>Parity input audit</b></div>")
        parts.append(
            f"<div class='status-line'>Calls usable: {cq['usable']}/{cq['total']} "
            f"| no2s={cq['no_two_sided_quote']} | crossed={cq['crossed_or_locked']} | "
            f"wide={cq['wide_spread']} | bad_mid={cq['bad_mid']}</div>"
        )
        parts.append(
            f"<div class='status-line'>Puts usable: {pq['usable']}/{pq['total']} "
            f"| no2s={pq['no_two_sided_quote']} | crossed={pq['crossed_or_locked']} | "
            f"wide={pq['wide_spread']} | bad_mid={pq['bad_mid']}</div>"
        )
        parts.append(
            f"<div class='status-line'>Common usable strikes: {pdiag['common_usable_strikes']} "
            f"| near-spot: {pdiag['near_spot_candidates']} "
            f"| final ATM: {pdiag['final_atm_strikes']}</div>"
        )
        parts.append(
            f"<div class='status-line'>Parity method: {pdiag['parity_method']} "
            f"| median={pdiag['simple_median_spot']} "
            f"| weighted={pdiag['weighted_median_spot']}</div>"
        )

    return "<div class='status-box'>" + "".join(parts) + "</div>"


def build_stats_html(stats, levels, date_label, spot_source, spot, regime_info):
    gr_color = "#00c853" if stats["gex_ratio"] > 0.5 else "#ff1744"
    ng_color = "#ff1744" if stats["net_gex"] < 0 else "#00c853"
    src_short = spot_source.split("(")[0].strip()

    d_cw = levels["call_wall"] - spot
    d_pw = levels["put_wall"] - spot
    d_zg = levels["zero_gamma"] - spot

    zg_type = levels.get("zero_gamma_type", "Unknown")
    zg_method = levels.get("zero_gamma_method", "unknown")
    zg_abs = levels.get("zero_gamma_abs_gex")
    zg_type_color = "#00e5ff" if levels.get("zero_gamma_is_true_crossing", False) else "#ffd600"
    zg_abs_text = fmt_gex(zg_abs) if zg_abs is not None else "n/a"

    return f"""
    <table class="stats-table">
      <tr class="header"><td colspan="2"><b>SPX</b></td><td colspan="2">{date_label}</td></tr>

      <tr class="thick-bottom">
        <td class="lbl">GEX Ratio</td><td class="val" style="color:{gr_color}">{stats['gex_ratio']:.2f}</td>
        <td class="lbl">Net GEX</td><td class="val" style="color:{ng_color}">{stats['net_gex_fmt']}</td>
      </tr>

      <tr><td class="lbl">Call OI</td><td class="val" colspan="3" style="color:#00c853">{stats['call_oi']} @ {stats['call_oi_strike']:.0f}</td></tr>
      <tr><td class="lbl">Pos GEX</td><td class="val" colspan="3" style="color:#00c853">{stats['pos_gex']} @ {stats['pos_gex_strike']:.0f}</td></tr>
      <tr><td class="lbl">Zero Gamma</td><td class="val" colspan="3" style="color:white">{levels['zero_gamma']:.2f}</td></tr>
      <tr><td class="lbl">Zero Γ Type</td><td class="val" colspan="3" style="color:{zg_type_color}">{zg_type} ({zg_method})</td></tr>
      <tr><td class="lbl">Zero Γ residual</td><td class="val" colspan="3" style="color:{zg_type_color}">{zg_abs_text}</td></tr>      
      <tr><td class="lbl">Neg GEX</td><td class="val" colspan="3" style="color:#ff1744">{stats['neg_gex']} @ {stats['neg_gex_strike']:.0f}</td></tr>
      <tr><td class="lbl">Put OI</td><td class="val" colspan="3" style="color:#ff1744">{stats['put_oi']} @ {stats['put_oi_strike']:.0f}</td></tr>
      <tr><td class="lbl">P/C OI Ratio</td><td class="val" colspan="3">{stats['pc_ratio']:.2f}</td></tr>

      <tr class="thick-top">
        <td class="lbl">Call IV</td><td class="val" style="color:#00c853">{stats['call_iv']:.1f}%</td>
        <td class="lbl">Put IV</td><td class="val" style="color:#ff1744">{stats['put_iv']:.1f}%</td>
      </tr>

      <tr class="thick-top">
        <td class="lbl" style="font-size:9px">→ Call Wall</td>
        <td class="val" style="color:#69f0ae;font-size:9px">{d_cw:+.0f} pts</td>
        <td class="lbl" style="font-size:9px">→ Put Wall</td>
        <td class="val" style="color:#ff8a80;font-size:9px">{d_pw:+.0f} pts</td>
      </tr>

      <tr>
        <td class="lbl" style="font-size:9px">→ Zero Γ</td>
        <td class="val" style="color:#00e5ff;font-size:9px">{d_zg:+.0f} pts</td>
        <td class="lbl" style="color:#666;font-size:9px">Spot src</td>
        <td class="val" style="color:#888;font-size:9px">{src_short}</td>
      </tr>

      <tr>
        <td class="lbl" style="font-size:9px">Gamma regime</td>
        <td class="val" style="font-size:9px;color:{regime_info['color']}">{regime_info['regime']}</td>
        <td class="lbl" style="font-size:9px">Coverage</td>
        <td class="val" style="font-size:9px">{stats['coverage_ratio']*100:.1f}%</td>
      </tr>

      <tr>
        <td class="lbl" style="font-size:9px">Used contracts</td>
        <td class="val" style="font-size:9px">{stats['used_option_count']:,}</td>
        <td class="lbl" style="font-size:9px">Failed exps</td>
        <td class="val" style="font-size:9px">{stats['failed_exp_count']}</td>
      </tr>
    </table>
    """


def gex_cell_color(v, abs_max):
    if abs_max == 0:
        return "#1a1a2e"
    ratio = max(-1, min(1, v / abs_max))
    if ratio >= 0:
        g = int(50 + 155 * ratio)
        return f"rgb(0, {g}, 0)"
    r = int(50 + 205 * abs(ratio))
    return f"rgb({r}, 0, 0)"


def iv_cell_color(v, vmin, vmax):
    if vmax <= vmin or v == 0:
        return "#1a1a2e"
    ratio = (v - vmin) / (vmax - vmin)
    g = int(40 + 180 * ratio)
    return f"rgb(0, {g}, 0)"


def build_heatmap_html(hm_df, spot, title, is_iv=False):
    if hm_df is None or hm_df.empty:
        return f"<div class='hm-title'>{title}</div><div class='empty-note'>No data</div>"

    hm = hm_df.iloc[::-1]
    strikes = hm.index.values
    cols = hm.columns.tolist()
    vals = hm.values
    nr, nc = vals.shape

    if is_iv:
        nonzero = vals[vals > 0]
        vmin = nonzero.min() if len(nonzero) > 0 else 0
        vmax = nonzero.max() if len(nonzero) > 0 else 1
    else:
        abs_max = max(np.abs(vals[vals != 0]).max(), 1) if np.any(vals != 0) else 1

    html = f'<div class="hm-title">{title}</div><table class="hm-table"><tr><th>Strike</th>'
    for c in cols:
        html += f"<th>{c}</th>"
    html += "</tr>"

    for r in range(nr):
        strike = strikes[r]
        is_spot = abs(strike - spot) < 3
        ss = "color:#ffd600;font-weight:bold" if is_spot else "color:#aaa"
        html += f'<tr><td class="hm-strike" style="{ss}">{strike:.0f}</td>'
        for c in range(nc):
            v = vals[r, c]
            bg = iv_cell_color(v, vmin, vmax) if is_iv else gex_cell_color(v, abs_max)
            txt = (f"+{v:.1f}%" if v > 0 else "") if is_iv else fmt_gex_val(v)
            html += f'<td class="hm-cell" style="background:{bg}">{txt}</td>'
        html += "</tr>"

    html += "</table>"
    return html

def build_sensitivity_html(sensitivity_df):
    if sensitivity_df is None or sensitivity_df.empty:
        return "<div class='hm-title'>Zero Gamma Sensitivity</div><div class='empty-note'>No data</div>"

    html = '<div class="hm-title">Zero Gamma Sensitivity</div>'
    html += '<table class="hm-table">'
    html += "<tr><th>Shock</th><th>Spot</th><th>Zero Γ</th><th>Gap</th><th>Type</th><th>Regime</th></tr>"

    for _, row in sensitivity_df.iterrows():
        shock_text = f"{row['shock_pct']*100:+.2f}%"
        gap = row["spot_minus_zero_gamma"]
        gap_color = "#00c853" if gap > 0 else "#ff5252" if gap < 0 else "#00e5ff"

        ztype = row.get("zero_gamma_type", "Unknown")
        html += (
            "<tr>"
            f"<td>{shock_text}</td>"
            f"<td>{row['shocked_spot']:.2f}</td>"
            f"<td>{row['zero_gamma']:.2f}</td>"
            f"<td style='color:{gap_color};font-weight:bold'>{gap:+.2f}</td>"
            f"<td>{ztype}</td>"
            f"<td>{row['regime']}</td>"
            "</tr>"
        )

    html += "</table>"
    return html

def build_strike_support_html(strike_support_df, max_rows=8):
    if strike_support_df is None or strike_support_df.empty:
        return "<div class='hm-title'>Strike Support</div><div class='empty-note'>No data</div>"

    top = strike_support_df.head(max_rows)

    html = '<div class="hm-title">Strike Support</div>'
    html += '<table class="hm-table">'
    html += "<tr><th>Strike</th><th>Score</th><th>Label</th><th>Exps</th><th>OI</th></tr>"

    for _, row in top.iterrows():
        label = row["support_label"]
        color = "#00c853" if label == "High" else "#ffd600" if label == "Moderate" else "#ff5252"
        html += (
            "<tr>"
            f"<td>{int(row['strike'])}</td>"
            f"<td style='color:{color};font-weight:bold'>{row['support_score']:.1f}</td>"
            f"<td style='color:{color};font-weight:bold'>{label}</td>"
            f"<td>{int(row['supporting_expirations'])}</td>"
            f"<td>{fmt_oi(row['total_oi'])}</td>"
            "</tr>"
        )

    html += "</table>"
    return html

def build_expiration_support_html(expiration_support_df):
    if expiration_support_df is None or expiration_support_df.empty:
        return "<div class='hm-title'>Expiration Support</div><div class='empty-note'>No data</div>"

    html = '<div class="hm-title">Expiration Support</div>'
    html += '<table class="hm-table">'
    html += "<tr><th>Exp</th><th>Score</th><th>Label</th><th>Strikes</th><th>OI</th></tr>"

    for _, row in expiration_support_df.iterrows():
        label = row["support_label"]
        color = "#00c853" if label == "High" else "#ffd600" if label == "Moderate" else "#ff5252"
        html += (
            "<tr>"
            f"<td>{row['expiration']}</td>"
            f"<td style='color:{color};font-weight:bold'>{row['support_score']:.1f}</td>"
            f"<td style='color:{color};font-weight:bold'>{label}</td>"
            f"<td>{int(row['strikes_used'])}</td>"
            f"<td>{fmt_oi(row['total_oi'])}</td>"
            "</tr>"
        )

    html += "</table>"
    return html

def build_wall_credibility_html(wall_credibility_info):
    if not wall_credibility_info:
        return "<div class='hm-title'>Wall Credibility</div><div class='empty-note'>No data</div>"

    rows = [
        wall_credibility_info.get("call_wall"),
        wall_credibility_info.get("put_wall"),
        wall_credibility_info.get("zero_gamma"),
    ]
    rows = [r for r in rows if r]

    if not rows:
        return "<div class='hm-title'>Wall Credibility</div><div class='empty-note'>No data</div>"

    name_map = {
        "call_wall": "Call Wall",
        "put_wall": "Put Wall",
        "zero_gamma": "Zero Gamma",
    }

    html = '<div class="hm-title">Wall Credibility</div>'
    html += '<table class="hm-table">'
    html += "<tr><th>Level</th><th>Value</th><th>Score</th><th>Label</th></tr>"

    for row in rows:
        label = row.get("label", "Low")
        color = "#00c853" if label == "High" else "#ffd600" if label == "Moderate" else "#ff5252"
        html += (
            "<tr>"
            f"<td>{name_map.get(row.get('level_name'), row.get('level_name'))}</td>"
            f"<td>{row.get('level_value')}</td>"
            f"<td style='color:{color};font-weight:bold'>{row.get('score')}</td>"
            f"<td style='color:{color};font-weight:bold'>{label}</td>"
            "</tr>"
        )

    html += "</table>"

    # brief notes
    for row in rows:
        name = name_map.get(row.get("level_name"), row.get("level_name"))
        reasons = row.get("reasons", [])[:2]
        for reason in reasons:
            html += f"<div class='status-line'>• <b>{name}:</b> {reason}</div>"

    return html

def build_scenarios_html(scenarios_df):
    if scenarios_df is None or scenarios_df.empty:
        return "<div class='hm-title'>Scenario Engine</div><div class='empty-note'>No data</div>"

    html = '<div class="hm-title">Scenario Engine</div>'
    html += '<table class="hm-table">'
    html += "<tr><th>Scenario</th><th>Call W</th><th>Put W</th><th>Zero Γ</th><th>Regime</th></tr>"

    for _, row in scenarios_df.iterrows():
        regime = row["gamma_regime"]
        regime_color = "#00c853" if regime == "Positive Gamma" else "#ff5252" if regime == "Negative Gamma" else "#00e5ff"

        html += (
            "<tr>"
            f"<td>{row['scenario']}</td>"
            f"<td>{row['call_wall']:.0f}</td>"
            f"<td>{row['put_wall']:.0f}</td>"
            f"<td>{row['zero_gamma']:.2f}</td>"
            f"<td style='color:{regime_color};font-weight:bold'>{regime}</td>"
            "</tr>"
        )

    html += "</table>"
    return html

def build_expected_move_html(em_info):
    """Build the Expected Move sidebar panel."""
    if em_info is None:
        return ""

    em = em_info.get("expected_move", {})
    on = em_info.get("overnight_move", {})
    cl = em_info.get("classification", {})
    spy = em_info.get("spy_proxy")
    lctx = em_info.get("level_context")

    if em.get("expected_move_pts") is None:
        return ""

    # Classification color
    bias = cl.get("bias", "")
    if bias in ("range-bound", "mean-revert"):
        cls_color = "#00c853"
    elif bias in ("directional", "continued-trend"):
        cls_color = "#ff5252"
    elif bias == "uncertain":
        cls_color = "#ffd600"
    else:
        cls_color = "#aaa"

    # Move ratio color
    ratio = cl.get("move_ratio")
    if ratio is not None:
        if ratio < 0.40:
            ratio_color = "#00c853"
        elif ratio < 0.70:
            ratio_color = "#ffd600"
        else:
            ratio_color = "#ff5252"
    else:
        ratio_color = "#aaa"

    # Overnight direction arrow
    on_pts = on.get("overnight_move_pts")
    if on_pts is not None:
        on_arrow = "▲" if on_pts > 0 else "▼" if on_pts < 0 else "–"
        on_color = "#00c853" if on_pts >= 0 else "#ff5252"
    else:
        on_arrow = ""
        on_color = "#aaa"

    parts = []
    parts.append("<div class='status-box' style='border-color:#6f7cff;'>")
    parts.append("<div class='status-line'><b style='color:#9fa8ff;font-size:12px;'>&#9889; Expected Move &mdash; 0DTE</b></div>")

    # Straddle & range
    straddle = em.get("straddle", {})
    if straddle:
        parts.append(
            f"<div class='status-line'><b>ATM Straddle:</b> "
            f"<span style='color:#e0e0ff;font-weight:bold;'>{em['expected_move_pts']:.1f} pts</span> "
            f"({em['expected_move_pct']:.2f}%) "
            f"<span style='color:#888;'>@ K={straddle.get('strike', '?')}</span></div>"
        )
        parts.append(
            f"<div class='status-line'><b>Expected Range:</b> "
            f"<span style='color:#ff8a80;'>${em['lower_level']:.0f}</span> &mdash; "
            f"<span style='color:#69f0ae;'>${em['upper_level']:.0f}</span></div>"
        )

    # Overnight move
    if on_pts is not None:
        parts.append(
            f"<div class='status-line'><b>Overnight Move:</b> "
            f"<span style='color:{on_color};font-weight:bold;'>"
            f"{on_arrow} {on_pts:+.1f} pts ({on['overnight_move_pct']:+.2f}%)</span></div>"
        )

    # SPY proxy
    if spy:
        parts.append(
            f"<div class='status-line'><b>SPY Pre-mkt:</b> "
            f"${spy['spy_price']:.2f} ({spy['spy_move_pct']:+.2f}%) "
            f"&rarr; ~{spy['implied_spx_move_pts']:+.1f} SPX pts</div>"
        )

    # Move ratio bar
    if ratio is not None:
        bar_width = min(int(ratio * 100), 100)
        parts.append(
            f"<div class='status-line'><b>Vol Budget Used:</b> "
            f"<span style='color:{ratio_color};font-weight:bold;'>{ratio*100:.0f}%</span> "
            f"<span style='color:#888;'>({cl['move_ratio_label']})</span></div>"
        )
        parts.append(
            f"<div style='background:#333;border-radius:3px;height:6px;margin:2px 0 6px 0;'>"
            f"<div style='background:{ratio_color};width:{bar_width}%;height:6px;border-radius:3px;'></div></div>"
        )

    # Session classification
    if cl.get("classification"):
        parts.append(
            f"<div class='status-line'><b>Session Type:</b> "
            f"<span style='color:{cls_color};font-weight:bold;font-size:12px;'>"
            f"{cl['classification']}</span></div>"
        )
        if cl.get("description"):
            parts.append(
                f"<div class='status-line' style='color:#bbb;font-size:9px;line-height:1.4;margin:2px 0 4px 0;'>"
                f"{cl['description']}</div>"
            )
        if cl.get("historical_tendencies"):
            tendency = cl["historical_tendencies"][0]
            parts.append(
                f"<div class='status-line'><b>Tendency:</b> "
                f"<span style='color:#cfd3ff;'>{tendency}</span></div>"
            )

    # Zero gamma context
    if lctx and lctx.get("zero_gamma_within_em") is not None:
        zg_in = "inside" if lctx["zero_gamma_within_em"] else "outside"
        zg_color = "#00e5ff" if lctx["zero_gamma_within_em"] else "#ffd600"
        parts.append(
            f"<div class='status-line'><b>Zero &Gamma;:</b> "
            f"<span style='color:{zg_color};'>{zg_in}</span> expected range "
            f"({lctx['zero_gamma_distance_to_spot']:+.1f} pts from spot)</div>"
        )

    parts.append("</div>")
    return "\n".join(parts)


def _em_level_shapes_bar(em_info):
    """Return Plotly shapes for EM upper/lower on the bar chart (horizontal lines on y-axis)."""
    if em_info is None:
        return [], []
    em = em_info.get("expected_move", {})
    upper = em.get("upper_level")
    lower = em.get("lower_level")
    if upper is None or lower is None:
        return [], []

    shapes = []
    annotations = []
    for val, label, color in [
        (upper, "EM+", "#b388ff"),
        (lower, "EM-", "#b388ff"),
    ]:
        shapes.append({
            "type": "line", "xref": "paper", "yref": "y",
            "x0": 0, "x1": 1, "y0": float(val), "y1": float(val),
            "line": {"color": color, "width": 1.2, "dash": "dot"},
            "layer": "below",
        })
        annotations.append({
            "x": 0.99, "xref": "paper",
            "y": float(val), "yref": "y",
            "text": f"<b>{label} ${val:.0f}</b>",
            "font": {"color": color, "size": 8},
            "showarrow": False, "xanchor": "right", "yanchor": "bottom", "yshift": 2,
        })
    return shapes, annotations


def _em_level_shapes_profile(em_info):
    """Return Plotly shapes for EM upper/lower on the profile chart (vertical lines on x-axis)."""
    if em_info is None:
        return [], []
    em = em_info.get("expected_move", {})
    upper = em.get("upper_level")
    lower = em.get("lower_level")
    if upper is None or lower is None:
        return [], []

    shapes = []
    annotations = []
    for val, label, color in [
        (upper, "EM+", "#b388ff"),
        (lower, "EM-", "#b388ff"),
    ]:
        shapes.append({
            "type": "line", "xref": "x", "yref": "paper",
            "x0": float(val), "x1": float(val), "y0": 0, "y1": 1,
            "line": {"color": color, "width": 1.2, "dash": "dot"},
            "layer": "below",
        })
        annotations.append({
            "x": float(val), "xref": "x",
            "y": 1.0, "yref": "paper",
            "text": f"<b>{label}</b>",
            "font": {"color": color, "size": 9},
            "showarrow": False, "yanchor": "top", "yshift": -4,
        })
    return shapes, annotations


def build_dashboard(gex_df, hm_gex, hm_iv, stats, levels, profile_df, sensitivity_df, strike_support_df, expiration_support_df, wall_credibility_info, scenarios_df, spot, spot_source, date_label, num_exps, spot_info=None, run_metadata=None, expected_move_info=None):
    df = gex_df.copy().sort_values("strike").reset_index(drop=True)
    if df.empty:
        print("No data to plot.")
        return

    df["strike"] = df["strike"].round().astype(int)

    s_min = int(np.floor(df["strike"].min() / 5) * 5)
    s_max = int(np.ceil(df["strike"].max() / 5) * 5)
    all_5s = np.arange(s_min, s_max + 5, 5, dtype=int)
    full = pd.DataFrame({"strike": all_5s})

    df = (
        full.merge(df, on="strike", how="left")
        .fillna(0)
        .sort_values("strike")
        .reset_index(drop=True)
    )

    strikes = df["strike"].values
    net_gex = df["net_gex"].values
    exp_text = f"{date_label} ({num_exps} exps)" if num_exps > 1 else date_label

    regime_info = gex_engine.get_gamma_regime_text(spot, levels["zero_gamma"])

    bar_chart_json = build_bar_chart_json(df, strikes, net_gex, levels, spot)
    profile_chart_json = build_profile_chart_json(profile_df, levels, spot, regime_info)
    stats_html = build_stats_html(stats, levels, date_label, spot_source, spot, regime_info)
    status_html = build_status_html(stats, spot_info=spot_info, run_metadata=run_metadata)
    wall_credibility_html = build_wall_credibility_html(wall_credibility_info)
    scenarios_html = build_scenarios_html(scenarios_df)
    strike_support_html = build_strike_support_html(strike_support_df)
    expiration_support_html = build_expiration_support_html(expiration_support_df)
    gex_hm_html = build_heatmap_html(hm_gex, spot, "Net GEX Heatmap", is_iv=False)
    sensitivity_html = build_sensitivity_html(sensitivity_df)
    iv_hm_html = build_heatmap_html(hm_iv, spot, "IV Heatmap", is_iv=True)
    expected_move_html = build_expected_move_html(expected_move_info)

    # Inject expected-move levels into both charts
    em_bar_shapes, em_bar_annots = _em_level_shapes_bar(expected_move_info)
    em_prof_shapes, em_prof_annots = _em_level_shapes_profile(expected_move_info)

    if em_bar_shapes:
        fig1 = json.loads(bar_chart_json)
        fig1["layout"]["shapes"].extend(em_bar_shapes)
        fig1["layout"]["annotations"].extend(em_bar_annots)
        bar_chart_json = json.dumps(fig1)

    if em_prof_shapes:
        fig2 = json.loads(profile_chart_json)
        fig2["layout"]["shapes"].extend(em_prof_shapes)
        fig2["layout"]["annotations"].extend(em_prof_annots)
        profile_chart_json = json.dumps(fig2)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SPX GEX — {exp_text}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    background: #1a1a2e;
    color: white;
    font-family: 'Segoe UI', Arial, sans-serif;
    display: flex;
    height: 100vh;
    overflow: hidden;
  }}
  #left {{
    flex: 1;
    min-width: 0;
    display: flex;
    flex-direction: column;
  }}
  #left-header {{
    text-align: center;
    padding: 8px 0 2px 0;
    font-size: 15px;
    font-weight: bold;
  }}
  #tab-bar {{
    display: flex;
    gap: 6px;
    padding: 6px 10px 4px 10px;
    border-bottom: 1px solid #333;
  }}
  .tab-btn {{
    background: #22263f;
    color: #cfd3ff;
    border: 1px solid #454b7a;
    border-radius: 7px;
    padding: 6px 10px;
    font-size: 12px;
    cursor: pointer;
  }}
  .tab-btn.active {{
    background: #35408c;
    color: white;
    border-color: #6f7cff;
  }}
  .chart-panel {{
    width: 100%;
    height: calc(100vh - 72px);
    display: none;
  }}
  .chart-panel.active {{
    display: block;
  }}
  #chart1, #chart2 {{
    width: 100%;
    height: 100%;
  }}
  #right {{
    width: 360px;
    min-width: 360px;
    overflow-y: auto;
    padding: 6px 8px;
    border-left: 1px solid #444;
  }}
  .stats-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
    margin-bottom: 10px;
  }}
  .stats-table td {{
    padding: 4px 6px;
    border: 1px solid #333;
  }}
  .stats-table .header {{
    background: #1a3a5c;
    font-size: 12px;
    text-align: center;
  }}
  .stats-table .header td:first-child {{ text-align: left; }}
  .stats-table .header td:last-child {{ text-align: right; }}
  .stats-table .lbl {{ color: #888; }}
  .stats-table .val {{ text-align: right; font-weight: bold; }}
  .stats-table .thick-bottom td {{ border-bottom: 2px solid #666; }}
  .stats-table .thick-top td {{ border-top: 2px solid #666; }}

  .status-box {{
    border: 1px solid #444;
    background: #20203a;
    padding: 8px 9px;
    margin-bottom: 10px;
    font-size: 10px;
    line-height: 1.45;
  }}
  .status-line {{
    margin-bottom: 4px;
    color: #ddd;
  }}
  .status-line.warn {{
    color: #ffb3b3;
  }}

  .hm-title {{
    font-size: 12px;
    font-weight: bold;
    margin: 8px 0 4px 0;
  }}
  .hm-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 9px;
  }}
  .hm-table th {{
    background: #2a2a4a;
    color: white;
    padding: 3px 2px;
    font-size: 9px;
    border: 1px solid #333;
    text-align: center;
  }}
  .hm-table td {{
    padding: 2px 3px;
    border: 1px solid #222;
    text-align: center;
  }}
  .hm-cell {{
    color: white;
    font-weight: bold;
    font-size: 8.5px;
  }}
  .hm-strike {{
    color: #aaa;
    font-size: 9px;
    background: #222;
    text-align: center;
  }}
  .empty-note {{
    color: #999;
    font-size: 10px;
    padding: 4px 0 8px 0;
  }}
</style>
</head>
<body>

<div id="left">
  <div id="left-header">SPX Gamma Exposure — {exp_text}</div>

  <div id="tab-bar">
    <button class="tab-btn active" onclick="showTab('gexTab', this)">Strike GEX</button>
    <button class="tab-btn" onclick="showTab('profileTab', this)">GEX Profile</button>
  </div>

  <div id="gexTab" class="chart-panel active">
    <div id="chart1"></div>
  </div>

  <div id="profileTab" class="chart-panel">
    <div id="chart2"></div>
  </div>
</div>

<div id="right">
  {stats_html}
  {expected_move_html}
  {status_html}
  {wall_credibility_html}
  {scenarios_html}
  {strike_support_html}
  {expiration_support_html}
  {gex_hm_html}
  {sensitivity_html}
  {iv_hm_html}
</div>

<script>
  function showTab(tabId, btn) {{
    document.querySelectorAll('.chart-panel').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    btn.classList.add('active');

    setTimeout(function() {{
      Plotly.Plots.resize('chart1');
      Plotly.Plots.resize('chart2');
    }}, 60);
  }}

  var fig1 = {bar_chart_json};
  Plotly.newPlot('chart1', fig1.data, fig1.layout, {{
    responsive: true,
    displayModeBar: false,
    scrollZoom: false
  }});

  var fig2 = {profile_chart_json};
  Plotly.newPlot('chart2', fig2.data, fig2.layout, {{
    responsive: true,
    displayModeBar: false,
    scrollZoom: false
  }});
</script>

</body>
</html>"""

    out_path = os.path.join(os.getcwd(), "gex_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    metadata_path = None
    if run_metadata is not None:
        metadata_path = os.path.join(os.getcwd(), "gex_run_metadata.json")
        write_run_metadata_json(run_metadata, metadata_path)

    webbrowser.open(f"file://{os.path.abspath(out_path)}")
    print(f"\nDashboard saved to {out_path}")
    if metadata_path:
        print(f"Run metadata saved to {metadata_path}")
