"""
Expected-move HTML rendering for the GEX dashboard.

Builds the EM sidebar panel and EM-level Plotly shapes for overlaying
on bar and profile charts.
"""
from __future__ import annotations

from phase1.dashboard import fmt_gex


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
