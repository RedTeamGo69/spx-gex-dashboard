"""
Sidebar rendering functions extracted from streamlit_app.py.
"""
from __future__ import annotations

import streamlit as st

from theme import COLORS


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar rendering
# ─────────────────────────────────────────────────────────────────────────────
def _render_move_display(overnight, market_ctx):
    """Render today's SPX move (or the retrospective session move when the
    cash market is closed). Pre-market SPY proxy / ES overnight display
    has been removed."""
    on_pts = overnight.get("overnight_move_pts")
    on_pct = overnight.get("overnight_move_pct")
    if on_pts is None:
        return
    label = "Today's Move" if market_ctx == "live" else "Session Move"
    arrow = "🟢 ▲" if on_pts >= 0 else "🔴 ▼"
    pct_str = f" ({on_pct:+.2f}%)" if on_pct is not None else ""
    st.markdown(f"**{label}:** {arrow} **{on_pts:+.1f} pts**{pct_str}")


def _render_classification(classification, level_ctx):
    """Render session classification and zero gamma context."""
    ratio = classification.get("move_ratio")
    if ratio is not None:
        pct = min(ratio * 100, 100)
        label = classification.get("move_ratio_label", "")
        st.markdown(f"**Vol Budget Used:** {pct:.0f}% ({label})")
        st.progress(min(ratio, 1.0))

    if classification.get("classification"):
        bias = classification.get("bias", "")
        signal_strength = classification.get("signal_strength", "weak")
        bucket_acc = classification.get("bucket_accuracy")

        # Visual weight is now driven by calibrated signal strength, not
        # by the classification's gut-feel bias. Weak signals get a grey
        # icon regardless of bias so the sidebar stops giving a decisive
        # green/red rendering to a near-coin-flip bucket.
        if signal_strength == "strong":
            if bias in ("range-bound", "mean-revert"):
                cls_icon = "🟢"
            elif bias in ("directional", "continued-trend"):
                cls_icon = "🔴"
            else:
                cls_icon = "🟡"
        elif signal_strength == "moderate":
            cls_icon = "🟡"
        else:
            # Weak or unknown strength — explicitly un-decisive rendering
            cls_icon = "⚪"

        # Tag the classification label with its calibrated accuracy so
        # the trader can see the confidence at a glance.
        if bucket_acc is not None:
            acc_tag = f" _(hist {bucket_acc*100:.0f}%)_"
        else:
            acc_tag = ""
        st.markdown(f"### {cls_icon} {classification['classification']}{acc_tag}")

        if signal_strength == "weak":
            st.caption(
                "⚠️ **Weak signal** — calibrated accuracy barely above random. "
                "Do not size based on this classification alone."
            )

        if classification.get("description"):
            st.caption(classification["description"])
        if classification.get("historical_tendencies"):
            st.markdown(f"**Tendencies:** {classification['historical_tendencies'][0]}")
        if classification.get("confidence_note"):
            st.caption(classification["confidence_note"])

    if level_ctx and level_ctx.get("zero_gamma_within_em") is not None:
        inside = "✅ inside" if level_ctx["zero_gamma_within_em"] else "⚠️ outside"
        st.markdown(
            f"**Zero Γ:** {inside} expected range "
            f"({level_ctx['zero_gamma_distance_to_spot']:+.1f} pts from spot)"
        )


def render_expected_move_panel(em_analysis, ticker="SPX"):
    em_data = em_analysis.get("expected_move", {})
    overnight = em_analysis.get("overnight_move", {})
    classification = em_analysis.get("classification", {})
    level_ctx = em_analysis.get("level_context")
    market_ctx = em_analysis.get("market_context", "live")

    if em_data.get("expected_move_pts") is None:
        st.caption("Expected move data not available.")
        return

    # Show the ACTUAL straddle DTE instead of always labeling "0DTE". On
    # weekends and after-hours, the nearest expiration is 1+ days away, and
    # the straddle reflects more vol than a true same-day EM would.
    straddle_info = em_data.get("straddle") or {}
    straddle_dte = straddle_info.get("dte")
    if straddle_dte is None or straddle_dte == 0:
        dte_label = "0DTE"
    else:
        dte_label = f"{straddle_dte}DTE"
    st.markdown(f"#### ⚡ Expected Move — {dte_label}")
    if straddle_dte and straddle_dte > 0:
        st.caption(
            f"⚠ Using {straddle_dte}DTE straddle (no same-day expiration "
            f"available) — reflects ~√{straddle_dte}× more vol than a true "
            f"0DTE would. Session classification is scaled accordingly."
        )

    # Straddle & range
    straddle = em_data.get("straddle", {})
    em_pts_val = em_data.get("expected_move_pts", 0) or 0
    em_pct_val = em_data.get("expected_move_pct", 0) or 0
    em_lower = em_data.get("lower_level")
    em_upper = em_data.get("upper_level")
    straddle_html = (
        '<div class="level-grid" style="grid-template-columns: 1fr 1fr;">'
        f'<div class="level-card"><div class="lbl">ATM Straddle</div><div class="val">{em_pts_val:.1f} pts</div><div class="lbl">{em_pct_val:.2f}%</div></div>'
        f'<div class="level-card"><div class="lbl">Strike</div><div class="val">${straddle.get("strike", "?")}</div></div>'
        '</div>'
    )
    st.markdown(straddle_html, unsafe_allow_html=True)

    if em_lower is not None and em_upper is not None:
        st.markdown(
            f"**Expected Range:** "
            f":red[${em_lower:.0f}] — :green[${em_upper:.0f}]"
        )

    _render_move_display(overnight, market_ctx)
    _render_classification(classification, level_ctx)

    st.divider()


def render_key_levels(levels, spot, regime_info, confidence_info, staleness_info):
    conf_score = confidence_info.get("score", 0)
    conf_label = confidence_info.get("label", "?")
    conf_color = COLORS["positive"] if conf_label == "High" else COLORS["warning"] if conf_label == "Moderate" else COLORS["negative"]

    spot_c = COLORS["spot"]
    cw_c = COLORS["call_wall"]
    pw_c = COLORS["put_wall"]
    zg_c = COLORS["zero_gamma"]
    html = (
        '<div class="level-grid">'
        f'<div class="level-card"><div class="lbl">Spot</div><div class="val" style="color:{spot_c};">${spot:.2f}</div></div>'
        f'<div class="level-card"><div class="lbl">Call Wall</div><div class="val" style="color:{cw_c};">${levels["call_wall"]:.0f}</div></div>'
        f'<div class="level-card"><div class="lbl">Put Wall</div><div class="val" style="color:{pw_c};">${levels["put_wall"]:.0f}</div></div>'
        f'<div class="level-card"><div class="lbl">Zero Gamma</div><div class="val" style="color:{zg_c};">${levels["zero_gamma"]:.2f}</div></div>'
        f'<div class="level-card"><div class="lbl">Regime</div><div class="val" style="color:{regime_info["color"]};font-size:12px;">{regime_info["regime"]}</div></div>'
        f'<div class="level-card"><div class="lbl">Confidence</div><div class="val" style="color:{conf_color};">{conf_score:.0f}</div><div class="lbl" style="color:{conf_color};">{conf_label}</div></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    if not levels.get("zero_gamma_is_true_crossing", True):
        zg_type = levels.get("zero_gamma_type", "Fallback node")
        if zg_type == "Rescued crossing":
            st.warning(
                "⚠️ Zero gamma is a **rescued crossing** — the coarse sweep "
                "missed a sign change and we only caught one by refining "
                "around the min-|GEX| node. Fragile to single-strike moves."
            )
        else:
            st.warning(
                "⚠️ Zero gamma is a fallback estimate — no true sign-change "
                "crossing was found in the sweep range. Use with caution."
            )

    st.caption(
        "**Dealer positioning assumption:** GEX models assume dealers are net short calls and net short puts "
        "(the standard retail convention). In reality, dealer positioning varies by strike — institutional "
        "overlays (collars, risk reversals) and retail put-selling can invert the sign at specific strikes. "
        "Open interest updates once daily (EOD), so intraday flow is not reflected. "
        "Treat GEX levels as probabilistic zones, not hard barriers."
    )


def render_wall_credibility(wall_cred):
    if not wall_cred:
        return
    st.markdown("#### Wall Credibility")
    for key, label in [("call_wall", "Call Wall"), ("put_wall", "Put Wall"), ("zero_gamma", "Zero Γ")]:
        info = wall_cred.get(key, {})
        if not info:
            continue
        score = info.get("score", 0)
        lbl = info.get("label", "?")
        color = "🟢" if lbl == "High" else "🟡" if lbl == "Moderate" else "🔴"
        st.markdown(f"{color} **{label}:** {score:.0f}/100 ({lbl})")
        reasons = info.get("reasons", [])
        for r in reasons[:2]:
            st.caption(f"  • {r}")


def render_gex_stream(stats, levels, spot):
    """Sidebar GEX Stream panel — key metrics at a glance."""
    st.markdown("#### 📡 GEX Stream")

    gex_ratio = stats.get("gex_ratio")  # may be None — see gex_engine.py
    net_gex = stats.get("net_gex", 0)
    pc_ratio = stats.get("pc_ratio", 0)
    call_iv = stats.get("call_iv", 0)
    put_iv = stats.get("put_iv", 0)

    # Color coding. gex_ratio is None in undefined cases: all-positive (+∞)
    # paints positive, all-negative (0) paints negative, both-empty is muted.
    if gex_ratio is None:
        # We can't distinguish "+∞" from "empty chain" at this point, but
        # the net_gex sign is a reliable proxy when there IS data.
        gr_color = COLORS["positive"] if net_gex > 0 else COLORS["negative"]
    else:
        gr_color = COLORS["positive"] if gex_ratio > 1 else COLORS["negative"]
    ng_color = COLORS["positive"] if net_gex > 0 else COLORS["negative"]
    cw_c = COLORS["call_wall"]
    pw_c = COLORS["put_wall"]
    zg_c = COLORS["zero_gamma"]
    text_w = COLORS["text_white"]
    text_m = COLORS["text_muted"]

    # Format net GEX
    ng_fmt = stats.get("net_gex_fmt", f"{net_gex:.0f}")

    # Net Charm / trading-hour — dealer-book $-delta drift per hour of
    # the session under the engine's SqueezeMetrics sign convention
    # (dealer long calls, short puts):
    #   positive net_charm  →  book gains delta  →  dealer SELLS  →
    #                          repelling / distributing flow (RED)
    #   negative net_charm  →  book loses delta  →  dealer BUYS  →
    #                          supportive / pinning flow (GREEN)
    # Note the inverted color logic vs Net GEX — there, positive = long
    # gamma = stabilizing (green); here, positive = repelling (red).
    # Amber label ties the KPI to the chart overlay line of the same
    # color.
    net_charm_per_hour = stats.get("net_charm_per_hour", 0.0)
    nc_fmt = stats.get("net_charm_per_hour_fmt", f"{net_charm_per_hour:,.0f}")
    if net_charm_per_hour > 0:
        nc_color = COLORS["negative"]   # repelling
    elif net_charm_per_hour < 0:
        nc_color = COLORS["positive"]   # supportive
    else:
        nc_color = COLORS["text_muted"]
    charm_accent = COLORS["charm_line"]

    # GEX Ratio display. None is the "undefined" sentinel from gex_engine
    # (all-positive regime → +∞, or empty chain). Disambiguate by checking
    # net_gex sign: positive net_gex with None ratio means +∞; otherwise
    # show "—" for truly undefined.
    if gex_ratio is None:
        if net_gex > 0:
            gr_display = "∞"
        elif net_gex < 0:
            gr_display = "0.00"   # all-negative, fall through to the usual low-ratio display
        else:
            gr_display = "—"
        gr_sigma_str = ""
    else:
        gr_display = f"{gex_ratio:.2f}"
        gr_sigma = abs(gex_ratio - 1.0) / 0.5
        gr_sigma_str = f"{gr_sigma:.1f}σ"

    stream_html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:12px;margin-bottom:8px;">
      <tr>
        <td style="color:{text_m};padding:3px 6px;">GEX Ratio</td>
        <td style="color:{gr_color};font-weight:bold;text-align:right;padding:3px 6px;">{gr_display}</td>
        <td style="color:{text_m};font-size:10px;text-align:right;padding:3px 6px;">{gr_sigma_str}</td>
        <td style="color:{text_m};padding:3px 6px;">Net GEX</td>
        <td style="color:{ng_color};font-weight:bold;text-align:right;padding:3px 6px;">{ng_fmt}</td>
      </tr>
      <tr style="border-top:1px solid #333;">
        <td style="color:{text_m};padding:3px 6px;">Call OI</td>
        <td colspan="2" style="color:{cw_c};font-weight:bold;text-align:right;padding:3px 6px;">{stats.get("call_oi", "0")} @ {stats.get("call_oi_strike", 0):.0f}</td>
        <td colspan="2" style="color:{text_m};"></td>
      </tr>
      <tr>
        <td style="color:{text_m};padding:3px 6px;">Pos GEX</td>
        <td colspan="2" style="color:{cw_c};font-weight:bold;text-align:right;padding:3px 6px;">{stats.get("pos_gex", "0")} @ {stats.get("pos_gex_strike", 0):.0f}</td>
        <td colspan="2" style="color:{text_m};"></td>
      </tr>
      <tr>
        <td style="color:{text_m};padding:3px 6px;">Zero Gamma</td>
        <td colspan="4" style="color:{zg_c};font-weight:bold;text-align:right;padding:3px 6px;">{levels.get("zero_gamma", 0):,.2f}</td>
      </tr>
      <tr>
        <td style="color:{text_m};padding:3px 6px;">Neg GEX</td>
        <td colspan="2" style="color:{pw_c};font-weight:bold;text-align:right;padding:3px 6px;">{stats.get("neg_gex", "0")} @ {stats.get("neg_gex_strike", 0):.0f}</td>
        <td colspan="2" style="color:{text_m};"></td>
      </tr>
      <tr>
        <td style="color:{text_m};padding:3px 6px;">Put OI</td>
        <td colspan="2" style="color:{pw_c};font-weight:bold;text-align:right;padding:3px 6px;">{stats.get("put_oi", "0")} @ {stats.get("put_oi_strike", 0):.0f}</td>
        <td colspan="2" style="color:{text_m};"></td>
      </tr>
      <tr style="border-top:1px solid #333;">
        <td style="color:{text_m};padding:3px 6px;">Call IV</td>
        <td style="color:{text_w};font-weight:bold;text-align:right;padding:3px 6px;">{call_iv:.1f}%</td>
        <td style="color:{text_m};padding:3px 6px;"></td>
        <td style="color:{text_m};padding:3px 6px;">Put IV</td>
        <td style="color:{text_w};font-weight:bold;text-align:right;padding:3px 6px;">{put_iv:.1f}%</td>
      </tr>
      <tr style="border-top:1px solid #333;">
        <td style="color:{text_m};padding:3px 6px;">P/C OI Ratio</td>
        <td colspan="4" style="color:{text_w};font-weight:bold;text-align:right;padding:3px 6px;">{pc_ratio:.2f}</td>
      </tr>
      <tr style="border-top:1px solid #333;">
        <td style="color:{charm_accent};padding:3px 6px;">Net Charm/hr</td>
        <td colspan="4" style="color:{nc_color};font-weight:bold;text-align:right;padding:3px 6px;">{nc_fmt}</td>
      </tr>
    </table>
    """
    st.markdown(stream_html, unsafe_allow_html=True)


def render_data_quality(stats, staleness_info):
    st.markdown("#### Data Quality")

    fresh = staleness_info.get("freshness_score", 0)
    fresh_lbl = staleness_info.get("freshness_label", "?")
    fresh_color = COLORS["positive"] if fresh_lbl == "High" else COLORS["warning"] if fresh_lbl == "Moderate" else COLORS["negative"]

    coverage = stats.get("coverage_ratio", 0)
    cov_color = COLORS["positive"] if coverage >= 0.95 else COLORS["warning"] if coverage >= 0.85 else COLORS["negative"]

    html = (
        '<div class="level-grid">'
        f'<div class="level-card"><div class="lbl">Used Options</div><div class="val">{stats.get("used_option_count", 0):,}</div></div>'
        f'<div class="level-card"><div class="lbl">Coverage</div><div class="val" style="color:{cov_color};">{coverage*100:.1f}%</div></div>'
        f'<div class="level-card"><div class="lbl">Freshness</div><div class="val" style="color:{fresh_color};">{fresh:.0f}</div><div class="lbl" style="color:{fresh_color};">{fresh_lbl}</div></div>'
        f'<div class="level-card"><div class="lbl">Direct IV</div><div class="val">{stats.get("direct_iv_count", 0):,}</div></div>'
        f'<div class="level-card"><div class="lbl">Synthetic IV</div><div class="val">{stats.get("synthetic_iv_count", 0):,}</div></div>'
        f'<div class="level-card"><div class="lbl">Skipped (quality)</div><div class="val">{stats.get("skipped_count", 0):,}</div></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)

    # Detailed filter breakdown
    range_filt = stats.get("range_filtered_count", 0)
    zero_oi = stats.get("zero_oi_filtered_count", 0)
    if range_filt or zero_oi:
        st.caption(f"Filtered: {range_filt:,} out-of-range strikes, {zero_oi:,} zero-OI contracts")

    # Volume amplification disclosure
    # --------------------------------------------------------------
    # GEX weights are max(OI, volume), which "unstales" EOD-reported
    # OI but overstates magnitude on churny 0DTE days where intraday
    # volume swamps settled OI. Surface the amplification ratio so
    # the trader knows when wall magnitudes are inflated. A ratio of
    # 1.0 means size==OI everywhere (no amplification). 2.0 means
    # the GEX scale is 2× what a pure-OI computation would produce.
    vol_ratio = stats.get("vol_amplification_ratio")
    vol_pct = stats.get("vol_dominated_pct", 0.0) or 0.0
    if vol_ratio is not None and vol_ratio > 1.10:
        # Only warn when amplification is material (>10% lift).
        if vol_ratio >= 1.75:
            va_color = COLORS["negative"]
            va_word = "heavy"
        elif vol_ratio >= 1.30:
            va_color = COLORS["warning"]
            va_word = "elevated"
        else:
            va_color = COLORS["text_secondary"]
            va_word = "mild"
        st.markdown(
            f"<div style='font-size:11px;color:{COLORS['text_muted']};margin-top:6px;'>"
            f"⚠ Volume amplification: "
            f"<span style='color:{va_color};font-weight:bold;'>"
            f"{vol_ratio:.2f}× ({va_word})</span> — "
            f"{vol_pct*100:.0f}% of strikes have today's volume > settled OI. "
            f"Wall magnitudes are inflated vs an OI-only computation; "
            f"wall <em>locations</em> are still reliable."
            f"</div>",
            unsafe_allow_html=True,
        )
