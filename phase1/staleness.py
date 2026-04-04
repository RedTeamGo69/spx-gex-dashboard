from __future__ import annotations

from phase1.config import (
    STALE_FRESHNESS_HIGH_THRESHOLD,
    STALE_FRESHNESS_MODERATE_THRESHOLD,
    STALE_NO_TWO_SIDED_RATIO_WARN,
    STALE_WIDE_SPREAD_RATIO_WARN,
    STALE_CROSSED_RATIO_WARN,
    STALE_MIN_HARD_FILTER_PASS,
    SUPPORT_MODERATE_THRESHOLD,
)


def _ratio(num, den):
    if den is None or den <= 0:
        return 0.0
    return float(num) / float(den)


def _label_from_score(score):
    if score >= STALE_FRESHNESS_HIGH_THRESHOLD:
        return "High"
    if score >= STALE_FRESHNESS_MODERATE_THRESHOLD:
        return "Moderate"
    return "Low"


def build_staleness_info(calendar_snapshot: dict, spot_info: dict, stats: dict, has_0dte: bool = False) -> dict:
    """
    Build a market-data freshness / stale-risk assessment using observable signals.

    Important:
    This is not true per-contract timestamp freshness unless your data vendor exposes it.
    It is a defensive proxy based on market context + quote quality + filter survivorship.
    """
    score = 100.0
    reasons = []
    defenses_triggered = []

    market_open = bool(spot_info.get("market_open"))
    spot_source = spot_info.get("source", "") or ""
    parity_attempted = bool(spot_info.get("parity_attempted"))
    parity_chain_status = spot_info.get("parity_chain_status")
    pdiag = spot_info.get("parity_diagnostics") or {}

    # Market closed => not stale exactly, but not suitable for live intraday execution
    if not market_open:
        score -= 18
        reasons.append("Market is closed; spot is not a live intraday executable reference.")
        defenses_triggered.append("market_closed_context")

    # Forced or degraded Tradier spot usage
    if spot_source.startswith("tradier (forced, market closed)"):
        score -= 6
        reasons.append("Spot source was forced to Tradier because the market was closed.")
        defenses_triggered.append("forced_tradier_spot")
    elif spot_source.startswith("tradier") and market_open:
        score -= 18
        reasons.append("Market open but implied parity did not supply the reference spot.")
        defenses_triggered.append("tradier_spot_during_market_hours")

    # Parity chain failure
    if parity_attempted and parity_chain_status not in (None, "ok"):
        score -= 20
        reasons.append("Parity chain failed or was unavailable.")
        defenses_triggered.append("parity_chain_failure")

    call_q = pdiag.get("call_quality") or {}
    put_q = pdiag.get("put_quality") or {}

    for side_name, q in [("calls", call_q), ("puts", put_q)]:
        total = q.get("total", 0) or 0
        no2s_ratio = _ratio(q.get("no_two_sided_quote", 0), total)
        wide_ratio = _ratio(q.get("wide_spread", 0), total)
        # Only penalize truly crossed quotes (bid > ask); locked quotes (bid == ask)
        # are valid tight markets and should not trigger staleness warnings.
        crossed_ratio = _ratio(q.get("crossed", 0), total)

        if no2s_ratio > STALE_NO_TWO_SIDED_RATIO_WARN:
            score -= 8
            reasons.append(f"{side_name.capitalize()} have a high no-two-sided-quote ratio ({no2s_ratio*100:.1f}%).")
            defenses_triggered.append(f"{side_name}_no_two_sided_warn")

        if wide_ratio > STALE_WIDE_SPREAD_RATIO_WARN:
            score -= 8
            reasons.append(f"{side_name.capitalize()} have a high wide-spread ratio ({wide_ratio*100:.1f}%).")
            defenses_triggered.append(f"{side_name}_wide_spread_warn")

        if crossed_ratio > STALE_CROSSED_RATIO_WARN:
            score -= 10
            reasons.append(f"{side_name.capitalize()} have a high crossed-quote ratio ({crossed_ratio*100:.1f}%).")
            defenses_triggered.append(f"{side_name}_crossed_warn")

    hard_pass = pdiag.get("hard_filter_pass_count")
    if parity_attempted and hard_pass is not None and hard_pass < STALE_MIN_HARD_FILTER_PASS:
        score -= 12
        reasons.append(f"Only {hard_pass} strikes survived final parity filters.")
        defenses_triggered.append("weak_parity_survivorship")

    failed_exp_count = stats.get("failed_exp_count", 0) or 0
    if failed_exp_count > 0:
        score -= min(15, failed_exp_count * 6)
        reasons.append(f"{failed_exp_count} expiration(s) failed to load.")
        defenses_triggered.append("failed_expirations")

    strike_support_avg = stats.get("strike_support_avg")
    if strike_support_avg is not None and strike_support_avg < SUPPORT_MODERATE_THRESHOLD:
        score -= 10
        reasons.append(f"Average strike support is weak ({strike_support_avg:.1f}).")
        defenses_triggered.append("weak_strike_support")

    fragile_strike_count = stats.get("fragile_strike_count", 0) or 0
    if fragile_strike_count >= 3:
        score -= 6
        reasons.append(f"There are {fragile_strike_count} fragile strikes in the selected range.")
        defenses_triggered.append("many_fragile_strikes")

    # 0DTE OI staleness: OI updates once daily (EOD), so intraday 0DTE positioning
    # can diverge significantly from what the OI field reflects.
    if has_0dte and market_open:
        score -= 10
        reasons.append(
            "0DTE OI is end-of-day data — intraday opening/closing flow is not reflected. "
            "Actual gamma positioning may differ substantially from what EOD OI shows."
        )
        defenses_triggered.append("0dte_oi_staleness")

    score = max(0.0, min(100.0, round(score, 1)))
    label = _label_from_score(score)

    if label == "High":
        trading_guidance = "Freshness/quote-quality conditions look acceptable for analysis."
    elif label == "Moderate":
        trading_guidance = "Use caution: market-data quality is mixed."
    else:
        trading_guidance = "Degraded: treat outputs cautiously and avoid over-trusting precise levels."

    if not reasons:
        reasons = ["No major stale-data defenses were triggered."]

    return {
        "freshness_score": score,
        "freshness_label": label,
        "reasons": reasons[:6],
        "defenses_triggered": defenses_triggered,
        "trading_guidance": trading_guidance,
        "spot_source": spot_source,
        "market_open": market_open,
    }