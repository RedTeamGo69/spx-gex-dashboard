from __future__ import annotations


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def build_run_confidence(stats: dict, spot_info: dict, staleness_info: dict | None = None) -> dict:
    score = 100.0
    reasons = []

    source = spot_info.get("source", "") or ""
    pdiag = spot_info.get("parity_diagnostics") or {}

    if source.startswith("vendor (forced, market closed)"):
        score -= 5
        reasons.append("Market closed: using vendor quote spot instead of live parity.")
    elif source.startswith("vendor"):
        score -= 12
        reasons.append("Reference spot did not use implied parity.")
    elif source.startswith("implied median"):
        score -= 3
        reasons.append("Using simple median parity instead of weighted parity.")

    coverage = stats.get("coverage_ratio")
    if coverage is not None:
        if coverage < 0.80:
            score -= 20
            reasons.append(f"Low coverage ratio ({coverage*100:.1f}%).")
        elif coverage < 0.90:
            score -= 12
            reasons.append(f"Moderate coverage ratio ({coverage*100:.1f}%).")
        elif coverage < 0.97:
            score -= 6
            reasons.append(f"Coverage slightly reduced ({coverage*100:.1f}%).")

    failed_exp_count = stats.get("failed_exp_count", 0) or 0
    if failed_exp_count > 0:
        penalty = min(20, failed_exp_count * 8)
        score -= penalty
        reasons.append(f"{failed_exp_count} expiration(s) failed to load.")

    synth_reject = stats.get("synthetic_fit_reject_count", 0) or 0
    if synth_reject > 0:
        penalty = min(12, synth_reject * 2)
        score -= penalty
        reasons.append(f"{synth_reject} synthetic-IV fit(s) were rejected.")

    synth_used = stats.get("synthetic_iv_count", 0) or 0
    used_total = max(stats.get("used_option_count", 0) or 0, 1)
    synth_share = synth_used / used_total
    if synth_share > 0.10:
        score -= 8
        reasons.append(f"Synthetic IV share is elevated ({synth_share*100:.1f}%).")
    elif synth_share > 0.03:
        score -= 4
        reasons.append(f"Synthetic IV share is non-trivial ({synth_share*100:.1f}%).")

    max_fit_err = stats.get("synthetic_fit_max_rel_error")
    if max_fit_err is not None:
        if max_fit_err > 0.05:
            score -= 8
            reasons.append(f"Max synthetic fit error is high ({max_fit_err*100:.2f}%).")
        elif max_fit_err > 0.03:
            score -= 4
            reasons.append(f"Max synthetic fit error is noticeable ({max_fit_err*100:.2f}%).")

    common_usable = pdiag.get("common_usable_strikes")
    hard_pass = pdiag.get("hard_filter_pass_count")
    simple_med = pdiag.get("simple_median_spot")
    weighted_med = pdiag.get("weighted_median_spot")

    if common_usable is not None and common_usable < 5 and spot_info.get("parity_attempted"):
        score -= 5
        reasons.append(f"Only {common_usable} common usable parity strikes.")

    if hard_pass is not None and hard_pass < 3 and spot_info.get("parity_attempted"):
        score -= 8
        reasons.append(f"Only {hard_pass} strikes survived final parity filters.")

    if (
        simple_med is not None
        and weighted_med is not None
        and abs(simple_med - weighted_med) > 5
    ):
        score -= 5
        reasons.append(
            f"Median vs weighted parity differ by {abs(simple_med - weighted_med):.2f} pts."
        )

    if staleness_info:
        freshness_score = staleness_info.get("freshness_score")
        if freshness_score is not None:
            if freshness_score < 60:
                score -= 12
                reasons.append(f"Market-data freshness is degraded ({freshness_score:.1f}/100).")
            elif freshness_score < 75:
                score -= 6
                reasons.append(f"Market-data freshness is mixed ({freshness_score:.1f}/100).")

    score = round(_clamp(score, 0.0, 100.0), 1)

    if score >= 85:
        label = "High"
    elif score >= 70:
        label = "Moderate"
    else:
        label = "Low"

    if not reasons:
        reasons = ["No major confidence penalties triggered."]

    return {
        "score": score,
        "label": label,
        "reasons": reasons[:6],
        "coverage_ratio": coverage,
        "synthetic_share": synth_share,
        "failed_exp_count": failed_exp_count,
        "spot_source": source,
    }