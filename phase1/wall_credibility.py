from __future__ import annotations

import math
import pandas as pd


def _clamp(x, lo=0.0, hi=100.0):
    return max(lo, min(hi, x))


def _label(score):
    if score >= 85:
        return "High"
    if score >= 70:
        return "Moderate"
    return "Low"


def _nearest_strike_row(strike_support_df: pd.DataFrame, target: float):
    if strike_support_df is None or strike_support_df.empty:
        return None

    df = strike_support_df.copy()
    df["distance_to_target"] = (df["strike"].astype(float) - float(target)).abs()
    row = df.sort_values(["distance_to_target", "support_score"], ascending=[True, False]).iloc[0]
    return row.to_dict()


def _score_wall_from_support(level_name: str, wall_level: float, strike_support_df: pd.DataFrame,
                             confidence_info: dict, staleness_info: dict):
    score = 100.0
    reasons = []

    row = _nearest_strike_row(strike_support_df, wall_level)
    if row is None:
        return {
            "level_name": level_name,
            "level_value": round(float(wall_level), 2),
            "score": 35.0,
            "label": "Low",
            "anchor_strike": None,
            "anchor_distance": None,
            "support_score": None,
            "support_label": None,
            "reasons": ["No strike-support data available near this wall."],
        }

    anchor_strike = float(row["strike"])
    distance = abs(anchor_strike - float(wall_level))
    support_score = float(row["support_score"])
    support_label = str(row["support_label"])
    supporting_expirations = int(row.get("supporting_expirations", 0) or 0)
    total_oi = float(row.get("total_oi", 0.0) or 0.0)

    # Main local support anchor
    score = 0.60 * support_score + 40.0

    # Distance penalty if wall doesn't line up tightly with supported strike
    if distance > 20:
        score -= 18
        reasons.append(f"Nearest supported strike is {distance:.1f} pts away.")
    elif distance > 10:
        score -= 10
        reasons.append(f"Nearest supported strike is {distance:.1f} pts away.")
    elif distance > 5:
        score -= 4
        reasons.append(f"Nearest supported strike is {distance:.1f} pts away.")
    else:
        reasons.append(f"Wall aligns closely with supported strike {anchor_strike:.0f}.")

    if support_label == "Low":
        score -= 14
        reasons.append("Nearest strike support is low.")
    elif support_label == "Moderate":
        score -= 5
        reasons.append("Nearest strike support is moderate.")
    else:
        reasons.append("Nearest strike support is high.")

    if supporting_expirations <= 1:
        score -= 6
        reasons.append("Wall support is concentrated in only one expiration.")
    elif supporting_expirations >= 3:
        reasons.append("Wall support is spread across multiple expirations.")

    if total_oi <= 0:
        score -= 8
        reasons.append("Very limited OI support near this wall.")

    # Mild overlays from overall run quality
    conf_score = float(confidence_info.get("score", 75.0) or 75.0)
    freshness_score = float(staleness_info.get("freshness_score", 75.0) or 75.0)

    score = 0.80 * score + 0.12 * conf_score + 0.08 * freshness_score
    score = round(_clamp(score), 1)

    return {
        "level_name": level_name,
        "level_value": round(float(wall_level), 2),
        "score": score,
        "label": _label(score),
        "anchor_strike": anchor_strike,
        "anchor_distance": round(distance, 2),
        "support_score": round(support_score, 1),
        "support_label": support_label,
        "reasons": reasons[:5],
    }


def _score_zero_gamma(levels: dict, sensitivity_df: pd.DataFrame, strike_support_df: pd.DataFrame,
                      confidence_info: dict, staleness_info: dict):
    zero_gamma = float(levels["zero_gamma"])
    score = 100.0
    reasons = []

    if sensitivity_df is None or sensitivity_df.empty:
        score = 45.0
        reasons.append("No zero-gamma sensitivity analysis available.")
        return {
            "level_name": "zero_gamma",
            "level_value": round(zero_gamma, 2),
            "score": score,
            "label": _label(score),
            "anchor_strike": None,
            "anchor_distance": None,
            "support_score": None,
            "support_label": None,
            "zg_range": None,
            "regime_consistency": None,
            "reasons": reasons,
        }

    zg_range = float(sensitivity_df["zero_gamma"].max() - sensitivity_df["zero_gamma"].min())
    base_regime = sensitivity_df.loc[sensitivity_df["shock_pct"] == 0.0, "regime"]
    base_regime = base_regime.iloc[0] if len(base_regime) else sensitivity_df["regime"].iloc[len(sensitivity_df)//2]
    regime_consistency = float((sensitivity_df["regime"] == base_regime).mean())

    # start from stability
    score = 100.0
 
    if not levels.get("zero_gamma_is_true_crossing", True):
        score -= 20
        reasons.append("Reported zero gamma is a fallback node, not a true sign-change crossing.")    

    if zg_range > 40:
        score -= 30
        reasons.append(f"Zero gamma shifts a lot across spot shocks (range {zg_range:.2f} pts).")
    elif zg_range > 25:
        score -= 18
        reasons.append(f"Zero gamma is somewhat unstable across shocks (range {zg_range:.2f} pts).")
    elif zg_range > 15:
        score -= 8
        reasons.append(f"Zero gamma has moderate shock sensitivity (range {zg_range:.2f} pts).")
    else:
        reasons.append(f"Zero gamma is fairly stable across shocks (range {zg_range:.2f} pts).")

    if regime_consistency < 0.60:
        score -= 18
        reasons.append(f"Gamma regime flips frequently across shocks ({regime_consistency*100:.1f}% consistency).")
    elif regime_consistency < 0.80:
        score -= 8
        reasons.append(f"Gamma regime consistency is mixed ({regime_consistency*100:.1f}%).")
    else:
        reasons.append(f"Gamma regime remains mostly consistent ({regime_consistency*100:.1f}%).")

    # local strike anchor near zero gamma
    row = _nearest_strike_row(strike_support_df, zero_gamma)
    if row is not None:
        anchor_strike = float(row["strike"])
        distance = abs(anchor_strike - zero_gamma)
        support_score = float(row["support_score"])
        support_label = str(row["support_label"])

        score = 0.75 * score + 0.25 * support_score

        if distance > 20:
            score -= 10
            reasons.append(f"Nearest supported strike is {distance:.1f} pts from zero gamma.")
        elif distance > 10:
            score -= 5
            reasons.append(f"Nearest supported strike is {distance:.1f} pts from zero gamma.")
        else:
            reasons.append(f"Zero gamma sits near supported strike {anchor_strike:.0f}.")
    else:
        anchor_strike = None
        distance = None
        support_score = None
        support_label = None
        score -= 10
        reasons.append("No supported strike found near zero gamma.")

    conf_score = float(confidence_info.get("score", 75.0) or 75.0)
    freshness_score = float(staleness_info.get("freshness_score", 75.0) or 75.0)
    score = 0.80 * score + 0.12 * conf_score + 0.08 * freshness_score
    score = round(_clamp(score), 1)

    return {
        "level_name": "zero_gamma",
        "level_value": round(zero_gamma, 2),
        "score": score,
        "label": _label(score),
        "anchor_strike": anchor_strike,
        "anchor_distance": round(distance, 2) if distance is not None else None,
        "support_score": round(support_score, 1) if support_score is not None else None,
        "support_label": support_label,
        "zg_range": round(zg_range, 2),
        "regime_consistency": round(regime_consistency, 4),
        "reasons": reasons[:5],
    }


def build_wall_credibility(levels: dict, strike_support_df: pd.DataFrame, sensitivity_df: pd.DataFrame,
                           confidence_info: dict, staleness_info: dict) -> dict:
    call_wall = _score_wall_from_support(
        "call_wall",
        levels["call_wall"],
        strike_support_df,
        confidence_info,
        staleness_info,
    )
    put_wall = _score_wall_from_support(
        "put_wall",
        levels["put_wall"],
        strike_support_df,
        confidence_info,
        staleness_info,
    )
    zero_gamma = _score_zero_gamma(
        levels,
        sensitivity_df,
        strike_support_df,
        confidence_info,
        staleness_info,
    )

    return {
        "call_wall": call_wall,
        "put_wall": put_wall,
        "zero_gamma": zero_gamma,
    }