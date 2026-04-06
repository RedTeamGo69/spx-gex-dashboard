"""
GEX wall detection and gamma regime classification.

Identifies call walls, put walls, and zero-gamma levels from the GEX
strike distribution. Uses cluster-based wall identification for robustness.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from phase1.config import DEFAULT_RISK_FREE_RATE
from phase1.zero_gamma import (
    zero_gamma_sweep_details,
    _estimate_atm_iv,
    _compute_per_expiry_zero_gamma,
)


def get_gamma_regime_text(spot, zero_gamma):
    dist = spot - zero_gamma
    abs_dist = abs(dist)

    if dist > 0:
        regime = "Positive Gamma"
        color = "#00c853"
        note = "0DTE read: above zero gamma, price action is more likely to pin, mean-revert, or slow down near key levels."
    elif dist < 0:
        regime = "Negative Gamma"
        color = "#ff5252"
        note = "0DTE read: below zero gamma, moves can expand faster, breakouts can carry farther, and reversals may be less stable."
    else:
        regime = "At Zero Gamma"
        color = "#00e5ff"
        note = "0DTE read: right at the flip zone, market behavior can change quickly between pinning and expansion."

    return {
        "regime": regime,
        "color": color,
        "distance_text": f"{dist:+.2f} pts from zero gamma",
        "note": note,
        "abs_distance": abs_dist,
    }


def _find_wall_cluster(df_subset, sort_col, ascending, cluster_radius=25, top_n=5):
    """
    Find a wall zone by clustering the top N strikes by GEX magnitude.

    Instead of just picking the single highest-GEX strike, we find the top N
    strikes, identify which ones cluster within `cluster_radius` points of the
    peak, and return a GEX-weighted centroid of that cluster.

    Returns dict with centroid, peak strike, cluster strikes, and total GEX.
    """
    if df_subset.empty:
        return None

    sorted_df = df_subset.sort_values(sort_col, ascending=ascending)
    top = sorted_df.head(top_n)

    peak_strike = float(top.iloc[0]["strike"])
    peak_gex = float(top.iloc[0]["net_gex"])

    # Find strikes that cluster near the peak
    cluster_mask = (top["strike"] - peak_strike).abs() <= cluster_radius
    cluster = top[cluster_mask]

    if len(cluster) <= 1:
        return {
            "strike": peak_strike,
            "gex": peak_gex,
            "centroid": peak_strike,
            "cluster_strikes": [peak_strike],
            "cluster_gex_total": peak_gex,
            "is_cluster": False,
        }

    # GEX-weighted centroid
    weights = cluster["net_gex"].abs().values
    strikes = cluster["strike"].values.astype(float)
    w_sum = weights.sum()
    centroid = float(np.average(strikes, weights=weights)) if w_sum > 0 else peak_strike

    return {
        "strike": peak_strike,
        "gex": peak_gex,
        "centroid": round(centroid, 2),
        "cluster_strikes": sorted(strikes.tolist()),
        "cluster_gex_total": float(cluster["net_gex"].sum()),
        "is_cluster": True,
    }


def find_key_levels(gex_df, spot, all_options=None, r=DEFAULT_RISK_FREE_RATE):
    if gex_df.empty:
        return {
            "call_wall": spot,
            "call_wall_gex": 0.0,
            "put_wall": spot,
            "put_wall_gex": 0.0,
            "zero_gamma": spot,
        }

    pos = gex_df[gex_df["net_gex"] > 0]
    neg = gex_df[gex_df["net_gex"] < 0]

    # Cluster-based wall identification
    cw_cluster = _find_wall_cluster(pos if not pos.empty else gex_df, "net_gex", ascending=False)
    pw_cluster = _find_wall_cluster(neg if not neg.empty else gex_df, "net_gex", ascending=True)

    if not pos.empty:
        cw = pos.loc[pos["net_gex"].idxmax()]
    else:
        cw = gex_df.loc[gex_df["net_gex"].idxmax()]

    if not neg.empty:
        pw = neg.loc[neg["net_gex"].idxmin()]
    else:
        pw = gex_df.loc[gex_df["net_gex"].idxmin()]

    if all_options:
        # Compute ATM IV for dynamic sweep range
        atm_iv = _estimate_atm_iv(all_options, spot)
        zg_details = zero_gamma_sweep_details(all_options, spot, r=r, atm_iv=atm_iv)
        zg = zg_details["zero_gamma"]
        print(
            f"  Zero gamma (sweep): ${zg:.2f} "
            f"[{zg_details['zero_gamma_type']}, {zg_details['method']}]"
        )
    else:
        s = gex_df.sort_values("strike").reset_index(drop=True)
        s["cum"] = s["net_gex"].cumsum()
        zg = spot
        best_dist = float("inf")
        for i in range(1, len(s)):
            if s.loc[i - 1, "cum"] * s.loc[i, "cum"] < 0:
                s1, s2 = s.loc[i - 1, "strike"], s.loc[i, "strike"]
                p, c = abs(s.loc[i - 1, "cum"]), abs(s.loc[i, "cum"])
                cr = s1 + (s2 - s1) * p / (p + c)
                if abs(cr - spot) < best_dist:
                    best_dist = abs(cr - spot)
                    zg = cr
        zg_details = {
            "zero_gamma": round(float(zg), 2),
            "is_true_crossing": False,
            "zero_gamma_type": "Fallback node",
            "method": "cumulative_fallback",
            "final_abs_gex": None,
        }
        print(f"  Zero gamma (cumulative fallback): ${zg:.2f}")

    # Per-expiry zero-gamma (0DTE vs rest)
    nearest_exp = None
    if all_options:
        exps_in_options = set(o[5] for o in all_options if len(o) > 5)
        if exps_in_options:
            nearest_exp = min(exps_in_options)
    per_exp_zg = _compute_per_expiry_zero_gamma(all_options, spot, r, nearest_exp)

    if per_exp_zg["nearest_exp_zero_gamma"] is not None:
        print(
            f"  Zero gamma (0DTE {nearest_exp}): ${per_exp_zg['nearest_exp_zero_gamma']:.2f} "
            f"({per_exp_zg['nearest_exp_option_count']} opts)"
        )
    if per_exp_zg["other_exp_zero_gamma"] is not None:
        print(
            f"  Zero gamma (other exps): ${per_exp_zg['other_exp_zero_gamma']:.2f} "
            f"({per_exp_zg['other_exp_option_count']} opts)"
        )

    return {
        "call_wall": float(cw["strike"]),
        "call_wall_gex": float(cw["net_gex"]),
        "call_wall_cluster": cw_cluster,
        "put_wall": float(pw["strike"]),
        "put_wall_gex": float(pw["net_gex"]),
        "put_wall_cluster": pw_cluster,
        "zero_gamma": round(float(zg_details['zero_gamma']), 2),
        "zero_gamma_is_true_crossing": bool(zg_details["is_true_crossing"]),
        "zero_gamma_type": zg_details["zero_gamma_type"],
        "zero_gamma_method": zg_details["method"],
        "zero_gamma_abs_gex": zg_details["final_abs_gex"],
        "per_exp_zero_gamma": per_exp_zg,
    }
