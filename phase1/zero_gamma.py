"""
Zero-gamma sweep logic — finds the price where total GEX crosses zero.

Includes coarse/fine sweep, profile curve computation, per-expiry zero-gamma
decomposition, and sensitivity analysis under spot shocks.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import norm

from phase1.config import (
    ZG_SWEEP_RANGE_PCT,
    ZG_SWEEP_STEP,
    ZG_FINE_STEP,
    ZG_SWEEP_DYNAMIC,
    ZG_SWEEP_MIN_RANGE_PCT,
    ZG_SWEEP_MAX_RANGE_PCT,
    ZG_SWEEP_IV_SCALE,
    PROFILE_RANGE_PCT,
    PROFILE_STEP,
    DEFAULT_RISK_FREE_RATE,
)
from phase1.gex_engine import bs_gamma_vec
from phase1.model_inputs import bs_gamma


def _sweep_gex_at_prices(all_options, test_prices, r):
    """
    Compute total signed GEX proxy at each test price.

    Returns a numpy array (not a list) so callers can safely use np.argmin, etc.
    """
    test_prices = np.asarray(test_prices, dtype=float)

    if len(test_prices) == 0:
        return np.array([], dtype=float)

    if not all_options:
        return np.zeros(len(test_prices), dtype=float)

    K_arr = np.array([o[0] for o in all_options], dtype=float)
    oi_arr = np.array([o[1] for o in all_options], dtype=float)
    iv_arr = np.array([o[2] for o in all_options], dtype=float)
    sign_arr = np.array([o[3] for o in all_options], dtype=float)
    T_arr = np.array([o[4] for o in all_options], dtype=float)

    gamma_matrix = bs_gamma_vec(test_prices, K_arr, T_arr, r, iv_arr)
    weights = sign_arr * oi_arr * 100.0
    total_gex = (gamma_matrix * weights).sum(axis=1) * (test_prices ** 2)

    return total_gex


def _find_nearest_crossing_details(test_prices, gex_values, spot):
    best = None

    for i in range(1, len(test_prices)):
        p1 = float(test_prices[i - 1])
        p2 = float(test_prices[i])
        g1 = float(gex_values[i - 1])
        g2 = float(gex_values[i])

        crossing = None

        if g1 == 0.0:
            crossing = p1
        elif g2 == 0.0:
            crossing = p2
        elif g1 * g2 < 0:
            a, b = abs(g1), abs(g2)
            crossing = p1 + (p2 - p1) * a / (a + b)

        if crossing is None:
            continue

        dist = abs(crossing - float(spot))

        if best is None or dist < best["distance_to_spot"]:
            best = {
                "crossing": float(crossing),
                "distance_to_spot": float(dist),
                "left_price": p1,
                "right_price": p2,
                "left_gex": g1,
                "right_gex": g2,
            }

    return best


def _find_nearest_crossing(test_prices, gex_values, spot):
    details = _find_nearest_crossing_details(test_prices, gex_values, spot)
    return None if details is None else details["crossing"]


def _compute_sweep_range_pct(atm_iv=None):
    """Compute sweep range pct, optionally scaling by ATM IV."""
    if ZG_SWEEP_DYNAMIC and atm_iv and atm_iv > 0:
        return max(ZG_SWEEP_MIN_RANGE_PCT, min(ZG_SWEEP_MAX_RANGE_PCT, atm_iv * ZG_SWEEP_IV_SCALE))
    return ZG_SWEEP_RANGE_PCT


def zero_gamma_sweep_details(all_options, spot, r=DEFAULT_RISK_FREE_RATE, atm_iv=None):
    """
    Return rich diagnostics for zero-gamma solving.

    A true zero gamma requires a sign change in total GEX across price.
    If no sign change exists in the sweep window, we fall back to the
    minimum-absolute-GEX node and mark it as a fallback.

    If atm_iv is provided and ZG_SWEEP_DYNAMIC is True, the sweep range
    scales with volatility for better coverage in high-vol environments.
    """
    if not all_options:
        return {
            "zero_gamma": round(float(spot), 2),
            "is_true_crossing": False,
            "zero_gamma_type": "Fallback node",
            "method": "no_options",
            "coarse_crossing_found": False,
            "fine_crossing_found": False,
            "final_abs_gex": None,
            "sweep_low": None,
            "sweep_high": None,
        }

    range_pct = _compute_sweep_range_pct(atm_iv)
    lo = float(spot) * (1 - range_pct)
    hi = float(spot) * (1 + range_pct)

    coarse_prices = np.arange(lo, hi + ZG_SWEEP_STEP, ZG_SWEEP_STEP, dtype=float)
    coarse_gex = _sweep_gex_at_prices(all_options, coarse_prices, r)
    coarse_cross = _find_nearest_crossing_details(coarse_prices, coarse_gex, spot)

    if coarse_cross is not None:
        fine_lo = coarse_cross["crossing"] - ZG_SWEEP_STEP
        fine_hi = coarse_cross["crossing"] + ZG_SWEEP_STEP
        fine_prices = np.arange(fine_lo, fine_hi + ZG_FINE_STEP, ZG_FINE_STEP, dtype=float)
        fine_gex = _sweep_gex_at_prices(all_options, fine_prices, r)
        fine_cross = _find_nearest_crossing_details(fine_prices, fine_gex, spot)

        if fine_cross is not None:
            return {
                "zero_gamma": round(float(fine_cross["crossing"]), 2),
                "is_true_crossing": True,
                "zero_gamma_type": "True crossing",
                "method": "crossing_fine",
                "coarse_crossing_found": True,
                "fine_crossing_found": True,
                "final_abs_gex": 0.0,
                "sweep_low": round(lo, 2),
                "sweep_high": round(hi, 2),
            }

        return {
            "zero_gamma": round(float(coarse_cross["crossing"]), 2),
            "is_true_crossing": True,
            "zero_gamma_type": "True crossing",
            "method": "crossing_coarse",
            "coarse_crossing_found": True,
            "fine_crossing_found": False,
            "final_abs_gex": 0.0,
            "sweep_low": round(lo, 2),
            "sweep_high": round(hi, 2),
        }

    coarse_gex = np.asarray(coarse_gex)
    if len(coarse_gex) == 0:
        return {
            "zero_gamma": round(float(spot), 2),
            "is_true_crossing": False,
            "zero_gamma_type": "Fallback node",
            "method": "empty_sweep",
            "coarse_crossing_found": False,
            "fine_crossing_found": False,
            "final_abs_gex": None,
            "sweep_low": round(lo, 2),
            "sweep_high": round(hi, 2),
        }

    min_idx = int(np.argmin(np.abs(coarse_gex)))
    fallback_center = float(coarse_prices[min_idx])

    fine_lo = fallback_center - ZG_SWEEP_STEP
    fine_hi = fallback_center + ZG_SWEEP_STEP
    fine_prices = np.arange(fine_lo, fine_hi + ZG_FINE_STEP, ZG_FINE_STEP, dtype=float)
    fine_gex = _sweep_gex_at_prices(all_options, fine_prices, r)

    fine_cross = _find_nearest_crossing_details(fine_prices, fine_gex, spot)
    if fine_cross is not None:
        return {
            "zero_gamma": round(float(fine_cross["crossing"]), 2),
            "is_true_crossing": True,
            "zero_gamma_type": "True crossing",
            "method": "crossing_from_fallback_refine",
            "coarse_crossing_found": False,
            "fine_crossing_found": True,
            "final_abs_gex": 0.0,
            "sweep_low": round(lo, 2),
            "sweep_high": round(hi, 2),
        }

    fine_min_idx = int(np.argmin(np.abs(fine_gex)))
    fallback_node = float(fine_prices[fine_min_idx])
    fallback_abs_gex = float(abs(fine_gex[fine_min_idx]))

    return {
        "zero_gamma": round(fallback_node, 2),
        "is_true_crossing": False,
        "zero_gamma_type": "Fallback node",
        "method": "min_abs_fallback",
        "coarse_crossing_found": False,
        "fine_crossing_found": False,
        "final_abs_gex": round(fallback_abs_gex, 4),
        "sweep_low": round(lo, 2),
        "sweep_high": round(hi, 2),
    }


def zero_gamma_sweep(all_options, spot, r=DEFAULT_RISK_FREE_RATE, atm_iv=None):
    return zero_gamma_sweep_details(all_options, spot, r=r, atm_iv=atm_iv)["zero_gamma"]


def compute_gex_profile_curve(all_options, spot, r=DEFAULT_RISK_FREE_RATE, atm_iv=None):
    if not all_options:
        return pd.DataFrame(columns=["price", "total_gex"])

    range_pct = _compute_sweep_range_pct(atm_iv) if atm_iv else PROFILE_RANGE_PCT
    lo = math.floor(spot * (1 - range_pct))
    hi = math.ceil(spot * (1 + range_pct))
    prices = np.arange(lo, hi + PROFILE_STEP, PROFILE_STEP, dtype=float)
    total_gex = _sweep_gex_at_prices(all_options, prices, r)

    return pd.DataFrame(
        {
            "price": prices,
            "total_gex": total_gex,
        }
    )


def _estimate_atm_iv(all_options, spot):
    """
    Estimate ATM IV from nearest options to spot using inverse-distance weighting.

    Uses the 4 nearest options (by strike) with valid IV, weighted by
    proximity to spot so that truly ATM options dominate the estimate.
    """
    if not all_options:
        return None
    nearest = sorted(all_options, key=lambda o: abs(o[0] - spot))[:4]
    valid = [(o[0], o[2]) for o in nearest if o[2] > 0]
    if not valid:
        return None
    strikes, ivs = zip(*valid)
    distances = [max(abs(k - spot), 0.01) for k in strikes]
    weights = [1.0 / d for d in distances]
    w_sum = sum(weights)
    return float(sum(iv * w / w_sum for iv, w in zip(ivs, weights)))


def _compute_per_expiry_zero_gamma(all_options, spot, r, nearest_exp=None):
    """
    Compute zero-gamma separately for the nearest expiry (typically 0DTE) and
    the remaining expirations. This reveals whether intraday gamma is dominated
    by 0DTE flow vs. multi-day positions.

    Returns dict with 'nearest_exp_zero_gamma' and 'other_exp_zero_gamma', or
    None values if insufficient data.
    """
    if not all_options or nearest_exp is None:
        return {
            "nearest_exp": nearest_exp,
            "nearest_exp_zero_gamma": None,
            "nearest_exp_option_count": 0,
            "other_exp_zero_gamma": None,
            "other_exp_option_count": 0,
        }

    nearest_opts = [o for o in all_options if len(o) > 5 and o[5] == nearest_exp]
    other_opts = [o for o in all_options if len(o) > 5 and o[5] != nearest_exp]

    result = {
        "nearest_exp": nearest_exp,
        "nearest_exp_zero_gamma": None,
        "nearest_exp_option_count": len(nearest_opts),
        "other_exp_zero_gamma": None,
        "other_exp_option_count": len(other_opts),
    }

    if len(nearest_opts) >= 4:
        atm_iv = _estimate_atm_iv(nearest_opts, spot)
        zg = zero_gamma_sweep(nearest_opts, spot, r=r, atm_iv=atm_iv)
        result["nearest_exp_zero_gamma"] = round(float(zg), 2)

    if len(other_opts) >= 4:
        atm_iv = _estimate_atm_iv(other_opts, spot)
        zg = zero_gamma_sweep(other_opts, spot, r=r, atm_iv=atm_iv)
        result["other_exp_zero_gamma"] = round(float(zg), 2)

    return result


def compute_zero_gamma_sensitivity(all_options, spot, r=DEFAULT_RISK_FREE_RATE, shock_percents=None):
    """
    Recompute zero gamma under a few spot shocks to judge stability.
    """
    if shock_percents is None:
        shock_percents = [-0.005, -0.0025, 0.0, 0.0025, 0.005]

    rows = []
    for shock in shock_percents:
        shocked_spot = float(spot) * (1.0 + float(shock))

        if all_options:
            zg_details = zero_gamma_sweep_details(all_options, shocked_spot, r=r)
            zg = zg_details["zero_gamma"]
        else:
            zg_details = {
                "zero_gamma": round(shocked_spot, 2),
                "is_true_crossing": False,
                "zero_gamma_type": "Fallback node",
                "method": "no_options",
                "final_abs_gex": None,
            }
            zg = zg_details["zero_gamma"]

        gap = round(shocked_spot - zg, 2)

        if gap > 0:
            regime = "Positive Gamma"
        elif gap < 0:
            regime = "Negative Gamma"
        else:
            regime = "At Zero Gamma"

        rows.append(
            {
                "shock_pct": float(shock),
                "shocked_spot": round(shocked_spot, 2),
                "zero_gamma": round(float(zg), 2),
                "spot_minus_zero_gamma": gap,
                "regime": regime,
                "zero_gamma_type": zg_details["zero_gamma_type"],
                "zero_gamma_method": zg_details["method"],
                "residual_abs_gex": zg_details["final_abs_gex"],
            }
        )

    return pd.DataFrame(rows)
