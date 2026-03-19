from __future__ import annotations

import numpy as np
from scipy.stats import norm

from phase1.config import (
    SYNTH_IV_MIN,
    SYNTH_IV_MAX,
    SYNTH_FIT_MAX_REL_ERROR,
    HYBRID_IV_MODE,
)


def bs_gamma(S, K, T, r, sigma):
    """
    Scalar Black-Scholes gamma.
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def infer_iv_from_gamma(target_gamma, S, K, T, r, low=SYNTH_IV_MIN, high=SYNTH_IV_MAX, steps=70):
    """
    Infer a synthetic IV that matches vendor gamma at current spot.

    BS gamma(sigma) is unimodal (rises to a peak, then falls), so naive bisection
    can fail. We locate the peak, then use Brent's method on the sub-interval(s)
    that bracket the target. Prefers the lower-sigma (left) solution.

    Returns:
        inferred sigma (float) or 0.0 if no reasonable solution.
    """
    from scipy.optimize import brentq

    if target_gamma <= 0 or S <= 0 or K <= 0 or T <= 0:
        return 0.0

    def gamma_residual(sigma):
        return bs_gamma(S, K, T, r, sigma) - target_gamma

    # Dense grid to find the peak of gamma(sigma) and bracket solutions
    n_grid = 200
    grid = np.linspace(low, high, n_grid)
    gammas = np.array([bs_gamma(S, K, T, r, s) for s in grid])

    peak_idx = int(np.argmax(gammas))
    peak_gamma = float(gammas[peak_idx])

    # Target is above the peak — no solution exists
    if target_gamma > peak_gamma * 1.02:
        return 0.0

    # Try left side of peak first (lower sigma — more financially meaningful)
    solutions = []
    for seg_start, seg_end in [(0, peak_idx), (peak_idx, n_grid - 1)]:
        if seg_end <= seg_start:
            continue
        seg_gammas = gammas[seg_start:seg_end + 1]
        seg_sigmas = grid[seg_start:seg_end + 1]

        # Find sub-intervals where residual changes sign
        residuals = seg_gammas - target_gamma
        for i in range(len(residuals) - 1):
            if residuals[i] * residuals[i + 1] < 0:
                try:
                    sol = brentq(gamma_residual, float(seg_sigmas[i]), float(seg_sigmas[i + 1]),
                                 xtol=1e-8, rtol=1e-8, maxiter=100)
                    solutions.append(sol)
                except ValueError:
                    continue
            elif abs(residuals[i]) < 1e-12:
                solutions.append(float(seg_sigmas[i]))

    if not solutions:
        # Grid fallback for near-peak targets
        idx = int(np.argmin(np.abs(gammas - target_gamma)))
        best_sigma = float(grid[idx])
        best_gamma = float(gammas[idx])
        if best_gamma > 0 and abs(best_gamma - target_gamma) / target_gamma < 0.08:
            return best_sigma
        return 0.0

    # Prefer the lower-sigma solution (left side of gamma peak)
    return float(min(solutions))


def fit_synthetic_iv(target_gamma, S, K, T, r):
    """
    Fit synthetic IV from vendor gamma and score the fit.

    Returns a dict:
    {
        "accepted": bool,
        "iv": float,
        "target_gamma": float,
        "fitted_gamma": float,
        "rel_error": float | None,
        "reason": str,
    }
    """
    if target_gamma <= 0 or S <= 0 or K <= 0 or T <= 0:
        return {
            "accepted": False,
            "iv": 0.0,
            "target_gamma": target_gamma,
            "fitted_gamma": 0.0,
            "rel_error": None,
            "reason": "invalid_inputs",
        }

    iv = infer_iv_from_gamma(target_gamma, S, K, T, r)
    if iv <= 0:
        return {
            "accepted": False,
            "iv": 0.0,
            "target_gamma": target_gamma,
            "fitted_gamma": 0.0,
            "rel_error": None,
            "reason": "no_solution",
        }

    fitted_gamma = float(bs_gamma(S, K, T, r, iv))
    rel_error = float(abs(fitted_gamma - target_gamma) / max(target_gamma, 1e-12))
    accepted = bool(rel_error <= SYNTH_FIT_MAX_REL_ERROR)

    return {
        "accepted": accepted,
        "iv": float(iv),
        "target_gamma": float(target_gamma),
        "fitted_gamma": fitted_gamma,
        "rel_error": rel_error,
        "reason": "accepted" if accepted else "fit_error_too_high",
    }


def prepare_option_for_model(opt, sign, T, spot, r):
    """
    Rich evaluation path for model input selection.

    Returns a dict:
    {
      "accepted": bool,
      "reason": str,
      "normalized": dict | None,
      "synthetic_fit_rel_error": float | None,
      "synthetic_target_gamma": float | None,
      "synthetic_fitted_gamma": float | None,
    }
    """
    K = opt["strike"]
    oi = opt["openInterest"]
    iv = opt.get("impliedVolatility", 0.0) or 0.0
    vendor_gamma = opt.get("vendorGamma", 0.0) or 0.0

    if iv > 0:
        gamma_now = bs_gamma(spot, K, T, r, iv)
        return {
            "accepted": True,
            "reason": "direct_iv",
            "synthetic_fit_rel_error": None,
            "synthetic_target_gamma": None,
            "synthetic_fitted_gamma": None,
            "normalized": {
                "strike": K,
                "oi": oi,
                "iv": iv,
                "sign": sign,
                "T": T,
                "gamma_now": gamma_now,
                "iv_source": "direct_iv",
                "synthetic_fit_rel_error": None,
            },
        }

    if HYBRID_IV_MODE and vendor_gamma > 0:
        fit = fit_synthetic_iv(vendor_gamma, spot, K, T, r)
        if fit["accepted"]:
            return {
                "accepted": True,
                "reason": "synthetic_iv",
                "synthetic_fit_rel_error": fit["rel_error"],
                "synthetic_target_gamma": fit["target_gamma"],
                "synthetic_fitted_gamma": fit["fitted_gamma"],
                "normalized": {
                    "strike": K,
                    "oi": oi,
                    "iv": fit["iv"],
                    "sign": sign,
                    "T": T,
                    "gamma_now": fit["fitted_gamma"],
                    "iv_source": "synthetic_iv",
                    "synthetic_fit_rel_error": fit["rel_error"],
                },
            }

        return {
            "accepted": False,
            "reason": f"synthetic_{fit['reason']}",
            "synthetic_fit_rel_error": fit["rel_error"],
            "synthetic_target_gamma": fit["target_gamma"],
            "synthetic_fitted_gamma": fit["fitted_gamma"],
            "normalized": None,
        }

    return {
        "accepted": False,
        "reason": "no_model_input",
        "synthetic_fit_rel_error": None,
        "synthetic_target_gamma": None,
        "synthetic_fitted_gamma": None,
        "normalized": None,
    }


def normalize_option_for_model(opt, sign, T, spot, r):
    """
    Backward-compatible wrapper.
    """
    result = prepare_option_for_model(opt, sign, T, spot, r)
    return result["normalized"]
