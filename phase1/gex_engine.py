from __future__ import annotations

from datetime import datetime
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

from phase1.config import (
    STRIKE_RANGE_PCT,
    HEATMAP_STRIKES,
    T_FLOOR,
    ZG_SWEEP_RANGE_PCT,
    ZG_SWEEP_STEP,
    ZG_FINE_STEP,
    PROFILE_RANGE_PCT,
    PROFILE_STEP,
    HYBRID_IV_MODE,
    NY_TZ,
    DEFAULT_RISK_FREE_RATE,
)
from phase1.model_inputs import prepare_option_for_model, bs_gamma
from phase1.market_clock import compute_time_to_expiry_years
from phase1.liquidity import build_strike_support_df, build_expiration_support_df


def fmt_gex(v):
    if abs(v) >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1000:
        return f"{v/1000:.1f}K"
    if abs(v) >= 1:
        return f"{v:.1f}"
    if abs(v) >= 0.01:
        return f"{v:.2f}"
    return f"{v:.3f}"


def fmt_oi(v):
    v = abs(v)  # OI should never be negative; guard defensively
    if v >= 1_000_000:
        return f"{v/1_000_000:.2f}M"
    if v >= 1000:
        return f"{v/1000:.1f}K"
    return f"{v:.0f}"


def unique_preserve_order(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def bs_gamma_vec(S_arr, K_arr, T_arr, r, sigma_arr):
    """
    Vectorized Black-Scholes gamma.

    S_arr: (N,)
    K_arr: (M,)
    T_arr: (M,)
    sigma_arr: (M,)

    Returns: (N, M)
    """
    S = np.asarray(S_arr, dtype=float).reshape(-1, 1)
    K = np.asarray(K_arr, dtype=float)
    T = np.asarray(T_arr, dtype=float)
    sigma = np.asarray(sigma_arr, dtype=float)

    gamma = np.zeros((S.shape[0], K.shape[0]), dtype=float)

    opt_valid = (K > 0) & (T > 0) & (sigma > 0)
    if not np.any(opt_valid):
        return gamma

    K_v = K[opt_valid]
    T_v = T[opt_valid]
    sig_v = sigma[opt_valid]

    sqrt_T = np.sqrt(T_v)
    d1 = (np.log(S / K_v) + (r + 0.5 * sig_v**2) * T_v) / (sig_v * sqrt_T)
    g = norm.pdf(d1) / (S * sig_v * sqrt_T)

    gamma[:, opt_valid] = g
    return gamma


def calculate_all(client, ticker, target_exps, spot, heatmap_exps, r=DEFAULT_RISK_FREE_RATE, now=None):
    """
    Hybrid mode:
      - use direct IV if available
      - otherwise infer synthetic IV from vendor gamma if possible

    Returns:
        gex_df, heatmap_gex, heatmap_iv, stats, all_options
    """
    lower = spot * (1 - STRIKE_RANGE_PCT)
    upper = spot * (1 + STRIKE_RANGE_PCT)

    now = now or datetime.now(NY_TZ)
    now_ny = now.astimezone(NY_TZ) if now.tzinfo is not None else now.replace(tzinfo=NY_TZ)

    agg = {}
    per_exp_gex = {}
    per_exp_iv = {}

    total_call_oi = 0.0
    total_put_oi = 0.0
    max_call_oi_strike = (0.0, 0)
    max_put_oi_strike = (0.0, 0)

    all_exps = unique_preserve_order(target_exps + (heatmap_exps or []))
    first_exp_call_ivs = []
    first_exp_put_ivs = []

    all_options = []   # (strike, oi, iv, sign, T)
    support_records = []

    failed_expirations = []

    used_option_count = 0
    direct_iv_count = 0
    synthetic_iv_count = 0
    synthetic_fit_accept_count = 0
    synthetic_fit_reject_count = 0
    no_model_input_count = 0
    skipped_count = 0
    skipped_oi = 0.0
    synthetic_fit_rel_errors = []

    client.prefetch_chains(ticker, all_exps)

    for i, exp in enumerate(all_exps):
        T, _exp_close = compute_time_to_expiry_years(exp, ts=now_ny, floor=T_FLOOR)

        entry = client.get_chain_cached(ticker, exp)
        calls_raw = entry["calls"]
        puts_raw = entry["puts"]

        if entry["status"] != "ok":
            failed_expirations.append(exp)
            print(f"  [{i+1}/{len(all_exps)}] {exp}  — FAILED ({entry.get('error', 'unknown error')})")
            continue

        exp_gex = {}
        exp_iv = {}

        for raw_opt, sign in [(c, +1) for c in calls_raw] + [(p, -1) for p in puts_raw]:
            K = raw_opt["strike"]
            if K < lower or K > upper:
                continue

            oi = raw_opt["openInterest"]
            if oi <= 0 or np.isnan(oi):
                continue

            prep = prepare_option_for_model(raw_opt, sign, T, spot, r)
            norm_opt = prep["normalized"]

            if not prep["accepted"]:
                skipped_count += 1
                skipped_oi += oi

                if prep["reason"].startswith("synthetic_"):
                    synthetic_fit_reject_count += 1
                elif prep["reason"] == "no_model_input":
                    no_model_input_count += 1

                continue

            model_iv = norm_opt["iv"]
            gamma_now = norm_opt["gamma_now"]
            gex = sign * oi * gamma_now * 100.0

            exp_gex[K] = exp_gex.get(K, 0.0) + gex
            exp_iv.setdefault(K, []).append(model_iv)

            if exp in target_exps:
                if K not in agg:
                    agg[K] = {"call_gex": 0.0, "put_gex": 0.0, "call_oi": 0.0, "put_oi": 0.0}

                if sign == +1:
                    agg[K]["call_gex"] += gex
                    agg[K]["call_oi"] += oi
                    total_call_oi += oi
                    if oi > max_call_oi_strike[0]:
                        max_call_oi_strike = (oi, K)
                    if exp == target_exps[0]:
                        first_exp_call_ivs.append(model_iv)
                else:
                    agg[K]["put_gex"] += gex
                    agg[K]["put_oi"] += oi
                    total_put_oi += oi
                    if oi > max_put_oi_strike[0]:
                        max_put_oi_strike = (oi, K)
                    if exp == target_exps[0]:
                        first_exp_put_ivs.append(model_iv)

                all_options.append((K, oi, model_iv, sign, T))
                used_option_count += 1
                bid = raw_opt.get("bid", 0.0) or 0.0
                ask = raw_opt.get("ask", 0.0) or 0.0
                spread = float(ask - bid) if bid > 0 and ask > 0 and ask >= bid else np.nan

                support_records.append(
                    {
                        "expiration": exp,
                        "strike": round(K, 2),
                        "oi": float(oi),
                        "iv_source": norm_opt["iv_source"],
                        "spread": spread,
                        "is_call": bool(sign == +1),
                        "is_put": bool(sign == -1),
                        "net_gex": float(gex),
                        "abs_gex": float(abs(gex)),
                    }
                )                

                if norm_opt["iv_source"] == "direct_iv":
                    direct_iv_count += 1
                elif norm_opt["iv_source"] == "synthetic_iv":
                    synthetic_iv_count += 1
                    synthetic_fit_accept_count += 1
                    if norm_opt.get("synthetic_fit_rel_error") is not None:
                        synthetic_fit_rel_errors.append(norm_opt["synthetic_fit_rel_error"])

        if heatmap_exps and exp in heatmap_exps:
            per_exp_gex[exp] = exp_gex
            per_exp_iv[exp] = {K: np.mean(vs) * 100.0 for K, vs in exp_iv.items()}

        print(f"  [{i+1}/{len(all_exps)}] {exp}  (T={T*365.25:.2f}d)  — {len(calls_raw)}C / {len(puts_raw)}P")

    rows = []
    for K in sorted(agg):
        d = agg[K]
        rows.append(
            {
                "strike": round(K, 2),
                "call_oi": d["call_oi"],
                "put_oi": d["put_oi"],
                "call_gex": d["call_gex"],
                "put_gex": d["put_gex"],
                "net_gex": d["call_gex"] + d["put_gex"],
            }
        )

    gex_df = pd.DataFrame(rows)

    all_strikes = sorted(agg.keys())
    if all_strikes:
        nearest_idx = min(range(len(all_strikes)), key=lambda idx: abs(all_strikes[idx] - spot))
        half = HEATMAP_STRIKES // 2
        start = max(0, nearest_idx - half)
        end = min(len(all_strikes), start + HEATMAP_STRIKES)
        start = max(0, end - HEATMAP_STRIKES)
        hm_strikes = all_strikes[start:end]
    else:
        hm_strikes = []

    hm_gex_data = {}
    hm_iv_data = {}
    for exp_str in sorted(per_exp_gex):
        col = datetime.strptime(exp_str, "%Y-%m-%d").strftime("%m/%d")
        hm_gex_data[col] = [per_exp_gex[exp_str].get(K, 0.0) for K in hm_strikes]
        hm_iv_data[col] = [per_exp_iv.get(exp_str, {}).get(K, 0.0) for K in hm_strikes]

    heatmap_gex = pd.DataFrame(hm_gex_data, index=hm_strikes)
    heatmap_iv = pd.DataFrame(hm_iv_data, index=hm_strikes)

    net_gex_total = gex_df["net_gex"].sum() if not gex_df.empty else 0.0
    pos_gex = gex_df[gex_df["net_gex"] > 0]
    neg_gex = gex_df[gex_df["net_gex"] < 0]

    pos_max = pos_gex.loc[pos_gex["net_gex"].idxmax()] if not pos_gex.empty else None
    neg_min = neg_gex.loc[neg_gex["net_gex"].idxmin()] if not neg_gex.empty else None

    coverage_total_used_oi = total_call_oi + total_put_oi
    coverage_total_chain_oi = coverage_total_used_oi + skipped_oi
    coverage_ratio = (coverage_total_used_oi / coverage_total_chain_oi) if coverage_total_chain_oi > 0 else 0.0

    stats = {
        "net_gex": net_gex_total,
        "net_gex_fmt": fmt_gex(net_gex_total),
        "gex_ratio": abs(pos_gex["net_gex"].sum() / neg_gex["net_gex"].sum())
        if not neg_gex.empty and neg_gex["net_gex"].sum() != 0
        else 0.0,
        "pc_ratio": total_put_oi / total_call_oi if total_call_oi > 0 else 0.0,
        "call_oi": fmt_oi(total_call_oi),
        "call_oi_strike": max_call_oi_strike[1],
        "put_oi": fmt_oi(total_put_oi),
        "put_oi_strike": max_put_oi_strike[1],
        "pos_gex": fmt_gex(pos_max["net_gex"]) if pos_max is not None else "0",
        "pos_gex_strike": round(float(pos_max["strike"]), 2) if pos_max is not None else 0,
        "neg_gex": fmt_gex(neg_min["net_gex"]) if neg_min is not None else "0",
        "neg_gex_strike": round(float(neg_min["strike"]), 2) if neg_min is not None else 0,
        "call_iv": np.mean(first_exp_call_ivs) * 100.0 if first_exp_call_ivs else 0.0,
        "put_iv": np.mean(first_exp_put_ivs) * 100.0 if first_exp_put_ivs else 0.0,
        "used_option_count": int(used_option_count),
        "direct_iv_count": int(direct_iv_count),
        "synthetic_iv_count": int(synthetic_iv_count),
        "synthetic_fit_accept_count": int(synthetic_fit_accept_count),
        "synthetic_fit_reject_count": int(synthetic_fit_reject_count),
        "no_model_input_count": int(no_model_input_count),
        "synthetic_fit_avg_rel_error": float(np.mean(synthetic_fit_rel_errors)) if synthetic_fit_rel_errors else None,
        "synthetic_fit_max_rel_error": float(np.max(synthetic_fit_rel_errors)) if synthetic_fit_rel_errors else None,
        "skipped_count": int(skipped_count),
        "skipped_oi": skipped_oi,
        "failed_expirations": failed_expirations,
        "failed_exp_count": len(failed_expirations),
        "coverage_ratio": coverage_ratio,
        "hybrid_iv_mode": HYBRID_IV_MODE,
    }

    strike_support_df = build_strike_support_df(support_records, selected_exp_count=len(target_exps))
    expiration_support_df = build_expiration_support_df(support_records)

    if not strike_support_df.empty:
        weights = np.maximum(strike_support_df["abs_net_gex"].values.astype(float), 1.0)
        stats["strike_support_avg"] = float(np.average(strike_support_df["support_score"].values.astype(float), weights=weights))
        stats["fragile_strike_count"] = int((strike_support_df["support_label"] == "Low").sum())
    else:
        stats["strike_support_avg"] = None
        stats["fragile_strike_count"] = 0

    if not expiration_support_df.empty:
        stats["expiration_support_avg"] = float(expiration_support_df["support_score"].mean())
    else:
        stats["expiration_support_avg"] = None

    return gex_df, heatmap_gex, heatmap_iv, stats, all_options, strike_support_df, expiration_support_df


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
    total_gex = (gamma_matrix * weights).sum(axis=1)

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


def zero_gamma_sweep_details(all_options, spot, r=DEFAULT_RISK_FREE_RATE):
    """
    Return rich diagnostics for zero-gamma solving.

    A true zero gamma requires a sign change in total GEX across price.
    If no sign change exists in the sweep window, we fall back to the
    minimum-absolute-GEX node and mark it as a fallback.
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

    lo = float(spot) * (1 - ZG_SWEEP_RANGE_PCT)
    hi = float(spot) * (1 + ZG_SWEEP_RANGE_PCT)

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


def zero_gamma_sweep(all_options, spot, r=DEFAULT_RISK_FREE_RATE):
    return zero_gamma_sweep_details(all_options, spot, r=r)["zero_gamma"]


def compute_gex_profile_curve(all_options, spot, r=DEFAULT_RISK_FREE_RATE):
    if not all_options:
        return pd.DataFrame(columns=["price", "total_gex"])

    lo = math.floor(spot * (1 - PROFILE_RANGE_PCT))
    hi = math.ceil(spot * (1 + PROFILE_RANGE_PCT))
    prices = np.arange(lo, hi + PROFILE_STEP, PROFILE_STEP, dtype=float)
    total_gex = _sweep_gex_at_prices(all_options, prices, r)

    return pd.DataFrame(
        {
            "price": prices,
            "total_gex": total_gex,
        }
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

    if not pos.empty:
        cw = pos.loc[pos["net_gex"].idxmax()]
    else:
        cw = gex_df.loc[gex_df["net_gex"].idxmax()]

    if not neg.empty:
        pw = neg.loc[neg["net_gex"].idxmin()]
    else:
        pw = gex_df.loc[gex_df["net_gex"].idxmin()]

    if all_options:
        zg_details = zero_gamma_sweep_details(all_options, spot, r=r)
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

    return {
        "call_wall": float(cw["strike"]),
        "call_wall_gex": float(cw["net_gex"]),
        "put_wall": float(pw["strike"]),
        "put_wall_gex": float(pw["net_gex"]),
        "zero_gamma": round(float(zg_details['zero_gamma']), 2),
        "zero_gamma_is_true_crossing": bool(zg_details["is_true_crossing"]),
        "zero_gamma_type": zg_details["zero_gamma_type"],
        "zero_gamma_method": zg_details["method"],
        "zero_gamma_abs_gex": zg_details["final_abs_gex"],
    }

def compute_strike_gex_from_all_options(all_options, spot, r=DEFAULT_RISK_FREE_RATE):
    """
    Recompute strike-by-strike GEX for a shocked spot / rate using the
    normalized option universe stored in all_options.

    all_options entries are:
        (strike, oi, iv, sign, T)
    """
    if not all_options:
        return pd.DataFrame(columns=["strike", "call_gex", "put_gex", "net_gex"])

    agg = {}

    for K, oi, iv, sign, T in all_options:
        gamma_now = bs_gamma(float(spot), float(K), float(T), float(r), float(iv))
        gex = float(sign) * float(oi) * float(gamma_now) * 100.0

        if K not in agg:
            agg[K] = {"call_gex": 0.0, "put_gex": 0.0}

        if sign > 0:
            agg[K]["call_gex"] += gex
        else:
            agg[K]["put_gex"] += gex

    rows = []
    for K in sorted(agg):
        call_gex = float(agg[K]["call_gex"])
        put_gex = float(agg[K]["put_gex"])
        rows.append(
            {
                "strike": round(K, 2),
                "call_gex": call_gex,
                "put_gex": put_gex,
                "net_gex": call_gex + put_gex,
            }
        )

    return pd.DataFrame(rows)

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