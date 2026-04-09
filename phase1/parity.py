from __future__ import annotations

from datetime import datetime
import numpy as np

from phase1.config import (
    MAX_PARITY_SPREAD,
    PARITY_NEAR_SPOT_CANDIDATES,
    PARITY_FINAL_STRIKES,
    PARITY_RELATIVE_BAND,
    PARITY_HARD_LOW_MULTIPLIER,
    PARITY_HARD_HIGH_MULTIPLIER,
    MIN_PARITY_STRIKES,
    PARITY_METHOD,
    PARITY_WEIGHT_EPS,
    PARITY_SPREAD_WEIGHT_POWER,
    PARITY_DISTANCE_SIGMA_PCT,
    DEFAULT_RISK_FREE_RATE,
)

from phase1.quote_filters import usable_for_parity, quote_mid, summarize_quote_quality
from phase1.market_clock import is_cash_market_open, compute_time_to_expiry_years

def weighted_median(values, weights):
    """
    Weighted median for 1D numeric arrays.

    When the 50% cumulative-weight cutoff falls inside an observation's weight
    block, that observation is the median. When it falls exactly on a boundary
    between two observations, we average the two to avoid systematic bias
    (analogous to how np.median averages the two middle values for even-length
    arrays).
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    if len(values) == 0:
        return None

    if len(values) != len(weights):
        raise ValueError("values and weights must have same length")

    if np.any(weights < 0):
        raise ValueError("weights must be nonnegative")

    total_weight = weights.sum()
    if total_weight <= 0:
        return float(np.median(values))

    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum_w = np.cumsum(w)

    cutoff = 0.5 * total_weight

    # Find the first index where cumulative weight >= cutoff
    idx = np.searchsorted(cum_w, cutoff, side="left")
    idx = min(idx, len(v) - 1)

    # Interpolate only when cutoff falls exactly on a cumulative boundary,
    # meaning the weight of observation idx is fully accumulated and the
    # true median sits between v[idx] and v[idx+1].
    if idx < len(v) - 1 and np.isclose(cum_w[idx], cutoff):
        return float((v[idx] + v[idx + 1]) / 2.0)

    return float(v[idx])


def parity_candidate_weight(strike, vendor_spot, combined_spread):
    """
    Build a simple liquidity-and-distance weight:
    - tighter spread => higher weight
    - closer to spot => higher weight
    """
    spread_component = 1.0 / ((combined_spread + PARITY_WEIGHT_EPS) ** PARITY_SPREAD_WEIGHT_POWER)

    sigma = max(abs(vendor_spot) * PARITY_DISTANCE_SIGMA_PCT, 1.0)
    dist = abs(strike - vendor_spot)
    distance_component = np.exp(-0.5 * (dist / sigma) ** 2)

    return float(spread_component * distance_component)

def _compute_implied_spot_core(calls, puts, vendor_spot, r=DEFAULT_RISK_FREE_RATE, T=None):
    """
    Core parity engine with diagnostics.
    """
    diagnostics = {
        "call_quality": summarize_quote_quality(calls or [], MAX_PARITY_SPREAD),
        "put_quality": summarize_quote_quality(puts or [], MAX_PARITY_SPREAD),
        "common_usable_strikes": 0,
        "near_spot_candidates": 0,
        "final_atm_strikes": 0,
        "relative_band_pass_count": 0,
        "hard_filter_pass_count": 0,
        "parity_method": PARITY_METHOD,
        "simple_median_spot": None,
        "weighted_median_spot": None,
        "selected_weight_sum": 0.0,
    }

    if not calls or not puts:
        return {
            "spot": vendor_spot,
            "source": "vendor (delayed)",
            "diagnostics": diagnostics,
        }

    call_by_k = {}
    for c in calls:
        if usable_for_parity(c, MAX_PARITY_SPREAD):
            call_by_k[c["strike"]] = c

    put_by_k = {}
    for p in puts:
        if usable_for_parity(p, MAX_PARITY_SPREAD):
            put_by_k[p["strike"]] = p

    common_strikes = sorted(set(call_by_k.keys()) & set(put_by_k.keys()))
    diagnostics["common_usable_strikes"] = len(common_strikes)

    if not common_strikes:
        return {
            "spot": vendor_spot,
            "source": "vendor (delayed)",
            "diagnostics": diagnostics,
        }

    common_strikes.sort(key=lambda k: abs(k - vendor_spot))
    candidates = common_strikes[:PARITY_NEAR_SPOT_CANDIDATES]
    diagnostics["near_spot_candidates"] = len(candidates)

    def combined_spread(K):
        c_spread = call_by_k[K]["ask"] - call_by_k[K]["bid"]
        p_spread = put_by_k[K]["ask"] - put_by_k[K]["bid"]
        return c_spread + p_spread

    candidates.sort(key=combined_spread)
    atm_strikes = candidates[:PARITY_FINAL_STRIKES]
    diagnostics["final_atm_strikes"] = len(atm_strikes)

    discount = np.exp(-r * T) if T and T > 0 else 1.0

    implied_prices = []
    implied_weights = []

    for K in atm_strikes:
        c_mid = quote_mid(call_by_k[K])
        p_mid = quote_mid(put_by_k[K])

        if c_mid is None or p_mid is None:
            continue

        s_implied = c_mid - p_mid + K * discount

        if s_implied > 0 and vendor_spot > 0:
            if abs(s_implied - vendor_spot) / vendor_spot < PARITY_RELATIVE_BAND:
                implied_prices.append(s_implied)
                implied_weights.append(parity_candidate_weight(K, vendor_spot, combined_spread(K)))

    diagnostics["relative_band_pass_count"] = len(implied_prices)

    if len(implied_prices) < MIN_PARITY_STRIKES:
        return {
            "spot": vendor_spot,
            "source": "vendor (delayed)",
            "diagnostics": diagnostics,
        }

    implied_prices = np.array(implied_prices, dtype=float)
    implied_weights = np.array(implied_weights, dtype=float)

    hard_mask = (
        (implied_prices > vendor_spot * PARITY_HARD_LOW_MULTIPLIER) &
        (implied_prices < vendor_spot * PARITY_HARD_HIGH_MULTIPLIER)
    )

    implied_prices = implied_prices[hard_mask]
    implied_weights = implied_weights[hard_mask]

    diagnostics["hard_filter_pass_count"] = len(implied_prices)

    if len(implied_prices) < MIN_PARITY_STRIKES:
        return {
            "spot": vendor_spot,
            "source": "vendor (parity filtered)",
            "diagnostics": diagnostics,
        }

    simple_median = float(np.median(implied_prices))
    weighted_med = weighted_median(implied_prices, implied_weights)

    diagnostics["simple_median_spot"] = round(simple_median, 2)
    diagnostics["weighted_median_spot"] = round(weighted_med, 2) if weighted_med is not None else None
    diagnostics["selected_weight_sum"] = float(implied_weights.sum())

    if PARITY_METHOD == "median":
        result = simple_median
        source = f"implied median ({len(implied_prices)} strikes)"
    else:
        result = weighted_med if weighted_med is not None else simple_median
        source = f"implied weighted median ({len(implied_prices)} strikes)"

    return {
        "spot": round(float(result), 2),
        "source": source,
        "diagnostics": diagnostics,
    }  


def compute_implied_spot(calls, puts, vendor_spot, r=DEFAULT_RISK_FREE_RATE, T=None):
    """
    Compatibility wrapper.
    """
    core = _compute_implied_spot_core(calls, puts, vendor_spot, r=r, T=T)
    return core["spot"], core["source"]


def get_reference_spot_details(
    ticker,
    nearest_exp,
    get_spot_price_func,
    get_chain_cached_func,
    r=DEFAULT_RISK_FREE_RATE,
    now=None,
):
    """
    Full reference-spot decision engine.
    """
    now = now or datetime.now()

    vendor_spot = float(get_spot_price_func(ticker))
    details = {
        "spot": round(vendor_spot, 2),
        "source": "vendor (default)",
        "vendor_spot": round(vendor_spot, 2),
        "implied_spot": None,
        "market_open": is_cash_market_open(now),
        "parity_attempted": False,
        "parity_chain_status": None,
        "nearest_exp": nearest_exp,
        "T_years": None,
        "parity_diagnostics": None,
        "expiration_close_ny": None,
    }

    # Always compute nearest expiration close metadata,
    # even if we later force vendor spot because the market is closed.
    T, exp_close = compute_time_to_expiry_years(nearest_exp, ts=now, floor=0.0)
    details["T_years"] = T
    details["expiration_close_ny"] = exp_close.isoformat() if exp_close is not None else None

    if not details["market_open"]:
        details["source"] = "vendor (forced, market closed)"
        return details

    entry = get_chain_cached_func(ticker, nearest_exp)
    details["parity_attempted"] = True
    details["parity_chain_status"] = entry.get("status", "unknown")

    if entry.get("status") != "ok":
        details["source"] = "vendor (parity chain failed)"
        return details

    calls = entry["calls"]
    puts = entry["puts"]

    core = _compute_implied_spot_core(calls, puts, vendor_spot, r=r, T=T)

    details["spot"] = round(float(core["spot"]), 2)
    details["source"] = core["source"]
    details["parity_diagnostics"] = core["diagnostics"]

    if core["source"].startswith("implied"):
        details["implied_spot"] = round(float(core["spot"]), 2)

    return details


def get_reference_spot(
    ticker,
    nearest_exp,
    get_spot_price_func,
    get_chain_cached_func,
    r=DEFAULT_RISK_FREE_RATE,
    now=None,
):
    details = get_reference_spot_details(
        ticker=ticker,
        nearest_exp=nearest_exp,
        get_spot_price_func=get_spot_price_func,
        get_chain_cached_func=get_chain_cached_func,
        r=r,
        now=now,
    )
    return details["spot"], details["source"]
