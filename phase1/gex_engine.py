from __future__ import annotations

from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import norm

from phase1.config import (
    COMPUTATION_RANGE_PCT,
    T_FLOOR,
    HYBRID_IV_MODE,
    NY_TZ,
    DEFAULT_RISK_FREE_RATE,
    TRADING_HOURS_PER_YEAR,
)
from phase1.model_inputs import prepare_option_for_model
from phase1.market_clock import compute_time_to_expiry_years
from phase1.liquidity import build_strike_support_df, build_expiration_support_df
from phase1.rates import interpolate_rate

DAYS_PER_YEAR_CAL = 365.25  # calendar-day conversion for FRED rate lookup


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
    r:     scalar OR (M,) per-option rate array (for term-structure
           interpolation — zero_gamma_sweep passes a curve-interpolated
           rate per option so each option uses the rate appropriate for
           its own DTE instead of a flat 3M scalar).
    sigma_arr: (M,)

    Returns: (N, M)
    """
    S = np.asarray(S_arr, dtype=float).reshape(-1, 1)
    K = np.asarray(K_arr, dtype=float)
    T = np.asarray(T_arr, dtype=float)
    sigma = np.asarray(sigma_arr, dtype=float)

    # r can be scalar or an array aligned to K/T/sigma. We only filter it
    # in the array case so the scalar fast-path stays untouched.
    r_is_array = (np.ndim(r) > 0)
    r_array = np.asarray(r, dtype=float) if r_is_array else None

    gamma = np.zeros((S.shape[0], K.shape[0]), dtype=float)

    opt_valid = (K > 0) & (T > 0) & (sigma > 0)
    if not np.any(opt_valid):
        return gamma

    K_v = K[opt_valid]
    T_v = T[opt_valid]
    sig_v = sigma[opt_valid]
    r_v = r_array[opt_valid] if r_is_array else r

    sqrt_T = np.sqrt(T_v)
    d1 = (np.log(S / K_v) + (r_v + 0.5 * sig_v**2) * T_v) / (sig_v * sqrt_T)
    g = norm.pdf(d1) / (S * sig_v * sqrt_T)

    gamma[:, opt_valid] = g
    return gamma


def calculate_all(client, ticker, target_exps, spot, r=DEFAULT_RISK_FREE_RATE,
                  now=None, r_curve=None):
    """
    Hybrid mode:
      - use direct IV if available
      - otherwise infer synthetic IV from vendor gamma if possible

    Parameters:
        r:       flat-rate back-compat scalar. Used as the fallback when
                 `r_curve` is not provided (existing behavior — the same
                 rate for every expiration).
        r_curve: optional term-structure dict {dte_days: rate} from
                 fetch_risk_free_rate()["curve"]. When provided, each
                 expiration iteration looks up the rate for its own DTE
                 via interpolate_rate(), so 0DTE uses ~1M, 45DTE uses
                 something between 1M and 3M, etc. This removes the
                 flat-3M bias that distorts OpEx-cycle BS gamma.

    Returns:
        gex_df, stats, all_options, strike_support_df, expiration_support_df
    """
    # Both bar chart and all_options use the same ±8% computation range
    # so the chart shows the full gamma landscape including deep OTM tails
    lower = spot * (1 - COMPUTATION_RANGE_PCT)
    upper = spot * (1 + COMPUTATION_RANGE_PCT)

    now = now or datetime.now(NY_TZ)
    now_ny = now.astimezone(NY_TZ) if now.tzinfo is not None else now.replace(tzinfo=NY_TZ)

    agg = {}

    total_call_oi = 0.0
    total_put_oi = 0.0
    max_call_oi_strike = (0.0, 0)
    max_put_oi_strike = (0.0, 0)

    all_exps = unique_preserve_order(target_exps)
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
    range_filtered_count = 0
    zero_oi_filtered_count = 0
    synthetic_fit_rel_errors = []
    # Volume-amplification tracking: sum of OI vs sum of max(OI, volume)
    # across the strikes that actually entered the GEX aggregation. The
    # ratio tells the UI how much of today's GEX magnitude comes from
    # intraday flow vs settled open interest — on high-churn 0DTE days
    # this can be 2-4×, which means the wall magnitudes are systematically
    # overstated vs what a pure-OI computation would show.
    total_used_oi_only = 0.0
    total_used_max = 0.0
    vol_dominated_strikes = 0  # count of strikes where volume > OI

    client.prefetch_chains(ticker, all_exps)

    expired_exp_count = 0

    for i, exp in enumerate(all_exps):
        T, exp_close = compute_time_to_expiry_years(exp, ts=now_ny, floor=T_FLOOR)

        # Skip expirations that have already closed for the day.  After
        # 4:00 PM ET on an expiration day, compute_time_to_expiry_years
        # returns T = 0 and then floors it at T_FLOOR (~1 minute in years).
        # Black-Scholes ATM gamma goes as 1/sqrt(T), so that tiny floor
        # produces a gamma value ~100x larger than a live 4DTE expiration's
        # gamma — a single expired 0DTE strike near spot then swamps every
        # other expiration in the aggregation, making after-hours "This
        # week" / "OpEx Cycle" / "Custom" views look like the only thing
        # that exists is the dead 0DTE session.  Expired options have zero
        # gamma left for dealers to hedge (the contracts have settled), so
        # drop them from the current GEX picture entirely.
        if exp_close is not None and exp_close <= now_ny:
            expired_exp_count += 1
            print(f"  [{i+1}/{len(all_exps)}] {exp}  — skipped (expired at {exp_close.strftime('%H:%M %Z')})")
            continue

        # Per-expiration rate: when a curve is available, interpolate to
        # this expiration's own DTE so BS gamma and any rate-sensitive
        # downstream math use the right point on the term structure
        # instead of the flat 3M scalar.
        r_for_exp = interpolate_rate(r_curve, T * DAYS_PER_YEAR_CAL, fallback=r)

        entry = client.get_chain_cached(ticker, exp)
        calls_raw = entry["calls"]
        puts_raw = entry["puts"]

        if entry["status"] != "ok":
            failed_expirations.append(exp)
            print(f"  [{i+1}/{len(all_exps)}] {exp}  — FAILED ({entry.get('error', 'unknown error')})")
            continue

        for raw_opt, sign in [(c, +1) for c in calls_raw] + [(p, -1) for p in puts_raw]:
            K = raw_opt["strike"]
            if K < lower or K > upper:
                range_filtered_count += 1
                continue

            oi_raw = raw_opt["openInterest"]
            oi = 0.0 if (np.isnan(oi_raw) or oi_raw < 0) else float(oi_raw)
            volume = float(raw_opt.get("volume", 0.0) or 0.0)

            # Effective GEX weight = max(OI, today's volume).  Tradier
            # reports open_interest as the previous day's settlement number,
            # which is especially stale for 0DTE strikes where this morning's
            # flow dwarfs yesterday's EOD OI.  Using the max of the two
            # "unstales" OI without double-counting: on strikes where
            # today's volume is heavier than yesterday's settled OI (active
            # 0DTE / freshly opened weeklies) we get credit for live flow,
            # and on quiet far-OTM strikes the max collapses back to OI
            # unchanged (volume == 0).  NaN / negative OI from the vendor
            # is coerced to zero so a strike with OI=0 but positive volume
            # can still show up.
            size = max(oi, volume)
            if size <= 0:
                zero_oi_filtered_count += 1
                continue

            # Track size vs OI-only so the UI can flag over-painting
            # from heavy intraday churn. Only accumulate after the
            # acceptance check below fires, see the `if prep["accepted"]`
            # branch. We stash the values now and commit them once the
            # strike is actually used.
            _this_strike_oi = oi
            _this_strike_size = size
            _this_strike_volume_dominates = volume > oi

            prep = prepare_option_for_model(raw_opt, sign, T, spot, r_for_exp)
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
            charm_now = norm_opt.get("charm_now", 0.0)
            gex = sign * size * gamma_now * 100.0 * spot * spot
            # Dollar charm exposure: dealer-book Δ drift per unit of T,
            # scaled to notional ($). Same sign convention as GEX — a call
            # (sign=+1) with positive charm means the assumed-short-call
            # dealer book is shedding delta as time passes, so dealers
            # must buy the underlying to stay hedged (positive CEX →
            # supportive flow). Units are $-delta per trading-year;
            # divide by 252 downstream for a per-trading-day figure.
            cex = sign * size * charm_now * 100.0 * spot

            if exp in target_exps:
                if K not in agg:
                    agg[K] = {
                        "call_gex": 0.0, "put_gex": 0.0,
                        "call_oi": 0.0, "put_oi": 0.0,
                        "call_cex": 0.0, "put_cex": 0.0,
                    }

                if sign == +1:
                    agg[K]["call_gex"] += gex
                    agg[K]["call_cex"] += cex
                    agg[K]["call_oi"] += oi
                    total_call_oi += oi
                    if oi > max_call_oi_strike[0]:
                        max_call_oi_strike = (oi, K)
                    if exp == target_exps[0]:
                        first_exp_call_ivs.append(model_iv)
                else:
                    agg[K]["put_gex"] += gex
                    agg[K]["put_cex"] += cex
                    agg[K]["put_oi"] += oi
                    total_put_oi += oi
                    if oi > max_put_oi_strike[0]:
                        max_put_oi_strike = (oi, K)
                    if exp == target_exps[0]:
                        first_exp_put_ivs.append(model_iv)

                # all_options gets the full wider range for sweep/profile/scenarios.
                # Use *size* (max of OI and volume) as the weight so zero-gamma
                # sweeps and per-expiry GEX projections line up with the live
                # volume-inclusive GEX computed above — otherwise the sweep would
                # revert to OI-only and disagree with the bar chart on 0DTE.
                all_options.append((K, size, model_iv, sign, T, exp))
                used_option_count += 1
                # Commit the per-strike amplification counters (now that the
                # strike has actually been accepted into the aggregation).
                total_used_oi_only += _this_strike_oi
                total_used_max     += _this_strike_size
                if _this_strike_volume_dominates:
                    vol_dominated_strikes += 1
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
                "call_charm": d["call_cex"],
                "put_charm": d["put_cex"],
                "net_charm": d["call_cex"] + d["put_cex"],
            }
        )

    gex_df = pd.DataFrame(rows)

    net_gex_total = gex_df["net_gex"].sum() if not gex_df.empty else 0.0
    net_charm_total = gex_df["net_charm"].sum() if not gex_df.empty else 0.0
    pos_gex = gex_df[gex_df["net_gex"] > 0]
    neg_gex = gex_df[gex_df["net_gex"] < 0]

    pos_max = pos_gex.loc[pos_gex["net_gex"].idxmax()] if not pos_gex.empty else None
    neg_min = neg_gex.loc[neg_gex["net_gex"].idxmin()] if not neg_gex.empty else None

    coverage_total_used_oi = total_call_oi + total_put_oi
    coverage_total_chain_oi = coverage_total_used_oi + skipped_oi
    coverage_ratio = (coverage_total_used_oi / coverage_total_chain_oi) if coverage_total_chain_oi > 0 else 0.0

    # gex_ratio = |Σ positive GEX / Σ negative GEX|
    #   > 1.0 → positive GEX dominates (calls heavier than puts)
    #   < 1.0 → negative GEX dominates (puts heavier than calls)
    # Returns None in undefined cases so the UI can disambiguate. Prior
    # code returned 0.0 for both "all positive GEX" (+∞) and "all negative
    # GEX" (0), collapsing those opposite regimes into the same displayed
    # value.
    pos_sum = float(pos_gex["net_gex"].sum()) if not pos_gex.empty else 0.0
    neg_sum = float(neg_gex["net_gex"].sum()) if not neg_gex.empty else 0.0
    if pos_gex.empty and neg_gex.empty:
        gex_ratio_val = None  # empty chain / bad data
    elif neg_gex.empty or neg_sum == 0:
        gex_ratio_val = None  # all positive — ratio is +∞, UI can show "∞"
    elif pos_gex.empty or pos_sum == 0:
        gex_ratio_val = 0.0   # all negative — semantically "no positive side"
    else:
        gex_ratio_val = abs(pos_sum / neg_sum)

    # Charm is annualized against the engine's trading-time T base
    # (1 year == 252 trading days × 6.5 hours — see TRADING_HOURS_PER_YEAR
    # and USE_TRADING_TIME in config.py), so per-day divides by 252 and
    # per-hour divides by TRADING_HOURS_PER_YEAR (1638). Per-hour is the
    # metric the UI surfaces because 0DTE charm flow accelerates
    # specifically in the afternoon session — a per-day figure averages
    # that acceleration away.
    net_charm_per_day = net_charm_total / 252.0
    net_charm_per_hour = net_charm_total / float(TRADING_HOURS_PER_YEAR)

    stats = {
        "net_gex": net_gex_total,
        "net_gex_fmt": fmt_gex(net_gex_total),
        "net_charm": float(net_charm_total),
        "net_charm_fmt": fmt_gex(net_charm_total),
        "net_charm_per_day": float(net_charm_per_day),
        "net_charm_per_day_fmt": fmt_gex(net_charm_per_day),
        "net_charm_per_hour": float(net_charm_per_hour),
        "net_charm_per_hour_fmt": fmt_gex(net_charm_per_hour),
        "gex_ratio": gex_ratio_val,
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
        "range_filtered_count": int(range_filtered_count),
        "zero_oi_filtered_count": int(zero_oi_filtered_count),
        "failed_expirations": failed_expirations,
        "failed_exp_count": len(failed_expirations),
        "expired_exp_count": int(expired_exp_count),
        "coverage_ratio": coverage_ratio,
        "hybrid_iv_mode": HYBRID_IV_MODE,
        # Volume amplification diagnostics
        # ------------------------------------------------------------
        # GEX weights use max(OI, volume) to "unstale" yesterday's
        # settled OI on churny 0DTE strikes. When volume dominates
        # across many strikes, the aggregated GEX magnitude is
        # systematically higher than a pure-OI computation would show
        # (wall magnitudes over-painted, zero-gamma shifted toward the
        # higher-volume side). These three fields let the UI surface
        # that so the trader knows the magnitudes are inflated.
        #
        #   vol_amplification_ratio   : sum(max(OI,vol)) / sum(OI)
        #                               > 1.0 means volume is boosting
        #                               the GEX scale above OI-only.
        #   vol_dominated_strike_count: # strikes where volume > OI
        #   vol_dominated_pct         : that count as a fraction of
        #                               total used strikes.
        "vol_amplification_ratio": (
            float(total_used_max / total_used_oi_only)
            if total_used_oi_only > 0 else None
        ),
        "vol_dominated_strike_count": int(vol_dominated_strikes),
        "vol_dominated_pct": (
            float(vol_dominated_strikes / used_option_count)
            if used_option_count > 0 else 0.0
        ),
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

    return gex_df, stats, all_options, strike_support_df, expiration_support_df


# ── Zero-gamma, key levels, regime — re-exported from split modules ──
from phase1.zero_gamma import (  # noqa: F401
    _sweep_gex_at_prices,
    _find_nearest_crossing_details,
    _find_nearest_crossing,
    _compute_sweep_range_pct,
    zero_gamma_sweep_details,
    zero_gamma_sweep,
)
from phase1.key_levels import (  # noqa: F401
    get_gamma_regime_text,
    _find_wall_cluster,
    find_key_levels,
)