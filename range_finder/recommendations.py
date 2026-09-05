"""One forecast/plan/tier/display path for the UI and unattended capture.

The displayed (pre-EM-floor) strike is the production recommendation. The
EM-floored ladder remains context; it must not silently widen this study.
"""
from range_finder.har_model import forecast_next_week, PI_ALPHA
from range_finder.spread_levels import build_spread_plan, build_spread_tiers
from range_finder.gex_policy import uses_disabled_gex_feature

TIER_KEYS = ("lower_pi", "point", "pi_upper", "effective")
TIER_LABELS = ("Lower PI", "Point Estimate", "80% PI Upper", "Effective (+buffer)")


def displayed_tier(tier):
    return {
        "put_short": tier.model_put_short if tier.model_put_short is not None else tier.put_short,
        "call_short": tier.model_call_short if tier.model_call_short is not None else tier.call_short,
        "put_spreads": tier.model_put_spreads or tier.put_spreads,
        "call_spreads": tier.model_call_spreads or tier.call_spreads,
    }


def tier_bands(tiers):
    return {key: (round(float(displayed_tier(t)["put_short"]), 2),
                  round(float(displayed_tier(t)["call_short"]), 2))
            for key, t in zip(TIER_KEYS, tiers)}


def chain_entry_to_quotes(entry):
    from phase1.quote_filters import filter_to_preferred_root
    calls, puts, _ = filter_to_preferred_root(entry.get("calls", []), entry.get("puts", []))
    quotes = {}
    for side, options in (("call", calls), ("put", puts)):
        for opt in options:
            quote = quotes.setdefault(opt["strike"], {})
            for field in ("bid", "ask"):
                quote[f"{side}_{field}"] = opt.get(field, 0.0) or 0.0
    return quotes


def build_recommendations(*, result, feature_row, feature_cols, reference,
                          vix, week_start, ticker, side_share_q=None,
                          chain_quotes=None, weekly_em=None, conn=None,
                          model_name=None):
    if uses_disabled_gex_feature(feature_cols):
        raise ValueError("Saved fit uses disabled GEX feature; refit required")
    forecast = forecast_next_week(result, feature_row, feature_cols, reference,
                                  alpha=PI_ALPHA, side_share_q=side_share_q)
    from range_finder.conformal import maybe_apply_conformal, CONFORMAL_ENABLED
    if CONFORMAL_ENABLED:
        forecast = maybe_apply_conformal(forecast, conn, ticker, model_name)
    plan = build_spread_plan(forecast=forecast, feature_row=feature_row,
                            week_start=week_start, vix_level=vix, ticker=ticker,
                            chain_quotes=chain_quotes, side_share_q=side_share_q)
    tiers = build_spread_tiers(forecast=forecast, plan=plan, spx_ref=reference,
                              vix_level=vix, chain_quotes=chain_quotes,
                              ticker=ticker, weekly_em=weekly_em)
    return forecast, plan, tiers
