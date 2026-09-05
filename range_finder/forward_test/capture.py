"""Admission, model identity and immutable production recommendations."""
from datetime import datetime, timedelta
import math
import re

import numpy as np
import pandas as pd

from phase1.ticker_config import register_dynamic_config
from range_finder import gex_policy
from range_finder.feature_builder import assess_path_provenance, COVID_START, COVID_END
from range_finder.har_model import (MODEL_SPECS, TRAIN_WINDOW_YEARS, production_feature_columns,
                                   fit_production_model, estimate_side_share_quantile, train_window_min_date)
from range_finder.model_persistence import SCHEMA_VERSION, IncompatibleModelError
from range_finder.recommendations import (TIER_KEYS, TIER_LABELS, build_recommendations,
                                          displayed_tier, chain_entry_to_quotes)
from .config import MODELS, UNIVERSE, digest, model_methodology_id
from .provider import finite_price, frame_records


def validate_saved_fit(payload, ticker, model, week):
    """A legacy or wrong-ticker fit may never be admitted into this study.

    Capture currently builds fresh fits in memory. This gate also makes the
    contract explicit for any future saved-fit adapter, without pickle fallbacks.
    """
    if (payload.get("schema_version") != SCHEMA_VERSION or payload.get("ticker") != ticker
            or payload.get("model_name") != model or not payload.get("training_cutoff")):
        raise IncompatibleModelError("Saved fit identity/training cutoff is incompatible")
    if gex_policy.uses_disabled_gex_feature(payload.get("feature_cols")):
        raise IncompatibleModelError("Saved fit contains disabled GEX")
    if pd.Timestamp(payload["training_cutoff"]).date() >= week.monday:
        raise IncompatibleModelError("Saved fit includes target-week or later outcomes")


def completed_features(features, week, as_of=None):
    if features.index.has_duplicates:
        raise ValueError("Duplicate feature weeks")
    cutoff = pd.Timestamp(week.monday)
    minimum = pd.Timestamp(train_window_min_date(as_of=as_of or week.capture_start))
    train = features.loc[(features.index < cutoff) & (features.index >= minimum)].copy()
    train = train.loc[(train.index < COVID_START) | (train.index > COVID_END)]
    if cutoff not in features.index:
        raise ValueError("Target-week feature row unavailable")
    row = features.loc[cutoff].copy()
    if not assess_path_provenance(row, cutoff)[0]:
        raise ValueError("Feature path does not come from the prior completed week")
    # No target-week high/low/range may enter the forecast or its audit vector.
    row = row.drop(labels=["log_range", "range_pct"], errors="ignore")
    return train, row


def contract_chain(entry, ticker, expiration):
    root = "SPXW" if ticker == "SPX" else ticker
    result = {"calls": [], "puts": []}
    for side, letter in (("calls", "C"), ("puts", "P")):
        for opt in entry.get(side, []):
            if opt.get("root") != root:
                continue
            symbol = opt.get("symbol") or ""
            match = re.fullmatch(r"([A-Z]+)(\d{6})([CP])(\d{8})", symbol)
            if (not match or match[1] != root or match[2] != expiration[2:].replace("-", "")
                    or match[3] != letter or int(match[4]) / 1000 != opt.get("strike")
                    or opt.get("expiration_date") != expiration or opt.get("contract_size") != 100):
                continue
            result[side].append(opt)
    if not result["calls"] or not result["puts"]:
        raise ValueError(f"Missing verified standard {root} contracts for {expiration}; AM/adjusted/unknown roots excluded")
    return result, root


def capture_model(prepared, model, week, model_version, now):
    ticker = prepared["ticker"]
    if ticker not in UNIVERSE or model not in MODELS or set(MODEL_SPECS) != set(MODELS):
        raise ValueError("Production model roster changed; review study configuration")
    from range_finder.conformal import CONFORMAL_ENABLED
    if gex_policy.GEX_LIVE_SPREAD_INFLUENCE_ENABLED or CONFORMAL_ENABLED:
        raise ValueError("Study requires current GEX-off/conformal-off policy")
    if not week.admits(now):
        raise ValueError("Missed capture window")
    train, row = completed_features(prepared["features"], week, now)
    if model == "M4_full" and any(k in prepared.get("source_errors", {}) for k in ("macro", "VIX9D", "VIX3M")):
        raise ValueError("M4_full input source unavailable: " + str(prepared["source_errors"]))
    cols = production_feature_columns(train, model)
    if len(cols) < 2 or any(not np.isfinite(float(row.get(c, np.nan))) for c in cols):
        raise ValueError(f"{model}: insufficient or missing current feature inputs; no mean substitution")
    clean = train[["log_range"] + cols].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) <= max(20, len(cols) + 2):
        raise ValueError(f"{model}: only {len(clean)} completed eligible training rows")
    result = fit_production_model(train, cols, model_name=model)
    history = prepared["weekly"].loc[prepared["weekly"].index < pd.Timestamp(week.monday)]
    side = estimate_side_share_quantile(history)
    chain, root = contract_chain(prepared["chain"], ticker, prepared["expiration"])
    quotes = chain_entry_to_quotes(chain)
    cfg = register_dynamic_config(ticker, strikes=list(quotes), spot=prepared["reference"], instrument_type="stock")
    design_version = model_methodology_id(model_version, model, cols)
    reference, vix = prepared["reference"], prepared["vix"]
    if not finite_price(reference) or not finite_price(vix):
        raise ValueError("Invalid opening anchor")
    # Weekly EM is informational in today's displayed production methodology.
    # Preserve the display fallback behavior by passing no artificial EM floor.
    forecast, plan, tiers = build_recommendations(
        result=result, feature_row=row, feature_cols=cols, reference=reference,
        vix=vix, week_start=str(week.monday), ticker=ticker, side_share_q=side["q"],
        chain_quotes=quotes, weekly_em=None)
    if not all(math.isfinite(forecast[k]) and 0 <= forecast[k] < 1
               for k in ("lower_pct", "point_pct", "upper_pct")):
        raise ValueError("Invalid forecast bounds")
    contracts = {side: {o["strike"]: o for o in chain[side + "s"]} for side in ("put", "call")}
    fitted = {"parameters": dict(result.params), "covariance": result.cov_params().values.tolist(),
              "residual_scale": float(result.scale), "df_resid": float(result.df_resid), "nobs": int(result.nobs)}
    inputs = {"ticker": ticker, "model": model, "model_version": design_version, "methodology_id": model_version,
              "schema_version": SCHEMA_VERSION, "feature_columns": cols,
              "omitted_features": [c for c in MODEL_SPECS[model] if c not in cols],
              "features": frame_records(clean), "feature_row": frame_records(row.to_frame().T),
              "fit": fitted, "side_share": side, "ticker_config": cfg,
              "training_end_week": str(clean.index.max().date()),
              "training_cutoff": training_outcome_close(clean.index.max().date()).isoformat(),
              "feature_cutoff": trading_week_prior_close(week).isoformat(),
              "sources": prepared["source_times"], "source_errors": prepared.get("source_errors", {}),
              "raw_inputs": prepared.get("raw_inputs", {}), "anchor_source": prepared["anchor_source"],
              "anchor_at": prepared["anchor_at"], "reference": reference, "vix": vix,
              "gex_policy": "disabled_pending_calibration", "conformal_enabled": False}
    forecasts = []
    for key, label, tier in zip(TIER_KEYS, TIER_LABELS, tiers):
        shown = displayed_tier(tier)
        put, call = shown["put_short"], shown["call_short"]
        if not (finite_price(put) and finite_price(call) and put < reference < call):
            raise ValueError(f"{label}: invalid short strikes")
        if put not in contracts["put"] or call not in contracts["call"]:
            raise ValueError(f"{label}: short strike not listed in verified chain")
        shorts, wings = {}, {}
        for side in ("put", "call"):
            shorts[side] = contracts[side][shown[f"{side}_short"]]
            wings[side] = [{"long_strike": s.long_strike, "width": s.wing_width,
                            "symbol": contracts[side][s.long_strike]["symbol"]}
                           for s in shown[f"{side}_spreads"] if s.long_strike in contracts[side]]
        forecasts.append({"tier": key, "tier_label": label, "ticker": ticker, "model": model,
                          "model_version": design_version, "methodology_id": model_version, "fit_id": digest(fitted),
                          "expiration": prepared["expiration"], "contract_root": root,
                          "settlement_convention": "PM cash European" if ticker == "SPX" else "physical American",
                          "interpretation": "weekly-close range study" if ticker == "SPX" else "theoretical expiration proxy",
                          "scheduled_at": week.capture_start.isoformat(), "captured_at": now.isoformat(),
                          "available_at": now.isoformat(), "prepared_at": prepared["prepared_at"],
                          "evaluation_close": week.evaluation_close.isoformat(),
                          "feature_cutoff": inputs["feature_cutoff"], "training_cutoff": inputs["training_cutoff"],
                          "data_delay_seconds": prepared["data_delay_seconds"],
                          "anchor_at": prepared["anchor_at"], "reference": reference,
                          "forecast": forecast, "buffer_pct": plan.buffer_pct, "buffer_reason": plan.buffer_reason,
                          "side_share_q": plan.side_share_q, "range_pct": tier.range_pct,
                          "put_short": put, "call_short": call, "range_width_ratio": (call - put) / reference,
                          "short_contracts": shorts, "wings": wings,
                          "recommended_width": plan.recommended_width, "gex_enabled": False})
    return inputs, forecasts


def trading_week_prior_close(week):
    from range_finder.trading_week import trading_week
    return trading_week(week.monday - timedelta(days=7)).evaluation_close


def training_outcome_close(day):
    from range_finder.trading_week import trading_week
    return trading_week(day).evaluation_close
