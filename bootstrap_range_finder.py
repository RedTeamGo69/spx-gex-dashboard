#!/usr/bin/env python3
"""
Range Finder bootstrap — one-shot fresh-start setup.

Run this after a Postgres wipe, after a HAR feature-definition change, or
when first deploying the app. It:

  1. Creates every range_finder table (idempotent — CREATE IF NOT EXISTS).
  2. Pulls 6 years of weekly SPX/VIX OHLC from yfinance and upserts to
     weekly_spx.
  3. Pulls FRED macro (10Y, 2Y, FedFunds) if FRED_API_KEY is set.
  4. Builds event flags (FOMC / CPI / NFP / OpEx) from the static calendars.
  5. Rebuilds the full feature matrix (model_features table) using the
     canonical HAR lag structure.
  6. Fits the HAR model across all specs (M1_baseline..M4_full) and reports
     out-of-sample R² + MAE for each.
  7. Saves the best-spec model to saved_models so the Spread Finder tab
     loads it on first open.

Requires:
  DATABASE_URL  — Postgres connection string (Neon, etc.)
  FRED_API_KEY  — optional, for macro features

Usage:
  DATABASE_URL=postgres://... FRED_API_KEY=... python bootstrap_range_finder.py
  DATABASE_URL=postgres://... python bootstrap_range_finder.py --skip-fred

No Tradier token needed — this only touches yfinance + FRED + Postgres.
GEX values will be NaN for all historical weeks (expected; fresh Monday
cron runs will start populating gex_inputs going forward).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
_log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-fred", action="store_true",
                        help="Skip FRED macro fetch (yield_spread/fed_funds features will be NULL)")
    parser.add_argument("--model", default="M3_extended",
                        choices=["M1_baseline", "M2_vix", "M3_extended", "M4_full", "M5_garch", "M6_regime"],
                        help="Which model spec to fit and save (default: M3_extended)")
    parser.add_argument("--years", type=int, default=6,
                        help="Years of history to fetch (default: 6)")
    args = parser.parse_args()

    # ── Validate env ──
    if not os.environ.get("DATABASE_URL", "").strip():
        _log.error("DATABASE_URL not set — cannot connect to Postgres.")
        _log.error("Set it: export DATABASE_URL='postgres://user:pass@host/db'")
        sys.exit(1)

    fred_key = os.environ.get("FRED_API_KEY", "").strip()
    if not fred_key and not args.skip_fred:
        _log.warning("FRED_API_KEY not set — macro features will be NULL. "
                     "Pass --skip-fred to suppress this warning.")

    # ── Imports ──
    from range_finder.db import get_connection, init_all_tables
    from range_finder.data_collector import (
        fetch_spx_vix, save_spx_vix,
        fetch_fred_macro, save_fred_macro,
        build_event_flags, print_summary,
    )
    from range_finder.feature_builder import build_features, get_features
    from range_finder.har_model import (
        MODEL_SPECS, time_series_split, fit_model, evaluate_oos,
    )
    from range_finder.model_persistence import save_model

    # ── Connect + init tables ──
    _log.info("Step 1/7  Connecting to Postgres and initializing tables...")
    conn = get_connection()
    init_all_tables(conn)

    # ── SPX/VIX history ──
    _log.info(f"Step 2/7  Fetching {args.years} years of weekly SPX/VIX from yfinance...")
    try:
        df_spx = fetch_spx_vix(years=args.years)
        n_written = save_spx_vix(conn, df_spx)
        _log.info(f"  ✓ {n_written} weekly rows upserted into weekly_spx")
    except Exception as e:
        _log.error(f"  ✗ yfinance fetch failed: {e}")
        _log.error("  Cannot proceed without weekly_spx data. Check network / yfinance.")
        sys.exit(1)

    # ── FRED macro ──
    if fred_key and not args.skip_fred:
        _log.info("Step 3/7  Fetching FRED macro (DGS10, DGS2, DFF)...")
        try:
            df_macro = fetch_fred_macro(years=args.years)
            n_macro = save_fred_macro(conn, df_macro)
            _log.info(f"  ✓ {n_macro} daily rows upserted into macro_daily")
        except Exception as e:
            _log.warning(f"  ⚠ FRED fetch failed: {e} (continuing without macro features)")
    else:
        _log.info("Step 3/7  Skipping FRED macro fetch")

    # ── Event flags ──
    _log.info("Step 4/7  Building event flags (FOMC / CPI / NFP / OpEx)...")
    try:
        build_event_flags(conn)
        _log.info("  ✓ event_flags populated")
    except Exception as e:
        _log.warning(f"  ⚠ event flag build failed: {e}")

    # ── Feature matrix rebuild ──
    _log.info("Step 5/7  Rebuilding feature matrix (canonical HAR lag structure)...")
    try:
        df_feat = build_features(conn)
        _log.info(f"  ✓ {len(df_feat)} feature rows written to model_features")
    except Exception as e:
        _log.error(f"  ✗ Feature rebuild failed: {e}")
        sys.exit(1)

    if df_feat.empty:
        _log.error("  ✗ Feature matrix is empty — cannot fit model.")
        sys.exit(1)

    # ── Fit all specs and print a comparison table ──
    _log.info("Step 6/7  Fitting all model specs and comparing OOS metrics...")

    results = {}
    for spec_name in ["M1_baseline", "M2_vix", "M3_extended", "M4_full"]:
        feat_cols = MODEL_SPECS.get(spec_name, [])
        # Drop features that have too few non-null rows (eg. gex_normalized on
        # a fresh DB — there won't be any historical GEX values)
        avail_cols = [c for c in feat_cols if c in df_feat.columns and df_feat[c].notna().sum() > 20]
        dropped = set(feat_cols) - set(avail_cols)
        if dropped:
            _log.info(f"  {spec_name}: dropping insufficient-data features: {sorted(dropped)}")

        if not avail_cols:
            _log.warning(f"  {spec_name}: no usable features, skipping")
            continue

        try:
            X_train, X_test, y_train, y_test = time_series_split(
                df_feat, feature_cols=avail_cols
            )
            result = fit_model(X_train, y_train, model_name=spec_name)
            metrics = evaluate_oos(result, X_test, y_test, model_name=spec_name)
            results[spec_name] = {
                "result": result,
                "metrics": metrics,
                "features": avail_cols,
            }
        except Exception as e:
            _log.warning(f"  {spec_name} fit failed: {e}")

    if not results:
        _log.error("No specs fit successfully — aborting.")
        sys.exit(1)

    # Print comparison table
    print()
    print("=" * 80)
    print("  MODEL COMPARISON — fresh HAR rebuild")
    print("=" * 80)
    print(f"  {'Spec':<16} {'OOS R²':>10} {'MAE %':>10} {'Direction':>12} {'N features':>12}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*12} {'-'*12}")
    for spec, info in results.items():
        m = info["metrics"]
        print(
            f"  {spec:<16} "
            f"{m['oos_r2']:>10.4f} "
            f"{m['mae_pct']*100:>9.2f}% "
            f"{m['direction_acc']:>11.2%} "
            f"{len(info['features']):>12}"
        )
    print()

    # ── Save the user-selected model ──
    if args.model not in results:
        _log.error(f"Requested --model {args.model} did not fit successfully. "
                   f"Available: {list(results.keys())}")
        sys.exit(1)

    _log.info(f"Step 7/7  Saving {args.model} to saved_models...")
    chosen = results[args.model]
    save_model(
        chosen["result"],
        chosen["features"],
        args.model,
        chosen["metrics"],
        conn=conn,
    )
    _log.info(f"  ✓ Model {args.model} saved.")

    # ── Final summary ──
    print()
    print_summary(conn)
    _log.info("Bootstrap complete. The Spread Finder tab will now load the saved model.")
    _log.info("Next scheduled cron run (or manual workflow_dispatch) will start populating")
    _log.info("gex_inputs and weekly_setup going forward.")


if __name__ == "__main__":
    main()
