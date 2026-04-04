# =============================================================================
# har_model.py
# Weekly SPX Range Prediction Model — HAR Regression Module
#
# Fits a Heterogeneous Autoregressive (HAR) model to predict weekly SPX
# range_pct. Uses statsmodels OLS for full inference (t-stats, p-values,
# prediction intervals). Produces point estimates + 80% CI for spread placement.
# =============================================================================

import sqlite3
import logging
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

from range_finder.data_collector import DB_PATH, init_db
from range_finder.feature_builder import (
    init_features_table,
    build_features,
    get_features,
    get_feature_for_week,
    create_gex_table,
    print_feature_summary,
)

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# =============================================================================
# CONFIG
# =============================================================================

TEST_SIZE = 0.20
PI_ALPHA = 0.20

# Model save path
MODEL_DIR = Path(__file__).parent / "models"

# =============================================================================
# FEATURE SETS PER MODEL SPEC
# =============================================================================

HAR_CORE = ["har_d1", "har_w", "har_m"]

MODEL_SPECS = {
    "M1_baseline": HAR_CORE,

    "M2_vix": HAR_CORE + [
        "vix_close",
        "vix_implied_range",
    ],

    "M3_extended": HAR_CORE + [
        "vix_close",
        "vix_implied_range",
        "hv_ratio",
        "event_count",
        "spx_return_lag1",
        "abs_return_lag1",
    ],

    "M4_full": HAR_CORE + [
        "vix_close",
        "vix_implied_range",
        "hv_ratio",
        "event_count",
        "spx_return_lag1",
        "abs_return_lag1",
        "vix_ts_slope",
        "yield_spread",
        # gex_normalized added dynamically if GEX data is present
        # (continuous feature, replacing the old binary gex_flag)
    ],
}


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

def time_series_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    feature_cols: list[str] = None,
    target_col: str = "log_range",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Chronological train/test split — NEVER random shuffle.
    """
    cols_needed = [target_col] + (feature_cols or [])
    clean = df[cols_needed].dropna()

    n_test  = max(1, int(len(clean) * test_size))
    n_train = len(clean) - n_test

    train = clean.iloc[:n_train]
    test  = clean.iloc[n_train:]

    X_train = sm.add_constant(train[feature_cols])
    X_test  = sm.add_constant(test[feature_cols])
    y_train = train[target_col]
    y_test  = test[target_col]

    log.info(
        f"Split: {len(train)} train rows "
        f"({train.index.min().date()} -> {train.index.max().date()})  |  "
        f"{len(test)} test rows "
        f"({test.index.min().date()} -> {test.index.max().date()})"
    )
    return X_train, X_test, y_train, y_test


# =============================================================================
# MODEL FITTING
# =============================================================================

def fit_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "HAR",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit OLS with HC3 heteroskedasticity-robust standard errors."""
    model = sm.OLS(y_train, X_train)
    result = model.fit(cov_type="HC3")

    log.info(f"\n{'='*60}")
    log.info(f"  {model_name} — IN-SAMPLE FIT")
    log.info(f"{'='*60}")
    log.info(f"  R²        : {result.rsquared:.4f}")
    log.info(f"  Adj R²    : {result.rsquared_adj:.4f}")
    log.info(f"  AIC       : {result.aic:.2f}")
    log.info(f"  BIC       : {result.bic:.2f}")
    log.info(f"  N (train) : {int(result.nobs)}")
    log.info(result.summary2().tables[1].to_string())

    return result


# =============================================================================
# OUT-OF-SAMPLE EVALUATION
# =============================================================================

def evaluate_oos(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "HAR",
) -> dict:
    """Evaluate out-of-sample performance."""
    y_pred_log = result.predict(X_test)

    y_pred_pct = np.exp(y_pred_log)
    y_true_pct = np.exp(y_test)

    oos_r2 = _oos_r2(y_test, y_pred_log)
    mae    = mean_absolute_error(y_test, y_pred_log)
    rmse   = math.sqrt(mean_squared_error(y_test, y_pred_log))
    mae_pct = mean_absolute_error(y_true_pct, y_pred_pct)

    # Directional accuracy
    y_lag  = y_test.shift(1).dropna()
    y_pred_aligned = y_pred_log[y_lag.index]
    y_true_aligned = y_test[y_lag.index]
    direction_acc = (
        np.sign(y_pred_aligned - y_lag.values) ==
        np.sign(y_true_aligned - y_lag.values)
    ).mean()

    metrics = {
        "model":         model_name,
        "oos_r2":        oos_r2,
        "mae_log":       mae,
        "rmse_log":      rmse,
        "mae_pct":       mae_pct,
        "direction_acc": direction_acc,
        "n_test":        len(y_test),
    }

    log.info(f"\n{'='*60}")
    log.info(f"  {model_name} — OUT-OF-SAMPLE EVALUATION")
    log.info(f"{'='*60}")
    log.info(f"  OOS R²            : {oos_r2:.4f}")
    log.info(f"  MAE (log_range)   : {mae:.4f}")
    log.info(f"  RMSE (log_range)  : {rmse:.4f}")
    log.info(f"  MAE (range_pct)   : {mae_pct:.4f}  ({mae_pct*100:.2f}%)")
    log.info(f"  Directional acc   : {direction_acc:.2%}")
    log.info(f"  N (test)          : {len(y_test)}")

    return metrics


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison table across all model specs."""
    df = pd.DataFrame(results.values())
    df = df.sort_values("oos_r2", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 75)
    print("  MODEL COMPARISON — OUT-OF-SAMPLE")
    print("=" * 75)
    print(df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x),
    ))
    print("=" * 75 + "\n")

    return df


def _oos_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Out-of-sample R² (Campbell & Thompson 2008)."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot)


# =============================================================================
# RESIDUAL DIAGNOSTICS
# =============================================================================

def run_diagnostics(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    model_name: str = "HAR",
) -> dict:
    """Run standard OLS assumption checks on the fitted model."""
    from statsmodels.stats.stattools import durbin_watson, jarque_bera
    from statsmodels.stats.diagnostic import het_breuschpagan

    resids = result.resid

    dw     = durbin_watson(resids)
    jb_val, jb_p, _, _ = jarque_bera(resids)
    bp_lm, bp_p, _, _  = het_breuschpagan(resids, result.model.exog)
    cond_num = result.condition_number

    diag = {
        "durbin_watson":  dw,
        "jarque_bera_p":  jb_p,
        "breusch_pagan_p": bp_p,
        "condition_number": cond_num,
    }

    log.info(f"\n{'='*60}")
    log.info(f"  {model_name} — RESIDUAL DIAGNOSTICS")
    log.info(f"{'='*60}")
    log.info(f"  Durbin-Watson     : {dw:.3f}  (target ~2.0; <1.5 -> autocorrelation)")
    log.info(f"  Jarque-Bera p     : {jb_p:.4f}  (< 0.05 -> non-normal residuals)")
    log.info(f"  Breusch-Pagan p   : {bp_p:.4f}  (< 0.05 -> heteroskedastic errors)")
    log.info(f"  Condition number  : {cond_num:.1f}  (> 30 -> multicollinearity concern)")

    return diag


# =============================================================================
# FORECAST + PREDICTION INTERVAL
# =============================================================================

def forecast_next_week(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    feature_row: pd.Series,
    feature_cols: list[str],
    spx_close: float,
    alpha: float = PI_ALPHA,
) -> dict:
    """
    Generate a point forecast + prediction interval for the upcoming week.
    """
    # Build feature vector, using training-data means for missing values
    # (0.0 would be wrong for log-transformed features like har_d1/har_w/har_m)
    train_means = {}
    try:
        exog_df = pd.DataFrame(result.model.exog, columns=result.model.exog_names)
        train_means = exog_df.mean().to_dict()
    except Exception:
        pass

    vals = {}
    for col in feature_cols:
        v = feature_row.get(col)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            vals[col] = float(v)
        else:
            vals[col] = float(train_means.get(col, 0.0))

    X_new = pd.DataFrame([vals], columns=feature_cols)
    X_new = sm.add_constant(X_new, has_constant="add")

    # Align columns to training frame
    X_new = X_new.reindex(columns=result.model.exog_names, fill_value=0.0)

    # Prediction + interval
    pred   = result.get_prediction(X_new)
    frame  = pred.summary_frame(alpha=alpha)

    log_point = float(frame["mean"].iloc[0])
    log_lower = float(frame["obs_ci_lower"].iloc[0])
    log_upper = float(frame["obs_ci_upper"].iloc[0])

    # Back-transform log -> range_pct
    point_pct = math.exp(log_point)
    lower_pct = max(0.0, math.exp(log_lower))
    upper_pct = math.exp(log_upper)

    # Price levels (symmetric around the open)
    half_point = point_pct / 2
    half_upper = upper_pct / 2

    vix_implied = float(feature_row.get("vix_implied_range", 0) or 0)

    forecast = {
        "point_pct":       round(point_pct,  4),
        "lower_pct":       round(lower_pct,  4),
        "upper_pct":       round(upper_pct,  4),
        "point_upper_px":  round(spx_close * (1 + half_point), 2),
        "point_lower_px":  round(spx_close * (1 - half_point), 2),
        "pi_upper_px":     round(spx_close * (1 + half_upper), 2),
        "pi_lower_px":     round(spx_close * (1 - half_upper), 2),
        "spx_ref_close":   spx_close,
        "vix_implied_pct": round(vix_implied, 4),
        "model_vs_vix":    round(point_pct - vix_implied, 4),
        "confidence_level": int((1 - alpha) * 100),
        "alpha":           alpha,
    }

    return forecast


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    feature_cols: list[str],
    model_name: str,
    metrics: dict,
    conn=None,
):
    """Save the fitted model + metadata to database (or pickle fallback)."""
    payload = {
        "result":       result,
        "feature_cols": feature_cols,
        "model_name":   model_name,
        "metrics":      metrics,
        "fitted_at":    datetime.now(timezone.utc).isoformat(),
    }

    blob = pickle.dumps(payload)
    now = datetime.now(timezone.utc).isoformat()

    if conn is not None:
        try:
            from range_finder.db import get_backend
            if get_backend() == "postgres":
                import psycopg2
                conn.execute("""
                    INSERT INTO saved_models (model_name, model_data, fitted_at, updated_at)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(model_name) DO UPDATE SET
                        model_data = EXCLUDED.model_data,
                        fitted_at  = EXCLUDED.fitted_at,
                        updated_at = EXCLUDED.updated_at
                """, (model_name, psycopg2.Binary(blob), payload["fitted_at"], now))
            else:
                conn.execute("""
                    INSERT INTO saved_models (model_name, model_data, fitted_at, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(model_name) DO UPDATE SET
                        model_data = excluded.model_data,
                        fitted_at  = excluded.fitted_at,
                        updated_at = excluded.updated_at
                """, (model_name, blob, payload["fitted_at"], now))
            conn.commit()
            log.info(f"Model saved to database: {model_name}")
            return
        except Exception as e:
            log.warning(f"DB save failed, falling back to pickle: {e}")

    # Fallback: pickle to disk
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / f"{model_name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    log.info(f"Model saved to disk: {path}")


def load_model(model_name: str, conn=None) -> dict:
    """Load a previously saved model from database (or pickle fallback)."""
    if conn is not None:
        try:
            from range_finder.db import get_backend
            if get_backend() == "postgres":
                cur = conn.execute(
                    "SELECT model_data, fitted_at FROM saved_models WHERE model_name = %s",
                    (model_name,)
                )
            else:
                cur = conn.execute(
                    "SELECT model_data, fitted_at FROM saved_models WHERE model_name = ?",
                    (model_name,)
                )
            row = cur.fetchone()
            if row:
                blob = row[0]
                if isinstance(blob, memoryview):
                    blob = bytes(blob)
                payload = pickle.loads(blob)
                log.info(f"Model loaded from database: {model_name}  (fitted {payload['fitted_at']})")
                return payload
        except Exception as e:
            log.warning(f"DB load failed, trying pickle fallback: {e}")

    # Fallback: pickle from disk
    path = MODEL_DIR / f"{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model found for {model_name}")

    with open(path, "rb") as f:
        payload = pickle.load(f)

    log.info(f"Model loaded from disk: {path}  (fitted {payload['fitted_at']})")
    return payload


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_full_pipeline(
    conn: sqlite3.Connection,
    spx_close: float = None,
    next_week_start: str = None,
    preferred_model: str = "M3_extended",
) -> dict:
    """End-to-end: load features -> fit all specs -> compare -> forecast."""
    # --- Load features ---
    df = get_features(conn)
    if df.empty:
        raise RuntimeError("model_features is empty — run feature_builder.py first")

    log.info(f"Loaded {len(df)} feature rows for modeling")

    # --- Determine if GEX is available ---
    # Use continuous gex_normalized for richer signal than the deprecated binary gex_flag
    # BUG FIX: use a local copy of the feature list instead of mutating the global
    gex_available = "gex_normalized" in df.columns and df["gex_normalized"].notna().sum() > 20
    local_specs = {k: list(v) for k, v in MODEL_SPECS.items()}
    if gex_available:
        log.info("GEX data available — adding gex_normalized (continuous) to M4_full")
        if "gex_normalized" not in local_specs["M4_full"]:
            local_specs["M4_full"].append("gex_normalized")
    else:
        log.info("GEX data sparse — GEX features excluded from M4_full")

    # --- Fit and evaluate all specs ---
    all_metrics  = {}
    all_results  = {}

    for spec_name, feat_cols in local_specs.items():
        available = [c for c in feat_cols if c in df.columns and df[c].notna().sum() > 20]
        if len(available) < len(feat_cols):
            missing = set(feat_cols) - set(available)
            log.warning(f"{spec_name}: skipping missing features {missing}")
        feat_cols = available

        try:
            X_train, X_test, y_train, y_test = time_series_split(df, feature_cols=feat_cols)
            result = fit_model(X_train, y_train, model_name=spec_name)
            metrics = evaluate_oos(result, X_test, y_test, model_name=spec_name)
            all_metrics[spec_name] = metrics
            all_results[spec_name] = (result, feat_cols)
        except Exception as e:
            log.error(f"Failed to fit {spec_name}: {e}")

    # --- Comparison table ---
    compare_models(all_metrics)

    # --- Diagnostics on preferred model ---
    if preferred_model not in all_results:
        preferred_model = list(all_results.keys())[0]
        log.warning(f"Preferred model unavailable — falling back to {preferred_model}")

    best_result, best_features = all_results[preferred_model]
    run_diagnostics(best_result, model_name=preferred_model)

    # --- Save preferred model ---
    save_model(best_result, best_features, preferred_model, all_metrics.get(preferred_model, {}))

    # --- Live forecast ---
    if spx_close is None:
        from range_finder.data_collector import get_weekly_spx
        wkly = get_weekly_spx(conn)
        spx_close = float(wkly["spx_close"].iloc[-1])
        log.info(f"Using most recent SPX close from DB: {spx_close:,.2f}")

    if next_week_start is None:
        today = datetime.today()
        days_ahead = (7 - today.weekday()) % 7 or 7
        next_week_start = (today + pd.Timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        log.info(f"Forecasting for week starting: {next_week_start}")

    feature_row = get_feature_for_week(conn, next_week_start)
    if feature_row is None:
        log.warning(
            f"No feature row for {next_week_start} — "
            "using most recent available row for demonstration"
        )
        feature_row = df.iloc[-1]

    forecast = forecast_next_week(
        best_result,
        feature_row,
        best_features,
        spx_close,
        alpha=PI_ALPHA,
    )

    return forecast
