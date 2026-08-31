# =============================================================================
# har_model.py
# Weekly SPX Range Prediction Model — HAR Regression Module
#
# Fits a Heterogeneous Autoregressive (HAR) model to predict weekly SPX
# range_pct. Uses statsmodels OLS for full inference (t-stats, p-values,
# prediction intervals). Produces point estimates + 80% CI for spread placement.
# =============================================================================

import logging
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error

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

# Weekly training window in years. This pins the READ path (every production
# fit filters model_features to this many years via train_window_min_date),
# so backfilling deeper history into weekly_spx — e.g. the 10y
# history_experiment — can never silently retrain production on a longer
# window. Flip this ONLY when the walk-forward comparison
# (python -m range_finder.history_experiment) clears the adoption gate.
TRAIN_WINDOW_YEARS = 6


def train_window_min_date(years: int = None) -> str:
    """ISO date `years` (default TRAIN_WINDOW_YEARS) back from today —
    the min_date every production fit passes to get_features."""
    from datetime import timedelta as _td
    yrs = years if years is not None else TRAIN_WINDOW_YEARS
    return (datetime.today() - _td(days=int(yrs * 365.25))).strftime("%Y-%m-%d")

# Minimum non-null observations required to fold a feature into a fit.
# These are cadence-specific: the gate counts ROWS at whatever cadence the
# caller's frame carries, so "20" means 20 WEEKS (~5 months) for the weekly
# specs but would mean just 20 trading DAYS (~1 month) for a daily-cadence
# caller — far too low a bar for a stable OLS coefficient. Daily-cadence
# callers should pass min_obs=DEFAULT_MIN_DAYS_FOR_FIT instead (the removed
# 0DTE daily HAR did; the gate keeps the capability).
#
# Weekly default (20): rule-of-thumb that HC3-robust OLS needs ~10
# obs/feature plus headroom. Daily default (126): half a trading year —
# proportionally the same information content as the weekly bar.
# `gex_normalized` uses a lower bar because it's a single-coefficient
# addition to an already-populated spec, and waiting 20 weeks (~5 months)
# before live GEX flows into strike placement is overkill for a stable-sign
# regressor.
DEFAULT_MIN_WEEKS_FOR_FIT = 20
DEFAULT_MIN_DAYS_FOR_FIT = 126
GEX_MIN_WEEKS_FOR_FIT = 12

_FEATURE_MIN_WEEKS_OVERRIDES = {
    "gex_normalized": GEX_MIN_WEEKS_FOR_FIT,
}


def feature_has_enough_data(df, col: str, min_obs: int | None = None) -> bool:
    """True if `col` exists in `df` and has enough non-null observations to
    include in a regression fit.

    The count is cadence-agnostic — it's the CALLER's job to pass a
    ``min_obs`` matching their frame's cadence (weekly callers omit it and
    get DEFAULT_MIN_WEEKS_FOR_FIT; daily callers pass
    DEFAULT_MIN_DAYS_FOR_FIT). Per-feature overrides in
    `_FEATURE_MIN_WEEKS_OVERRIDES` take precedence over ``min_obs``."""
    if col not in df.columns:
        return False
    default = min_obs if min_obs is not None else DEFAULT_MIN_WEEKS_FOR_FIT
    threshold = _FEATURE_MIN_WEEKS_OVERRIDES.get(col, default)
    return bool(df[col].notna().sum() > threshold)


# =============================================================================
# FEATURE SETS PER MODEL SPEC
# =============================================================================

HAR_CORE = ["har_d1", "har_w", "har_m"]

MODEL_SPECS = {
    "M1_baseline": HAR_CORE,

    # vix_implied_range was dropped from every spec below (2026-06): it is a
    # deterministic rescale of vix_close (vix_close / sqrt(52) / 100 * Brownian
    # range factor), so corr(vix_close, vix_implied_range) = 1.0000 exactly.
    # Carrying both made the design matrix singular (condition number ~1.4e10)
    # and every coefficient uninterpretable, with NO predictive benefit — the
    # columns span the same space, so OLS predictions are identical either way.
    # vix_implied_range is still computed/stored and used OUTSIDE the regression
    # (VRP / expected-move display), it is just no longer a model feature.
    "M2_vix": HAR_CORE + [
        "vix_close",
    ],

    "M3_extended": HAR_CORE + [
        "vix_close",
        "hv_ratio",
        "event_count",
        "spx_return_lag1",
        "abs_return_lag1",
    ],

    "M4_full": HAR_CORE + [
        "vix_close",
        "hv_ratio",
        "event_count",
        "spx_return_lag1",
        "abs_return_lag1",
        "vix_ts_slope",
        "yield_spread",
        # gex_normalized added dynamically if GEX data is present
        # (continuous feature, replacing the old binary gex_flag)
    ],

    # M5_garch (HAR + VIX + weekly GARCH(1,1) fit) and M6_regime (HAR + VIX +
    # high-vol-regime interactions) were both removed.
    #
    # M6 was dropped (2026-06) after walk-forward OOS validation showed its
    # regime interaction terms (har_d1_x_regime, har_w_x_regime) were collinear
    # with their parent HAR lags (VIF 18-48, every term statistically
    # insignificant) and actually HURT out-of-sample R² versus the simpler
    # specs on every ticker — it was dominated by M3_extended on index products
    # and by M2_vix on single names. The `high_vol_regime` column and the two
    # interaction terms are no longer consumed by any spec (high_vol_regime is
    # still computed/stored, just unused). Old saved M6 fits are deleted in
    # db.init_all_tables; all other fits are invalidated via SCHEMA_VERSION.
}


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

def _complete_model_frame(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "log_range",
) -> pd.DataFrame:
    """Rows eligible for fitting: selected features plus a realized target."""
    if not feature_cols:
        raise ValueError("feature_cols must contain at least one feature")
    return df[[target_col] + feature_cols].dropna()


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
    clean = _complete_model_frame(df, feature_cols or [], target_col=target_col)

    # A single sparse feature silently guts the sample: dropna() is row-wise,
    # so e.g. gex_normalized (only populated for recent weeks) can shrink a
    # 300-row frame to a few dozen without any signal to the caller. Surface
    # it loudly with the per-column NaN counts so the offending feature is
    # identifiable at a glance.
    if len(clean) < max(40, int(len(df) * 0.5)) and len(df) > 0:
        _null_counts = df[cols_needed].isna().sum().sort_values(ascending=False)
        _worst = ", ".join(f"{c}={int(n)}" for c, n in _null_counts.head(3).items() if n)
        log.warning(
            f"time_series_split: dropna kept only {len(clean)}/{len(df)} rows "
            f"— sparse feature(s) are starving the fit (worst NaN counts: {_worst})"
        )

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


def fit_production_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str = "HAR",
    target_col: str = "log_range",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Refit a forecasting model on every completed eligible observation.

    Holdout evaluation happens before this function is called. Reusing the
    same complete-row filter as ``time_series_split`` guarantees that the
    production fit is exactly validation-train plus holdout, while partial
    current-week and forecast scaffold rows remain excluded by their NULL
    targets.
    """
    clean = _complete_model_frame(df, feature_cols, target_col=target_col)
    X_full = sm.add_constant(clean[feature_cols])
    y_full = clean[target_col]
    return fit_model(X_full, y_full, model_name=f"{model_name} production")


def fit_model_wls(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    half_life: int = 52,
    model_name: str = "HAR-WLS",
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Fit WLS with exponential recency weighting (half_life in weeks)."""
    n = len(y_train)
    decay = np.log(2) / half_life
    # Most recent observation gets weight 1.0, older observations decay
    weights = np.exp(-decay * np.arange(n)[::-1])
    # Normalize to mean 1.0: statsmodels' WLS scale (and with it every
    # prediction interval) is estimated from the weighted residuals, and
    # un-normalized weights averaging ~0.25 shrank the residual scale ~4x —
    # quoting PIs materially narrower than the data supports. Normalization
    # leaves the coefficient estimates untouched (WLS is invariant to a
    # scalar weight rescaling) and only repairs the scale.
    weights = weights / weights.mean()

    model = sm.WLS(y_train, X_train, weights=weights)
    result = model.fit(cov_type="HC3")

    log.info(f"\n{'='*60}")
    log.info(f"  {model_name} — IN-SAMPLE FIT (WLS, half-life={half_life})")
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
    y_train_mean: "float | None" = None,
) -> dict:
    """Evaluate out-of-sample performance.

    y_train_mean: pass the TRAINING target mean for a proper Campbell-
    Thompson OOS R² benchmark; omitted → legacy test-mean benchmark.
    """
    y_pred_log = result.predict(X_test)

    y_pred_pct = np.exp(y_pred_log)
    y_true_pct = np.exp(y_test)

    oos_r2 = _oos_r2(y_test, y_pred_log, benchmark=y_train_mean)
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


def fit_validation_and_production(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str = "HAR",
    target_col: str = "log_range",
) -> tuple[
    sm.regression.linear_model.RegressionResultsWrapper,
    sm.regression.linear_model.RegressionResultsWrapper,
    dict,
]:
    """Evaluate a train-only fit, then build the separate forecasting fit."""
    X_train, X_test, y_train, y_test = time_series_split(
        df, feature_cols=feature_cols, target_col=target_col
    )
    validation_result = fit_model(X_train, y_train, model_name=model_name)
    metrics = evaluate_oos(
        validation_result, X_test, y_test, model_name=model_name
    )
    production_result = fit_production_model(
        df,
        feature_cols=feature_cols,
        model_name=model_name,
        target_col=target_col,
    )
    return validation_result, production_result, metrics


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


def _oos_r2(y_true: pd.Series, y_pred: pd.Series, benchmark=None) -> float:
    """Out-of-sample R² (Campbell & Thompson 2008).

    benchmark: the null model's forecast — a scalar (train mean) or a Series
    aligned to y_true (per-row train means from a walk-forward). Campbell-
    Thompson benchmarks against the HISTORICAL mean; the legacy default of
    y_true.mean() (the TEST-set mean) hands the null model look-ahead
    information and understates the model's relative skill. None keeps the
    legacy behavior for callers that have no train context.
    """
    ss_res = ((y_true - y_pred) ** 2).sum()
    if benchmark is None:
        bench = y_true.mean()
    elif np.isscalar(benchmark):
        bench = benchmark
    else:
        bench = pd.Series(benchmark).reindex(y_true.index)
        bench = bench.fillna(y_true.mean())
    ss_tot = ((y_true - bench) ** 2).sum()
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
# ENHANCEMENT COMPARISON — walk-forward OOS evaluation
# =============================================================================

# Improvement gates: only keep enhancements that clear these thresholds
_MIN_R2_IMPROVEMENT = 0.02    # OOS R² must improve by at least this
_MIN_MAPE_IMPROVEMENT = 0.003  # MAE_pct must improve by at least this (0.3pp)


def compare_enhancements(
    conn,
    baseline_spec: str = "M3_extended",
    exclude_covid: bool = True,
    ticker: str = "SPX",
) -> pd.DataFrame:
    """Walk-forward OOS comparison of model enhancements vs baseline.

    Historically this claimed "walk-forward" while actually running a single
    80/20 split; it is now a thin wrapper over walkforward.compare_configs,
    which genuinely refits on an expanding window. Tests WLS recency
    weighting vs the OLS baseline and applies the adoption gate (R²/MAE
    improvement thresholds + coverage guard). Returns the comparison
    DataFrame with a 'keep' column.
    """
    # Lazy import — walkforward imports from this module.
    from range_finder.walkforward import WalkForwardConfig, compare_configs

    df = get_features(conn, min_date=train_window_min_date(),
                      exclude_covid=exclude_covid, ticker=ticker)
    if df.empty:
        raise RuntimeError(f"model_features is empty for {ticker} — run feature_builder.py first")

    feat_cols = tuple(
        c for c in MODEL_SPECS.get(baseline_spec, [])
        if feature_has_enough_data(df, c)
    )
    if len(feat_cols) < 2:
        log.error(f"{baseline_spec}: too few available features ({list(feat_cols)})")
        return pd.DataFrame()

    configs = [
        WalkForwardConfig(label="Baseline (OLS)", feature_cols=feat_cols),
        WalkForwardConfig(label="Baseline (WLS)", feature_cols=feat_cols,
                          fit_fn_name="wls", fit_kwargs={"half_life": 52}),
    ]
    return compare_configs(df, configs, baseline_label="Baseline (OLS)")


# =============================================================================
# FORECAST + PREDICTION INTERVAL
# =============================================================================

# =============================================================================
# SIDE-SHARE QUANTILE — empirical anchor-position calibration
# =============================================================================

# Tail mass for the side-share quantile. 0.20 → the 80th percentile of how
# much of a week's range lands on one side of the open (q ≈ 0.84 on SPX).
# Chosen from the 150-week walk-forward (side_placement_experiment.py,
# 2026-07-16): any-touch 45.3% under the legacy /2 split vs 10.0% at
# beta=0.20 — which matches the 10% tail mass the 90% one-sided PI is
# audited against, at an average band cost of 3.41%/side vs 2.02%. Deeper
# betas bought little (beta=0.10 → 7.3% touch for +0.37% width) and thin
# the credit further.
SIDE_SHARE_BETA = 0.20

# Below this many completed weeks the estimate is noise — fall back to the
# legacy symmetric half-split rather than trust a thin quantile.
SIDE_SHARE_MIN_OBS = 52


def estimate_side_share_quantile(
    weekly_df: pd.DataFrame,
    beta: float = SIDE_SHARE_BETA,
    min_obs: int = SIDE_SHARE_MIN_OBS,
) -> dict:
    """Empirical (1-beta) quantile of the share of a week's range on one side
    of the open — the honest replacement for the symmetric /2 split.

    The range forecast says how WIDE the week trades, not WHERE the open sits
    inside that range. Splitting the forecast half-up/half-down protects the
    strikes only when the open lands mid-range — and it doesn't: even for a
    driftless walk the open concentrates near one extreme (arcsine law), so a
    trending week breaches a half-split strike while the realized range stays
    inside the forecast. s_up = (H-O)/(H-L) measures that anchor position per
    completed week; its (1-beta) pooled quantile q is the per-side range
    share that historically contained all but a beta-tail of weeks. Callers
    place bands at ref*(1 ± u*q) instead of ref*(1 ± u/2).

    Pools s_up with s_dn = 1 - s_up (up/down symmetry doubles the sample and
    forces a direction-neutral q). Only the caller-supplied rows are used —
    pass training-window rows for walk-forward work, everything-to-date live.

    Returns {"q": float|None, "n": int, "beta": beta}; q is None (caller
    falls back to /2) when fewer than min_obs valid weeks exist.
    """
    cols = ("spx_open", "spx_high", "spx_low")
    if weekly_df is None or weekly_df.empty or any(c not in weekly_df.columns for c in cols):
        return {"q": None, "n": 0, "beta": beta}

    o = weekly_df["spx_open"].astype(float)
    h = weekly_df["spx_high"].astype(float)
    l = weekly_df["spx_low"].astype(float)
    rng = h - l
    valid = (rng > 0) & o.notna() & h.notna() & l.notna() & (h >= o) & (o >= l)
    if int(valid.sum()) < min_obs:
        return {"q": None, "n": int(valid.sum()), "beta": beta}

    s_up = ((h - o) / rng)[valid].clip(0.0, 1.0)
    pooled = pd.concat([s_up, 1.0 - s_up], ignore_index=True)
    q = float(pooled.quantile(1.0 - beta))
    # q below 0.5 would place strikes TIGHTER than the legacy split — only a
    # pathological sample does that; never allow it.
    return {"q": round(max(q, 0.5), 4), "n": int(valid.sum()), "beta": beta}


def forecast_next_week(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    feature_row: pd.Series,
    feature_cols: list[str],
    spx_close: float,
    alpha: float = PI_ALPHA,
    side_share_q: "float | None" = None,
) -> dict:
    """
    Generate a point forecast + prediction interval for the upcoming week.

    side_share_q: per-side range share from estimate_side_share_quantile().
    None keeps the legacy symmetric half-split for the price levels.
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

    # Back-transform log -> range_pct.
    #
    # DELIBERATE (documented after the 2026-07 audit): exp(E[log range]) is
    # the conditional MEDIAN of the range, not its mean — the lognormal mean
    # would need a Jensen correction (× exp(sigma²/2), ~+2-4% here). For
    # strike placement the median is the right anchor: it's the
    # equal-probability center of the range distribution, and the PI bounds
    # (which placement actually rides) are quantiles, which back-transform
    # exactly under any monotone map. Do NOT add a smearing correction to
    # point_pct without re-running calibration — every implied-vs-forecast
    # ratio in the repo (feature_builder.vix_implied_range, model_vs_vix,
    # the calibration audit) was measured against this median convention.
    point_pct = math.exp(log_point)
    lower_pct = max(0.0, math.exp(log_lower))
    upper_pct = math.exp(log_upper)

    # Price levels around the open. NOTE: pi_lower_px below deliberately
    # derives from upper_pct, not a "half_lower" — the model forecasts a
    # RANGE (high-low), so the PI-UPPER range bound maps to one price band
    # and pi_lower_px is that band's floor. lower_pct (the narrow-range PI
    # bound) is a different, tighter band used by the Lower-PI tier in
    # build_spread_tiers, not a lower price here.
    #
    # The per-side share defaults to the legacy symmetric 1/2. When the
    # caller provides side_share_q (estimate_side_share_quantile), each side
    # gets u*q of the range instead — because a range says nothing about
    # WHERE the open sits inside it, and empirically the open concentrates
    # near one extreme, a half-split band is touched far more often than the
    # range-coverage number suggests.
    side_share = 0.5
    if side_share_q is not None and 0.5 <= float(side_share_q) <= 1.0:
        side_share = float(side_share_q)

    half_point = point_pct * side_share
    half_upper = upper_pct * side_share

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
        "side_share_q":    round(side_share, 4),
    }

    return forecast


# =============================================================================
# MODEL PERSISTENCE — re-exported from model_persistence.py
# =============================================================================

from range_finder.model_persistence import save_model, load_model  # noqa: F401
from range_finder.model_persistence import MODEL_DIR  # noqa: F401


# =============================================================================
# FULL PIPELINE
# =============================================================================

def run_full_pipeline(
    conn,
    spx_close: float = None,
    next_week_start: str = None,
    preferred_model: str = "M3_extended",
    exclude_covid: bool = True,
    ticker: str = "SPX",
) -> dict:
    """End-to-end: load features -> fit all specs -> compare -> forecast."""
    # --- Load features (window pinned to TRAIN_WINDOW_YEARS) ---
    df = get_features(conn, min_date=train_window_min_date(),
                      exclude_covid=exclude_covid, ticker=ticker)
    if df.empty:
        raise RuntimeError(f"model_features is empty for {ticker} — run feature_builder.py first")

    log.info(f"Loaded {len(df)} feature rows for modeling ({ticker})")

    # --- Determine if GEX is available ---
    # Use continuous gex_normalized for richer signal than the deprecated binary gex_flag
    # BUG FIX: use a local copy of the feature list instead of mutating the global
    gex_available = feature_has_enough_data(df, "gex_normalized")
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
        available = [c for c in feat_cols if feature_has_enough_data(df, c)]
        if len(available) < len(feat_cols):
            missing = set(feat_cols) - set(available)
            log.warning(f"{spec_name}: skipping missing features {missing}")
        feat_cols = available

        if len(feat_cols) < 2:
            log.warning(f"{spec_name}: fewer than 2 features available, skipping")
            continue

        try:
            validation_result, production_result, metrics = (
                fit_validation_and_production(
                    df, feature_cols=feat_cols, model_name=spec_name
                )
            )
            all_metrics[spec_name] = metrics
            all_results[spec_name] = (
                validation_result, production_result, feat_cols
            )
        except Exception as e:
            log.error(f"Failed to fit {spec_name}: {e}")

    # --- Comparison table ---
    compare_models(all_metrics)

    # --- Diagnostics on preferred model ---
    if preferred_model not in all_results:
        preferred_model = list(all_results.keys())[0]
        log.warning(f"Preferred model unavailable — falling back to {preferred_model}")

    validation_result, production_result, best_features = all_results[preferred_model]
    run_diagnostics(validation_result, model_name=preferred_model)

    # --- Save preferred model ---
    save_model(production_result, best_features, preferred_model,
               all_metrics.get(preferred_model, {}), ticker=ticker)

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

    feature_row = get_feature_for_week(conn, next_week_start, ticker=ticker)
    if feature_row is None:
        log.warning(
            f"No feature row for {next_week_start} — "
            "using most recent available row for demonstration"
        )
        feature_row = df.iloc[-1]

    forecast = forecast_next_week(
        production_result,
        feature_row,
        best_features,
        spx_close,
        alpha=PI_ALPHA,
    )

    return forecast
