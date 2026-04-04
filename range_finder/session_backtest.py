# =============================================================================
# session_backtest.py
# Backtest session classification thresholds against historical SPX data.
#
# Uses daily SPX OHLC + VIX to compute:
#   - Overnight gap (open vs prev close)
#   - VIX-implied expected move (EM)
#   - Move ratio (gap / EM)
#   - Realized intraday range (high - low) / open
#
# Grid-searches over threshold pairs to find optimal boundaries that
# separate low-range from high-range days. Outputs recommended thresholds.
#
# Run standalone:  python -m range_finder.session_backtest
# =============================================================================

import math
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# DATA FETCH
# =============================================================================

def fetch_backtest_data(years: int = 4) -> pd.DataFrame:
    """
    Pull daily SPX + VIX data and compute session classification features.

    Returns a DataFrame with one row per trading day:
        - overnight_gap_pts: open - prev_close
        - overnight_gap_abs: abs(overnight_gap_pts)
        - vix_prev_close: previous day's VIX close
        - em_pts: VIX-implied daily expected move in points
        - move_ratio: overnight_gap_abs / em_pts
        - intraday_range_pct: (high - low) / open
        - intraday_range_pts: high - low
    """
    end = datetime.today()
    start = end - timedelta(days=years * 365 + 30)

    log.info(f"Fetching SPX daily OHLC ({years} years)...")
    spx = yf.download("^GSPC", start=start, end=end, interval="1d", progress=False)
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    log.info(f"Fetching VIX daily close ({years} years)...")
    vix = yf.download("^VIX", start=start, end=end, interval="1d", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)

    df = pd.DataFrame({
        "spx_open": spx["Open"],
        "spx_high": spx["High"],
        "spx_low": spx["Low"],
        "spx_close": spx["Close"],
        "vix_close": vix["Close"],
    })
    df.dropna(inplace=True)

    # Lag VIX and SPX close to represent "known at market open" info
    df["prev_close"] = df["spx_close"].shift(1)
    df["vix_prev_close"] = df["vix_close"].shift(1)

    # Overnight gap
    df["overnight_gap_pts"] = df["spx_open"] - df["prev_close"]
    df["overnight_gap_abs"] = df["overnight_gap_pts"].abs()

    # VIX-implied daily expected move: VIX / sqrt(252) / 100 * prev_close
    df["em_pts"] = (df["vix_prev_close"] / math.sqrt(252)) / 100 * df["prev_close"]

    # Move ratio
    df["move_ratio"] = df["overnight_gap_abs"] / df["em_pts"]

    # Realized intraday range (what happened after the open)
    df["intraday_range_pct"] = (df["spx_high"] - df["spx_low"]) / df["spx_open"]
    df["intraday_range_pts"] = df["spx_high"] - df["spx_low"]

    # Drop first row (no prev_close) and any NaN/inf
    df.dropna(inplace=True)
    df = df[np.isfinite(df["move_ratio"])]
    df = df[df["em_pts"] > 0]

    log.info(f"Backtest dataset: {len(df)} trading days")
    return df


# =============================================================================
# THRESHOLD OPTIMIZATION
# =============================================================================

def evaluate_thresholds(df: pd.DataFrame, low_thresh: float, high_thresh: float) -> dict:
    """
    Classify days by move_ratio thresholds and measure how well they
    separate low-range from high-range realized intraday outcomes.

    A good threshold pair should:
    - "low" days have meaningfully lower median intraday range
    - "high" days have meaningfully higher median intraday range
    - Clear separation between buckets
    """
    low_mask = df["move_ratio"] < low_thresh
    mid_mask = (df["move_ratio"] >= low_thresh) & (df["move_ratio"] <= high_thresh)
    high_mask = df["move_ratio"] > high_thresh

    n_low = low_mask.sum()
    n_mid = mid_mask.sum()
    n_high = high_mask.sum()

    if n_low < 10 or n_high < 10:
        return None

    range_low = df.loc[low_mask, "intraday_range_pct"]
    range_mid = df.loc[mid_mask, "intraday_range_pct"]
    range_high = df.loc[high_mask, "intraday_range_pct"]

    median_low = range_low.median()
    median_mid = range_mid.median() if n_mid > 5 else np.nan
    median_high = range_high.median()

    # Separation score: how much higher is the high-bucket range vs low-bucket
    separation = median_high - median_low
    # Relative separation (normalized by overall median)
    overall_median = df["intraday_range_pct"].median()
    rel_separation = separation / overall_median if overall_median > 0 else 0

    # Fraction of "low" days that actually had below-median range
    accuracy_low = (range_low < overall_median).mean()
    # Fraction of "high" days that actually had above-median range
    accuracy_high = (range_high > overall_median).mean()

    # Combined score: separation * accuracy
    combined_score = rel_separation * (accuracy_low + accuracy_high) / 2

    return {
        "low_thresh": low_thresh,
        "high_thresh": high_thresh,
        "n_low": int(n_low),
        "n_mid": int(n_mid),
        "n_high": int(n_high),
        "median_range_low": round(median_low * 100, 4),
        "median_range_mid": round(median_mid * 100, 4) if not np.isnan(median_mid) else None,
        "median_range_high": round(median_high * 100, 4),
        "separation_pct": round(separation * 100, 4),
        "rel_separation": round(rel_separation, 4),
        "accuracy_low": round(accuracy_low, 4),
        "accuracy_high": round(accuracy_high, 4),
        "combined_score": round(combined_score, 4),
    }


def grid_search_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Grid search over threshold pairs. Returns sorted results.
    """
    results = []

    low_candidates = np.arange(0.25, 0.56, 0.05)
    high_candidates = np.arange(0.55, 0.86, 0.05)

    for low_t in low_candidates:
        for high_t in high_candidates:
            if high_t <= low_t + 0.05:
                continue
            result = evaluate_thresholds(df, low_t, high_t)
            if result is not None:
                results.append(result)

    results_df = pd.DataFrame(results)
    results_df.sort_values("combined_score", ascending=False, inplace=True)
    results_df.reset_index(drop=True, inplace=True)

    return results_df


# =============================================================================
# DETAILED STATS FOR CHOSEN THRESHOLDS
# =============================================================================

def detailed_bucket_stats(df: pd.DataFrame, low_thresh: float, high_thresh: float) -> dict:
    """
    Produce detailed statistics for the chosen threshold pair.
    """
    low_mask = df["move_ratio"] < low_thresh
    mid_mask = (df["move_ratio"] >= low_thresh) & (df["move_ratio"] <= high_thresh)
    high_mask = df["move_ratio"] > high_thresh

    stats = {}
    for label, mask in [("low", low_mask), ("moderate", mid_mask), ("high", high_mask)]:
        subset = df[mask]
        if len(subset) == 0:
            continue
        stats[label] = {
            "count": len(subset),
            "pct_of_days": round(len(subset) / len(df) * 100, 1),
            "median_range_pct": round(subset["intraday_range_pct"].median() * 100, 3),
            "mean_range_pct": round(subset["intraday_range_pct"].mean() * 100, 3),
            "p25_range_pct": round(subset["intraday_range_pct"].quantile(0.25) * 100, 3),
            "p75_range_pct": round(subset["intraday_range_pct"].quantile(0.75) * 100, 3),
            "median_move_ratio": round(subset["move_ratio"].median(), 3),
        }
    return stats


# =============================================================================
# MAIN
# =============================================================================

def run_backtest(years: int = 4) -> dict:
    """
    Full backtest pipeline. Returns recommended thresholds and stats.
    """
    df = fetch_backtest_data(years=years)

    log.info("Running grid search over threshold pairs...")
    results = grid_search_thresholds(df)

    if results.empty:
        log.warning("No valid threshold pairs found (insufficient data?).")
        return {"error": "no_results", "dataset_size": len(df)}

    best = results.iloc[0]
    low_t = float(best["low_thresh"])
    high_t = float(best["high_thresh"])

    log.info(f"\nOptimal thresholds: low={low_t:.2f}, high={high_t:.2f}")
    log.info(f"Combined score: {best['combined_score']:.4f}")
    log.info(f"Separation: {best['separation_pct']:.3f}%")
    log.info(f"Accuracy (low bucket): {best['accuracy_low']:.1%}")
    log.info(f"Accuracy (high bucket): {best['accuracy_high']:.1%}")

    stats = detailed_bucket_stats(df, low_t, high_t)

    print("\n" + "=" * 65)
    print("  SESSION CLASSIFICATION BACKTEST RESULTS")
    print("=" * 65)
    print(f"  Dataset: {len(df)} trading days ({years} years)")
    print(f"  Optimal low threshold:  {low_t:.2f}  (move_ratio < {low_t:.2f} = 'low' overnight move)")
    print(f"  Optimal high threshold: {high_t:.2f}  (move_ratio > {high_t:.2f} = 'high' overnight move)")
    print()
    for label, s in stats.items():
        print(f"  [{label.upper()}] {s['count']} days ({s['pct_of_days']}%)")
        print(f"    Median intraday range: {s['median_range_pct']:.3f}%")
        print(f"    IQR range: {s['p25_range_pct']:.3f}% – {s['p75_range_pct']:.3f}%")
        print(f"    Median move ratio: {s['median_move_ratio']:.3f}")
        print()
    print(f"  Top 5 threshold pairs by combined score:")
    print(results.head(5)[["low_thresh", "high_thresh", "combined_score",
                           "accuracy_low", "accuracy_high", "separation_pct"]].to_string(index=False))
    print("=" * 65)

    return {
        "recommended_low_threshold": low_t,
        "recommended_high_threshold": high_t,
        "best_result": best.to_dict(),
        "bucket_stats": stats,
        "top_5": results.head(5).to_dict(orient="records"),
        "dataset_size": len(df),
    }


if __name__ == "__main__":
    run_backtest(years=4)
