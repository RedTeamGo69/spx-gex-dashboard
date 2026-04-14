# =============================================================================
# feature_builder.py
# Weekly SPX Range Prediction Model — Feature Engineering Module
#
# Reads raw data from Postgres (populated by data_collector.py) and produces
# a clean, model-ready feature matrix saved to the `model_features` table.
# =============================================================================

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from range_finder.data_collector import (
    get_weekly_spx,
    get_macro_daily,
    get_event_flags,
    init_db,
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
# DAILY SPX — needed for HV calculation
# =============================================================================

def fetch_daily_spx(years: int = 6) -> pd.DataFrame:
    """Pull daily SPX closes from yfinance for HV calculation."""
    end   = datetime.today()
    start = end - timedelta(days=years * 365)

    log.info("Fetching daily SPX closes for HV calculation...")
    raw = yf.download("^GSPC", start=start, end=end, interval="1d", progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Close"]].copy()
    df.columns = ["spx_close"]
    df.index = pd.to_datetime(df.index).normalize()
    df.dropna(inplace=True)

    log.info(f"Daily SPX: {len(df)} rows")
    return df


def compute_hv_windows(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling historical volatility (annualized) from daily log returns.
    HV formula: std(log_returns, window) * sqrt(252)
    """
    df = daily_df.copy()
    df["log_ret"] = np.log(df["spx_close"] / df["spx_close"].shift(1))

    df["hv5"]  = df["log_ret"].rolling(5,  min_periods=4).std()  * math.sqrt(252)
    df["hv10"] = df["log_ret"].rolling(10, min_periods=8).std()  * math.sqrt(252)
    df["hv20"] = df["log_ret"].rolling(20, min_periods=15).std() * math.sqrt(252)

    # Resample to weekly — take the LAST value of each week (Friday close HV)
    weekly_hv = df[["hv5", "hv10", "hv20"]].resample("W-FRI").last()
    weekly_hv.index = weekly_hv.index - pd.offsets.Week(weekday=0)  # shift to Monday
    weekly_hv.index.name = "week_start"

    log.info(f"HV windows computed: {len(weekly_hv)} weekly rows")
    return weekly_hv


# =============================================================================
# VIX TERM STRUCTURE
# =============================================================================

def fetch_vix_term_structure(years: int = 6) -> pd.DataFrame:
    """Pull weekly closes for VIX9D and VIX3M from yfinance."""
    end   = datetime.today()
    start = end - timedelta(days=years * 365)

    log.info("Fetching VIX9D and VIX3M for term structure...")

    vix9d_raw = yf.download("^VIX9D", start=start, end=end, interval="1wk", progress=False)
    vix3m_raw = yf.download("^VIX3M", start=start, end=end, interval="1wk", progress=False)

    def extract_close(raw, name):
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            log.warning(f"{name} returned empty — will be NULL in features")
            return pd.Series(dtype=float, name=name)
        s = raw["Close"].copy()
        s.name = name
        s.index = pd.to_datetime(s.index).normalize()
        return s

    vix9d = extract_close(vix9d_raw, "vix9d_close")
    vix3m = extract_close(vix3m_raw, "vix3m_close")

    df = pd.DataFrame({"vix9d_close": vix9d, "vix3m_close": vix3m})
    df.index.name = "week_start"

    df["vix_ts_slope"] = df["vix3m_close"] - df["vix9d_close"]

    log.info(f"VIX term structure: {len(df)} weekly rows")
    return df


# =============================================================================
# MACRO — resample daily FRED to weekly
# =============================================================================

def resample_macro_to_weekly(macro_df: pd.DataFrame) -> pd.DataFrame:
    """Resample daily FRED data to weekly frequency (Friday → Monday index)."""
    weekly = macro_df[["yield_spread", "fed_funds"]].resample("W-FRI").last()
    weekly.index = weekly.index - pd.offsets.Week(weekday=0)
    weekly.index.name = "week_start"
    weekly.ffill(inplace=True)
    return weekly


# =============================================================================
# HAR FEATURE COMPUTATION
# =============================================================================

def compute_har_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the three HAR components from the range_pct series.

    Follows the canonical Corsi (2009) HAR-RV structure adapted for weekly
    data: each component ends at lag 1 so the forecast at week t uses only
    information known at the close of week t-1. The components OVERLAP by
    design — this is how standard HAR captures vol cascade across horizons.

    har_d1 — range_pct at lag 1                  (prior week)
    har_w  — mean of lags 1..5   (rolling 5)     (~1 month of weeks)
    har_m  — mean of lags 1..20  (rolling 20)    (~5 months of weeks)
    """
    df = weekly_df[["range_pct"]].copy()

    # shift(1) ensures the lag-1 value is the most recent observation used.
    lagged = df["range_pct"].shift(1)

    df["har_d1"] = lagged

    df["har_w"] = lagged.rolling(5, min_periods=3).mean()

    df["har_m"] = lagged.rolling(20, min_periods=10).mean()

    return df[["har_d1", "har_w", "har_m"]]


# =============================================================================
# GEX PLACEHOLDER
# =============================================================================

def load_gex_inputs(conn, ticker: str = "SPX") -> pd.DataFrame:
    """Load GEX values from the gex_inputs table if it exists.

    Filters by ticker so the HAR feature builder only sees rows that
    match the underlying it was trained on. The feature builder
    normalizes by SPX's spx_open² (see build_features), so mixing XSP
    rows into the input here would produce a ~100× scale mismatch on
    gex_normalized for any week that XSP wrote last.
    """
    try:
        df = pd.read_sql_query(
            "SELECT week_start, gex FROM gex_inputs WHERE ticker = ? "
            "ORDER BY week_start ASC",
            conn,
            params=(ticker,),
            parse_dates=["week_start"],
        )
        df.set_index("week_start", inplace=True)
        log.info(f"GEX inputs loaded: {len(df)} rows (ticker={ticker})")
        return df
    except Exception:
        log.info("gex_inputs table not found — GEX features will be NULL")
        return pd.DataFrame(columns=["gex"])


def upsert_gex(conn, week_start: str, gex: float, notes: str = "",
               ticker: str = "SPX") -> None:
    """Insert or update a single GEX value for a given week and ticker."""
    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO gex_inputs (week_start, ticker, gex, notes, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(week_start, ticker) DO UPDATE SET
            gex        = excluded.gex,
            notes      = excluded.notes,
            updated_at = excluded.updated_at
    """, (week_start, ticker, gex, notes, now))
    conn.commit()
    log.info(f"GEX upserted: {week_start} {ticker} → {gex:,.0f}")


# =============================================================================
# MAIN FEATURE BUILD
# =============================================================================

def build_features(conn, exclude_covid: bool = True) -> pd.DataFrame:
    """
    Assemble the full model-ready feature matrix and save to model_features.
    Every feature is lagged so that at row t (target week), you only
    use information observable at Friday close of week t-1.
    """
    log.info("Building feature matrix...")

    # --- Load base data ---
    weekly  = get_weekly_spx(conn)
    macro   = get_macro_daily(conn)
    events  = get_event_flags(conn)

    # --- Fetch supplemental data ---
    daily_spx = fetch_daily_spx(years=6)
    vix_ts    = fetch_vix_term_structure(years=6)

    # --- HAR components ---
    har = compute_har_features(weekly)

    # --- HV windows ---
    hv = compute_hv_windows(daily_spx)

    # --- Macro weekly ---
    macro_wk = resample_macro_to_weekly(macro)

    # --- GEX ---
    # Feature matrix is built against SPX weekly OHLC (yfinance) and
    # normalized by spx_open², so only SPX GEX rows are meaningful here.
    # Filtering explicitly avoids the XSP-row scale mismatch.
    gex_df = load_gex_inputs(conn, ticker="SPX")

    # --- VIX implied range ---
    # VIX is annualized 1-SD vol in percent. De-annualize by sqrt(52) for the
    # weekly 1-SD, then scale by the Brownian high-low range factor
    # E[H-L] = 2·sigma·sqrt(2/pi) ≈ 1.5958·sigma (Feller 1951 / Parkinson 1980)
    # so the column is directly comparable to realized range_pct = (H-L)/open.
    # Previously this was just 1-SD weekly vol (biased ~37% low as a range
    # predictor); the fix rescales it so UI comparisons like model_vs_vix and
    # the "trust the model" warning in spread_levels.py stop firing spuriously
    # on nearly every week. OLS is scale-invariant so re-fitting the HAR
    # model just rescales its coefficient on this feature — predictive power
    # is unchanged.
    _BM_RANGE_FACTOR = 2.0 * math.sqrt(2.0 / math.pi)
    weekly["vix_implied_range"] = (
        (weekly["vix_close"] / math.sqrt(52)) / 100 * _BM_RANGE_FACTOR
    )

    # --- SPX return lags ---
    weekly["spx_return_lag1"] = weekly["spx_return"].shift(1)
    weekly["abs_return_lag1"] = weekly["spx_return_lag1"].abs()

    # --- Assemble ---
    df = weekly[[
        "range_pct", "log_range",
        "vix_close", "vix_implied_range",
        "spx_return_lag1", "abs_return_lag1",
    ]].copy()

    # VIX close needs to be lagged — we observe prior Friday's VIX
    df["vix_close"]         = df["vix_close"].shift(1)
    df["vix_implied_range"] = df["vix_implied_range"].shift(1)

    # Join HAR
    df = df.join(har, how="left")

    # Join HV (already on Monday index)
    hv_lagged = hv.shift(1)
    df = df.join(hv_lagged, how="left")
    df["hv_ratio"] = df["hv5"] / df["hv20"]

    # --- High-vol regime detection ---
    # Binary flag: 1 if trailing 4-week average VIX > 20
    df["high_vol_regime"] = (df["vix_close"].rolling(4, min_periods=2).mean() > 20).astype(int)

    # Join VIX term structure (lag 1 week)
    vix_ts_lagged = vix_ts.shift(1)
    df = df.join(vix_ts_lagged, how="left")
    df["vix_wk_ratio"] = df["vix_close"] / df["vix3m_close"]

    # Join macro (lag 1 week)
    macro_lagged = macro_wk.shift(1)
    df = df.join(macro_lagged, how="left")

    # Join GEX (Monday open — same week, no lag needed)
    if not gex_df.empty:
        df = df.join(gex_df[["gex"]], how="left")
        # Continuous GEX feature: normalize raw GEX by spot² to cancel the
        # spot² scaling embedded in the GEX formula
        # (gex_engine.py:174 — sign × OI × gamma × 100 × spot²).
        #
        # Without this cancellation, as SPX rose from ~3000 to ~6800 over the
        # last 6 years, the same underlying dealer gamma positioning would
        # produce a ~5× larger gex_normalized value — turning this into a
        # non-stationary "how high is SPX" proxy rather than a "what is the
        # positioning" signal. Dividing by spot² (using the week's SPX open
        # as the reference) gives a time-stationary feature that OLS can
        # meaningfully regress on. Multiplied by 1e4 purely to bring the
        # magnitude into the same order as the other features — OLS is
        # scale-invariant so the exact constant doesn't affect the fit.
        df["gex_normalized"] = df["gex"] / (weekly["spx_open"] ** 2) * 1e4
    else:
        df["gex"]            = np.nan
        df["gex_normalized"] = np.nan

    # Join event flags
    df = df.join(
        events[["has_fomc", "has_cpi", "has_nfp", "has_opex", "event_count"]],
        how="left"
    )
    for col in ["has_fomc", "has_cpi", "has_nfp", "has_opex", "event_count"]:
        df[col] = df[col].fillna(0).astype(int)

    # Drop rows with insufficient lag history
    df.dropna(subset=["har_d1", "har_w", "har_m", "vix_close", "log_range"], inplace=True)

    # Exclude COVID crash period if requested (default True)
    if exclude_covid:
        covid_start = pd.Timestamp("2020-03-01")
        covid_end = pd.Timestamp("2020-09-30")
        pre_filter = len(df)
        df = df[(df.index < covid_start) | (df.index > covid_end)]
        removed = pre_filter - len(df)
        if removed:
            log.info(f"exclude_covid: removed {removed} rows (2020-03-01 to 2020-09-30)")

    log.info(f"Feature matrix: {len(df)} rows x {len(df.columns)} columns")

    # --- Save to DB ---
    _save_features(conn, df)

    return df


def _save_features(conn, df: pd.DataFrame) -> None:
    """Upsert the feature matrix into model_features."""
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    rows = 0

    for week_start, row in df.iterrows():
        cur.execute("""
            INSERT INTO model_features (
                week_start,
                log_range, range_pct,
                har_d1, har_w, har_m,
                vix_close, vix_implied_range,
                vix9d_close, vix3m_close, vix_ts_slope, vix_wk_ratio,
                hv5, hv10, hv20, hv_ratio,
                high_vol_regime,
                gex, gex_normalized,
                yield_spread, fed_funds,
                spx_return_lag1, abs_return_lag1,
                has_fomc, has_cpi, has_nfp, has_opex, event_count,
                updated_at
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            ON CONFLICT(week_start) DO UPDATE SET
                log_range           = excluded.log_range,
                range_pct           = excluded.range_pct,
                har_d1              = excluded.har_d1,
                har_w               = excluded.har_w,
                har_m               = excluded.har_m,
                vix_close           = excluded.vix_close,
                vix_implied_range   = excluded.vix_implied_range,
                vix9d_close         = excluded.vix9d_close,
                vix3m_close         = excluded.vix3m_close,
                vix_ts_slope        = excluded.vix_ts_slope,
                vix_wk_ratio        = excluded.vix_wk_ratio,
                hv5                 = excluded.hv5,
                hv10                = excluded.hv10,
                hv20                = excluded.hv20,
                hv_ratio            = excluded.hv_ratio,
                high_vol_regime     = excluded.high_vol_regime,
                gex                 = excluded.gex,
                gex_normalized      = excluded.gex_normalized,
                yield_spread        = excluded.yield_spread,
                fed_funds           = excluded.fed_funds,
                spx_return_lag1     = excluded.spx_return_lag1,
                abs_return_lag1     = excluded.abs_return_lag1,
                has_fomc            = excluded.has_fomc,
                has_cpi             = excluded.has_cpi,
                has_nfp             = excluded.has_nfp,
                has_opex            = excluded.has_opex,
                event_count         = excluded.event_count,
                updated_at          = excluded.updated_at
        """, (
            week_start.strftime("%Y-%m-%d"),
            _f(row, "log_range"),        _f(row, "range_pct"),
            _f(row, "har_d1"),           _f(row, "har_w"),           _f(row, "har_m"),
            _f(row, "vix_close"),        _f(row, "vix_implied_range"),
            _f(row, "vix9d_close"),      _f(row, "vix3m_close"),
            _f(row, "vix_ts_slope"),     _f(row, "vix_wk_ratio"),
            _f(row, "hv5"),              _f(row, "hv10"),            _f(row, "hv20"),
            _f(row, "hv_ratio"),
            _i(row, "high_vol_regime"),
            _f(row, "gex"),              _f(row, "gex_normalized"),
            _f(row, "yield_spread"),     _f(row, "fed_funds"),
            _f(row, "spx_return_lag1"),  _f(row, "abs_return_lag1"),
            _i(row, "has_fomc"),         _i(row, "has_cpi"),
            _i(row, "has_nfp"),          _i(row, "has_opex"),
            _i(row, "event_count"),
            now,
        ))
        rows += 1

    conn.commit()
    log.info(f"model_features: {rows} rows upserted")


# =============================================================================
# READERS
# =============================================================================

def get_features(conn, min_date: str = None, exclude_covid: bool = False) -> pd.DataFrame:
    """Load the full model_features table as a DataFrame."""
    query = "SELECT * FROM model_features ORDER BY week_start ASC"
    df = pd.read_sql_query(query, conn, parse_dates=["week_start"])
    df.set_index("week_start", inplace=True)

    if min_date:
        df = df[df.index >= pd.to_datetime(min_date)]

    if exclude_covid:
        covid_start = pd.Timestamp("2020-03-01")
        covid_end = pd.Timestamp("2020-09-30")
        df = df[(df.index < covid_start) | (df.index > covid_end)]

    return df


def get_feature_for_week(conn, week_start: str) -> pd.Series | None:
    """Fetch the feature row for a specific week."""
    df = pd.read_sql_query(
        "SELECT * FROM model_features WHERE week_start = ?",
        conn,
        params=(week_start,),
        parse_dates=["week_start"],
    )
    if df.empty:
        log.warning(f"No feature row found for week_start={week_start}")
        return None
    df.set_index("week_start", inplace=True)
    return df.iloc[0]


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def print_feature_summary(df: pd.DataFrame) -> None:
    """Quick console summary of the feature matrix."""
    print("\n" + "=" * 65)
    print("  FEATURE BUILDER — FEATURE MATRIX SUMMARY")
    print("=" * 65)
    print(f"  Rows      : {len(df)}")
    print(f"  Date range: {df.index.min().date()} → {df.index.max().date()}")
    print(f"  Columns   : {len(df.columns)}")
    print()

    null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    if not null_pct.empty:
        print("  NULL % by column (non-zero only):")
        for col, pct in null_pct.items():
            print(f"    {col:<25} {pct:.1f}%")
    else:
        print("  No nulls in feature matrix.")

    print()
    print("  Target (log_range) stats:")
    print(df["log_range"].describe().to_string(float_format="{:.4f}".format))
    print("=" * 65 + "\n")


# =============================================================================
# UTILITY
# =============================================================================

def _f(row: pd.Series, col: str) -> float | None:
    """Safe float extractor."""
    val = row.get(col)
    if val is None:
        return None
    try:
        f = float(val)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


def _i(row: pd.Series, col: str) -> int | None:
    """Safe int extractor."""
    val = _f(row, col)
    return None if val is None else int(val)
