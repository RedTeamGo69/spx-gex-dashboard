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

# The COVID-crash window excluded from training (both at build time and by
# exclude_covid readers). Single-sourced here so experiments that toggle the
# exclusion (range_finder.covid_experiment) filter the exact same rows.
COVID_START = pd.Timestamp("2020-03-01")
COVID_END = pd.Timestamp("2020-09-30")


def assess_path_provenance(feature_row: pd.Series, forecast_week) -> tuple:
    """Return whether a forecast row uses the immediately prior week's path.

    Monday's build may create next week's scaffold before the current weekly
    high/low path is complete.  The row is structurally valid for storage, but
    it is not valid to serve after the UI rolls to that next week on Friday.
    ``path_source_week`` makes that distinction explicit instead of treating
    mere row existence (or a recent ``updated_at``) as proof of freshness.
    """
    required_week = pd.Timestamp(forecast_week)
    if required_week.tzinfo is not None:
        required_week = required_week.tz_localize(None)
    required_week = required_week.normalize() - pd.Timedelta(days=7)

    raw_source = feature_row.get("path_source_week")
    if raw_source is None or pd.isna(raw_source):
        return False, None, required_week
    try:
        source_week = pd.Timestamp(raw_source)
        if source_week.tzinfo is not None:
            source_week = source_week.tz_localize(None)
        source_week = source_week.normalize()
    except (TypeError, ValueError):
        return False, None, required_week
    return source_week == required_week, source_week, required_week


# =============================================================================
# DATABASE — add model_features table
# =============================================================================

def init_features_table(conn) -> None:
    """
    Ensure the model_features table exists.
    Now handled by db.init_all_tables() — this is kept for backwards compatibility.
    """
    pass  # Tables created in db.init_all_tables()


# =============================================================================
# DAILY SPX — needed for HV calculation
# =============================================================================

def fetch_daily_spx(years: int = 6) -> pd.DataFrame:
    """Pull daily SPX closes from yfinance for HV calculation."""
    return fetch_daily_underlying("^GSPC", years=years, label="SPX",
                                   close_col="spx_close")


def fetch_daily_underlying(yf_symbol: str, years: int = 6,
                            label: str = None,
                            close_col: str = "spx_close",
                            tradier_symbol: str = None) -> pd.DataFrame:
    """Pull daily closes for any underlying — Tradier primary, yfinance fallback.

    Returns a DataFrame indexed by date with one column whose name defaults to
    ``spx_close`` so the existing ``compute_hv_windows`` code path works
    unchanged on any ticker. Pass a different ``close_col`` if you need to
    distinguish multiple series in the same DataFrame.
    """
    label = label or yf_symbol
    log.info(f"Fetching daily {label} closes for HV calculation...")

    # bar_sources handles source selection + validation + fallback. The
    # label doubles as the Tradier symbol for app tickers (SPX/QQQ/...);
    # callers can override via tradier_symbol.
    try:
        from range_finder.bar_sources import fetch_daily_bars
        bars = fetch_daily_bars(yf_symbol, tradier_symbol or label, years, label)
    except Exception as e:
        log.warning(f"Daily {label}: all bar sources failed ({e})")
        return pd.DataFrame(columns=[close_col])

    if bars.empty or "close" not in bars.columns:
        log.warning(f"Daily {label}: no data from any source")
        return pd.DataFrame(columns=[close_col])

    df = bars[["close"]].copy()
    df.columns = [close_col]
    df.dropna(inplace=True)

    log.info(f"Daily {label}: {len(df)} rows")
    return df


def compute_hv_windows(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rolling historical volatility (annualized) from daily log returns.
    HV formula: std(log_returns, window) * sqrt(252)

    Operates on the first column of the DataFrame regardless of name, so it
    works for SPX (spx_close), QQQ (close), or any single-ticker daily series.
    """
    df = daily_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["hv5", "hv10", "hv20"])
    close_col = df.columns[0]
    df["log_ret"] = np.log(df[close_col] / df[close_col].shift(1))

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
    """Pull weekly closes for VIX9D and VIX3M — Cboe primary, yfinance fallback.

    Cboe's official CSVs carry VIX9D back to 2011 and VIX3M back to 2009
    (far deeper than yfinance's reliable coverage), which also unlocks longer
    training windows for the history-depth experiment.
    """
    end   = datetime.today()
    start = end - timedelta(days=years * 365)

    log.info("Fetching VIX9D and VIX3M for term structure (Cboe primary)...")

    def fetch_series(cboe_index: str, yf_symbol: str, name: str) -> pd.Series:
        # Primary: official Cboe daily history resampled to Monday weeks.
        try:
            from range_finder.cboe_data import fetch_cboe_weekly_closes
            s = fetch_cboe_weekly_closes(cboe_index, name)
            return s[s.index >= pd.Timestamp(start)]
        except Exception as e:
            log.warning(f"Cboe {cboe_index} unavailable ({e}) — "
                        f"falling back to yfinance {yf_symbol}")
        # Fallback: yfinance weekly bars.
        raw = yf.download(yf_symbol, start=start, end=end, interval="1wk", progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        if raw.empty:
            log.warning(f"{name} returned empty — will be NULL in features")
            return pd.Series(dtype=float, name=name)
        s = raw["Close"].copy()
        s.name = name
        s.index = pd.to_datetime(s.index).normalize()
        return s

    vix9d = fetch_series("VIX9D", "^VIX9D", "vix9d_close")
    vix3m = fetch_series("VIX3M", "^VIX3M", "vix3m_close")

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


def create_gex_table(conn) -> None:
    """Ensure the gex_inputs table exists.
    Now handled by db.init_all_tables() — kept for backwards compatibility."""
    pass  # Tables created in db.init_all_tables()


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

def _ny_now() -> pd.Timestamp:
    """Current NY wall-clock time. Module-level so tests can freeze it."""
    return pd.Timestamp.now(tz="America/New_York")


def _lag_one_week(frame: pd.DataFrame) -> pd.DataFrame:
    """Lag a weekly Monday-indexed frame by one week by shifting its INDEX
    forward 7 days: the value observed in week W becomes the feature joined
    at week W+1. On the gapless Monday indexes produced by the resample
    helpers this is value-for-value identical to ``.shift(1)`` row-shifting,
    but unlike a row-shift it also emits an index entry one week PAST the
    frame's last row — which is what lets the appended forecast row in
    ``build_features`` pick up HV / term-structure / macro features instead
    of landing on NaN."""
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        return frame.shift(1)
    return frame.shift(1, freq="7D")


def _load_weekly_for_ticker(conn, ticker: str) -> pd.DataFrame:
    """Return weekly OHLC + vol-proxy data for a ticker, normalised to the
    legacy SPX-style column names (``spx_open``, ``vix_close``, ``spx_return``,
    etc.) so the downstream feature pipeline can stay column-name-stable
    across SPX and the own-HAR tickers (QQQ / AMZN / AMD).

    SPX/XSP read ``weekly_spx`` (the existing table). Own-HAR tickers read
    ``weekly_underlying`` (per-ticker schema). A scaled mini (XSP→SPX)
    reads its parent's rows — returns/log-ranges are identical across the /10
    scale, so the parent's features serve the mini directly. The vol-proxy
    column is VIX for SPX/stocks and VXN for QQQ/NDX — see
    phase1.ticker_config.vol_proxy_yf.
    """
    from phase1.ticker_config import feature_source_ticker
    src = feature_source_ticker(ticker)
    if src == "SPX":
        return get_weekly_spx(conn)

    # Per-ticker path: read weekly_underlying for the source ticker (the parent
    # for a scaled mini, else the ticker itself) and rename to legacy column
    # names so the downstream pipeline stays unchanged.
    from range_finder.data_collector import get_weekly_underlying
    raw = get_weekly_underlying(conn, ticker=src)
    if raw.empty:
        log.warning(f"weekly_underlying empty for {src}")
        return raw
    return raw.rename(columns={
        "open":            "spx_open",
        "high":            "spx_high",
        "low":             "spx_low",
        "close":           "spx_close",
        "volume":          "spx_volume",
        "vol_proxy_open":  "vix_open",
        "vol_proxy_high":  "vix_high",
        "vol_proxy_low":   "vix_low",
        "vol_proxy_close": "vix_close",
        "return_pct":      "spx_return",
    })


def _load_daily_for_ticker(ticker: str, years: int = 6) -> pd.DataFrame:
    """Return daily closes for HV computation for one ticker.

    Returns a DataFrame with column ``spx_close`` regardless of ticker so
    ``compute_hv_windows`` works unchanged.
    """
    from phase1.ticker_config import get_config, feature_source_ticker
    src = feature_source_ticker(ticker)
    if src == "SPX":
        return fetch_daily_spx(years=years)
    cfg = get_config(src)
    return fetch_daily_underlying(cfg["yf_symbol"], years=years, label=src,
                                   close_col="spx_close")


def _load_earnings_flags(conn, ticker: str) -> pd.DataFrame:
    """Return earnings_flags rows for a ticker, indexed by week_start.

    Single-stock tickers (AMZN/AMD) have rows here; index/ETF tickers don't,
    so the join produces all-zero ``has_earnings`` columns.
    """
    try:
        df = pd.read_sql_query(
            "SELECT week_start, has_earnings FROM earnings_flags WHERE ticker = ?",
            conn,
            params=(ticker,),
            parse_dates=["week_start"],
        )
    except Exception:
        return pd.DataFrame(columns=["has_earnings"])
    if df.empty:
        return pd.DataFrame(columns=["has_earnings"])
    df.set_index("week_start", inplace=True)
    return df


def build_features(conn, exclude_covid: bool = True,
                    ticker: str = "SPX",
                    history_years: int = 6) -> pd.DataFrame:
    """
    Assemble the full model-ready feature matrix and save to model_features.
    Every feature is lagged so that at row t (target week), you only
    use information observable at Friday close of week t-1.

    For SPX (and XSP, which shares SPX features) this is the original
    SPX/VIX pipeline. For own-HAR tickers (QQQ / AMZN / AMD) the weekly OHLC
    and vol-proxy come from the per-ticker ``weekly_underlying`` table; the
    rest of the pipeline (HAR, HV, macro, events, GEX normalisation) is
    column-name-stable thanks to ``_load_weekly_for_ticker`` renaming on the way in.
    """
    log.info(f"Building feature matrix for {ticker}...")

    # --- Load base data (per-ticker for own-HAR; SPX/XSP share SPX rows) ---
    weekly  = _load_weekly_for_ticker(conn, ticker)
    macro   = get_macro_daily(conn)
    events  = get_event_flags(conn)
    earnings = _load_earnings_flags(conn, ticker)

    if weekly.empty:
        log.warning(f"weekly OHLC empty for {ticker} — skipping feature build")
        return pd.DataFrame()

    # --- Forecast-row scaffold for the upcoming week ---
    # Every feature is lagged one week, so the features of week W+1 are fully
    # known once week W's bar exists — but the matrix used to end at the last
    # week that HAD a bar, so a row for the upcoming week never existed and
    # Fri-Sun forecasts silently reused the current week's row (features
    # lagged one week further back than necessary, event flags from the
    # wrong week). Appending an all-NaN placeholder week lets the shift(1)
    # pipeline below populate W+1's features naturally; its target
    # (range_pct / log_range) stays NULL until that week trades, which keeps
    # it out of every fit (time_series_split drops NaN targets).
    weekly = weekly.sort_index()
    last_bar_week = weekly.index.max()
    forecast_week = last_bar_week + pd.Timedelta(days=7)
    weekly.loc[forecast_week] = np.nan

    # Is the latest bar a completed week or the in-progress one? yfinance
    # includes the current partial week (Monday-partial on Mondays), and its
    # (H-L)/open over a fraction of the week is NOT a valid weekly-range
    # target. Tracked here; the target gets nulled after assembly below.
    _last_bar_friday_close = (
        last_bar_week + pd.Timedelta(days=4, hours=16)
    ).tz_localize("America/New_York")
    last_bar_is_complete = _ny_now() >= _last_bar_friday_close

    # --- Partial-bar quarantine for lagged PATH statistics ---
    # The scaffold row's har_d1 is shift(1) of range_pct — i.e. the LAST bar's
    # range. On the Monday 9:31 cron that bar is minutes old: its high-low is
    # a running max-min that has had no time to accumulate, so its range_pct
    # (~0.3% after two minutes vs ~2% for a typical full week) is not "fresh
    # information", it is a structurally biased-low observation. Fed to the
    # scaffold it poisons har_d1 (100% of the value), har_w (1/5) and har_m
    # (1/20), and that too-narrow W+1 forecast is then served all week to the
    # Spread Finder. The same bias applies to spx_return (a partial
    # |return| is systematically smaller). LEVEL observations (vix_close =
    # the latest VIX print) are genuinely fresh and stay live — only path
    # statistics get proxied by the prior COMPLETED week's values.
    # When the last bar is complete, har_source IS weekly and the output is
    # byte-identical — historical builds and walk-forward runs are untouched.
    har_source = weekly
    forecast_path_source_week = last_bar_week if last_bar_is_complete else None
    if not last_bar_is_complete:
        _prior_weeks = weekly.index[weekly.index < last_bar_week]
        if len(_prior_weeks) > 0:
            har_source = weekly.copy()
            _prior_week = _prior_weeks.max()
            forecast_path_source_week = _prior_week
            for _col in ("range_pct", "log_range", "spx_return"):
                if _col in har_source.columns:
                    har_source.loc[last_bar_week, _col] = weekly.loc[_prior_week, _col]
            log.info(
                f"Partial-bar quarantine: {last_bar_week.date()} path stats "
                f"proxied by {_prior_week.date()} in the HAR lag source "
                f"(bar covers a fraction of the week)"
            )

    # --- Fetch supplemental data (depth follows history_years so the 10y
    # history experiment can build a full-depth matrix; production callers
    # leave the default 6) ---
    daily_underlying = _load_daily_for_ticker(ticker, years=history_years)
    # VIX9D/VIX3M term structure stays VIX-anchored across all tickers — it's
    # a macro-vol regime feature that single names also covary with. Cboe's
    # official CSVs carry both back past 2011, so any history_years the
    # experiment asks for is available (yfinance stays the fallback).
    vix_ts    = fetch_vix_term_structure(years=history_years)

    # --- HAR components (from the quarantined lag source) ---
    har = compute_har_features(har_source)

    # --- HV windows ---
    hv = compute_hv_windows(daily_underlying)

    # --- Macro weekly ---
    macro_wk = resample_macro_to_weekly(macro)

    # --- GEX (per-ticker — see save_gex_to_range_finder writes per ticker) ---
    gex_df = load_gex_inputs(conn, ticker=ticker)

    # --- Vol-proxy implied range ---
    # Vol proxy is annualized 1-SD vol in percent (VIX for SPX/stocks, VXN for
    # QQQ). De-annualize by sqrt(52) for the weekly 1-SD, then scale by the
    # Brownian high-low range factor E[H-L] = 2·sigma·sqrt(2/pi) ≈ 1.5958·sigma
    # (Feller 1951 / Parkinson 1980) so the column is directly comparable to
    # realized range_pct = (H-L)/open. Previously this was just 1-SD weekly vol
    # (biased ~37% low as a range predictor); the fix rescales it so UI
    # comparisons like model_vs_vix and the "trust the model" warning in
    # spread_levels.py stop firing spuriously on nearly every week. OLS is
    # scale-invariant so re-fitting the HAR model just rescales its coefficient
    # on this feature — predictive power is unchanged.
    _BM_RANGE_FACTOR = 2.0 * math.sqrt(2.0 / math.pi)
    weekly["vix_implied_range"] = (
        (weekly["vix_close"] / math.sqrt(52)) / 100 * _BM_RANGE_FACTOR
    )

    # --- Underlying return lags (column kept as spx_return_lag1 for SQL stability) ---
    # Shifted from har_source so the scaffold's lag-1 return is the prior
    # COMPLETED week's, never a partial-week return (see quarantine above).
    weekly["spx_return_lag1"] = har_source["spx_return"].shift(1)
    weekly["abs_return_lag1"] = weekly["spx_return_lag1"].abs()

    # --- Assemble ---
    df = weekly[[
        "range_pct", "log_range",
        "vix_close", "vix_implied_range",
        "spx_return_lag1", "abs_return_lag1",
    ]].copy()

    # Vol-proxy close needs to be lagged — we observe prior Friday's value
    df["vix_close"]         = df["vix_close"].shift(1)
    df["vix_implied_range"] = df["vix_implied_range"].shift(1)

    # Join HAR
    df = df.join(har, how="left")

    # Join HV (already on Monday index)
    hv_lagged = _lag_one_week(hv)
    df = df.join(hv_lagged, how="left")
    df["hv_ratio"] = df["hv5"] / df["hv20"]

    # --- High-vol regime detection ---
    # Binary flag: 1 if trailing 4-week average vol-proxy > 20
    df["high_vol_regime"] = (df["vix_close"].rolling(4, min_periods=2).mean() > 20).astype(int)

    # Join VIX term structure (lag 1 week)
    vix_ts_lagged = _lag_one_week(vix_ts)
    df = df.join(vix_ts_lagged, how="left")
    df["vix_wk_ratio"] = df["vix_close"] / df["vix3m_close"]

    # Join macro (lag 1 week)
    macro_lagged = _lag_one_week(macro_wk)
    df = df.join(macro_lagged, how="left")

    # Join GEX (Monday open — same week, no lag needed). Per-ticker filter
    # applied above means each ticker's GEX is normalised by its OWN open²
    # (cancelling the spot² scale embedded in the GEX formula at
    # gex_engine.py:174). Without this cancellation, time-series of
    # gex_normalized would track the level of the underlying rather than the
    # positioning signal — and would also be incomparable across tickers.
    if not gex_df.empty:
        df = df.join(gex_df[["gex"]], how="left")
        df["gex_flag"] = df["gex"].apply(_gex_flag)
        df["gex_normalized"] = df["gex"] / (weekly["spx_open"] ** 2) * 1e4
    else:
        df["gex"]            = np.nan
        df["gex_flag"]       = np.nan
        df["gex_normalized"] = np.nan

    # Join event flags
    df = df.join(
        events[["has_fomc", "has_cpi", "has_nfp", "has_opex", "event_count"]],
        how="left"
    )
    for col in ["has_fomc", "has_cpi", "has_nfp", "has_opex", "event_count"]:
        df[col] = df[col].fillna(0).astype(int)

    # Join single-stock earnings flag (only AMZN/AMD ever populates it)
    if not earnings.empty:
        df = df.join(earnings[["has_earnings"]], how="left")
    else:
        df["has_earnings"] = 0
    df["has_earnings"] = df["has_earnings"].fillna(0).astype(int)

    # Drop rows with insufficient lag history
    df.dropna(subset=["har_d1", "har_w", "har_m", "vix_close"], inplace=True)

    # Rows without a realized target are dropped EXCEPT the trailing
    # forecast row, whose target legitimately doesn't exist yet.
    df = df[df["log_range"].notna() | (df.index == forecast_week)]

    # Null the in-progress week's target: until its Friday session closes,
    # range_pct covers only part of the week and would otherwise feed the
    # fits (and OOS metrics) a systematically-too-small "weekly" range.
    # Its FEATURE columns are untouched — they're lagged from the prior,
    # completed week and remain valid. The next rebuild after Friday's
    # close upserts the real target over the NULL.
    if not last_bar_is_complete and last_bar_week in df.index:
        df.loc[last_bar_week, ["range_pct", "log_range"]] = np.nan

    # Record which weekly bar supplied each row's path statistics. Historical
    # and current-week rows use the immediately prior week. The next-week
    # scaffold is the exception while the latest bar is incomplete: its HAR
    # path is deliberately proxied from one week earlier, and that provenance
    # must survive persistence so Friday-Sunday consumers can reject it.
    df["path_source_week"] = df.index - pd.Timedelta(days=7)
    if forecast_week in df.index:
        df.loc[forecast_week, "path_source_week"] = forecast_path_source_week

    # Exclude COVID crash period if requested (default True)
    if exclude_covid:
        pre_filter = len(df)
        df = df[(df.index < COVID_START) | (df.index > COVID_END)]
        removed = pre_filter - len(df)
        if removed:
            log.info(f"exclude_covid: removed {removed} rows "
                     f"({COVID_START.date()} to {COVID_END.date()})")

    log.info(f"Feature matrix ({ticker}): {len(df)} rows x {len(df.columns)} columns")

    # --- Save to DB ---
    _save_features(conn, df, ticker=ticker)

    return df


def _gex_flag(gex_val) -> int | None:
    """
    Classify GEX into regime: +1 positive, 0 neutral, -1 negative.

    DEPRECATED: This binary flag is superseded by the continuous gex_normalized
    feature. Kept only for backward compatibility with existing DB rows.
    New code should use gex_normalized directly.
    """
    if gex_val is None or (isinstance(gex_val, float) and math.isnan(gex_val)):
        return None
    # Note: this threshold is approximate and not calibrated to Spot²-scaled GEX.
    # It exists only for legacy rows. The HAR model uses gex_normalized instead.
    if gex_val > 0:
        return 1
    elif gex_val < 0:
        return -1
    return 0


def _save_features(conn, df: pd.DataFrame, ticker: str = "SPX") -> None:
    """Upsert the feature matrix into model_features for a given ticker.

    `has_earnings` is read from the dataframe if present (for single-name
    stocks) and falls back to 0 otherwise — index/ETF tickers leave it 0.
    """
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.cursor()
    rows = 0

    for week_start, row in df.iterrows():
        cur.execute("""
            INSERT INTO model_features (
                week_start, ticker,
                log_range, range_pct,
                har_d1, har_w, har_m,
                vix_close, vix_implied_range,
                vix9d_close, vix3m_close, vix_ts_slope, vix_wk_ratio,
                hv5, hv10, hv20, hv_ratio,
                high_vol_regime,
                gex, gex_flag, gex_normalized,
                yield_spread, fed_funds,
                spx_return_lag1, abs_return_lag1,
                has_fomc, has_cpi, has_nfp, has_opex, event_count,
                has_earnings,
                path_source_week,
                updated_at
            ) VALUES (
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            ON CONFLICT(week_start, ticker) DO UPDATE SET
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
                gex_flag            = excluded.gex_flag,
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
                has_earnings        = excluded.has_earnings,
                path_source_week    = excluded.path_source_week,
                updated_at          = excluded.updated_at
        """, (
            week_start.strftime("%Y-%m-%d"), ticker,
            _f(row, "log_range"),        _f(row, "range_pct"),
            _f(row, "har_d1"),           _f(row, "har_w"),           _f(row, "har_m"),
            _f(row, "vix_close"),        _f(row, "vix_implied_range"),
            _f(row, "vix9d_close"),      _f(row, "vix3m_close"),
            _f(row, "vix_ts_slope"),     _f(row, "vix_wk_ratio"),
            _f(row, "hv5"),              _f(row, "hv10"),            _f(row, "hv20"),
            _f(row, "hv_ratio"),
            _i(row, "high_vol_regime"),
            _f(row, "gex"),              _i(row, "gex_flag"),        _f(row, "gex_normalized"),
            _f(row, "yield_spread"),     _f(row, "fed_funds"),
            _f(row, "spx_return_lag1"),  _f(row, "abs_return_lag1"),
            _i(row, "has_fomc"),         _i(row, "has_cpi"),
            _i(row, "has_nfp"),          _i(row, "has_opex"),
            _i(row, "event_count"),
            _i(row, "has_earnings"),
            _d(row, "path_source_week"),
            now,
        ))
        rows += 1

    conn.commit()
    log.info(f"model_features ({ticker}): {rows} rows upserted")


# =============================================================================
# READERS
# =============================================================================

def get_features(conn, min_date: str = None, exclude_covid: bool = False,
                 ticker: str = "SPX") -> pd.DataFrame:
    """Load the model_features table for a given ticker as a DataFrame.

    Defaults to SPX so existing call sites that haven't been updated yet keep
    seeing the SPX feature matrix.
    """
    query = "SELECT * FROM model_features WHERE ticker = ? ORDER BY week_start ASC"
    df = pd.read_sql_query(query, conn, params=(ticker,), parse_dates=["week_start"])
    df.set_index("week_start", inplace=True)

    if min_date:
        df = df[df.index >= pd.to_datetime(min_date)]

    if exclude_covid:
        df = df[(df.index < COVID_START) | (df.index > COVID_END)]

    # (The M6_regime interaction terms har_d1_x_regime / har_w_x_regime were
    # derived here when M6 existed. M6 was removed after walk-forward OOS
    # validation showed the interactions hurt OOS R²; no spec consumes them
    # now. high_vol_regime is still computed/stored above but is unused.)

    return df


def get_feature_for_week(conn, week_start: str, ticker: str = "SPX") -> pd.Series | None:
    """Fetch the feature row for a specific week + ticker."""
    df = pd.read_sql_query(
        "SELECT * FROM model_features WHERE week_start = ? AND ticker = ?",
        conn,
        params=(week_start, ticker),
        parse_dates=["week_start"],
    )
    if df.empty:
        log.warning(f"No feature row found for week_start={week_start} ticker={ticker}")
        return None
    df.set_index("week_start", inplace=True)
    # (M6_regime interaction terms were derived here when M6 existed; removed
    # along with the spec — see get_features() above.)
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


def _d(row: pd.Series, col: str) -> str | None:
    """Safe ISO-date extractor for persisted provenance fields."""
    val = row.get(col)
    if val is None or pd.isna(val):
        return None
    try:
        return pd.Timestamp(val).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None
