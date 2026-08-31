# =============================================================================
# bar_sources.py
# Underlying OHLC bar fetch with per-symbol source selection.
#
# Tradier /markets/history is the primary source (the app's authenticated
# broker API — no scraping, no silent Yahoo layout breakage, and verified in
# Phase-0 probes: 10y index history for SPX, split-adjusted single names,
# Monday-labeled weekly bars). yfinance remains the fallback for outages,
# missing tokens, or symbols Tradier can't serve.
#
# Output contract (both sources, both cadences): DataFrame indexed by a
# normalized tz-naive DatetimeIndex with float columns
#   open, high, low, close, volume
# sorted ascending. Callers (data_collector fetchers) do their own renaming
# to spx_* / prefixed schemas, so the save/upsert layers stay untouched.
# =============================================================================

import logging
from datetime import datetime, timedelta

import pandas as pd

log = logging.getLogger(__name__)

# Per-symbol primary source, keyed by TRADIER symbol. Seeded from the Phase-0
# probes (2026-07-01): SPX weekly history to 2016 ✓, VIX daily history ✓,
# AMZN split-adjusted ✓, QQQ current ✓ — everything the app trades goes
# Tradier-primary. Add {"SYM": "yfinance"} here to pin a symbol back to
# yfinance if a parity regression ever shows up.
SOURCE_MAP: dict[str, str] = {}
DEFAULT_SOURCE = "tradier"

# Daily trading gaps longer than this (calendar days) mean the series has a
# hole. Weekly cadence is validated against expected exchange-session weeks
# below; multiplying this value by seven was the bug that admitted ~98-day
# weekly holes.
_MAX_GAP_DAYS = 14

# Weekly bars must be Monday-labeled to match the weekly_spx convention.
_MIN_MONDAY_FRACTION = 0.95

_INTERVAL_TO_TRADIER = {"1wk": "weekly", "1d": "daily"}


def primary_source_for(tradier_symbol: str | None) -> str:
    """Resolve the primary bar source for a symbol ("tradier"/"yfinance")."""
    if not tradier_symbol:
        return "yfinance"
    return SOURCE_MAP.get(tradier_symbol.upper(), DEFAULT_SOURCE)


# =============================================================================
# FETCH — primary with validation, yfinance fallback
# =============================================================================

def fetch_weekly_bars(yf_symbol: str, tradier_symbol: str | None,
                      years: float, label: str) -> pd.DataFrame:
    """Weekly OHLCV bars, Monday-labeled. Tradier primary, yfinance fallback."""
    return _fetch_bars(yf_symbol, tradier_symbol, "1wk", years, label)


def fetch_daily_bars(yf_symbol: str, tradier_symbol: str | None,
                     years: float, label: str) -> pd.DataFrame:
    """Daily OHLCV bars. Tradier primary, yfinance fallback."""
    return _fetch_bars(yf_symbol, tradier_symbol, "1d", years, label)


def _request_window(years: float) -> tuple[datetime, datetime]:
    """Return the buffered history window requested from either source."""
    end = datetime.today()
    start = end - timedelta(days=int(years * 365 + 30))
    return start, end


def _fetch_bars(yf_symbol: str, tradier_symbol: str | None, interval: str,
                years: float, label: str) -> pd.DataFrame:
    start, end = _request_window(years)

    if primary_source_for(tradier_symbol) == "tradier":
        try:
            df = _fetch_tradier(tradier_symbol, interval, start, end)
            ok, reason = _validate_bars(df, interval, start=start, end=end)
            if ok:
                log.info(f"[{label}] {interval} bars via Tradier "
                         f"({tradier_symbol}): {len(df)} rows")
                return df
            log.warning(f"[{label}] Tradier {interval} bars rejected "
                        f"({reason}) — falling back to yfinance")
        except Exception as e:
            log.warning(f"[{label}] Tradier {interval} fetch failed: {e} — "
                        "falling back to yfinance")

    df = _fetch_yfinance(yf_symbol, interval, start, end)
    log.info(f"[{label}] {interval} bars via yfinance ({yf_symbol}): "
             f"{len(df)} rows")
    return df


def _fetch_tradier(symbol: str, interval: str,
                   start: datetime, end: datetime) -> pd.DataFrame:
    """Bars from Tradier /markets/history, shaped to the output contract."""
    from range_finder.data_collector import _tradier_token
    token = _tradier_token()
    if not token:
        raise RuntimeError("TRADIER_TOKEN not set")

    from phase1.data_client import TradierDataClient
    days = TradierDataClient(token).get_history(
        symbol,
        interval=_INTERVAL_TO_TRADIER[interval],
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if not days:
        raise RuntimeError(f"Tradier returned no {interval} history for {symbol}")

    df = pd.DataFrame(days)
    df.index = pd.DatetimeIndex(pd.to_datetime(df["date"])).normalize()
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df.sort_index()


def _fetch_yfinance(yf_symbol: str, interval: str,
                    start: datetime, end: datetime) -> pd.DataFrame:
    """Bars from yfinance, shaped to the output contract."""
    import yfinance as yf
    raw = yf.download(yf_symbol, start=start, end=end, interval=interval,
                      progress=False, timeout=60)
    if raw.empty:
        raise RuntimeError(f"yfinance returned empty {interval} data "
                           f"for {yf_symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.astype(float)   # yfinance Volume is int64 — match the Tradier path
    df.index = pd.DatetimeIndex(pd.to_datetime(df.index)).normalize()
    return df.sort_index()


# =============================================================================
# VALIDATION — a primary series must look sane before it replaces yfinance
# =============================================================================

def _expected_week_labels(start, end) -> pd.DatetimeIndex:
    """Monday labels for every cash-market week represented in [start, end]."""
    from phase1.config import CASH_CALENDAR
    from phase1.market_clock import get_schedule

    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize()
    schedule = get_schedule(
        CASH_CALENDAR,
        start_ts.date().isoformat(),
        end_ts.date().isoformat(),
    )
    if schedule.empty:
        return pd.DatetimeIndex([])
    sessions = pd.DatetimeIndex(pd.to_datetime(schedule.index)).tz_localize(None)
    labels = sessions.normalize() - pd.to_timedelta(sessions.weekday, unit="D")
    return pd.DatetimeIndex(labels.unique()).sort_values()


def _validate_weekly_cadence(
    df: pd.DataFrame,
    start=None,
    end=None,
) -> tuple[bool, str]:
    """Validate row cadence and requested coverage using exchange sessions."""
    actual = pd.DatetimeIndex(pd.to_datetime(df.index)).tz_localize(None).normalize()
    if actual.has_duplicates:
        return False, "duplicate weekly labels"

    requested_start = pd.Timestamp(start).normalize() if start is not None else actual.min()
    requested_end = pd.Timestamp(end).normalize() if end is not None else actual.max()
    expected = _expected_week_labels(requested_start, requested_end)
    if expected.empty:
        return True, ""

    actual_set = set(actual)
    missing = [label for label in expected if label not in actual_set]

    # The request may begin midweek, in which case the source is not required
    # to return the Monday-labeled bar whose week began before `start`.
    if expected[0] < requested_start and missing and missing[0] == expected[0]:
        missing = missing[1:]

    if not missing:
        return True, ""

    latest_expected = expected[-1]
    if latest_expected in missing and actual.max() < latest_expected:
        return False, (
            f"stale tail: latest weekly label is {actual.max():%Y-%m-%d}, "
            f"requires {latest_expected:%Y-%m-%d} for requested end"
        )

    if missing[0] < actual.min():
        return False, (
            f"requested start not covered: first weekly label is "
            f"{actual.min():%Y-%m-%d}; missing {missing[0]:%Y-%m-%d}"
        )

    labels = ", ".join(label.strftime("%Y-%m-%d") for label in missing[:3])
    suffix = "..." if len(missing) > 3 else ""
    return False, f"missing {len(missing)} expected weekly observation(s): {labels}{suffix}"


def _validate_bars(
    df: pd.DataFrame,
    interval: str,
    start=None,
    end=None,
) -> tuple[bool, str]:
    """Sanity-check a bar series. Returns (ok, reason-if-not)."""
    if df is None or df.empty:
        return False, "empty frame"

    for col in ("open", "high", "low", "close"):
        vals = df[col].dropna()
        if vals.empty or (vals <= 0).any():
            return False, f"non-positive or missing {col} values"

    if len(df) >= 2 and interval != "1wk":
        max_gap = df.index.to_series().diff().dropna().max()
        if max_gap > pd.Timedelta(days=_MAX_GAP_DAYS):
            return False, f"gap of {max_gap.days} days in series"

    if interval == "1wk":
        monday_frac = (df.index.weekday == 0).mean()
        if monday_frac < _MIN_MONDAY_FRACTION:
            return False, (f"only {monday_frac:.0%} of weekly bars are "
                           "Monday-labeled")
        cadence_ok, cadence_reason = _validate_weekly_cadence(
            df, start=start, end=end,
        )
        if not cadence_ok:
            return False, cadence_reason

    return True, ""


# =============================================================================
# CLI — Tradier-vs-yfinance parity report (the pre-cutover gate)
# =============================================================================
# Usage:
#   TRADIER_TOKEN=... python -m range_finder.bar_sources
#
# Prints per-symbol comparisons of overlapping bars: range_pct MAE (what the
# HAR models actually consume) and worst-case close deviation (flags
# adjustment mismatches). Read-only.

_PARITY_SYMBOLS = [
    # (label, yf_symbol, tradier_symbol)
    ("SPX",  "^GSPC", "SPX"),
    ("QQQ",  "QQQ",   "QQQ"),
    ("AMZN", "AMZN",  "AMZN"),
    ("AMD",  "AMD",   "AMD"),
    ("SPY",  "SPY",   "SPY"),
    ("NDX",  "^NDX",  "NDX"),
]


def _parity_row(label: str, yf_symbol: str, tradier_symbol: str,
                interval: str, years: float) -> dict:
    end = datetime.today()
    start = end - timedelta(days=int(years * 365))
    t = _fetch_tradier(tradier_symbol, interval, start, end)
    y = _fetch_yfinance(yf_symbol, interval, start, end)

    common = t.index.intersection(y.index)
    if interval == "1wk" and len(common):
        common = common[:-1]  # drop the in-progress week
    t, y = t.loc[common], y.loc[common]

    t_range = (t["high"] - t["low"]) / t["open"]
    y_range = (y["high"] - y["low"]) / y["open"]
    close_dev = ((t["close"] - y["close"]).abs() / y["close"])

    return {
        "symbol": label,
        "interval": interval,
        "overlap": len(common),
        "range_pct_mae": float((t_range - y_range).abs().mean()),
        "max_close_dev": float(close_dev.max()),
        "worst_close_date": str(close_dev.idxmax().date()) if len(close_dev) else "-",
    }


def main() -> None:
    import os
    import sys

    logging.basicConfig(level=logging.WARNING)

    from range_finder.data_collector import _tradier_token
    if not (_tradier_token() or os.environ.get("TRADIER_TOKEN", "").strip()):
        print("TRADIER_TOKEN not set — cannot run parity check.")
        sys.exit(1)

    rows = []
    for label, yf_sym, tr_sym in _PARITY_SYMBOLS:
        for interval, years in (("1wk", 2.0), ("1d", 1.0)):
            try:
                rows.append(_parity_row(label, yf_sym, tr_sym, interval, years))
            except Exception as e:
                rows.append({"symbol": label, "interval": interval,
                             "overlap": 0, "range_pct_mae": float("nan"),
                             "max_close_dev": float("nan"),
                             "worst_close_date": f"FAILED: {e}"})

    df = pd.DataFrame(rows)
    print("\nTRADIER vs YFINANCE PARITY (overlapping completed bars)")
    print("range_pct MAE = model-relevant difference; close dev flags "
          "split/dividend adjustment mismatches\n")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nCutover guidance: range_pct MAE < 0.001 and max close dev < 2% "
          "= clean. A symbol failing that belongs in SOURCE_MAP as "
          '"yfinance".')


if __name__ == "__main__":
    main()
