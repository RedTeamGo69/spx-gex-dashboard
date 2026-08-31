"""Forecast-row scaffold in build_features.

The HAR pipeline lags every feature one week, so week W+1's features are
fully known the moment week W's bar exists — but the matrix used to end at
the last week that HAD a bar, forcing Fri-Sun forecasts onto the current
week's row (features one week staler than necessary). These tests pin the
new behavior:

  * an extra row appears at last_bar_week + 7d with features lagged from
    the latest bar and a NULL target,
  * the in-progress week's PARTIAL target is nulled so fits never train on
    a half-week "weekly range",
  * mid-week, the scaffold's PATH-statistic lags (har_d1, spx_return_lag1)
    are proxied by the prior COMPLETED week — a partial bar's high-low is a
    running max-min with structural downward bias, not fresh information —
    while LEVEL lags (vix_close) stay live from the latest print,
  * historical rows are value-for-value unchanged by the freq-based lag
    refactor,
  * fits exclude all NULL-target rows automatically.
"""
import math

import numpy as np
import pandas as pd
import pytest

import range_finder.feature_builder as fb
from range_finder.har_model import time_series_split, fit_model

# Fixed, deterministic calendar: 30 Mondays ending 2026-06-08 (a Monday).
LAST_MONDAY = pd.Timestamp("2026-06-08")
FORECAST_MONDAY = pd.Timestamp("2026-06-15")
WEEK_IDX = pd.date_range(end=LAST_MONDAY, periods=30, freq="W-MON")


def _make_weekly() -> pd.DataFrame:
    n = len(WEEK_IDX)
    open_ = 7000.0 + np.arange(n) * 10.0
    range_pct = 0.020 + 0.001 * (np.arange(n) % 7)
    high = open_ * (1 + range_pct / 2)
    low = open_ * (1 - range_pct / 2)
    close = open_ * (1 + 0.001 * ((np.arange(n) % 5) - 2))
    vix = 15.0 + (np.arange(n) % 8)
    df = pd.DataFrame(
        {
            "week_end": WEEK_IDX + pd.Timedelta(days=4),
            "spx_open": open_, "spx_high": high, "spx_low": low,
            "spx_close": close, "spx_volume": 1e9,
            "vix_open": vix, "vix_high": vix + 1, "vix_low": vix - 1,
            "vix_close": vix,
            "range_pts": high - low,
            "range_pct": range_pct,
            "log_range": np.log(range_pct),
            "spx_return": (close - open_) / open_,
        },
        index=WEEK_IDX,
    )
    df.index.name = "week_start"
    return df


def _make_daily() -> pd.DataFrame:
    days = pd.bdate_range(WEEK_IDX[0] - pd.Timedelta(days=120),
                          LAST_MONDAY + pd.Timedelta(days=4))
    rng = np.random.RandomState(7)
    close = 7000.0 * np.exp(np.cumsum(rng.normal(0, 0.008, len(days))))
    return pd.DataFrame({"spx_close": close}, index=days)


def _make_vix_ts() -> pd.DataFrame:
    n = len(WEEK_IDX)
    df = pd.DataFrame(
        {
            "vix9d_close": 14.0 + (np.arange(n) % 6),
            "vix3m_close": 18.0 + (np.arange(n) % 4),
        },
        index=WEEK_IDX,
    )
    df["vix_ts_slope"] = df["vix3m_close"] - df["vix9d_close"]
    df.index.name = "week_start"
    return df


def _make_macro() -> pd.DataFrame:
    days = pd.date_range(WEEK_IDX[0] - pd.Timedelta(days=30),
                         LAST_MONDAY + pd.Timedelta(days=4), freq="D")
    return pd.DataFrame(
        {"yield_spread": 0.5, "fed_funds": 4.25}, index=days,
    )


def _make_events() -> pd.DataFrame:
    # FOMC lands in the FORECAST week — the row that doesn't exist until
    # this scaffold builds it. Verifies forward event flags reach it.
    df = pd.DataFrame(
        {"has_fomc": 0, "has_cpi": 0, "has_nfp": 0, "has_opex": 0,
         "event_count": 0},
        index=list(WEEK_IDX) + [FORECAST_MONDAY],
    )
    df.loc[FORECAST_MONDAY, ["has_fomc", "event_count"]] = 1
    df.index.name = "week_start"
    return df


@pytest.fixture
def patched_builder(monkeypatch):
    weekly = _make_weekly()
    saved = {}
    monkeypatch.setattr(fb, "_load_weekly_for_ticker", lambda conn, ticker: weekly.copy())
    monkeypatch.setattr(fb, "_load_daily_for_ticker", lambda ticker, years=6: _make_daily())
    monkeypatch.setattr(fb, "fetch_vix_term_structure", lambda years=6: _make_vix_ts())
    monkeypatch.setattr(fb, "get_macro_daily", lambda conn: _make_macro())
    monkeypatch.setattr(fb, "get_event_flags", lambda conn: _make_events())
    monkeypatch.setattr(fb, "load_gex_inputs", lambda conn, ticker="SPX": pd.DataFrame(columns=["gex"]))
    monkeypatch.setattr(fb, "_load_earnings_flags", lambda conn, ticker: pd.DataFrame(columns=["has_earnings"]))
    monkeypatch.setattr(fb, "_save_features", lambda conn, df, ticker="SPX": saved.update({"df": df}))
    return weekly, saved


def _freeze_clock(monkeypatch, ts: str):
    frozen = pd.Timestamp(ts, tz="America/New_York")
    monkeypatch.setattr(fb, "_ny_now", lambda: frozen)


def test_forecast_row_appended_after_completed_week(patched_builder, monkeypatch):
    weekly, saved = patched_builder
    # Saturday after the 2026-06-08 week closed → bar is complete.
    _freeze_clock(monkeypatch, "2026-06-13 10:00")

    df = fb.build_features(conn=None, ticker="SPX")

    assert FORECAST_MONDAY in df.index
    row = df.loc[FORECAST_MONDAY]

    # Target unknown — must be NULL so fits skip it.
    assert pd.isna(row["log_range"]) and pd.isna(row["range_pct"])

    # Features lagged from the just-completed week.
    assert row["har_d1"] == pytest.approx(weekly.loc[LAST_MONDAY, "range_pct"])
    assert row["har_w"] == pytest.approx(weekly["range_pct"].iloc[-5:].mean())
    assert row["vix_close"] == pytest.approx(weekly.loc[LAST_MONDAY, "vix_close"])
    _bm = 2.0 * math.sqrt(2.0 / math.pi)
    assert row["vix_implied_range"] == pytest.approx(
        weekly.loc[LAST_MONDAY, "vix_close"] / math.sqrt(52) / 100 * _bm
    )
    assert row["spx_return_lag1"] == pytest.approx(weekly.loc[LAST_MONDAY, "spx_return"])

    # Joined (externally-indexed) features must reach the forecast row too.
    assert not pd.isna(row["hv5"])
    assert row["vix3m_close"] == pytest.approx(_make_vix_ts().loc[LAST_MONDAY, "vix3m_close"])
    assert not pd.isna(row["yield_spread"])

    # Forward calendar flags apply to the week being forecast.
    assert row["has_fomc"] == 1

    # Completed week keeps its realized target.
    assert not pd.isna(df.loc[LAST_MONDAY, "log_range"])

    # Persisted frame matches the returned one.
    assert FORECAST_MONDAY in saved["df"].index


def test_historical_rows_unchanged_by_freq_lag(patched_builder, monkeypatch):
    weekly, _ = patched_builder
    _freeze_clock(monkeypatch, "2026-06-13 10:00")
    df = fb.build_features(conn=None, ticker="SPX")

    vix_ts = _make_vix_ts()
    for w in WEEK_IDX[25:29]:
        assert df.loc[w, "har_d1"] == pytest.approx(
            weekly.loc[w - pd.Timedelta(days=7), "range_pct"])
        assert df.loc[w, "vix3m_close"] == pytest.approx(
            vix_ts.loc[w - pd.Timedelta(days=7), "vix3m_close"])
        assert df.loc[w, "vix_close"] == pytest.approx(
            weekly.loc[w - pd.Timedelta(days=7), "vix_close"])


def test_partial_week_target_is_nulled_midweek(patched_builder, monkeypatch):
    weekly, _ = patched_builder
    # Wednesday inside the 2026-06-08 week → its bar is still in progress.
    _freeze_clock(monkeypatch, "2026-06-10 12:00")

    df = fb.build_features(conn=None, ticker="SPX")

    # Partial target must not survive into the fit data...
    assert pd.isna(df.loc[LAST_MONDAY, "log_range"])
    # ...but the row's FEATURES (lagged from the prior completed week) stay.
    assert df.loc[LAST_MONDAY, "har_d1"] == pytest.approx(
        weekly.loc[LAST_MONDAY - pd.Timedelta(days=7), "range_pct"])

    # Forecast row still appends — but its PATH-statistic lags must come from
    # the prior COMPLETED week, never the in-progress bar: a partial week's
    # high-low is a running max-min (a Monday bar shows ~0.3% "weekly range"
    # after minutes of trading vs ~2% typical), so lagging it into har_d1
    # would bias the served W+1 forecast far too narrow for the whole week.
    assert FORECAST_MONDAY in df.index
    prior_week = LAST_MONDAY - pd.Timedelta(days=7)
    assert df.loc[FORECAST_MONDAY, "har_d1"] == pytest.approx(
        weekly.loc[prior_week, "range_pct"])
    assert df.loc[FORECAST_MONDAY, "spx_return_lag1"] == pytest.approx(
        weekly.loc[prior_week, "spx_return"])

    # har_w = mean of lags 1..5 with the partial week proxied by prior week:
    # [r(W-1), r(W-1), r(W-2), r(W-3), r(W-4)].
    r = weekly["range_pct"]
    expected_har_w = np.mean([
        r.loc[prior_week], r.loc[prior_week],
        r.loc[LAST_MONDAY - pd.Timedelta(days=14)],
        r.loc[LAST_MONDAY - pd.Timedelta(days=21)],
        r.loc[LAST_MONDAY - pd.Timedelta(days=28)],
    ])
    assert df.loc[FORECAST_MONDAY, "har_w"] == pytest.approx(expected_har_w)

    # LEVEL lags stay live: the partial week's vix_close is the latest VIX
    # print — genuinely fresh information, unlike a partial range.
    assert df.loc[FORECAST_MONDAY, "vix_close"] == pytest.approx(
        weekly.loc[LAST_MONDAY, "vix_close"])


def test_next_week_path_provenance_rolls_only_after_week_completion(
        patched_builder, monkeypatch):
    """A Monday scaffold must not masquerade as fresh after Friday rolls.

    The separate Monday-open strike anchor is represented by the immutable
    sentinel below: rebuilding path features after the weekly bar completes
    must not alter it.
    """
    weekly, _ = patched_builder
    monday_open_anchor = weekly.loc[LAST_MONDAY, "spx_open"]

    _freeze_clock(monkeypatch, "2026-06-08 09:31")
    monday_build = fb.build_features(conn=None, ticker="SPX")
    provisional = monday_build.loc[FORECAST_MONDAY]

    fresh, source_week, required_week = fb.assess_path_provenance(
        provisional, FORECAST_MONDAY,
    )
    assert fresh is False
    assert source_week == LAST_MONDAY - pd.Timedelta(days=7)
    assert required_week == LAST_MONDAY
    assert monday_open_anchor == weekly.loc[LAST_MONDAY, "spx_open"]

    _freeze_clock(monkeypatch, "2026-06-13 10:00")
    weekend_rebuild = fb.build_features(conn=None, ticker="SPX")
    refreshed = weekend_rebuild.loc[FORECAST_MONDAY]

    fresh, source_week, required_week = fb.assess_path_provenance(
        refreshed, FORECAST_MONDAY,
    )
    assert fresh is True
    assert source_week == required_week == LAST_MONDAY
    assert refreshed["har_d1"] == pytest.approx(
        weekly.loc[LAST_MONDAY, "range_pct"])
    assert monday_open_anchor == weekly.loc[LAST_MONDAY, "spx_open"]


def test_fit_excludes_null_target_rows(patched_builder, monkeypatch):
    _, _ = patched_builder
    _freeze_clock(monkeypatch, "2026-06-10 12:00")
    df = fb.build_features(conn=None, ticker="SPX")

    feature_cols = ["har_d1", "har_w", "har_m"]
    X_train, X_test, y_train, y_test = time_series_split(df, feature_cols=feature_cols)

    fit_idx = X_train.index.union(X_test.index)
    assert FORECAST_MONDAY not in fit_idx
    assert LAST_MONDAY not in fit_idx          # partial target excluded too
    assert not y_train.isna().any() and not y_test.isna().any()

    # And the fit itself runs clean on what's left.
    result = fit_model(X_train, y_train, model_name="test")
    assert np.isfinite(result.params).all()
