"""Bar source selection + validation (range_finder.bar_sources).

Pins the cutover safety contracts:

  * Tradier is primary; a failed/invalid Tradier series falls back to
    yfinance instead of erroring,
  * validation rejects empty, gapped, non-positive, and mislabeled series
    BEFORE they can displace yfinance data,
  * both sources produce the same output schema (lowercase ohlcv, normalized
    ascending index) so data_collector's renaming works either way,
  * SOURCE_MAP pins a symbol back to yfinance without touching fetch code.

No live API calls — TradierDataClient.get_history and yf.download are
mocked, mirroring phase1/tests/test_data_client.py conventions.
"""
from datetime import datetime

import pandas as pd
import pytest

import range_finder.bar_sources as bs


@pytest.fixture(autouse=True)
def fixed_request_window(monkeypatch):
    """Keep mocked fetch fixtures inside the request they claim to satisfy."""
    monkeypatch.setattr(
        bs, "_request_window",
        lambda years: (datetime(2026, 6, 15), datetime(2026, 7, 5)),
    )


# ── fixtures / helpers ─────────────────────────────────────────────────────────

def _tradier_days(dates_closes: dict) -> list[dict]:
    """Tradier /markets/history 'day' dicts."""
    return [
        {"date": d, "open": c - 1.0, "high": c + 1.0, "low": c - 2.0,
         "close": c, "volume": 1000}
        for d, c in dates_closes.items()
    ]


_WEEK_MONDAYS = {"2026-06-15": 100.0, "2026-06-22": 102.0, "2026-06-29": 104.0}
_DAILY_DATES = {"2026-06-29": 100.0, "2026-06-30": 101.0, "2026-07-01": 102.0}


class _FakeClient:
    def __init__(self, days):
        self._days = days

    def get_history(self, symbol, interval="daily", start=None, end=None):
        if isinstance(self._days, Exception):
            raise self._days
        return self._days


def _patch_tradier(monkeypatch, days) -> None:
    monkeypatch.setattr(bs, "_tradier_token_or_empty", lambda: "tok",
                        raising=False)
    import range_finder.data_collector as dc
    monkeypatch.setattr(dc, "_tradier_token", lambda: "tok")
    import phase1.data_client as pdc
    monkeypatch.setattr(pdc, "TradierDataClient", lambda tok: _FakeClient(days))


def _patch_yfinance(monkeypatch, dates_closes: dict) -> None:
    import yfinance as yf

    def _download(symbol, start=None, end=None, interval="1d",
                  progress=False, timeout=60):
        if not dates_closes:
            return pd.DataFrame()
        idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates_closes])
        closes = list(dates_closes.values())
        return pd.DataFrame({
            "Open": [c - 1.0 for c in closes], "High": [c + 1.0 for c in closes],
            "Low": [c - 2.0 for c in closes], "Close": closes,
            "Volume": [1000] * len(closes),
        }, index=idx)

    monkeypatch.setattr(yf, "download", _download)


# ── source selection ───────────────────────────────────────────────────────────

def test_tradier_is_primary(monkeypatch):
    _patch_tradier(monkeypatch, _tradier_days(_WEEK_MONDAYS))
    _patch_yfinance(monkeypatch, {"2026-06-15": 999.0})

    df = bs.fetch_weekly_bars("^GSPC", "SPX", 1.0, "SPX")

    assert df["close"].iloc[0] == pytest.approx(100.0)  # Tradier, not yfinance
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index[0] == pd.Timestamp("2026-06-15")


def test_tradier_error_falls_back(monkeypatch):
    _patch_tradier(monkeypatch, ConnectionError("api down"))
    _patch_yfinance(monkeypatch, _WEEK_MONDAYS)

    df = bs.fetch_weekly_bars("^GSPC", "SPX", 1.0, "SPX")

    assert len(df) == 3  # yfinance served the bars


def test_empty_tradier_falls_back(monkeypatch):
    _patch_tradier(monkeypatch, [])
    _patch_yfinance(monkeypatch, _DAILY_DATES)

    df = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")

    assert len(df) == 3


def test_no_token_goes_straight_to_yfinance(monkeypatch):
    import range_finder.data_collector as dc
    monkeypatch.setattr(dc, "_tradier_token", lambda: "")
    _patch_yfinance(monkeypatch, _DAILY_DATES)

    df = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")

    assert len(df) == 3


def test_source_map_pins_symbol_to_yfinance(monkeypatch):
    _patch_tradier(monkeypatch, _tradier_days(_WEEK_MONDAYS))
    _patch_yfinance(monkeypatch, {"2026-06-15": 999.0})
    monkeypatch.setitem(bs.SOURCE_MAP, "SPX", "yfinance")

    df = bs.fetch_weekly_bars("^GSPC", "SPX", 1.0, "SPX")

    assert df["close"].iloc[0] == pytest.approx(999.0)  # pinned to yfinance


def test_no_tradier_symbol_uses_yfinance(monkeypatch):
    _patch_yfinance(monkeypatch, _DAILY_DATES)

    df = bs.fetch_daily_bars("^WEIRD", None, 1.0, "WEIRD")

    assert len(df) == 3


# ── validation rejects bad primary series ──────────────────────────────────────

def test_gapped_series_falls_back(monkeypatch):
    gapped = _tradier_days({"2026-01-05": 100.0, "2026-03-02": 105.0})  # 8-week hole
    _patch_tradier(monkeypatch, gapped)
    _patch_yfinance(monkeypatch, _DAILY_DATES)

    df = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")

    assert df["close"].iloc[0] == pytest.approx(100.0)
    assert len(df) == 3  # yfinance won


def test_nonpositive_prices_fall_back(monkeypatch):
    bad = _tradier_days(_DAILY_DATES)
    bad[1]["close"] = 0.0
    _patch_tradier(monkeypatch, bad)
    _patch_yfinance(monkeypatch, {"2026-06-29": 55.0})

    df = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")

    assert df["close"].iloc[0] == pytest.approx(55.0)


def test_non_monday_weekly_labels_fall_back(monkeypatch):
    fridays = _tradier_days({"2026-06-19": 100.0, "2026-06-26": 102.0})
    _patch_tradier(monkeypatch, fridays)
    _patch_yfinance(monkeypatch, _WEEK_MONDAYS)

    df = bs.fetch_weekly_bars("^GSPC", "SPX", 1.0, "SPX")

    assert df.index[0] == pd.Timestamp("2026-06-15")  # yfinance's Mondays


def test_missing_expected_week_triggers_existing_fallback(monkeypatch):
    gapped = _tradier_days({"2026-06-15": 100.0, "2026-06-29": 104.0})
    _patch_tradier(monkeypatch, gapped)
    _patch_yfinance(monkeypatch, _WEEK_MONDAYS)

    df = bs.fetch_weekly_bars("^GSPC", "SPX", 1.0, "SPX")

    assert list(df.index) == list(pd.DatetimeIndex(_WEEK_MONDAYS))
    assert list(df["close"]) == list(_WEEK_MONDAYS.values())


def test_holiday_monday_weeks_still_pass():
    # A weekly series labeled entirely on Mondays validates even when a
    # session gap (holiday week) exists inside the window.
    idx = pd.DatetimeIndex(["2026-06-15", "2026-06-22", "2026-06-29"])
    df = pd.DataFrame({
        "open": [99.0, 101.0, 103.0], "high": [101.0, 103.0, 105.0],
        "low": [98.0, 100.0, 102.0], "close": [100.0, 102.0, 104.0],
        "volume": [0.0, 0.0, 0.0],   # index volume can legitimately be 0
    }, index=idx)

    ok, reason = bs._validate_bars(df, "1wk")

    assert ok, reason


def _weekly_frame(labels) -> pd.DataFrame:
    n = len(labels)
    return pd.DataFrame({
        "open": [100.0] * n,
        "high": [102.0] * n,
        "low": [99.0] * n,
        "close": [101.0] * n,
        "volume": [0.0] * n,
    }, index=pd.DatetimeIndex(labels))


def test_weekly_validation_accepts_normal_and_holiday_weeks():
    # MLK Day is Monday 2026-01-19; the weekly bar remains Monday-labeled
    # even though its first cash session is Tuesday.
    df = _weekly_frame(pd.date_range("2026-01-05", "2026-02-02", freq="W-MON"))

    ok, reason = bs._validate_bars(
        df, "1wk", start="2026-01-01", end="2026-02-08",
    )

    assert ok, reason


def test_weekly_validation_rejects_one_missing_interior_week():
    df = _weekly_frame(["2026-01-05", "2026-01-12", "2026-01-26"])

    ok, reason = bs._validate_bars(
        df, "1wk", start="2026-01-05", end="2026-02-01",
    )

    assert not ok
    assert "missing" in reason and "2026-01-19" in reason


@pytest.mark.parametrize(
    "labels",
    [
        ["2026-01-05", "2026-01-12", "2026-02-02"],
        ["2026-01-05", "2026-04-13"],
    ],
)
def test_weekly_validation_rejects_multiweek_and_thirteen_week_holes(labels):
    ok, reason = bs._validate_bars(
        _weekly_frame(labels), "1wk", start=labels[0], end=labels[-1],
    )

    assert not ok
    assert "missing" in reason


def test_weekly_validation_rejects_stale_requested_tail():
    df = _weekly_frame(pd.date_range("2026-01-05", "2026-01-19", freq="W-MON"))

    ok, reason = bs._validate_bars(
        df, "1wk", start="2026-01-01", end="2026-02-08",
    )

    assert not ok
    assert "stale tail" in reason


def test_weekly_validation_allows_only_the_partial_start_boundary():
    # Request starts Thursday Jan 1, so the Dec-29-labeled partial boundary
    # week may be absent. Starting another week later is insufficient.
    boundary_ok = _weekly_frame(
        pd.date_range("2026-01-05", "2026-02-02", freq="W-MON"),
    )
    late = _weekly_frame(
        pd.date_range("2026-01-12", "2026-02-02", freq="W-MON"),
    )

    ok, reason = bs._validate_bars(
        boundary_ok, "1wk", start="2026-01-01", end="2026-02-08",
    )
    late_ok, late_reason = bs._validate_bars(
        late, "1wk", start="2026-01-01", end="2026-02-08",
    )

    assert ok, reason
    assert not late_ok
    assert "requested start" in late_reason


# ── output schema parity between sources ───────────────────────────────────────

def test_both_sources_same_schema(monkeypatch):
    _patch_tradier(monkeypatch, _tradier_days(_DAILY_DATES))
    _patch_yfinance(monkeypatch, _DAILY_DATES)

    t = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")
    monkeypatch.setitem(bs.SOURCE_MAP, "SPX", "yfinance")
    y = bs.fetch_daily_bars("^GSPC", "SPX", 1.0, "SPX")

    assert list(t.columns) == list(y.columns)
    assert t.index.equals(y.index)
    assert (t.dtypes == y.dtypes).all()
