"""Bounded read-only Tradier/Cboe/FRED adapters. No broker order endpoints.

The study deliberately fails closed when a primary source is unavailable.
It does not change the interactive app's existing fallback behavior.
"""
from datetime import date, datetime, timedelta
from functools import cached_property
import math

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from phase1.data_client import TradierDataClient
from range_finder.cboe_data import fetch_cboe_index_history, resample_cboe_weekly
from range_finder.event_calendars import event_flag_rows, FOMC_DATES, CPI_DATES, NFP_DATES
from range_finder.feature_builder import build_features
from range_finder.trading_week import NY, UTC, listed_week_expiration, trading_week
from .config import UNIVERSE


def finite_price(value):
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def valid_ohlc(bar):
    if not isinstance(bar, dict) or not all(finite_price(bar.get(k)) for k in ("open", "high", "low", "close")):
        return False
    return float(bar["low"]) <= min(float(bar["open"]), float(bar["close"])) <= max(float(bar["open"]), float(bar["close"])) <= float(bar["high"])


def epoch_time(value):
    v = float(value)
    return datetime.fromtimestamp(v / 1000 if v > 1e11 else v, UTC)


def frame_records(frame):
    # Standard JSON float representation round-trips IEEE doubles. pandas'
    # to_json caps precision at 15 digits and can lose original fit inputs.
    def scalar(value):
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (date, datetime, pd.Timestamp)):
            return value.isoformat()
        return value.item() if isinstance(value, np.generic) else value
    return [{"date": scalar(idx), **{str(k): scalar(v) for k, v in row.items()}}
            for idx, row in frame.iterrows()]


class TradierProvider:
    def __init__(self, token, *, clock, declared_delay_seconds=0, history_loader=None):
        if not token:
            raise ValueError("TRADIER_TOKEN is required")
        if declared_delay_seconds not in (0, 900):
            raise ValueError("Data delay must be 0 or 900 seconds")
        self.client = TradierDataClient(token)
        self.clock = clock
        self.delay = declared_delay_seconds
        self.history_loader = history_loader
        self.http = requests.Session()
        self.http.headers.update(self.client.tradier_headers())
        self.http.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",), respect_retry_after_header=False)))
        self.source_times = {}

    def close(self):
        self.http.close()

    def _get(self, endpoint, **params):
        response = self.http.get(f"{self.client.base_url}/markets/{endpoint}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()

    def history(self, ticker, start, end, interval="daily"):
        block = self._get("history", symbol=ticker, interval=interval,
                          start=str(start), end=str(end)).get("history") or {}
        rows = block.get("day") or []
        return [rows] if isinstance(rows, dict) else rows

    @cached_property
    def quotes(self):
        block = self._get("quotes", symbols=",".join((*UNIVERSE, "VIX"))).get("quotes") or {}
        rows = block.get("quote") or []
        if isinstance(rows, dict):
            rows = [rows]
        return {r["symbol"]: r for r in rows}

    def checked_quote(self, ticker, session_day):
        quote = self.quotes.get(ticker)
        if not quote or quote.get("symbol") != ticker:
            raise ValueError(f"Missing quote for {ticker}")
        timestamp = epoch_time(quote.get("trade_date", 0))
        age = (self.clock() - timestamp).total_seconds()
        if timestamp.astimezone(NY).date() != session_day or not 0 <= age <= self.delay + 120:
            raise ValueError(f"Stale {ticker} quote; age={age:.0f}s")
        if not finite_price(quote.get("open")) or not finite_price(quote.get("last")):
            raise ValueError(f"Invalid {ticker} open/last")
        return quote, timestamp

    @cached_property
    def common(self):
        from range_finder.data_collector import fetch_fred_macro
        now = self.clock()
        week = trading_week(now.astimezone(NY).date())
        cutoff = pd.Timestamp(week.monday)
        previous = trading_week(week.monday - timedelta(days=7)).sessions[-1].day
        series = {}
        errors = {}
        for index in ("VIX", "VIX9D", "VIX3M"):
            try:
                daily = fetch_cboe_index_history(index)
                daily = daily[daily.index < cutoff]
                if daily.empty or daily.index.max().date() != previous:
                    raise ValueError("Prior final-session volatility observation missing")
                series[index] = resample_cboe_weekly(daily)
                self.source_times[index] = {"source": "Cboe daily_prices", "retrieved_at": self.clock().isoformat(),
                                            "last_session": str(daily.index.max().date())}
            except Exception as exc:
                errors[index] = type(exc).__name__ + ": history source unavailable or stale; prior final-session value required"
        if "VIX" not in series:
            raise ValueError("VIX history unavailable: " + errors["VIX"])
        try:
            macro = fetch_fred_macro(years=6)
            macro = macro[macro.index < cutoff]
            if macro.empty or macro.index.max().date() < previous - timedelta(days=7):
                raise ValueError("Macro observations are more than one week stale")
            self.source_times["macro"] = {"source": "FRED", "retrieved_at": self.clock().isoformat(),
                                           "last_session": str(macro.index.max().date())}
        except Exception as exc:
            macro = pd.DataFrame(columns=["yield_spread", "fed_funds"], index=pd.DatetimeIndex([]))
            errors["macro"] = type(exc).__name__ + ": FRED history unavailable"
        ts = pd.DataFrame({name: series.get(index, pd.DataFrame(columns=["close"]))["close"]
                           for index, name in (("VIX9D", "vix9d_close"), ("VIX3M", "vix3m_close"))})
        ts.index = pd.DatetimeIndex(ts.index)
        ts["vix_ts_slope"] = ts["vix3m_close"] - ts["vix9d_close"]
        return series["VIX"], ts, macro, errors

    def prepare(self, ticker, week):
        if ticker not in UNIVERSE:
            raise ValueError("Ticker outside study universe")
        now = self.clock()
        if not week.admits(now):
            raise ValueError("Outside prospective capture window")
        for name, dates in (("FOMC", FOMC_DATES), ("CPI", CPI_DATES), ("NFP", NFP_DATES)):
            if not dates or max(dates) < str(week.sessions[-1].day):
                raise ValueError(f"{name} event calendar expired; extend the existing calendar")
        start = week.monday - timedelta(days=int(6 * 365.25) + 30)
        previous = trading_week(week.monday - timedelta(days=7)).sessions[-1].day
        weekly_rows = self.history(ticker, start, previous, "weekly")
        daily_rows = self.history(ticker, start, previous)
        if not weekly_rows or not daily_rows:
            raise ValueError(f"Missing {ticker} training history")
        for rows in (weekly_rows, daily_rows):
            if any(not valid_ohlc(r) for r in rows):
                raise ValueError(f"Invalid {ticker} history OHLC")
        def frame(rows):
            f = pd.DataFrame(rows).set_index("date")
            f.index = pd.to_datetime(f.index)
            return f.sort_index().astype(float)
        weekly, daily = frame(weekly_rows), frame(daily_rows)
        if weekly.index.has_duplicates or daily.index.has_duplicates:
            raise ValueError("Duplicate historical session/week labels")
        weekly = weekly[weekly.index < pd.Timestamp(week.monday)]
        daily = daily[daily.index <= pd.Timestamp(previous)]
        from range_finder.bar_sources import _validate_bars
        ok, reason = _validate_bars(weekly, "1wk", start=pd.Timestamp(start), end=pd.Timestamp(previous))
        if not ok:
            raise ValueError(f"Invalid weekly cadence: {reason}")
        ok, reason = _validate_bars(daily, "1d", start=pd.Timestamp(start), end=pd.Timestamp(previous))
        if not ok:
            raise ValueError(f"Invalid daily history: {reason}")
        if daily.empty or daily.index.max().date() != previous or weekly.index.max().date() != week.monday - timedelta(days=7):
            raise ValueError("Stale history: prior completed trading week missing")
        # A single missing weekday silently changes the 5/20-session HV
        # windows. Validate the daily calendar, not just the largest gap.
        import pandas_market_calendars as mcal
        expected = mcal.get_calendar("NYSE").valid_days(start_date=daily.index.min().date(), end_date=previous)
        missing = expected.tz_localize(None).difference(daily.index)
        if len(missing):
            raise ValueError(f"Daily history missing {len(missing)} exchange session(s); first {missing[0].date()}")
        vix, ts, macro, errors = self.common
        if pd.Timestamp(week.monday - timedelta(days=7)) not in vix.index:
            raise ValueError("Stale VIX history: previous week missing")
        # Same data-column contract and derived OHLC targets as data_collector;
        # all feature transforms and forecasting below use the production code.
        base = weekly.rename(columns={k: f"spx_{k}" for k in ("open", "high", "low", "close", "volume")})
        base = base.join(vix.rename(columns={k: f"vix_{k}" for k in ("open", "high", "low", "close")}), how="left")
        base["range_pct"] = (base.spx_high - base.spx_low) / base.spx_open
        base["log_range"] = np.log(base.range_pct.where(base.range_pct > 0))
        base["spx_return"] = (base.spx_close - base.spx_open) / base.spx_open
        if self.history_loader is not None:
            older = self.history_loader(ticker)
            self.source_times[f"{ticker}_persisted_history"] = {
                "source": "existing Gamma Lens weekly history (read only)",
                "retrieved_at": self.clock().isoformat(), "rows": len(older)}
            if not older.empty:
                older = older.loc[older.index < base.index.min()]
                base = pd.concat([older.reindex(columns=base.columns), base]).sort_index()
        events = pd.DataFrame.from_dict(event_flag_rows(now), orient="index")
        events.index = pd.to_datetime(events.index)
        raw_inputs = {"weekly": base, "daily": daily[["close"]].rename(columns={"close": "spx_close"}),
                      "vix_ts": ts[ts.index < pd.Timestamp(week.monday)], "macro": macro,
                      "events": events, "earnings": pd.DataFrame(), "gex": pd.DataFrame()}
        features = build_features(None, ticker=ticker, inputs=raw_inputs, as_of=now, persist=False)
        # A long preparation/retry must refresh the batch rather than reusing
        # the first ticker's now-stale quote cache.
        self.__dict__.pop("quotes", None)
        quote, quote_at = self.checked_quote(ticker, week.sessions[0].day)
        vol_quote, vol_at = self.checked_quote("VIX", week.sessions[0].day)
        avail = self.client.get_expirations(ticker)
        expiry = listed_week_expiration(avail, week)
        if not expiry:
            raise ValueError("No listed final-session expiration in this trading week")
        chain = self.client.get_chain_once(ticker, expiry)
        if chain.get("status") != "ok":
            raise ValueError("Chain fetch failed")
        self.source_times[ticker] = {"source": "Tradier history/quotes/chains", "retrieved_at": self.clock().isoformat(),
                                     "quote_at": quote_at.isoformat(), "vix_quote_at": vol_at.isoformat(),
                                     "last_history_session": str(previous)}
        return {"ticker": ticker, "features": features, "weekly": base,
                "reference": round(float(quote["open"]), 2), "vix": round(float(vol_quote["open"]), 2),
                "anchor_source": "Tradier current-session daily open in quote", "anchor_at": week.sessions[0].open.isoformat(),
                "expiration": expiry, "chain": chain, "source_times": dict(self.source_times),
                "data_delay_seconds": self.delay, "source_errors": errors,
                "raw_inputs": {k: frame_records(f) for k, f in raw_inputs.items()},
                "prepared_at": self.clock().isoformat()}

    def observe(self, ticker, session, first_available=None):
        rows = self.history(ticker, session.day, session.day)
        exact = [r for r in rows if r.get("date") == session.day.isoformat()]
        daily = exact[0] if len(exact) == 1 else None
        minutes, error = [], None
        if first_available is not None:
            try:
                block = self._get("timesales", symbol=ticker, interval="1min", session_filter="open",
                                  start=first_available.astimezone(NY).strftime("%Y-%m-%d %H:%M"),
                                  end=session.close.astimezone(NY).strftime("%Y-%m-%d %H:%M")).get("series") or {}
                minute_rows = block.get("data") or []
                if isinstance(minute_rows, dict):
                    minute_rows = [minute_rows]
                for r in minute_rows:
                    when = epoch_time(r["timestamp"])
                    if session.open <= when < session.close:
                        minutes.append({"start": when.isoformat(), **{k: r.get(k) for k in ("open", "high", "low", "close")}})
            except Exception as exc:
                error = f"Minute data unavailable: {type(exc).__name__}"
        return {"ticker": ticker, "session": str(session.day), "daily": daily,
                "minutes": minutes, "minute_error": error, "source": "Tradier history + timesales (regular session)",
                "window_start": session.open.isoformat(), "window_end": session.close.isoformat(),
                "data_delay_seconds": self.delay,
                "settlement": {"status": "unverified", "value": None,
                               "reason": "No official contract settlement value in Tradier price bars"}}
