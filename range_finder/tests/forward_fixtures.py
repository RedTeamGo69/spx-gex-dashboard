"""Deterministic prices and production feature inputs; never a live study."""
from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd

from range_finder.feature_builder import build_features
from range_finder.trading_week import trading_week
from range_finder.forward_test.provider import frame_records


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def prepared_fixture(ticker, week, clock):
    ref = {"SPX": 6000.0, "SPY": 600.0, "AAPL": 200.0, "AMD": 150.0}[ticker]
    index = pd.date_range(end=pd.Timestamp(week.monday) - pd.Timedelta(days=7), periods=160, freq="W-MON")
    rng = np.random.default_rng(179)
    width = 0.012 + rng.uniform(0, 0.020, len(index))
    weekly = pd.DataFrame({"spx_open": ref, "spx_high": ref * (1 + width * .7),
                           "spx_low": ref * (1 - width * .3), "spx_close": ref * (1 + width * .2),
                           "range_pct": width, "log_range": np.log(width), "spx_return": rng.uniform(-.004, .01, len(index)),
                           "vix_close": 18 + rng.normal(0, 1, len(index))}, index=index)
    days = pd.bdate_range(index[0], index[-1] + pd.Timedelta(days=4))
    daily = pd.DataFrame({"spx_close": ref * np.exp(np.cumsum(rng.normal(0, .005, len(days))))}, index=days)
    macro = pd.DataFrame({"yield_spread": rng.normal(.5, .1, len(days)), "fed_funds": 4.0}, index=days)
    ts = pd.DataFrame({"vix9d_close": 18 + rng.normal(0, 1, len(index)),
                       "vix3m_close": 20 + rng.normal(0, 1, len(index))}, index=index)
    ts["vix_ts_slope"] = ts.vix3m_close - ts.vix9d_close
    events = pd.DataFrame({"has_fomc": 0, "has_cpi": 0, "has_nfp": 0, "has_opex": 0,
                           "event_count": [int(i % 4 == 0) for i in range(len(index) + 1)]},
                          index=index.append(pd.DatetimeIndex([pd.Timestamp(week.monday)])))
    raw = {"weekly": weekly, "daily": daily, "macro": macro, "vix_ts": ts, "events": events,
           "earnings": pd.DataFrame(), "gex": pd.DataFrame()}
    features = build_features(None, ticker=ticker, inputs=raw, as_of=clock(), persist=False)
    expiration = str(week.sessions[-1].day)
    root = "SPXW" if ticker == "SPX" else ticker
    increment = 5 if ticker == "SPX" else 1
    strikes = np.arange(ref * .65 // increment * increment, ref * 1.35, increment)
    chain = {"status": "ok", "calls": [], "puts": []}
    for side, code in (("calls", "C"), ("puts", "P")):
        for k in strikes:
            chain[side].append({"strike": float(k), "root": root, "expiration_date": expiration,
                                "symbol": f"{root}{expiration[2:].replace('-', '')}{code}{int(k*1000):08d}",
                                "bid": 2.0, "ask": 2.2, "contract_size": 100,
                                "bid_date": int(clock().timestamp() * 1000), "ask_date": int(clock().timestamp() * 1000)})
    return {"ticker": ticker, "features": features, "weekly": weekly, "reference": ref, "vix": 18.0,
            "anchor_source": "deterministic fixture daily open", "anchor_at": week.sessions[0].open.isoformat(),
            "expiration": expiration, "chain": chain, "source_times": {ticker: {"source": "fixture", "retrieved_at": clock().isoformat()}},
            "source_errors": {}, "data_delay_seconds": 0, "prepared_at": clock().isoformat(),
            "raw_inputs": {k: frame_records(v) for k, v in raw.items()}}


class FixtureProvider:
    def __init__(self, clock):
        self.clock = clock
        self.prepares = []
        self.observes = []
        self.fail_tickers = set()

    def prepare(self, ticker, week):
        self.prepares.append(ticker)
        if ticker in self.fail_tickers:
            raise ValueError("Fixture missing history")
        return prepared_fixture(ticker, week, self.clock)

    def observe(self, ticker, session, first_available=None):
        self.observes.append((ticker, session.day))
        ref = {"SPX": 6000., "SPY": 600., "AAPL": 200., "AMD": 150.}[ticker]
        bar = {"open": ref, "close": ref, "high": ref * 1.002, "low": ref * .998}
        minutes = []
        when = first_available.replace(second=0, microsecond=0) if first_available else session.close
        while when < session.close:
            minutes.append({"start": when.isoformat(), **bar})
            when += timedelta(minutes=1)
        return {"ticker": ticker, "session": str(session.day), "daily": {"date": str(session.day), **bar},
                "minutes": minutes, "minute_error": None, "source": "deterministic fixture",
                "window_start": session.open.isoformat(), "window_end": session.close.isoformat(),
                "settlement": {"status": "unverified", "value": None}}


def scoring_fixture(week, put=95., call=105., available=None):
    available = available or week.capture_start
    f = {"ticker": "SPY", "put_short": put, "call_short": call,
         "available_at": available.isoformat(), "expiration": str(week.sessions[-1].day), "contract_root": "SPY"}
    clock = Clock(week.evaluation_close + timedelta(hours=3))
    provider = FixtureProvider(clock)
    observations = {}
    for session in week.sessions:
        obs = provider.observe("SPY", session, available if session == week.sessions[0] else None)
        bar = {"open": 100., "high": 104., "low": 96., "close": 100.}
        obs["daily"] = {"date": str(session.day), **bar}
        for m in obs["minutes"]:
            m.update(bar)
        observations[str(session.day)] = {**obs, "collected_at": clock().isoformat(), "observation_id": str(session.day)}
    return f, observations, clock
