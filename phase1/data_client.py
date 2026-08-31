from __future__ import annotations

import logging
import random
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

_logger = logging.getLogger(__name__)

import threading

from phase1.config import (
    MAX_WORKERS,
    CHAIN_RETRIES,
    CHAIN_RETRY_SLEEP,
)


_safe_float_coercion_count = 0
_coercion_lock = threading.Lock()

def safe_float(x, default=0.0):
    global _safe_float_coercion_count
    try:
        return float(x)
    except (TypeError, ValueError):
        with _coercion_lock:
            _safe_float_coercion_count += 1
        return default


def resolve_quote_spot(quote) -> float:
    """Resolve one positive spot value as last -> close -> prevclose.

    Raw and normalized Tradier quotes share this semantic path. In
    particular, normalization turns a null ``last`` into 0.0, which must not
    prevent a valid after-hours ``close`` or ``prevclose`` fallback.
    """
    if not isinstance(quote, dict):
        return 0.0
    for field in ("last", "close", "prevclose"):
        value = safe_float(quote.get(field), 0.0)
        if value > 0:
            return value
    return 0.0


def get_coercion_count():
    return _safe_float_coercion_count

def reset_coercion_count():
    global _safe_float_coercion_count
    with _coercion_lock:
        _safe_float_coercion_count = 0


class TradierDataClient:
    def __init__(self, token: str, base_url: str = "https://api.tradier.com/v1"):
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.chain_cache = {}

    def tradier_headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json",
        }

    def get_spot_price(self, ticker="SPX"):
        r = requests.get(
            f"{self.base_url}/markets/quotes",
            headers=self.tradier_headers(),
            params={"symbols": ticker},
            timeout=10,
        )
        r.raise_for_status()
        # Tradier returns {"quotes": {"unmatched_symbols": {...}}} (no "quote"
        # key) for any symbol it can't price (e.g. an unrecognized or illiquid
        # symbol the user searched). A bare ["quote"] subscript raised
        # KeyError('quote') and surfaced as "Engine error: 'quote'"; degrade to
        # 0.0 so the engine reports "no data" cleanly instead of crashing.
        q = (r.json().get("quotes") or {}).get("quote")
        if isinstance(q, list):
            q = q[0] if q else None
        if not isinstance(q, dict):
            return 0.0
        return resolve_quote_spot(q)

    @staticmethod
    def _normalize_quote(q, fallback_symbol):
        return {
            "symbol": q.get("symbol", fallback_symbol),
            "last": safe_float(q.get("last", 0), 0.0),
            "close": safe_float(q.get("close", 0), 0.0),
            "prevclose": safe_float(q.get("prevclose", 0), 0.0),
            "open": safe_float(q.get("open", 0), 0.0),
            "high": safe_float(q.get("high", 0), 0.0),
            "low": safe_float(q.get("low", 0), 0.0),
            "bid": safe_float(q.get("bid", 0), 0.0),
            "ask": safe_float(q.get("ask", 0), 0.0),
            "change": safe_float(q.get("change", 0), 0.0),
            "change_pct": safe_float(q.get("change_percentage", 0), 0.0),
        }

    def get_full_quotes(self, tickers):
        """
        Batch quote lookup. Returns {symbol: quote_dict} for every symbol
        Tradier returned. Tradier's /markets/quotes accepts a comma-separated
        symbols list and returns all of them in a single response.
        """
        symbols = ",".join(tickers)
        r = requests.get(
            f"{self.base_url}/markets/quotes",
            headers=self.tradier_headers(),
            params={"symbols": symbols},
            timeout=10,
        )
        r.raise_for_status()
        # See get_spot_price: unmatched symbols come back without a "quote" key.
        # Return an empty map so get_full_quote() degrades to its zeroed-quote
        # fallback instead of raising KeyError('quote').
        q = (r.json().get("quotes") or {}).get("quote")
        if q is None:
            return {}
        if isinstance(q, dict):
            q = [q]
        out = {}
        for row in q:
            if not isinstance(row, dict):
                continue
            symbol = row.get("symbol")
            if not symbol:
                continue
            out[symbol] = self._normalize_quote(row, symbol)
        return out

    def get_full_quote(self, ticker="SPX"):
        """
        Return the full quote dict with prevclose, open, last, bid, ask, etc.

        Useful for computing overnight moves and pre-market context.
        """
        quotes = self.get_full_quotes([ticker])
        if ticker in quotes:
            return quotes[ticker]
        # Tradier dropped the symbol from the response — return a zeroed quote
        # with the requested ticker label so callers keep the shape they expect.
        return self._normalize_quote({}, ticker)

    def get_expirations(self, ticker="SPX"):
        r = requests.get(
            f"{self.base_url}/markets/options/expirations",
            headers=self.tradier_headers(),
            params={"symbol": ticker, "includeAllRoots": "true"},
            timeout=10,
        )
        r.raise_for_status()
        exp = r.json().get("expirations")
        # Tradier returns {"expirations": null} for a symbol that has no
        # listed options (or doesn't exist). Treat that as "no options"
        # rather than letting a None subscript blow up the caller.
        if not exp:
            return []
        e = exp.get("date")
        if not e:
            return []
        return sorted([e] if isinstance(e, str) else e)

    def get_history(self, symbol, interval="daily", start=None, end=None):
        """Historical OHLCV bars from /markets/history.

        ``interval`` is one of daily / weekly / monthly. ``start`` / ``end``
        are YYYY-MM-DD strings. Returns a list of day dicts
        (``{"date", "open", "high", "low", "close", "volume"}``) sorted the
        way Tradier returns them (ascending). Weekly bars are labeled with
        the week's Monday (verified against live data, incl. holiday
        Mondays). ``{"history": null}`` (unknown symbol / no entitlement /
        empty window) degrades to ``[]`` rather than raising, mirroring
        get_expirations.
        """
        params = {"symbol": symbol, "interval": interval,
                  "session_filter": "all"}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        r = requests.get(
            f"{self.base_url}/markets/history",
            headers=self.tradier_headers(),
            params=params,
            timeout=30,
        )
        r.raise_for_status()
        hist = r.json().get("history")
        if not hist or not isinstance(hist, dict):
            return []
        days = hist.get("day")
        if not days:
            return []
        return days if isinstance(days, list) else [days]

    def validate_ticker(self, symbol):
        """Validate that `symbol` is a real, optionable Tradier instrument.

        Returns a dict describing the instrument when it has a listed option
        chain (the only kind that yields GEX), or ``None`` when the symbol
        is unknown / has no options. Kept free of any UI framework so it can
        be cached at the Streamlit layer.

        Return shape::

            {"symbol": "AAPL", "type": "stock",
             "name": "Apple Inc", "has_options": True}
        """
        sym = (symbol or "").strip().upper()
        if not sym:
            return None

        # A non-empty expiration list is the authoritative "this symbol has
        # options" signal — no chain means no GEX, so we reject it here.
        try:
            exps = self.get_expirations(sym)
        except Exception:
            return None
        if not exps:
            return None

        # Enrich with instrument type / display name from the raw quote.
        # get_full_quotes() drops `type`/`description`, so read the quote
        # directly here. Failures degrade to sensible defaults rather than
        # rejecting an otherwise-valid optionable symbol.
        inst_type, name = "stock", sym
        try:
            r = requests.get(
                f"{self.base_url}/markets/quotes",
                headers=self.tradier_headers(),
                params={"symbols": sym},
                timeout=10,
            )
            r.raise_for_status()
            q = r.json().get("quotes", {}).get("quote")
            if isinstance(q, list):
                q = q[0] if q else {}
            if isinstance(q, dict):
                inst_type = (q.get("type") or "stock").lower()
                name = q.get("description") or sym
        except Exception:
            pass

        return {
            "symbol": sym,
            "type": inst_type,
            "name": name,
            "has_options": True,
        }

    @staticmethod
    def _parse_iv_from_greeks(greeks):
        if not greeks:
            return 0.0
        iv = greeks.get("mid_iv")
        if iv in (None, "", 0):
            iv = greeks.get("ask_iv")
        iv = safe_float(iv, 0.0)
        # Unit heuristic: Tradier normally sends decimal IV (0.18), but some
        # feeds slip percent units (18.0). Values in (3, 300] are treated as
        # percent and rescaled — a genuine 300%+ decimal IV essentially never
        # occurs on the underlyings this app trades, while blindly dividing
        # would turn a real percent-unit 350 into a fake 3.5 decimal (350%)
        # or a corrupt >300 reading into near-zero. Anything above 300 is
        # garbage either way: drop it (0.0 = missing → synthetic-IV path).
        if 3 < iv <= 300:
            _logger.debug("IV normalization: %.4f → %.4f (divided by 100)", iv, iv / 100.0)
            iv /= 100.0
        elif iv > 300:
            _logger.debug("IV rejected as garbage: %.4f (unit-ambiguous, > 300)", iv)
            iv = 0.0
        return max(iv, 0.0)

    def get_chain_once(self, ticker, expiration):
        """
        Fetch one options chain.

        Returns dict with:
            status: "ok" or "failed"
            calls: [...]
            puts: [...]
            error: optional string
        """
        try:
            r = requests.get(
                f"{self.base_url}/markets/options/chains",
                headers=self.tradier_headers(),
                params={"symbol": ticker, "expiration": expiration, "greeks": "true"},
                timeout=10,
            )
            r.raise_for_status()
        except (requests.RequestException, requests.Timeout) as e:
            return {
                "status": "failed",
                "calls": [],
                "puts": [],
                "error": str(e),
            }

        # --- Response shape validation ---
        try:
            d = r.json()
        except (ValueError, TypeError) as e:
            return {
                "status": "failed",
                "calls": [],
                "puts": [],
                "error": f"Invalid JSON: {e}",
            }

        if not isinstance(d, dict):
            return {
                "status": "failed",
                "calls": [],
                "puts": [],
                "error": f"Unexpected response type: {type(d).__name__}",
            }

        # "options" key missing or null → vendor returned no data for this expiration
        options_block = d.get("options")
        if options_block is None:
            return {
                "status": "ok",
                "calls": [],
                "puts": [],
                "error": None,
            }

        if not isinstance(options_block, dict):
            return {
                "status": "failed",
                "calls": [],
                "puts": [],
                "error": f"Malformed 'options' field: {type(options_block).__name__}",
            }

        option_list = options_block.get("option")
        if option_list is None:
            return {
                "status": "ok",
                "calls": [],
                "puts": [],
                "error": None,
            }

        if isinstance(option_list, dict):
            # Single option returned as a dict instead of a list
            option_list = [option_list]
        elif not isinstance(option_list, list):
            return {
                "status": "failed",
                "calls": [],
                "puts": [],
                "error": f"Malformed 'option' field: {type(option_list).__name__}",
            }

        calls = []
        puts = []

        for o in option_list:
            if not isinstance(o, dict):
                continue

            strike = round(safe_float(o.get("strike", 0), 0.0), 2)
            if strike <= 0:
                continue

            bid = safe_float(o.get("bid", 0), 0.0)
            ask = safe_float(o.get("ask", 0), 0.0)
            oi = safe_float(o.get("open_interest", 0), 0.0)
            volume = safe_float(o.get("volume", 0), 0.0)
            greeks = o.get("greeks") or {}
            if not isinstance(greeks, dict):
                greeks = {}
            iv = self._parse_iv_from_greeks(greeks)
            vendor_gamma = safe_float(greeks.get("gamma", 0), 0.0) if greeks else 0.0
            vendor_delta = safe_float(greeks.get("delta", 0), 0.0) if greeks else 0.0

            row = {
                "strike": strike,
                "openInterest": oi,
                "volume": volume,
                "impliedVolatility": iv,
                "vendorGamma": vendor_gamma,
                "vendorDelta": vendor_delta,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0,
                # OCC root (SPX vs SPXW, NDX vs NDXP, SPY vs adjusted SPY1).
                # On 3rd Fridays Tradier returns BOTH the AM-settled monthly
                # and the PM-settled weekly for index underlyings — two
                # different contracts per strike. Quote consumers must filter
                # to one root (quote_filters.filter_to_preferred_root); GEX
                # aggregation deliberately keeps every root's OI.
                "root": str(o.get("root_symbol") or "").upper(),
            }

            if o.get("option_type") == "call":
                calls.append(row)
            else:
                puts.append(row)

        return {
            "status": "ok",
            "calls": calls,
            "puts": puts,
            "error": None,
        }

    def get_chain_with_retry(self, ticker, expiration, retries=CHAIN_RETRIES, sleep_sec=CHAIN_RETRY_SLEEP):
        last_error = None
        for attempt in range(retries + 1):
            result = self.get_chain_once(ticker, expiration)
            # A status-ok chain with ZERO options for a LISTED expiration is a
            # transient vendor gap (Tradier occasionally returns a null
            # "options" block mid-session), not a settled contract — the
            # expired-expiration filter upstream already drops genuinely dead
            # exps before we get here. Treat empty-ok as retryable so a blip
            # doesn't get cached as a permanent hole that blocks the GEX/EM
            # computation for the whole session.
            if result["status"] == "ok" and not result["calls"] and not result["puts"]:
                last_error = "empty options block (listed expiration)"
                if attempt < retries:
                    backoff = sleep_sec * (2 ** attempt)
                    time.sleep(backoff + random.uniform(0, sleep_sec))
                    continue
                # Retries exhausted — surface as failed so callers skip it
                # loudly instead of silently treating it as "no gamma here".
                return {"status": "failed", "calls": [], "puts": [],
                        "error": last_error}
            if result["status"] == "ok":
                if attempt > 0:
                    print(f"    ✓ Recovered {expiration} on retry {attempt}")
                return result
            last_error = result.get("error")
            if attempt < retries:
                # Exponential backoff with jitter. Parallel threads would
                # otherwise retry in lockstep and re-hammer the API after a
                # 429; jitter de-synchronizes them.
                backoff = sleep_sec * (2 ** attempt)
                jitter = random.uniform(0, sleep_sec)
                time.sleep(backoff + jitter)

        return {
            "status": "failed",
            "calls": [],
            "puts": [],
            "error": last_error or "Unknown fetch failure",
        }

    def get_chain_cached(self, ticker, expiration):
        key = (ticker, expiration)
        if key not in self.chain_cache:
            self.chain_cache[key] = self.get_chain_with_retry(ticker, expiration)
        return self.chain_cache[key]

    def prefetch_chains(self, ticker, expirations):
        uncached = [e for e in expirations if (ticker, e) not in self.chain_cache]
        if not uncached:
            return

        print(f"  Fetching {len(uncached)} chains in parallel...")

        def _fetch(exp):
            return exp, self.get_chain_with_retry(ticker, exp)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_fetch, e): e for e in uncached}
            for future in as_completed(futures):
                exp = futures[future]
                try:
                    exp_res, result = future.result()
                    self.chain_cache[(ticker, exp_res)] = result
                except Exception as e:
                    self.chain_cache[(ticker, exp)] = {
                        "status": "failed",
                        "calls": [],
                        "puts": [],
                        "error": f"Thread exception: {e}",
                    }

    def clear_cache(self):
        self.chain_cache.clear()
