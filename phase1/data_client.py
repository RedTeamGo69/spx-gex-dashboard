from __future__ import annotations

import logging
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
        q = r.json()["quotes"]["quote"]
        if isinstance(q, list):
            q = q[0]
        return safe_float(q.get("last", q.get("close", 0)), 0.0)

    def get_full_quote(self, ticker="SPX"):
        """
        Return the full quote dict with prevclose, open, last, bid, ask, etc.

        Useful for computing overnight moves and pre-market context.
        """
        r = requests.get(
            f"{self.base_url}/markets/quotes",
            headers=self.tradier_headers(),
            params={"symbols": ticker},
            timeout=10,
        )
        r.raise_for_status()
        q = r.json()["quotes"]["quote"]
        if isinstance(q, list):
            q = q[0]
        return {
            "symbol": q.get("symbol", ticker),
            "last": safe_float(q.get("last", 0), 0.0),
            "prevclose": safe_float(q.get("prevclose", 0), 0.0),
            "open": safe_float(q.get("open", 0), 0.0),
            "high": safe_float(q.get("high", 0), 0.0),
            "low": safe_float(q.get("low", 0), 0.0),
            "bid": safe_float(q.get("bid", 0), 0.0),
            "ask": safe_float(q.get("ask", 0), 0.0),
            "change": safe_float(q.get("change", 0), 0.0),
            "change_pct": safe_float(q.get("change_percentage", 0), 0.0),
        }

    def get_expirations(self, ticker="SPX"):
        r = requests.get(
            f"{self.base_url}/markets/options/expirations",
            headers=self.tradier_headers(),
            params={"symbol": ticker, "includeAllRoots": "true"},
            timeout=10,
        )
        r.raise_for_status()
        e = r.json()["expirations"]["date"]
        return sorted([e] if isinstance(e, str) else e)

    @staticmethod
    def _parse_iv_from_greeks(greeks):
        if not greeks:
            return 0.0
        iv = greeks.get("mid_iv")
        if iv in (None, "", 0):
            iv = greeks.get("ask_iv")
        iv = safe_float(iv, 0.0)
        if iv > 3:
            _logger.debug("IV normalization: %.4f → %.4f (divided by 100)", iv, iv / 100.0)
            iv /= 100.0
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
            greeks = o.get("greeks") or {}
            if not isinstance(greeks, dict):
                greeks = {}
            iv = self._parse_iv_from_greeks(greeks)
            vendor_gamma = safe_float(greeks.get("gamma", 0), 0.0) if greeks else 0.0

            row = {
                "strike": strike,
                "openInterest": oi,
                "impliedVolatility": iv,
                "vendorGamma": vendor_gamma,
                "bid": bid,
                "ask": ask,
                "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0,
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
            if result["status"] == "ok":
                if attempt > 0:
                    print(f"    ✓ Recovered {expiration} on retry {attempt}")
                return result
            last_error = result.get("error")
            if attempt < retries:
                time.sleep(sleep_sec * (attempt + 1))

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
