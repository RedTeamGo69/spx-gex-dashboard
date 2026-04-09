from __future__ import annotations

import logging
import time
from datetime import datetime

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


_PUBLIC_BASE = "https://api.public.com"
_PUBLIC_AUTH_URL = f"{_PUBLIC_BASE}/userapiauthservice/personal/access-tokens"
_PUBLIC_GW = f"{_PUBLIC_BASE}/userapigateway"

# Maximum OSI symbols per Greeks batch request
_GREEKS_BATCH_SIZE = 100
# Maximum instruments per quotes batch request
_QUOTES_BATCH_SIZE = 50


class PublicDataClient:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.chain_cache: dict = {}

        # OAuth2 token state
        self._access_token: str | None = None
        self._token_expires_at: float = 0.0
        self._account_id: str | None = None
        self._token_lock = threading.Lock()

        # Cache for yfinance prevclose (date_str -> prevclose float)
        self._prevclose_cache: dict[str, float] = {}

    # ── Authentication ─────────────────────────────────────────────────

    def _ensure_auth(self):
        """Authenticate with Public.com and fetch account ID if needed."""
        with self._token_lock:
            # Refresh if token missing or expiring within 5 minutes
            if self._access_token and time.time() < self._token_expires_at - 300:
                return

            r = requests.post(
                _PUBLIC_AUTH_URL,
                headers={"Content-Type": "application/json"},
                json={"validityInMinutes": 60, "secret": self.secret_key},
                timeout=10,
            )
            r.raise_for_status()
            data = r.json()
            self._access_token = data["accessToken"]
            self._token_expires_at = time.time() + 60 * 60  # 60 minutes

            # Fetch account ID if not yet known
            if self._account_id is None:
                r2 = requests.get(
                    f"{_PUBLIC_GW}/trading/account",
                    headers=self._auth_headers_inner(),
                    timeout=10,
                )
                r2.raise_for_status()
                accounts = r2.json().get("accounts", [])
                if accounts:
                    self._account_id = accounts[0]["accountId"]
                else:
                    raise RuntimeError("No accounts found on Public.com")

    def _auth_headers_inner(self):
        """Return auth headers (assumes token is valid, no lock)."""
        return {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json",
        }

    def _auth_headers(self):
        """Ensure auth then return headers."""
        self._ensure_auth()
        return self._auth_headers_inner()

    # ── Quotes ─────────────────────────────────────────────────────────

    def get_spot_price(self, ticker="SPX"):
        r = requests.post(
            f"{_PUBLIC_GW}/marketdata/{self._account_id}/quotes",
            headers=self._auth_headers(),
            json={"instruments": [{"symbol": ticker, "type": "INDEX"}]},
            timeout=10,
        )
        r.raise_for_status()
        quotes = r.json().get("quotes", [])
        if not quotes:
            return 0.0
        return safe_float(quotes[0].get("last", 0), 0.0)

    def get_full_quote(self, ticker="SPX"):
        """
        Return the full quote dict with prevclose, open, last, bid, ask, etc.

        Public.com returns last for SPX but bid/ask/prevclose are often null.
        Uses yfinance as fallback for prevclose (needed for overnight move).
        """
        r = requests.post(
            f"{_PUBLIC_GW}/marketdata/{self._account_id}/quotes",
            headers=self._auth_headers(),
            json={"instruments": [{"symbol": ticker, "type": "INDEX"}]},
            timeout=10,
        )
        r.raise_for_status()
        quotes = r.json().get("quotes", [])
        q = quotes[0] if quotes else {}

        last = safe_float(q.get("last", 0), 0.0)
        prevclose = safe_float(q.get("previousClose", 0), 0.0)
        open_price = safe_float(q.get("open", 0), 0.0)
        high = safe_float(q.get("high", 0), 0.0)
        low = safe_float(q.get("low", 0), 0.0)
        bid = safe_float(q.get("bid", 0), 0.0)
        ask = safe_float(q.get("ask", 0), 0.0)

        # Fallback to yfinance for missing prevclose/open/high/low
        if prevclose <= 0 and ticker in ("SPX", "^SPX", "XSP"):
            yf_data = self._fetch_yfinance_data(ticker)
            if prevclose <= 0:
                prevclose = yf_data.get("prevclose", 0.0)
            if open_price <= 0:
                open_price = yf_data.get("open", 0.0)
            if high <= 0:
                high = yf_data.get("high", 0.0)
            if low <= 0:
                low = yf_data.get("low", 0.0)

        change = last - prevclose if last > 0 and prevclose > 0 else 0.0
        change_pct = (change / prevclose * 100.0) if prevclose > 0 else 0.0

        return {
            "symbol": ticker,
            "last": last,
            "prevclose": prevclose,
            "open": open_price,
            "high": high,
            "low": low,
            "bid": bid,
            "ask": ask,
            "change": change,
            "change_pct": change_pct,
        }

    def _fetch_yfinance_data(self, ticker="SPX"):
        """Fetch prevclose and OHLC from yfinance, cached by date."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"{ticker}_{today_str}"
        if cache_key in self._prevclose_cache:
            return self._prevclose_cache[cache_key]

        result = {"prevclose": 0.0, "open": 0.0, "high": 0.0, "low": 0.0}
        try:
            import yfinance as yf
            yf_ticker = "^GSPC" if ticker in ("SPX", "^SPX") else "^XSP" if ticker == "XSP" else ticker
            hist = yf.Ticker(yf_ticker).history(period="5d")
            if not hist.empty:
                result["prevclose"] = round(float(hist["Close"].dropna().iloc[-1]), 2)
                if len(hist) >= 2:
                    result["prevclose"] = round(float(hist["Close"].dropna().iloc[-2]), 2)
                # Today's OHLC if available
                last_row = hist.iloc[-1]
                result["open"] = round(float(last_row.get("Open", 0) or 0), 2)
                result["high"] = round(float(last_row.get("High", 0) or 0), 2)
                result["low"] = round(float(last_row.get("Low", 0) or 0), 2)
        except Exception as e:
            _logger.warning("yfinance fallback failed for %s: %s", ticker, e)

        self._prevclose_cache[cache_key] = result
        return result

    # ── Expirations ────────────────────────────────────────────────────

    def get_expirations(self, ticker="SPX"):
        r = requests.post(
            f"{_PUBLIC_GW}/marketdata/{self._account_id}/option-expirations",
            headers=self._auth_headers(),
            json={"instrument": {"symbol": ticker, "type": "INDEX"}},
            timeout=10,
        )
        r.raise_for_status()
        exps = r.json().get("expirations", [])
        return sorted([exps] if isinstance(exps, str) else exps)

    # ── IV normalization ───────────────────────────────────────────────

    @staticmethod
    def _parse_iv_from_greeks(greeks, is_0dte=False):
        """
        Parse implied volatility from Public.com Greeks response.

        Public returns IV in decimal for non-0DTE (e.g., 0.15 = 15%) but
        inflated values for 0DTE (e.g., 2.63) due to a different time
        convention for annualizing near-expiry IV.

        For 0DTE with IV > 1.5: zero it out so the synthetic IV fallback
        (infer_iv_from_gamma) handles it using vendor gamma instead.
        Normal SPX IV never exceeds ~1.0 even in extreme markets, so 1.5
        safely separates real IV from 0DTE artifacts.

        For non-0DTE with IV > 3.0: divide by 100 as defensive handling
        in case of percentage-format data.
        """
        if not greeks:
            return 0.0
        iv = safe_float(greeks.get("impliedVolatility", 0), 0.0)
        if is_0dte and iv > 1.5:
            _logger.debug("0DTE IV zeroed out (inflated): %.4f", iv)
            return 0.0
        if iv > 3.0:
            _logger.debug("IV normalization: %.4f -> %.4f (divided by 100)", iv, iv / 100.0)
            iv /= 100.0
        return max(iv, 0.0)

    # ── Greeks batch fetch ─────────────────────────────────────────────

    def _fetch_greeks_batch(self, osi_symbols):
        """
        Fetch Greeks for a list of OSI symbols. Returns {symbol: greeks_dict}.

        Batches requests to avoid exceeding API limits.
        """
        self._ensure_auth()
        result = {}
        for i in range(0, len(osi_symbols), _GREEKS_BATCH_SIZE):
            batch = osi_symbols[i:i + _GREEKS_BATCH_SIZE]
            try:
                r = requests.get(
                    f"{_PUBLIC_GW}/option-details/{self._account_id}/greeks",
                    headers=self._auth_headers_inner(),
                    params={"osiSymbols": batch},
                    timeout=20,
                )
                if r.status_code == 200:
                    for entry in r.json().get("greeks", []):
                        sym = entry.get("symbol", "")
                        result[sym] = entry.get("greeks", {})
                else:
                    _logger.warning("Greeks batch fetch failed: %s", r.status_code)
            except requests.RequestException as e:
                _logger.warning("Greeks batch fetch error: %s", e)
            if i + _GREEKS_BATCH_SIZE < len(osi_symbols):
                time.sleep(0.1)
        return result

    # ── Chain fetch ────────────────────────────────────────────────────

    def get_chain_once(self, ticker, expiration):
        """
        Fetch one options chain from Public.com.

        Fetches the chain (bid/ask/OI/volume) then Greeks separately,
        merges them, and returns the same normalized format.

        Returns dict with:
            status: "ok" or "failed"
            calls: [...]
            puts: [...]
            error: optional string
        """
        try:
            r = requests.post(
                f"{_PUBLIC_GW}/marketdata/{self._account_id}/option-chain",
                headers=self._auth_headers(),
                json={
                    "instrument": {"symbol": ticker, "type": "INDEX"},
                    "expirationDate": expiration,
                },
                timeout=20,
            )
            r.raise_for_status()
        except (requests.RequestException, requests.Timeout) as e:
            return {
                "status": "failed",
                "calls": [],
                "puts": [],
                "error": str(e),
            }

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

        raw_calls = d.get("calls", [])
        raw_puts = d.get("puts", [])

        if not raw_calls and not raw_puts:
            return {"status": "ok", "calls": [], "puts": [], "error": None}

        # Collect all OSI symbols for batch Greeks fetch
        all_symbols = []
        for entry in raw_calls + raw_puts:
            sym = (entry.get("instrument") or {}).get("symbol", "")
            if sym:
                all_symbols.append(sym)

        # Fetch Greeks for all symbols in batch
        greeks_map = self._fetch_greeks_batch(all_symbols) if all_symbols else {}

        # Determine if this is 0DTE
        today_str = datetime.now().strftime("%Y-%m-%d")
        is_0dte = expiration == today_str

        def _parse_entries(raw_entries):
            rows = []
            for entry in raw_entries:
                if not isinstance(entry, dict):
                    continue

                sym = (entry.get("instrument") or {}).get("symbol", "")

                # Parse strike from OSI symbol (last 8 digits / 1000)
                try:
                    strike = round(int(sym[-8:]) / 1000.0, 2)
                except (ValueError, IndexError):
                    continue
                if strike <= 0:
                    continue

                bid = safe_float(entry.get("bid", 0), 0.0)
                ask = safe_float(entry.get("ask", 0), 0.0)
                oi = safe_float(entry.get("openInterest", 0), 0.0)
                volume = safe_float(entry.get("volume", 0), 0.0)

                greeks = greeks_map.get(sym, {})
                iv = self._parse_iv_from_greeks(greeks, is_0dte=is_0dte)
                vendor_gamma = safe_float(greeks.get("gamma", 0), 0.0) if greeks else 0.0

                rows.append({
                    "strike": strike,
                    "openInterest": oi,
                    "volume": volume,
                    "impliedVolatility": iv,
                    "vendorGamma": vendor_gamma,
                    "bid": bid,
                    "ask": ask,
                    "mid": (bid + ask) / 2 if bid > 0 and ask > 0 else 0.0,
                })

            return rows

        calls = _parse_entries(raw_calls)
        puts = _parse_entries(raw_puts)

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
