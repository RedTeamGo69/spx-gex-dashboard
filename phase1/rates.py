from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import requests

from phase1.config import DEFAULT_RISK_FREE_RATE

# Disk cache for last successful rate fetch
_RATE_CACHE_PATH = Path(__file__).parent / ".rate_cache.json"


def _write_rate_cache(rate_dict: dict) -> None:
    """Persist a successful rate fetch to disk so it survives restarts."""
    try:
        payload = {**rate_dict, "cached_at": datetime.now(timezone.utc).isoformat()}
        _RATE_CACHE_PATH.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass  # Caching is best-effort — never fail the main flow


def _read_rate_cache() -> dict | None:
    """Read the most recently cached rate, or None if unavailable."""
    try:
        if not _RATE_CACHE_PATH.exists():
            return None
        payload = json.loads(_RATE_CACHE_PATH.read_text())
        if "rate" not in payload:
            return None
        return payload
    except Exception:
        return None


def parse_fred_rate_response(payload: dict):
    """
    Parse FRED DTB3 response payload.

    Returns:
        dict with keys: rate, source, label, as_of
        or None if no usable observation exists
    """
    obs = payload.get("observations", [])
    for o in obs:
        val = o.get("value", ".")
        if val != ".":
            rate = float(val) / 100
            date = o.get("date", "?")
            return {
                "rate": rate,
                "source": "fred_dtb3",
                "label": f"3M T-bill (FRED {date})",
                "as_of": date,
            }
    return None


def parse_treasury_rate_response(payload: dict):
    """
    Parse Treasury monthly average response payload.

    Returns:
        dict with keys: rate, source, label, as_of
        or None if no usable record exists
    """
    data = payload.get("data", [])
    if not data:
        return None

    rate_str = data[0].get("avg_interest_rate_amt")
    date = data[0].get("record_date", "unknown")
    if rate_str is None:
        return None

    rate = float(rate_str) / 100
    return {
        "rate": rate,
        "source": "treasury_monthly_average",
        "label": f"Treasury monthly average ({date})",
        "as_of": date,
    }


def _fetch_fred(fred_api_key: str):
    """Fetch from FRED DTB3. Returns parsed dict or None."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "DTB3",
        "api_key": fred_api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,
    }
    r = requests.get(url, params=params, timeout=5)
    if r.status_code == 200:
        return parse_fred_rate_response(r.json())
    return None


def _fetch_treasury():
    """Fetch from Treasury fiscal data API. Returns parsed dict or None."""
    url = (
        "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
        "v2/accounting/od/avg_interest_rates"
    )
    params = {
        "filter": "security_desc:eq:Treasury Bills",
        "sort": "-record_date",
        "page[size]": "1",
        "fields": "record_date,avg_interest_rate_amt",
    }
    r = requests.get(url, params=params, timeout=5)
    if r.status_code == 200:
        return parse_treasury_rate_response(r.json())
    return None


def fetch_risk_free_rate(fred_api_key: str, debug: bool = False):
    """
    Fetch current 3-month T-bill rate.

    Fires FRED and Treasury API calls in parallel to avoid sequential
    timeout stacking. FRED is preferred when both succeed.

    Final fallback: DEFAULT_RISK_FREE_RATE from config.

    Returns a dict:
    {
        "rate": float,
        "source": "...",
        "label": "...",
        "as_of": "YYYY-MM-DD" or None,
    }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    futures = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        has_fred_key = bool(fred_api_key and fred_api_key != "YOUR_FRED_KEY_HERE")
        if has_fred_key:
            futures[pool.submit(_fetch_fred, fred_api_key)] = "fred"
        futures[pool.submit(_fetch_treasury)] = "treasury"

        results = {}
        for future in as_completed(futures):
            label = futures[future]
            try:
                results[label] = future.result()
            except Exception as e:
                if debug:
                    print(f"  {label.upper()} API error: {e}")
                results[label] = None

    # Prefer FRED over Treasury
    for key in ["fred", "treasury"]:
        parsed = results.get(key)
        if parsed is not None:
            print(f"  Risk-free rate: {parsed['rate']*100:.2f}% ({parsed['label']})")
            _write_rate_cache(parsed)
            return parsed

    # Both APIs failed — try disk cache before hardcoded fallback
    cached = _read_rate_cache()
    if cached is not None:
        print(
            f"  Risk-free rate: {cached['rate']*100:.2f}% "
            f"(cached from {cached.get('as_of', 'unknown')} — live APIs unavailable)"
        )
        return {
            "rate": cached["rate"],
            "source": "cached_rate",
            "label": f"cached rate from {cached.get('as_of', 'unknown')}",
            "as_of": cached.get("as_of"),
        }

    print(f"  Risk-free rate: {DEFAULT_RISK_FREE_RATE*100:.2f}% (fallback default — no cache available)")
    return {
        "rate": DEFAULT_RISK_FREE_RATE,
        "source": "fallback_default",
        "label": f"fallback default {DEFAULT_RISK_FREE_RATE*100:.2f}%",
        "as_of": None,
    }
