from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import requests

from phase1.config import DEFAULT_RISK_FREE_RATE

# Disk cache for last successful rate fetch
_RATE_CACHE_PATH = Path(__file__).parent / ".rate_cache.json"

# Term-structure points we try to fetch from FRED constant-maturity Treasury
# series. Anchor days are the nominal DTE each series represents — used both
# for curve interpolation and for labeling the disk cache. The series are
# constant-maturity yields (DGS*), which are curve points suitable for
# linear interpolation across tenors. DTB3 stays as the scalar fallback for
# backward compat when the curve fetch fails.
TERM_STRUCTURE_SERIES = [
    ("DGS1MO", 30),
    ("DGS3MO", 91),
    ("DGS6MO", 182),
    ("DGS1",   365),
]


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


def interpolate_rate(curve, dte_days, fallback: float = DEFAULT_RISK_FREE_RATE) -> float:
    """
    Linearly interpolate a risk-free rate from a term-structure curve.

    Parameters:
        curve:     dict mapping DTE in days (int/float) → rate (float).
                   Can be None or empty, in which case `fallback` is returned.
        dte_days:  target DTE in calendar days. Values below the curve's min
                   DTE clamp to the shortest point; values above the max DTE
                   clamp to the longest point (no extrapolation — a flat
                   extension is safer than a linear extrapolation of a
                   nearly-flat yield curve).
        fallback:  value returned when the curve is unusable.

    Returns:
        A scalar float rate (decimal, e.g. 0.043 for 4.30%).

    Flat-rate back-compat: if `curve` is itself a numeric, it's returned
    directly so callers can pass a single rate without branching.
    """
    if isinstance(curve, (int, float)):
        return float(curve)
    if not curve:
        return float(fallback)

    try:
        dte_days = float(dte_days)
    except (TypeError, ValueError):
        return float(fallback)

    points = sorted(((float(k), float(v)) for k, v in curve.items()), key=lambda x: x[0])
    if not points:
        return float(fallback)

    # Clamp below / above the curve
    if dte_days <= points[0][0]:
        return points[0][1]
    if dte_days >= points[-1][0]:
        return points[-1][1]

    # Linear interpolation between the two bracketing points
    for i in range(len(points) - 1):
        lo_dte, lo_rate = points[i]
        hi_dte, hi_rate = points[i + 1]
        if lo_dte <= dte_days <= hi_dte:
            if hi_dte == lo_dte:
                return lo_rate
            frac = (dte_days - lo_dte) / (hi_dte - lo_dte)
            return lo_rate + frac * (hi_rate - lo_rate)

    return points[-1][1]  # shouldn't reach here


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


def _fetch_fred_series_latest(fred_api_key: str, series_id: str) -> float | None:
    """
    Fetch the most recent non-missing observation for a single FRED series.
    Returns the rate as a decimal (e.g. 0.043) or None on any failure.
    Used to build the term-structure curve in fetch_risk_free_rate_curve().
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": fred_api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        if r.status_code != 200:
            return None
        obs = r.json().get("observations", [])
        for o in obs:
            val = o.get("value", ".")
            if val != ".":
                return float(val) / 100
    except Exception:
        pass
    return None


def _fetch_fred_curve(fred_api_key: str) -> dict | None:
    """
    Fetch the full short-end Treasury term structure from FRED constant-
    maturity series (DGS1MO, DGS3MO, DGS6MO, DGS1).

    Returns a dict {dte_days: rate} with whichever points succeeded, or
    None if nothing was fetched. Callers pass this into interpolate_rate()
    to look up a rate per-expiration.

    We need at least two points for interpolation to be meaningful; if
    only one succeeds the caller should treat that as a flat curve or
    fall back to the single DTB3 scalar.
    """
    curve = {}
    for series_id, dte_days in TERM_STRUCTURE_SERIES:
        rate = _fetch_fred_series_latest(fred_api_key, series_id)
        if rate is not None:
            curve[dte_days] = rate
    return curve if curve else None


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
    Fetch current risk-free rate with term-structure curve when possible.

    Tries, in order:
      1. FRED DTB3 (3M T-bill) for the scalar rate. This stays as the
         primary scalar so existing callers that only want a single rate
         keep working unchanged.
      2. FRED constant-maturity curve (DGS1MO / DGS3MO / DGS6MO / DGS1)
         to populate a term structure for per-tenor interpolation. This
         is used by GEX / zero-gamma / parity to apply the right rate to
         each expiration instead of the flat 3M scalar. Even if one or
         two points fail, whatever comes back is used.
      3. If FRED is entirely unavailable: Treasury fiscal API (scalar),
         then disk cache, then DEFAULT_RISK_FREE_RATE.

    Returns a dict:
    {
        "rate":   float — scalar 3M for flat-rate back-compat,
        "source": str,
        "label":  str,
        "as_of":  "YYYY-MM-DD" or None,
        "curve":  dict {dte_days: rate} or None — multi-point curve
                  for interpolate_rate(). Callers that want the
                  tenor-aware rate should use this; None means fall
                  through to the scalar `rate`.
    }
    """
    import time

    has_fred_key = bool(fred_api_key and fred_api_key != "YOUR_FRED_KEY_HERE")

    # Try FRED first (with retry to handle rate limits from parallel matrix jobs)
    if has_fred_key:
        for attempt in range(3):
            try:
                parsed = _fetch_fred(fred_api_key)
                if parsed is not None:
                    # Scalar succeeded — try to fetch the curve alongside.
                    # Non-fatal: curve is a bonus, scalar is the primary.
                    try:
                        curve = _fetch_fred_curve(fred_api_key)
                    except Exception as e:
                        if debug:
                            print(f"  FRED curve fetch failed (non-fatal): {e}")
                        curve = None
                    parsed["curve"] = curve
                    if curve:
                        curve_summary = ", ".join(
                            f"{int(k)}d={v*100:.2f}%" for k, v in sorted(curve.items())
                        )
                        print(f"  Risk-free curve: {curve_summary}")
                    print(f"  Risk-free rate: {parsed['rate']*100:.2f}% ({parsed['label']})")
                    _write_rate_cache(parsed)
                    return parsed
            except Exception as e:
                if debug:
                    print(f"  FRED attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2 * (attempt + 1))  # 2s, 4s backoff

    # Fall back to Treasury only if FRED is unavailable
    try:
        parsed = _fetch_treasury()
        if parsed is not None:
            parsed["curve"] = None  # Treasury fallback is scalar-only
            print(f"  Risk-free rate: {parsed['rate']*100:.2f}% ({parsed['label']})")
            _write_rate_cache(parsed)
            return parsed
    except Exception as e:
        if debug:
            print(f"  Treasury API error: {e}")

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
            "curve": cached.get("curve"),
        }

    print(f"  Risk-free rate: {DEFAULT_RISK_FREE_RATE*100:.2f}% (fallback default — no cache available)")
    return {
        "rate": DEFAULT_RISK_FREE_RATE,
        "source": "fallback_default",
        "label": f"fallback default {DEFAULT_RISK_FREE_RATE*100:.2f}%",
        "as_of": None,
        "curve": None,
    }
