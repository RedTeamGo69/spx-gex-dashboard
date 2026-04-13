"""
ES Futures data fetcher.

Primary source: Yahoo Finance (ES=F) — ~10 min delayed, free, no API key.
Fallback: manual input via Streamlit sidebar.
"""
from __future__ import annotations


def fetch_es_from_yahoo() -> dict | None:
    """
    Fetch current ES futures quote from Yahoo Finance via yfinance.

    Returns dict with last, high, low, prevclose, change, or None on failure.
    The high/low are the current session's high/low (overnight session for
    pre-market use).

    Note: ~10 minute delay on futures data from Yahoo/CME licensing.
    """
    try:
        import yfinance as yf

        es = yf.Ticker("ES=F")
        info = es.info

        if not info or info.get("regularMarketPrice") is None:
            return None

        last = float(info.get("regularMarketPrice", 0))
        if last <= 0:
            return None

        prevclose = float(info.get("regularMarketPreviousClose") or info.get("previousClose") or 0)
        high = float(info.get("regularMarketDayHigh") or info.get("dayHigh") or 0)
        low = float(info.get("regularMarketDayLow") or info.get("dayLow") or 0)

        return {
            "last": round(last, 2),
            "prevclose": round(prevclose, 2),
            "high": round(high, 2) if high > 0 else None,
            "low": round(low, 2) if low > 0 else None,
            "change": round(last - prevclose, 2) if prevclose > 0 else None,
            "change_pct": round((last - prevclose) / prevclose * 100, 3) if prevclose > 0 else None,
            "source": "yahoo_es_f",
            "note": "~10 min delayed (CME via Yahoo Finance)",
        }

    except Exception as e:
        print(f"  Yahoo ES fetch failed: {e}")
        return None


def build_futures_context(
    es_last: float | None,
    es_high: float | None,
    es_low: float | None,
    spx_prevclose: float,
    source: str = "manual",
    es_prevclose: float | None = None,
) -> dict | None:
    """
    Build overnight futures context from ES data (Yahoo or manual).

    Parameters:
        es_last:        Current ES price
        es_high:        Overnight session high
        es_low:         Overnight session low
        spx_prevclose:  Yesterday's SPX cash close (from Tradier)
        source:         "yahoo_es_f" or "manual"
        es_prevclose:   Yesterday's ES settle/close, if known. When provided,
                        the overnight move is computed as es_last - es_prevclose
                        which cancels the ES-vs-SPX basis (typically 1-5 pts from
                        dividends/carry). When None (e.g. manual entry without
                        prior ES close), falls back to es_last - spx_prevclose
                        which carries the basis as a systematic bias. Yahoo
                        auto-fetch always provides es_prevclose.

    Returns a dict with overnight move, range info, and budget metrics,
    or None if insufficient data.
    """
    if es_last is None or es_last <= 0 or spx_prevclose <= 0:
        return None

    # Basis-clean path: use ES's own prior close so es_last - es_prevclose is
    # a pure ES move with no cross-instrument contamination. Since ES ≈ SPX
    # for moves (they move together in points), we use this as the SPX-
    # equivalent overnight move. spx_prevclose is still used below for the
    # percent-change normalization (it's the right denominator for "% move
    # relative to SPX level") and for display.
    if es_prevclose is not None and es_prevclose > 0:
        overnight_move_pts = es_last - es_prevclose
        basis_clean = True
    else:
        # Legacy fallback: cross-instrument subtraction. Carries the ES-SPX
        # basis (typically +1 to +5 pts on SPX) as a systematic bias in the
        # reported overnight move. Used only when es_prevclose is unavailable,
        # which today happens only on manual ES entry in the Streamlit UI.
        overnight_move_pts = es_last - spx_prevclose
        basis_clean = False

    overnight_move_pct = overnight_move_pts / spx_prevclose * 100

    result = {
        "es_last": round(es_last, 2),
        "es_high": round(es_high, 2) if es_high else None,
        "es_low": round(es_low, 2) if es_low else None,
        "es_prevclose": round(es_prevclose, 2) if es_prevclose else None,
        "spx_prevclose": round(spx_prevclose, 2),
        "overnight_move_pts": round(overnight_move_pts, 2),
        "overnight_move_pct": round(overnight_move_pct, 3),
        "direction": "up" if overnight_move_pts > 0 else "down" if overnight_move_pts < 0 else "flat",
        "basis_clean": basis_clean,
        "source": source,
    }

    # Overnight range context (if high/low available)
    if es_high and es_low and es_high > 0 and es_low > 0:
        range_pts = es_high - es_low
        # Express high/low as moves from the same reference as overnight_move_pts
        # so they line up with each other and with the move direction.
        ref = es_prevclose if basis_clean else spx_prevclose
        max_move_up = es_high - ref
        max_move_down = ref - es_low
        max_move = max(abs(max_move_up), abs(max_move_down))

        result["overnight_range_pts"] = round(range_pts, 2)
        result["overnight_high_move"] = round(max_move_up, 2)
        result["overnight_low_move"] = round(-max_move_down, 2)
        result["max_overnight_move"] = round(max_move, 2)
    else:
        result["overnight_range_pts"] = None
        result["overnight_high_move"] = None
        result["overnight_low_move"] = None
        result["max_overnight_move"] = None

    return result
