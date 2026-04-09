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
) -> dict | None:
    """
    Build overnight futures context from ES data (Yahoo or manual).

    Parameters:
        es_last:        Current ES price
        es_high:        Overnight session high
        es_low:         Overnight session low
        spx_prevclose:  Yesterday's SPX close (from vendor quote)
        source:         "yahoo_es_f" or "manual"

    Returns a dict with overnight move, range info, and budget metrics,
    or None if insufficient data.
    """
    if es_last is None or es_last <= 0 or spx_prevclose <= 0:
        return None

    # ES trades at a small premium/discount to SPX, but the *move* is
    # what matters — compute the ES move in points and apply it to SPX.
    overnight_move_pts = es_last - spx_prevclose
    overnight_move_pct = overnight_move_pts / spx_prevclose * 100

    result = {
        "es_last": round(es_last, 2),
        "es_high": round(es_high, 2) if es_high else None,
        "es_low": round(es_low, 2) if es_low else None,
        "spx_prevclose": round(spx_prevclose, 2),
        "overnight_move_pts": round(overnight_move_pts, 2),
        "overnight_move_pct": round(overnight_move_pct, 3),
        "direction": "up" if overnight_move_pts > 0 else "down" if overnight_move_pts < 0 else "flat",
        "source": source,
    }

    # Overnight range context (if high/low available)
    if es_high and es_low and es_high > 0 and es_low > 0:
        range_pts = es_high - es_low
        # Max move from prevclose (whichever extreme was further)
        max_move_up = es_high - spx_prevclose
        max_move_down = spx_prevclose - es_low
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
