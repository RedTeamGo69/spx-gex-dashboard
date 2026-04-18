from __future__ import annotations

import calendar as _cal
import math
from datetime import date, timedelta

import numpy as np

from phase1.quote_filters import quote_mid, has_two_sided_quote, is_crossed
from phase1.session_classifier import classify_session


# ── ATM straddle expected move ──────────────────────────────────────────────

def find_atm_straddle(calls: list[dict], puts: list[dict], spot: float):
    """
    Find the at-the-money straddle from 0DTE call and put chains.

    Picks the strike nearest to spot where both the call and put have
    usable two-sided quotes, then computes the straddle price (call mid + put mid).

    Returns a dict with straddle details, or None if no usable pair exists.
    """
    call_by_k = {}
    for c in calls:
        K = c["strike"]
        if has_two_sided_quote(c) and not is_crossed(c):
            call_by_k[K] = c

    put_by_k = {}
    for p in puts:
        K = p["strike"]
        if has_two_sided_quote(p) and not is_crossed(p):
            put_by_k[K] = p

    common_strikes = sorted(set(call_by_k.keys()) & set(put_by_k.keys()))
    if not common_strikes:
        return None

    # Sort by distance from spot
    common_strikes.sort(key=lambda k: abs(k - spot))

    # Try the nearest strikes until we find one with valid mids
    for K in common_strikes[:5]:
        c_mid = quote_mid(call_by_k[K])
        p_mid = quote_mid(put_by_k[K])

        if c_mid is not None and c_mid > 0 and p_mid is not None and p_mid > 0:
            straddle_price = c_mid + p_mid
            return {
                "strike": K,
                "call_mid": round(c_mid, 2),
                "put_mid": round(p_mid, 2),
                "straddle_price": round(straddle_price, 2),
                "call_bid": call_by_k[K].get("bid", 0),
                "call_ask": call_by_k[K].get("ask", 0),
                "put_bid": put_by_k[K].get("bid", 0),
                "put_ask": put_by_k[K].get("ask", 0),
                "distance_from_spot": round(abs(K - spot), 2),
            }

    return None


def compute_expected_move(straddle_info: dict | None, spot: float) -> dict:
    """
    Compute expected move levels from the ATM straddle.

    The expected move ≈ straddle price at ATM. This defines the "volatility
    budget" for the session.

    Returns upper/lower expected move levels and the move magnitude.
    """
    if straddle_info is None or spot <= 0:
        return {
            "expected_move_pts": None,
            "expected_move_pct": None,
            "upper_level": None,
            "lower_level": None,
            "straddle": None,
        }

    em = straddle_info["straddle_price"]
    return {
        "expected_move_pts": round(em, 2),
        "expected_move_pct": round(em / spot * 100, 3),
        "upper_level": round(spot + em, 2),
        "lower_level": round(spot - em, 2),
        "straddle": straddle_info,
    }


# ── Expiration finders for weekly / monthly EM ─────────────────────────────

def find_weekly_expiration(avail_exps: list[str], ref_date: date) -> str | None:
    """Find this Friday's expiration (or nearest weekly within 7 days)."""
    days_to_fri = (4 - ref_date.weekday()) % 7
    if days_to_fri == 0 and ref_date.weekday() != 4:
        days_to_fri = 7  # not actually Friday, wrap around
    friday = (ref_date + timedelta(days=days_to_fri)).strftime("%Y-%m-%d")

    # Exact Friday match
    if friday in avail_exps:
        return friday

    # Fallback: nearest expiration between tomorrow and 7 days out
    today_str = ref_date.strftime("%Y-%m-%d")
    cutoff = (ref_date + timedelta(days=7)).strftime("%Y-%m-%d")
    candidates = sorted(e for e in avail_exps if today_str < e <= cutoff)
    return candidates[-1] if candidates else None  # prefer the furthest in the week


def find_monthly_expiration(avail_exps: list[str], ref_date: date) -> str | None:
    """Find the standard monthly options expiration (3rd Friday of month)."""
    # Compute 3rd Friday of current month
    year, month = ref_date.year, ref_date.month
    first_day_weekday = _cal.weekday(year, month, 1)  # 0=Mon
    # First Friday: day offset from 1st to first Friday
    first_fri = 1 + (4 - first_day_weekday) % 7
    third_fri = first_fri + 14
    third_fri_date = date(year, month, third_fri)

    # If 3rd Friday has passed, use next month
    if third_fri_date < ref_date:
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
        first_day_weekday = _cal.weekday(year, month, 1)
        first_fri = 1 + (4 - first_day_weekday) % 7
        third_fri = first_fri + 14
        third_fri_date = date(year, month, third_fri)

    target = third_fri_date.strftime("%Y-%m-%d")
    if target in avail_exps:
        return target

    # Fallback: nearest available within 5 days of the 3rd Friday
    window_start = (third_fri_date - timedelta(days=5)).strftime("%Y-%m-%d")
    window_end = (third_fri_date + timedelta(days=5)).strftime("%Y-%m-%d")
    candidates = sorted(e for e in avail_exps if window_start <= e <= window_end)
    # Prefer the one closest to the 3rd Friday
    if candidates:
        candidates.sort(key=lambda e: abs((date.fromisoformat(e) - third_fri_date).days))
        return candidates[0]
    return None


def compute_em_for_expiration(client, ticker: str, expiration: str, spot: float) -> dict | None:
    """Compute Expected Move from the ATM straddle of a specific expiration.

    Returns the same dict shape as compute_expected_move(), or None on failure.
    """
    if not expiration:
        return None
    try:
        entry = client.get_chain_cached(ticker, expiration)
        if entry.get("status") != "ok":
            return None
        calls = entry.get("calls", [])
        puts = entry.get("puts", [])
        if not calls or not puts:
            return None
        straddle = find_atm_straddle(calls, puts, spot)
        em = compute_expected_move(straddle, spot)
        if em.get("expected_move_pts") is not None:
            em["expiration"] = expiration
            return em
    except Exception:
        pass
    return None


# ── Overnight move analysis ─────────────────────────────────────────────────

def compute_overnight_move(
    current_price: float,
    prev_close: float,
    source: str = "spx",
) -> dict:
    """
    Compute the overnight move from previous close to the current reference price.

    current_price: live spot (parity-implied or Tradier quote)
    prev_close: yesterday's SPX close from the Tradier quote
    source: label for what provided the current price
    """
    if prev_close <= 0 or current_price <= 0:
        return {
            "overnight_move_pts": None,
            "overnight_move_pct": None,
            "direction": None,
            "source": source,
        }

    move_pts = current_price - prev_close
    move_pct = move_pts / prev_close * 100

    if move_pts > 0:
        direction = "up"
    elif move_pts < 0:
        direction = "down"
    else:
        direction = "flat"

    return {
        "overnight_move_pts": round(move_pts, 2),
        "overnight_move_pct": round(move_pct, 3),
        "direction": direction,
        "source": source,
    }


# ── Full expected-move analysis ─────────────────────────────────────────────

def build_expected_move_analysis(
    spot: float,
    prev_close: float,
    zero_gamma: float,
    gamma_regime: str,
    calls_0dte: list[dict],
    puts_0dte: list[dict],
    market_open: bool = True,
    expiration: str | None = None,
    as_of: date | None = None,
) -> dict:
    """
    Full expected-move analysis combining the ATM straddle, the SPX move vs
    prevclose, and a session classification. Pre-market overnight context
    (SPY proxy, ES futures, overnight range) has been removed.

    Parameters:
        spot:             Current reference spot (parity-implied or Tradier)
        prev_close:       SPX previous close from Tradier quote
        zero_gamma:       Zero-gamma level from GEX engine
        gamma_regime:     "Positive Gamma" / "Negative Gamma" / "At Zero Gamma"
        calls_0dte:       Call chain for the 0DTE expiration (the nearest
                          available expiration — NOT necessarily 0DTE)
        puts_0dte:        Put chain for the same expiration
        market_open:      Whether the cash market is currently open
        expiration:       ISO date string of the expiration the straddle came
                          from ("YYYY-MM-DD"). When provided, the straddle DTE
                          is stored in em_info["straddle"]["dte"] so the UI
                          can label the card correctly — on weekends and
                          after-hours the "0DTE" straddle is actually a 1+ DTE
                          straddle which carries √2-ish more vol than a true
                          same-day straddle would.
        as_of:            Reference date for DTE calculation. Defaults to today.
    """
    # 1. ATM straddle and expected move
    straddle = find_atm_straddle(calls_0dte, puts_0dte, spot)
    em_info = compute_expected_move(straddle, spot)

    # Annotate the straddle with its actual DTE so the UI knows whether
    # it's a true same-day expected move or a longer-tenor fallback.
    if em_info.get("straddle") and expiration:
        try:
            exp_date = date.fromisoformat(expiration)
            ref = as_of or date.today()
            em_info["straddle"]["dte"] = max((exp_date - ref).days, 0)
            em_info["straddle"]["expiration"] = expiration
        except (ValueError, TypeError):
            pass

    # 2. SPX move from prevclose — live during the session, retrospective
    #    (full realized session move) once the cash market is closed.
    overnight = compute_overnight_move(spot, prev_close, source="spx_vs_prevclose")

    # 3. Session classification — always uses the SPX move.
    classification = classify_session(
        expected_move_pts=em_info["expected_move_pts"],
        overnight_move_pts=overnight.get("overnight_move_pts"),
        gamma_regime=gamma_regime,
    )
    classification["move_source"] = "spx"

    # 4. Market context
    if market_open:
        market_context = "live"
        context_note = None
    else:
        market_context = "afterhours"
        context_note = (
            "Market closed — the SPX move shown is retrospective "
            "(today's full realized session move from prevclose, not a "
            "forward-looking overnight gap). Re-check after the open."
        )

    # 5. Expected move levels relative to key GEX levels
    level_context = None
    if em_info["upper_level"] is not None:
        level_context = {
            "em_upper": em_info["upper_level"],
            "em_lower": em_info["lower_level"],
            "zero_gamma": round(zero_gamma, 2),
            "zero_gamma_within_em": (
                em_info["lower_level"] <= zero_gamma <= em_info["upper_level"]
            ),
            "zero_gamma_distance_to_spot": round(spot - zero_gamma, 2),
        }

    return {
        "expected_move": em_info,
        "overnight_move": overnight,
        "classification": classification,
        "level_context": level_context,
        "spot": round(spot, 2),
        "prev_close": round(prev_close, 2),
        "gamma_regime": gamma_regime,
        "market_context": market_context,
        "context_note": context_note,
    }
