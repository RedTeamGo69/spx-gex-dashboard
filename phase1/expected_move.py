from __future__ import annotations

import calendar as _cal
import math
from datetime import date, timedelta

import numpy as np

from phase1.quote_filters import quote_mid, has_two_sided_quote, is_crossed


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


# ── Move ratio and day classification ───────────────────────────────────────

_DAY_CLASSIFICATIONS = {
    ("low", "positive"):  {
        "label": "Pin Day",
        "description": (
            "Small overnight move + positive gamma. Dealer hedging tends to suppress volatility. "
            "Historically correlated with tighter intraday ranges and mean-reverting price action near major strikes."
        ),
        "bias": "range-bound",
        "historical_tendencies": [
            "historically correlated with range-bound, mean-reverting price action",
            "dealer hedging tends to create support/resistance at GEX walls",
            "lower realized vol relative to implied",
        ],
        "confidence_note": "This is a probabilistic tendency, not a guarantee. Confirm with price action at the open.",
    },
    ("low", "negative"):  {
        "label": "Trend Day",
        "description": (
            "Small overnight move + negative gamma. Most of the expected move budget remains available "
            "and dealer hedging can reinforce directional moves. Historically correlated with wider intraday ranges."
        ),
        "bias": "directional",
        "historical_tendencies": [
            "historically correlated with wider intraday ranges and sustained moves",
            "dealer hedging may amplify directional flow",
            "breakouts from the open tend to have more follow-through",
        ],
        "confidence_note": "Trend days are identified probabilistically. The direction is unknown — watch the first 30 min.",
    },
    ("high", "positive"): {
        "label": "Exhaustion Day",
        "description": (
            "Large overnight move + positive gamma. A significant portion of the expected move "
            "has already occurred. Dealer hedging tends to dampen further moves. "
            "Historically correlated with tighter ranges after the open."
        ),
        "bias": "mean-revert",
        "historical_tendencies": [
            "historically correlated with fading of the overnight move",
            "reduced intraday range as volatility budget is consumed",
            "price may consolidate near the open or drift back toward prior close",
        ],
        "confidence_note": "Large overnight gaps can extend further on news catalysts. This classification reflects typical behavior, not certainty.",
    },
    ("high", "negative"): {
        "label": "Extension Day",
        "description": (
            "Large overnight move + negative gamma. Despite consuming much of the expected move, "
            "dealer hedging can amplify further movement. Historically the most volatile session type. "
            "Exercise caution with position sizing."
        ),
        "bias": "continued-trend",
        "historical_tendencies": [
            "historically associated with the widest intraday ranges",
            "dealer hedging may add fuel to directional moves",
            "risk management is critical — stops and position sizing matter most",
        ],
        "confidence_note": "This is the highest-risk session type. Protect capital first.",
    },
}

# Thresholds for move ratio classification.
# Calibrated via: python -m range_finder.session_backtest (1022 days, 4 years)
# Best combined_score=0.2498, accuracy_low=55%, accuracy_high=73%
MOVE_RATIO_LOW_THRESHOLD = 0.30   # below 30% of EM = "low" overnight move  (63% of days)
MOVE_RATIO_HIGH_THRESHOLD = 0.85  # above 85% of EM = "high" overnight move (4% of days)


def classify_session(
    expected_move_pts: float | None,
    overnight_move_pts: float | None,
    gamma_regime: str,
) -> dict:
    """
    Classify the expected session behavior based on:
    - How much of the expected move has been consumed overnight
    - The gamma regime (positive / negative / at zero gamma)

    Returns a classification dict with label, description, bias, and strategies.
    """
    if expected_move_pts is None or overnight_move_pts is None or expected_move_pts <= 0:
        return {
            "classification": None,
            "move_ratio": None,
            "move_ratio_label": None,
            "gamma_bucket": None,
            "description": "Insufficient data to classify the session.",
            "bias": None,
            "historical_tendencies": [],
            "confidence_note": "",
        }

    move_ratio = abs(overnight_move_pts) / expected_move_pts

    if move_ratio < MOVE_RATIO_LOW_THRESHOLD:
        ratio_label = "low"
    elif move_ratio > MOVE_RATIO_HIGH_THRESHOLD:
        ratio_label = "high"
    else:
        ratio_label = "moderate"

    # Map gamma regime to bucket
    regime_lower = gamma_regime.lower().strip()
    if "positive" in regime_lower:
        gamma_bucket = "positive"
    elif "negative" in regime_lower:
        gamma_bucket = "negative"
    else:
        # At zero gamma — could go either way; treat as a blend
        gamma_bucket = "positive"  # at zero gamma: default to less volatile assumption

    # Moderate move ratio: blend characteristics
    if ratio_label == "moderate":
        base_key = ("low", gamma_bucket)
        base = _DAY_CLASSIFICATIONS[base_key]
        return {
            "classification": f"Mixed / {base['label']} Leaning",
            "move_ratio": round(move_ratio, 3),
            "move_ratio_label": ratio_label,
            "gamma_bucket": gamma_bucket,
            "description": (
                f"Overnight move consumed a moderate portion of the expected range "
                f"({move_ratio*100:.0f}%). The session could go either way — watch price "
                f"action near gamma levels to confirm direction."
            ),
            "bias": "uncertain",
            "historical_tendencies": ["no clear historical edge — wait for confirmation", "reduce position sizes"],
            "confidence_note": "Moderate move ratios have the least predictive value. Let the first 30 minutes resolve ambiguity.",
        }

    key = (ratio_label, gamma_bucket)
    info = _DAY_CLASSIFICATIONS.get(key, {})

    return {
        "classification": info.get("label", "Unknown"),
        "move_ratio": round(move_ratio, 3),
        "move_ratio_label": ratio_label,
        "gamma_bucket": gamma_bucket,
        "description": info.get("description", ""),
        "bias": info.get("bias"),
        "historical_tendencies": info.get("historical_tendencies", []),
        "confidence_note": info.get("confidence_note", ""),
    }


# ── Full expected-move analysis ─────────────────────────────────────────────

def build_expected_move_analysis(
    spot: float,
    prev_close: float,
    zero_gamma: float,
    gamma_regime: str,
    calls_0dte: list[dict],
    puts_0dte: list[dict],
    spy_quote: dict | None = None,
    market_open: bool = True,
    futures_context: dict | None = None,
) -> dict:
    """
    Full expected-move analysis combining all signals.

    Parameters:
        spot:             Current reference spot (parity-implied or Tradier)
        prev_close:       SPX previous close from Tradier quote
        zero_gamma:       Zero-gamma level from GEX engine
        gamma_regime:     "Positive Gamma" / "Negative Gamma" / "At Zero Gamma"
        calls_0dte:       Call chain for the 0DTE expiration
        puts_0dte:        Put chain for the 0DTE expiration
        spy_quote:        Optional SPY full quote for pre-market proxy
        market_open:      Whether the cash market is currently open
        futures_context:  Optional dict from futures_data.build_futures_context()

    Returns a comprehensive analysis dict.
    """
    # 1. ATM straddle and expected move
    straddle = find_atm_straddle(calls_0dte, puts_0dte, spot)
    em_info = compute_expected_move(straddle, spot)

    # 2. Overnight move — primary from SPX prevclose
    overnight = compute_overnight_move(spot, prev_close, source="spx_vs_prevclose")

    # 3. SPY pre-market proxy (if available)
    spy_overnight = None
    if spy_quote is not None and spy_quote.get("prevclose", 0) > 0:
        spy_current = spy_quote.get("last", 0) or spy_quote.get("bid", 0)
        if spy_current > 0:
            spy_move_pct = (spy_current - spy_quote["prevclose"]) / spy_quote["prevclose"] * 100
            implied_spx_move = spot * spy_move_pct / 100
            spy_overnight = {
                "spy_price": round(spy_current, 2),
                "spy_prevclose": round(spy_quote["prevclose"], 2),
                "spy_move_pct": round(spy_move_pct, 3),
                "implied_spx_move_pts": round(implied_spx_move, 2),
                "source": "spy_premarket_proxy",
            }

    # 4. Determine the best overnight move for classification
    #    Priority: ES futures > SPY proxy > SPX (when pre-market)
    #    During market hours: SPX is live and primary
    classification_move_pts = overnight.get("overnight_move_pts")
    classification_source = "spx"

    spx_move_is_stale = (
        not market_open
        and classification_move_pts is not None
        and abs(classification_move_pts) < 0.5
    )

    if not market_open and futures_context is not None:
        # ES futures available — best pre-market source
        classification_move_pts = futures_context["overnight_move_pts"]
        classification_source = f"es_futures ({futures_context['source']})"
    elif spx_move_is_stale and spy_overnight is not None:
        classification_move_pts = spy_overnight["implied_spx_move_pts"]
        classification_source = "spy_proxy"
    elif not market_open and classification_move_pts is not None and abs(classification_move_pts) > 0.5:
        classification_source = "spx_realized"

    # 5. Session classification
    classification = classify_session(
        expected_move_pts=em_info["expected_move_pts"],
        overnight_move_pts=classification_move_pts,
        gamma_regime=gamma_regime,
    )
    classification["move_source"] = classification_source

    # 6. Overnight range context (from ES futures)
    overnight_range = None
    if futures_context is not None and futures_context.get("overnight_range_pts") is not None:
        em_pts = em_info.get("expected_move_pts")
        max_move = futures_context["max_overnight_move"]
        overnight_range = {
            "es_high": futures_context["es_high"],
            "es_low": futures_context["es_low"],
            "range_pts": futures_context["overnight_range_pts"],
            "high_move_from_close": futures_context["overnight_high_move"],
            "low_move_from_close": futures_context["overnight_low_move"],
            "max_move_pts": max_move,
            "max_move_vs_em": round(max_move / em_pts, 3) if em_pts and em_pts > 0 else None,
        }

    # 7. Market context
    if market_open:
        market_context = "live"
        context_note = None
    elif futures_context is not None:
        market_context = "premarket"
        src_label = "ES futures (Yahoo, ~10 min delayed)" if "yahoo" in futures_context.get("source", "") else "ES futures (manual)"
        context_note = (
            f"Pre-market — overnight move from {src_label}. "
            "The ATM straddle reflects the nearest available chain. "
            "Re-check after the open for live classification."
        )
    elif spx_move_is_stale:
        market_context = "premarket"
        context_note = (
            "Pre-market — SPX is not trading. No ES futures data available. "
            "Enter ES price manually in the sidebar for accurate classification."
        )
    else:
        market_context = "afterhours"
        context_note = (
            "After hours — the overnight move reflects today's full realized session move, "
            "not the overnight gap. The classification is retrospective. "
            "Run again tomorrow morning before the open for a forward-looking signal."
        )

    # 8. Expected move levels relative to key GEX levels
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
        "spy_proxy": spy_overnight,
        "futures_context": futures_context,
        "overnight_range": overnight_range,
        "classification": classification,
        "level_context": level_context,
        "spot": round(spot, 2),
        "prev_close": round(prev_close, 2),
        "gamma_regime": gamma_regime,
        "market_context": market_context,
        "context_note": context_note,
    }
