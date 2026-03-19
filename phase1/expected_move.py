from __future__ import annotations

import math
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
            "Small overnight move + positive gamma. Dealers hedge against movement. "
            "Price tends to oscillate around major strikes. Volatility sellers often do well."
        ),
        "bias": "range-bound",
        "favored_strategies": ["premium selling", "iron condors", "iron butterflies"],
    },
    ("low", "negative"):  {
        "label": "Trend Day",
        "description": (
            "Small overnight move + negative gamma. Volatility budget is intact and dealer "
            "hedging reinforces directional moves. Higher probability of sustained trend."
        ),
        "bias": "directional",
        "favored_strategies": ["directional spreads", "trend following", "debit spreads"],
    },
    ("high", "positive"): {
        "label": "Exhaustion Day",
        "description": (
            "Large overnight move + positive gamma. Most of the volatility budget is used. "
            "Dealer hedging dampens further moves. Often produces tight ranges or pinning after the open."
        ),
        "bias": "mean-revert",
        "favored_strategies": ["premium selling", "fade extremes", "short strangles"],
    },
    ("high", "negative"): {
        "label": "Extension Day",
        "description": (
            "Large overnight move + negative gamma. Despite spending much of the volatility budget, "
            "dealer hedging amplifies movement. The session can still trend further, but manage risk carefully."
        ),
        "bias": "continued-trend",
        "favored_strategies": ["directional with tight stops", "protect existing positions"],
    },
}

# Thresholds for move ratio classification
MOVE_RATIO_LOW_THRESHOLD = 0.40   # below 40% of EM = "low" overnight move
MOVE_RATIO_HIGH_THRESHOLD = 0.70  # above 70% of EM = "high" overnight move


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
            "favored_strategies": [],
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
        gamma_bucket = "negative"  # conservative: assume more volatile

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
            "favored_strategies": ["wait for confirmation", "smaller position sizes"],
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
        "favored_strategies": info.get("favored_strategies", []),
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
) -> dict:
    """
    Full expected-move analysis combining all signals.

    Parameters:
        spot:           Current reference spot (parity-implied or Tradier)
        prev_close:     SPX previous close from Tradier quote
        zero_gamma:     Zero-gamma level from GEX engine
        gamma_regime:   "Positive Gamma" / "Negative Gamma" / "At Zero Gamma"
        calls_0dte:     Call chain for the 0DTE expiration
        puts_0dte:      Put chain for the 0DTE expiration
        spy_quote:      Optional SPY full quote for pre-market proxy
        market_open:    Whether the cash market is currently open

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
            # Scale SPY move to SPX points
            implied_spx_move = spot * spy_move_pct / 100
            spy_overnight = {
                "spy_price": round(spy_current, 2),
                "spy_prevclose": round(spy_quote["prevclose"], 2),
                "spy_move_pct": round(spy_move_pct, 3),
                "implied_spx_move_pts": round(implied_spx_move, 2),
                "source": "spy_premarket_proxy",
            }

    # 4. Determine the best overnight move for classification
    #    Pre-market: SPX spot == prevclose → overnight move is 0 (stale).
    #    Use SPY proxy as the real signal if available.
    #    After hours: SPX move is today's realized move (still useful but retrospective).
    classification_move_pts = overnight.get("overnight_move_pts")
    classification_source = "spx"

    spx_move_is_stale = (
        not market_open
        and classification_move_pts is not None
        and abs(classification_move_pts) < 0.5  # essentially zero
    )

    if spx_move_is_stale and spy_overnight is not None:
        # Pre-market: SPX hasn't moved, use SPY proxy
        classification_move_pts = spy_overnight["implied_spx_move_pts"]
        classification_source = "spy_proxy"
    elif not market_open and classification_move_pts is not None and abs(classification_move_pts) > 0.5:
        # After hours: SPX move is the full day's realized move
        classification_source = "spx_realized"

    # 5. Session classification
    classification = classify_session(
        expected_move_pts=em_info["expected_move_pts"],
        overnight_move_pts=classification_move_pts,
        gamma_regime=gamma_regime,
    )
    classification["move_source"] = classification_source

    # 6. Market context
    if market_open:
        market_context = "live"
        context_note = None
    elif spx_move_is_stale:
        market_context = "premarket"
        context_note = (
            "Pre-market — SPX is not trading. Overnight move is estimated from SPY pre-market data. "
            "The ATM straddle reflects the nearest available chain. "
            "Re-check after the open for live classification."
        )
    else:
        market_context = "afterhours"
        context_note = (
            "After hours — the overnight move reflects today's full realized session move, "
            "not the overnight gap. The classification is retrospective. "
            "Run again tomorrow morning before the open for a forward-looking signal."
        )

    # 7. Expected move levels relative to key GEX levels
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
        "classification": classification,
        "level_context": level_context,
        "spot": round(spot, 2),
        "prev_close": round(prev_close, 2),
        "gamma_regime": gamma_regime,
        "market_context": market_context,
        "context_note": context_note,
    }
