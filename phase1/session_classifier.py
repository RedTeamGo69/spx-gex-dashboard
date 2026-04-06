"""
Session day-type classification based on overnight move ratio and gamma regime.

Classifies sessions as Pin Day, Trend Day, Exhaustion Day, or Extension Day
based on the fraction of expected move consumed overnight and the dealer
gamma positioning.
"""
from __future__ import annotations


# ── Day classification lookup table ────────────────────────────────────────

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
