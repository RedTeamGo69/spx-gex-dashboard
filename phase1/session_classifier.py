"""
Session day-type classification based on overnight move ratio and gamma regime.

Classifies sessions as Pin Day, Trend Day, Exhaustion Day, or Extension Day
based on the fraction of expected move consumed overnight and the dealer
gamma positioning.

IMPORTANT — calibrated accuracy is modest:
    low bucket  : 55% accuracy (5pp above random)
    high bucket : 73% accuracy
    moderate    : no historical edge
The classifier is a probabilistic lean, not a reliable signal. Every
returned dict carries a numeric ``bucket_accuracy`` field (0..1) and a
``signal_strength`` label ("weak" / "moderate" / "strong") so UIs can
surface the calibrated confidence instead of rendering every cell with
decisive visual weight.
"""
from __future__ import annotations


# ── Bucket accuracy from session_backtest.py (1022 days, 4 years) ─────────
# These are shared across both gamma regimes within a ratio bucket — the
# backtest can only measure accuracy_low / accuracy_high, not the 4 cells
# independently. Populating the dict explicitly so consumers never have to
# recompute them.
_LOW_BUCKET_ACCURACY  = 0.55
_HIGH_BUCKET_ACCURACY = 0.73
_MODERATE_BUCKET_ACCURACY = 0.50  # no measured edge; treated as coin flip

# A signal is "strong" only if calibrated accuracy is meaningfully above
# the 50% random baseline. 65% is the threshold where the kelly edge is
# large enough to be actionable with typical spread-seller risk-reward.
_SIGNAL_STRONG_THRESHOLD = 0.65
_SIGNAL_WEAK_THRESHOLD   = 0.58


def _signal_strength_from_accuracy(acc: float) -> str:
    if acc >= _SIGNAL_STRONG_THRESHOLD:
        return "strong"
    if acc >= _SIGNAL_WEAK_THRESHOLD:
        return "moderate"
    return "weak"


# ── Day classification lookup table ────────────────────────────────────────

_DAY_CLASSIFICATIONS = {
    ("low", "positive"):  {
        "label": "Pin Day",
        "description": (
            "Small overnight move + positive gamma. Dealer hedging may suppress volatility. "
            "Historical lean toward tighter intraday ranges near major strikes, but the "
            "low-bucket classifier is only 55% accurate — treat as a weak tilt, not a setup."
        ),
        "bias": "range-bound",
        "historical_tendencies": [
            "weak historical tilt toward range-bound action (55% bucket accuracy)",
            "dealer hedging may create support/resistance at GEX walls",
            "lower realized vol relative to implied",
        ],
        "confidence_note": (
            "Weak signal (55% historical accuracy, only 5pp above random). "
            "Do not size based on this classification alone."
        ),
        "bucket_accuracy": _LOW_BUCKET_ACCURACY,
    },
    ("low", "negative"):  {
        "label": "Trend Day",
        "description": (
            "Small overnight move + negative gamma. Most of the expected move budget "
            "remains available and dealer hedging can reinforce directional moves. "
            "Historical lean toward wider intraday ranges, but the low-bucket "
            "classifier is only 55% accurate — treat as a weak tilt."
        ),
        "bias": "directional",
        "historical_tendencies": [
            "weak historical tilt toward wider intraday ranges (55% bucket accuracy)",
            "dealer hedging may amplify directional flow",
            "breakouts from the open may have more follow-through",
        ],
        "confidence_note": (
            "Weak signal (55% historical accuracy, only 5pp above random). "
            "Direction is unknown — watch the first 30 min before committing."
        ),
        "bucket_accuracy": _LOW_BUCKET_ACCURACY,
    },
    ("high", "positive"): {
        "label": "Exhaustion Day",
        "description": (
            "Large overnight move + positive gamma. A significant portion of the expected move "
            "has already occurred. Dealer hedging tends to dampen further moves. "
            "High-bucket classifier has 73% historical accuracy — moderate confidence."
        ),
        "bias": "mean-revert",
        "historical_tendencies": [
            "moderate historical tilt toward fading the overnight move (73% bucket accuracy)",
            "reduced intraday range as volatility budget is consumed",
            "price may consolidate near the open or drift back toward prior close",
        ],
        "confidence_note": (
            "Moderate signal (73% historical accuracy). Large overnight gaps can "
            "still extend on news catalysts — this reflects typical behavior, not certainty."
        ),
        "bucket_accuracy": _HIGH_BUCKET_ACCURACY,
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
            "moderate historical tilt toward widest intraday ranges (73% bucket accuracy)",
            "dealer hedging may add fuel to directional moves",
            "risk management is critical — stops and position sizing matter most",
        ],
        "confidence_note": (
            "Moderate signal (73% historical accuracy). Highest-risk session type — "
            "protect capital first regardless of directional read."
        ),
        "bucket_accuracy": _HIGH_BUCKET_ACCURACY,
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
            "at_zero_gamma": False,
            "description": "Insufficient data to classify the session.",
            "bias": None,
            "historical_tendencies": [],
            "confidence_note": "",
            "bucket_accuracy": None,
            "signal_strength": None,
        }

    move_ratio = abs(overnight_move_pts) / expected_move_pts

    if move_ratio < MOVE_RATIO_LOW_THRESHOLD:
        ratio_label = "low"
    elif move_ratio > MOVE_RATIO_HIGH_THRESHOLD:
        ratio_label = "high"
    else:
        ratio_label = "moderate"

    # Map gamma regime to bucket. "At zero gamma" can go either way, so
    # we tie-break to the less volatile assumption (positive) but keep
    # an at_zero_gamma flag so consumers can mute the signal rather
    # than trust an arbitrary coin-flip.
    regime_lower = gamma_regime.lower().strip()
    at_zero_gamma = False
    if "positive" in regime_lower:
        gamma_bucket = "positive"
    elif "negative" in regime_lower:
        gamma_bucket = "negative"
    else:
        gamma_bucket = "positive"
        at_zero_gamma = True

    # At zero gamma the regime could flip either way within the session,
    # so the chosen gamma_bucket is a tie-break, not a read. Downgrade
    # the signal by one step and append a note so UIs can render it muted.
    _ZG_DOWNGRADE = {"strong": "moderate", "moderate": "weak", "weak": "weak"}
    _ZG_NOTE = (
        " Spot is at zero gamma — the regime could flip intraday, so "
        "this classification is a tie-break rather than a read. Muted."
    )

    # Moderate move ratio: blend characteristics
    if ratio_label == "moderate":
        base_key = ("low", gamma_bucket)
        base = _DAY_CLASSIFICATIONS[base_key]
        signal = _signal_strength_from_accuracy(_MODERATE_BUCKET_ACCURACY)
        conf = (
            "No signal (50% historical accuracy — coin flip). "
            "Moderate move ratios have the least predictive value. "
            "Let the first 30 minutes resolve ambiguity."
        )
        if at_zero_gamma:
            signal = _ZG_DOWNGRADE[signal]
            conf = conf + _ZG_NOTE
        return {
            "classification": f"Mixed / {base['label']} Leaning",
            "move_ratio": round(move_ratio, 3),
            "move_ratio_label": ratio_label,
            "gamma_bucket": gamma_bucket,
            "at_zero_gamma": at_zero_gamma,
            "description": (
                f"Overnight move consumed a moderate portion of the expected range "
                f"({move_ratio*100:.0f}%). No historical edge in this bucket — "
                f"the classifier is at the coin-flip baseline. Watch price action "
                f"near gamma levels to confirm direction."
            ),
            "bias": "uncertain",
            "historical_tendencies": [
                "no measured historical edge — classifier is at random baseline",
                "wait for confirmation; reduce position sizes",
            ],
            "confidence_note": conf,
            "bucket_accuracy": _MODERATE_BUCKET_ACCURACY,
            "signal_strength": signal,
        }

    key = (ratio_label, gamma_bucket)
    info = _DAY_CLASSIFICATIONS.get(key, {})
    bucket_acc = info.get("bucket_accuracy", _MODERATE_BUCKET_ACCURACY)

    signal = _signal_strength_from_accuracy(bucket_acc)
    conf = info.get("confidence_note", "")
    if at_zero_gamma:
        signal = _ZG_DOWNGRADE[signal]
        conf = (conf + _ZG_NOTE).strip()

    return {
        "classification": info.get("label", "Unknown"),
        "move_ratio": round(move_ratio, 3),
        "move_ratio_label": ratio_label,
        "gamma_bucket": gamma_bucket,
        "at_zero_gamma": at_zero_gamma,
        "description": info.get("description", ""),
        "bias": info.get("bias"),
        "historical_tendencies": info.get("historical_tendencies", []),
        "confidence_note": conf,
        "bucket_accuracy": bucket_acc,
        "signal_strength": signal,
    }
