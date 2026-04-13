# =============================================================================
# gex_bridge.py
# Bridge between the GEX Dashboard and the Range Finder model.
#
# Takes live GEX data computed by the dashboard (spot, zero_gamma, call_wall,
# put_wall, net GEX, gamma regime) and feeds it into the range finder's
# gex_inputs table and spread buffer logic.
#
# This is the key integration point — the GEX dashboard produces real-time
# gamma exposure levels, and the range finder uses them to:
#   1. Adjust the gex_flag feature in the HAR model (M4_full spec)
#   2. Widen/tighten the spread buffer based on dealer positioning
#   3. Use GEX walls as additional guardrails for strike placement
# =============================================================================

import math
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional

from range_finder.data_collector import init_db
from range_finder.feature_builder import (
    init_features_table, create_gex_table, upsert_gex,
    get_features, get_feature_for_week,
)
from range_finder.spread_levels import SpreadPlan

log = logging.getLogger(__name__)


# =============================================================================
# GEX CONTEXT — structured data from the dashboard
# =============================================================================

@dataclass
class GEXContext:
    """Live GEX data extracted from the dashboard for the range finder."""
    spot:          float          # Current SPX spot price
    zero_gamma:    float          # Zero-gamma level (GEX flip point)
    call_wall:     float          # Positive GEX wall (resistance)
    put_wall:      float          # Negative GEX wall (support)
    gamma_regime:  str            # "positive", "negative", or "transition"
    net_gex:       Optional[float] = None  # Net gamma exposure ($ value)
    call_wall_gex: Optional[float] = None  # GEX at call wall
    put_wall_gex:  Optional[float] = None  # GEX at put wall


def extract_gex_context(levels: dict, spot: float, regime_info: dict) -> GEXContext:
    """
    Extract a GEXContext from the dashboard's computed levels and regime info.

    Args:
        levels      : dict from gex_engine.find_key_levels()
        spot        : current SPX spot price
        regime_info : dict from gex_engine.get_gamma_regime_text()
    """
    return GEXContext(
        spot         = spot,
        zero_gamma   = levels.get("zero_gamma", spot),
        call_wall    = levels.get("call_wall", spot + 50),
        put_wall     = levels.get("put_wall", spot - 50),
        gamma_regime = regime_info.get("regime", "unknown"),
        net_gex      = levels.get("net_gex"),
        call_wall_gex= levels.get("call_wall_gex"),
        put_wall_gex = levels.get("put_wall_gex"),
    )


# =============================================================================
# REGIME → GEX FLAG MAPPING
# =============================================================================

def regime_to_gex_flag(regime: str) -> int:
    """
    Map the dashboard's gamma regime string to the range finder's gex_flag integer.

    Dashboard regime values:
        "positive"   → dealers long gamma above zero-gamma, suppress moves → flag = +1
        "negative"   → dealers short gamma below zero-gamma, amplify moves → flag = -1
        "transition" → near zero-gamma, regime unclear                     → flag =  0
    """
    regime_lower = regime.lower().strip()
    if "positive" in regime_lower:
        return 1
    elif "negative" in regime_lower:
        return -1
    return 0


def compute_continuous_gex_features(gex_ctx: GEXContext) -> dict:
    """
    Compute continuous GEX context features for the notes field on
    gex_inputs rows — purely descriptive, NOT fed to the HAR model.

    The HAR model's gex_normalized feature is computed from raw dollar GEX
    in feature_builder.compute_har_features path (see feature_builder.py
    where gex is divided by spx_open²).

    Returns:
        gex_zg_distance_pct: (spot - zero_gamma) / spot — positive = positive gamma
        gex_wall_width_pct:  (call_wall - put_wall) / spot — wider = more room
    """
    spot = gex_ctx.spot
    if spot <= 0:
        return {
            "gex_zg_distance_pct": 0.0,
            "gex_wall_width_pct": 0.0,
        }

    zg_distance = (spot - gex_ctx.zero_gamma) / spot
    wall_width = (gex_ctx.call_wall - gex_ctx.put_wall) / spot

    return {
        "gex_zg_distance_pct": round(float(zg_distance), 6),
        "gex_wall_width_pct": round(float(wall_width), 6),
    }


def regime_to_gex_dollars(gex_ctx: GEXContext) -> float:
    """
    Convert the GEX context to a dollar-denominated GEX value for the
    range finder's gex_inputs table. Uses net_gex if available, otherwise
    derives a synthetic value from the call/put wall GEX magnitudes.
    """
    if gex_ctx.net_gex is not None:
        return gex_ctx.net_gex

    # Synthetic: use call_wall_gex + put_wall_gex as a proxy
    cw = gex_ctx.call_wall_gex or 0
    pw = gex_ctx.put_wall_gex or 0
    # Net GEX ≈ sum of dominant walls (call wall is positive, put wall is negative)
    return cw + pw


# =============================================================================
# SAVE GEX TO RANGE FINDER DB
# =============================================================================

def save_gex_to_range_finder(
    gex_ctx: GEXContext,
    conn = None,
) -> int:
    """
    Persist the live GEX data from the dashboard into the range finder's
    gex_inputs table. Returns the gex_flag value written.

    This is called after each GEX dashboard run so the range finder
    model has fresh GEX data for the current week.
    """
    if conn is None:
        conn = init_db()
        create_gex_table(conn)

    # Determine current week start (Monday)
    today = datetime.now(timezone.utc)
    days_since_monday = today.weekday()
    monday = today - timedelta(days=days_since_monday)
    week_start = monday.strftime("%Y-%m-%d")

    gex_dollars = regime_to_gex_dollars(gex_ctx)
    gex_flag = regime_to_gex_flag(gex_ctx.gamma_regime)
    continuous = compute_continuous_gex_features(gex_ctx)

    notes = (
        f"regime={gex_ctx.gamma_regime} | "
        f"zero_gamma={gex_ctx.zero_gamma:.0f} | "
        f"call_wall={gex_ctx.call_wall:.0f} | "
        f"put_wall={gex_ctx.put_wall:.0f} | "
        f"spot={gex_ctx.spot:.2f} | "
        f"zg_dist={continuous['gex_zg_distance_pct']:.4f} | "
        f"wall_w={continuous['gex_wall_width_pct']:.4f}"
    )

    upsert_gex(conn, week_start, gex_dollars, notes=notes)

    log.info(
        f"GEX bridge: saved for {week_start} — "
        f"gex=${gex_dollars:,.0f}, flag={gex_flag}, regime={gex_ctx.gamma_regime}"
    )
    return gex_flag


# =============================================================================
# GEX-ENHANCED SPREAD ADJUSTMENT
# =============================================================================

def adjust_spread_with_gex(
    plan: SpreadPlan,
    gex_ctx: GEXContext,
) -> dict:
    """
    Produce GEX-enhanced annotations for the spread plan.

    Uses the GEX walls (call_wall, put_wall) as additional reference points
    beyond the HAR model's statistical range. This gives the trader two
    perspectives: statistical (model) and microstructural (GEX).

    Returns a dict with:
        gex_call_wall_vs_short : how far the call wall is from the call short strike
        gex_put_wall_vs_short  : how far the put wall is from the put short strike
        call_strike_inside_wall: True if call short is inside the call wall (safer)
        put_strike_inside_wall : True if put short is inside the put wall (safer)
        gex_regime_label       : human-readable regime description
        gex_adjustment_notes   : list of actionable notes
    """
    call_short = plan.call_spreads[0].short_strike if plan.call_spreads else None
    put_short  = plan.put_spreads[0].short_strike  if plan.put_spreads  else None

    result = {
        "gex_call_wall":        gex_ctx.call_wall,
        "gex_put_wall":         gex_ctx.put_wall,
        "gex_zero_gamma":       gex_ctx.zero_gamma,
        "gex_regime":           gex_ctx.gamma_regime,
        "gex_regime_flag":      regime_to_gex_flag(gex_ctx.gamma_regime),
        "gex_adjustment_notes": [],
    }

    notes = result["gex_adjustment_notes"]

    if call_short is not None:
        dist = gex_ctx.call_wall - call_short
        result["gex_call_wall_vs_short"] = round(dist, 1)
        result["call_strike_inside_wall"] = call_short < gex_ctx.call_wall

        if call_short >= gex_ctx.call_wall:
            notes.append(
                f"Call short ({call_short:.0f}) is AT or ABOVE the call wall "
                f"({gex_ctx.call_wall:.0f}) — dealers may pin here, reducing breach risk"
            )
        elif dist / gex_ctx.spot < 0.004:  # ~0.4% of spot (~20pts SPX, ~2pts XSP)
            notes.append(
                f"Call short ({call_short:.0f}) is only {dist:.0f} pts below "
                f"call wall ({gex_ctx.call_wall:.0f}) — consider widening"
            )

    if put_short is not None:
        dist = put_short - gex_ctx.put_wall
        result["gex_put_wall_vs_short"] = round(dist, 1)
        result["put_strike_inside_wall"] = put_short > gex_ctx.put_wall

        if put_short <= gex_ctx.put_wall:
            notes.append(
                f"Put short ({put_short:.0f}) is AT or BELOW the put wall "
                f"({gex_ctx.put_wall:.0f}) — dealers may pin here, reducing breach risk"
            )
        elif dist / gex_ctx.spot < 0.004:  # ~0.4% of spot
            notes.append(
                f"Put short ({put_short:.0f}) is only {dist:.0f} pts above "
                f"put wall ({gex_ctx.put_wall:.0f}) — consider widening"
            )

    # Regime-specific notes
    if gex_ctx.gamma_regime.lower() == "positive":
        notes.append(
            "Positive GEX — dealer hedging suppresses moves. "
            "Spread buffer tightened proportionally to GEX magnitude."
        )
    elif gex_ctx.gamma_regime.lower() == "negative":
        notes.append(
            "Negative GEX — dealer hedging amplifies moves. "
            "Spread buffer widened proportionally to GEX magnitude. Exercise caution on position sizing."
        )

    # Zero-gamma proximity warning
    zg_dist = abs(gex_ctx.spot - gex_ctx.zero_gamma)
    zg_pct = zg_dist / gex_ctx.spot * 100
    if zg_pct < 0.5:
        notes.append(
            f"Spot ({gex_ctx.spot:.0f}) is very close to zero-gamma "
            f"({gex_ctx.zero_gamma:.0f}, {zg_pct:.2f}% away) — "
            f"regime could flip intraday. Monitor closely."
        )

    return result
