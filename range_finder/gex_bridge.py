# =============================================================================
# gex_bridge.py
# Bridge between the GEX Dashboard and the Range Finder model.
#
# Takes live GEX data computed by the dashboard (spot, zero_gamma, call_wall,
# put_wall, net GEX, gamma regime) and feeds it into research persistence and
# read-only Spread Finder context.
#
# This is the key integration point — the GEX dashboard produces real-time
# gamma exposure levels, and the range finder uses them to:
#   1. Preserve GEX inputs for pending historical calibration
#   2. Display regimes, walls, and zero-gamma as research diagnostics
#   3. Keep GEX separate from live forecasts, buffers, and strikes (BUG-06)
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
from range_finder.spread_levels import SpreadPlan, GEX_NEGATIVE_REGIME_WARNING

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

    Coerces missing / None wall values to a ±1% fallback so that downstream
    formatting code (which does ``f"{gex_ctx.call_wall:.0f}"``) doesn't
    crash with TypeError on force-mode / empty-chain paths where the
    upstream dict intentionally passes None.
    """
    def _coalesce(value, fallback):
        return fallback if value is None else value

    return GEXContext(
        spot         = spot,
        zero_gamma   = _coalesce(levels.get("zero_gamma"), spot),
        call_wall    = _coalesce(levels.get("call_wall"), round(spot * 1.01, 2)),
        put_wall     = _coalesce(levels.get("put_wall"),  round(spot * 0.99, 2)),
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
    ticker: str = "SPX",
) -> int:
    """
    Persist the live GEX data from the dashboard into the range finder's
    gex_inputs table. Returns the gex_flag value written.

    gex_inputs is keyed on (week_start, ticker), so XSP and SPX runs
    write to distinct rows and don't stomp on each other. The HAR
    feature builder only consumes SPX rows (see load_gex_inputs),
    but we still persist XSP rows for audit/debug purposes.

    This is called after each GEX dashboard run so the range finder
    model has fresh GEX data for the current week.
    """
    if conn is None:
        conn = init_db()
        create_gex_table(conn)

    # Determine the PLANNING week's Monday — on the EXCHANGE clock, not UTC.
    # A Sunday-19:00-ET run is already Monday in UTC part of the year (and a
    # Friday-21:00-ET run is UTC Saturday), so UTC bucketing filed weekend
    # GEX saves under the wrong week permanently. ET weekday is authoritative;
    # weekend runs (and Friday after the 16:00 close) describe positioning
    # INTO the upcoming week, matching the Spread Finder's Fri-Sun
    # next-week planning convention.
    from phase1.config import NY_TZ
    now_et = datetime.now(timezone.utc).astimezone(NY_TZ)
    wd = now_et.weekday()
    if wd >= 5 or (wd == 4 and now_et.hour >= 16):
        monday = now_et + timedelta(days=7 - wd)     # upcoming Monday
    else:
        monday = now_et - timedelta(days=wd)         # this week's Monday
    week_start = monday.strftime("%Y-%m-%d")

    gex_dollars = regime_to_gex_dollars(gex_ctx)
    gex_flag = regime_to_gex_flag(gex_ctx.gamma_regime)
    continuous = compute_continuous_gex_features(gex_ctx)

    notes = (
        f"ticker={ticker} | "
        f"regime={gex_ctx.gamma_regime} | "
        f"zero_gamma={gex_ctx.zero_gamma:.0f} | "
        f"call_wall={gex_ctx.call_wall:.0f} | "
        f"put_wall={gex_ctx.put_wall:.0f} | "
        f"spot={gex_ctx.spot:.2f} | "
        f"zg_dist={continuous['gex_zg_distance_pct']:.4f} | "
        f"wall_w={continuous['gex_wall_width_pct']:.4f}"
    )

    upsert_gex(conn, week_start, gex_dollars, notes=notes, ticker=ticker)

    log.info(
        f"GEX bridge: saved for {week_start} {ticker} — "
        f"gex=${gex_dollars:,.0f}, flag={gex_flag}, regime={gex_ctx.gamma_regime}"
    )
    return gex_flag


# =============================================================================
# GEX RESEARCH ANNOTATIONS
# =============================================================================

def adjust_spread_with_gex(
    plan: SpreadPlan,
    gex_ctx: GEXContext,
) -> dict:
    """
    Produce research-only GEX annotations for the spread plan.

    Uses the GEX walls (call_wall, put_wall) as additional reference points
    beyond the HAR model's statistical range. These diagnostics never mutate
    the plan, its buffer, or its strikes.

    Returns a dict with:
        gex_call_wall_vs_short : how far the call wall is from the call short strike
        gex_put_wall_vs_short  : how far the put wall is from the put short strike
        call_strike_inside_wall: True if call short is inside the call wall (safer)
        put_strike_inside_wall : True if put short is inside the put wall (safer)
        gex_regime_label       : human-readable regime description
        gex_adjustment_notes   : wall-proximity / zero-gamma notes only — the
                                 regime narrative is composed separately by
                                 reconcile_gex_warnings()
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
                f"({gex_ctx.call_wall:.0f}) — research context only; live strikes unchanged"
            )
        elif dist / gex_ctx.spot < 0.004:  # ~0.4% of spot (~20pts SPX, ~2pts XSP)
            notes.append(
                f"Call short ({call_short:.0f}) is only {dist:.0f} pts below "
                f"call wall ({gex_ctx.call_wall:.0f}) — research proximity flag; "
                "live strikes unchanged"
            )

    if put_short is not None:
        dist = put_short - gex_ctx.put_wall
        result["gex_put_wall_vs_short"] = round(dist, 1)
        result["put_strike_inside_wall"] = put_short > gex_ctx.put_wall

        if put_short <= gex_ctx.put_wall:
            notes.append(
                f"Put short ({put_short:.0f}) is AT or BELOW the put wall "
                f"({gex_ctx.put_wall:.0f}) — research context only; live strikes unchanged"
            )
        elif dist / gex_ctx.spot < 0.004:  # ~0.4% of spot
            notes.append(
                f"Put short ({put_short:.0f}) is only {dist:.0f} pts above "
                f"put wall ({gex_ctx.put_wall:.0f}) — research proximity flag; "
                "live strikes unchanged"
            )

    # Regime-story notes remain annotation-only. The single coherent regime
    # message (anchored vs live, including flips) is composed by
    # reconcile_gex_warnings() below; callers pass it this dict's gex_regime.

    # Zero-gamma proximity warning
    zg_dist = abs(gex_ctx.spot - gex_ctx.zero_gamma)
    zg_pct = zg_dist / gex_ctx.spot * 100
    if zg_pct < 0.5:
        notes.append(
            f"Spot ({gex_ctx.spot:.0f}) is very close to zero-gamma "
            f"({gex_ctx.zero_gamma:.0f}, {zg_pct:.2f}% away) — "
            "research context only; the live plan is unchanged."
        )

    return result


# =============================================================================
# GEX WARNING RECONCILIATION
# =============================================================================

# Stable prefixes of every message reconcile_gex_warnings can compose. Used to
# strip previously-composed messages so a second application is idempotent
# (a fragment rerun re-reconciling an already-reconciled warning list must not
# stack a second regime message).
_GEX_COMPOSED_PREFIXES = (
    "Positive GEX regime —",
    "Negative GEX regime —",   # also covers the legacy GEX_NEGATIVE_REGIME_WARNING
    "GEX regime flipped since the anchor:",
    "Anchored GEX regime was neutral",
    "Live GEX regime is",
)

_GEX_FLAG_WORD = {1: "positive", -1: "negative"}


def reconcile_gex_warnings(
    plan_warnings: list,
    gex_notes: list,
    anchored_gex_flag: Optional[int],
    live_regime: Optional[str],
    anchor_label: str = "Monday anchor",
) -> list:
    """Merge plan warnings and GEX annotation notes into one coherent list.

    The dashboard has both anchored and live gamma observations. Rendering
    them independently produced contradictory messages whenever the regime
    moved after the anchor. This function composes AT MOST ONE regime message
    and makes the BUG-06 production boundary explicit: GEX is research context
    and is not applied to the live buffer or strikes.

    Pure function (no Streamlit, no I/O) so both the weekly and 0DTE tabs can
    share it and it unit-tests trivially.

    Args:
        plan_warnings     : SpreadPlan.warnings (event/EM/credit warnings and
                            possibly the anchored GEX_NEGATIVE_REGIME_WARNING)
        gex_notes         : adjust_spread_with_gex()["gex_adjustment_notes"]
                            (wall-proximity / zero-gamma notes)
        anchored_gex_flag : SpreadPlan.gex_flag at the anchor (+1/0/-1), or
                            None when the feature row had no GEX data
        live_regime       : GEXContext.gamma_regime from the dashboard
                            (e.g. "Positive Gamma", "negative", "transition")
        anchor_label      : where the anchored features come from, for copy
                            ("Monday anchor" weekly, "session anchor" 0DTE)
    """
    out = [
        w for w in plan_warnings
        if w != GEX_NEGATIVE_REGIME_WARNING
        and not str(w).startswith(_GEX_COMPOSED_PREFIXES)
    ]
    for note in gex_notes:
        if note not in out:
            out.append(note)

    live_flag = regime_to_gex_flag(live_regime) if live_regime else 0
    anchored = anchored_gex_flag if anchored_gex_flag in (-1, 0, 1) else None
    policy_note = (
        "GEX is retained for research and is not applied to the live buffer "
        "or strikes."
    )

    msg = None
    if anchored in (1, -1) and live_flag != 0:
        if anchored == live_flag:
            if anchored == 1:
                msg = (
                    f"Positive GEX regime — anchored ({anchor_label}) and live "
                    f"dashboard agree. {policy_note}"
                )
            else:
                msg = (
                    f"Negative GEX regime — anchored ({anchor_label}) and live "
                    f"dashboard agree. {policy_note}"
                )
        else:
            msg = (
                f"GEX regime flipped since the anchor: {_GEX_FLAG_WORD[anchored]} "
                f"at the {anchor_label}, live dashboard now reads "
                f"{_GEX_FLAG_WORD[live_flag]}. {policy_note}"
            )
    elif anchored == 1:
        msg = (
            f"Positive GEX regime — anchored ({anchor_label}); dealer hedging "
            f"may suppress moves, while live regime is unclear. {policy_note}"
        )
    elif anchored == -1:
        msg = (
            f"Negative GEX regime — anchored ({anchor_label}); dealer hedging "
            f"may amplify moves, while live regime is unclear. {policy_note}"
        )
    elif live_flag != 0 and anchored == 0:
        msg = (
            f"Anchored GEX regime was neutral at the {anchor_label}; live "
            f"dashboard reads {_GEX_FLAG_WORD[live_flag]}. {policy_note}"
        )
    elif live_flag != 0:
        msg = (
            f"Live GEX regime is {_GEX_FLAG_WORD[live_flag]} — research context. "
            f"{policy_note}"
        )

    if msg:
        out.append(msg)
    return out
