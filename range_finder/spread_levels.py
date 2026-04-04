# =============================================================================
# spread_levels.py
# Weekly SPX Range Prediction Model — Spread Level Calculator
#
# Takes the forecast dict from har_model.py and converts it into
# actionable credit spread parameters.
# =============================================================================

import sqlite3
import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from range_finder.data_collector import DB_PATH, init_db

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================

SPX_STRIKE_INCREMENT = 5

DEFAULT_BUFFER_PCT = 0.003

EVENT_BUFFER_MULTIPLIERS = {
    "fomc":  1.50,
    "cpi":   1.35,
    "nfp":   1.20,
    "opex":  1.10,
}

GEX_BUFFER_ADJUSTMENTS = {
    1:    -0.001,   # Positive GEX -> tighten buffer
    0:     0.000,
    -1:   +0.003,   # Negative GEX -> widen buffer
    None:  0.000,
}

# Continuous GEX buffer: scale buffer proportionally to normalized GEX magnitude
# gex_normalized > 0 means positive gamma (tighten), < 0 means negative (widen)
GEX_CONTINUOUS_SCALE = 0.002  # buffer adjustment per unit of gex_normalized

MIN_SPREAD_WIDTH = {
    "normal":     20,
    "event_1":    25,
    "event_2":    30,
    "fomc_week":  35,
}

STANDARD_WING_WIDTHS = [15, 20, 25, 30, 40, 50]

MIN_CREDIT_RATIO = 0.05   # 5% — realistic for far-OTM weekly credit spreads

# Per-ticker configuration — XSP is 1/10th of SPX
TICKER_CONFIG = {
    "SPX": {
        "strike_increment": 5,
        "wing_widths": [15, 20, 25, 30, 40, 50],
        "min_spread_width": {"normal": 20, "event_1": 25, "event_2": 30, "fomc_week": 35},
    },
    "XSP": {
        "strike_increment": 1,
        "wing_widths": [1.5, 2, 2.5, 3, 4, 5],
        "min_spread_width": {"normal": 2, "event_1": 2.5, "event_2": 3, "fomc_week": 3.5},
    },
}


def get_ticker_config(ticker: str = "SPX") -> dict:
    """Get spread configuration for a given ticker."""
    return TICKER_CONFIG.get(ticker.upper(), TICKER_CONFIG["SPX"])


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class SpreadLeg:
    """One leg of a credit spread."""
    side:        str
    direction:   str
    strike:      float
    description: str     = ""


@dataclass
class SpreadSide:
    """One side of the iron condor (either the call spread or put spread)."""
    side:          str
    short_strike:  float
    long_strike:   float
    wing_width:    float
    short_pct:     float
    max_profit:    float
    max_loss:      float
    breakeven:     float
    credit_ratio:  float
    estimated_credit: float
    meets_min_credit: bool
    below_min_width:  bool = False


@dataclass
class SpreadPlan:
    """Complete weekly spread plan — both call and put sides, with context."""
    week_start:       str
    generated_at:     str

    spx_ref_close:    float
    spx_ref_open:     Optional[float]

    point_pct:        float
    upper_pct:        float
    lower_pct:        float
    vix_implied_pct:  float
    confidence_level: int

    buffer_pct:       float
    buffer_pts:       float
    buffer_reason:    str

    effective_range_pct:  float
    effective_upper_px:   float
    effective_lower_px:   float

    call_spreads:     list[SpreadSide]  = field(default_factory=list)
    put_spreads:      list[SpreadSide]  = field(default_factory=list)

    has_fomc:         int  = 0
    has_cpi:          int  = 0
    has_nfp:          int  = 0
    has_opex:         int  = 0
    event_count:      int  = 0
    gex_flag:         Optional[int] = None
    gex_regime:       str = "unknown"

    recommended_width: int = 25

    warnings:         list[str] = field(default_factory=list)


# =============================================================================
# BUFFER CALCULATION
# =============================================================================

def compute_buffer(
    forecast: dict,
    feature_row: "pd.Series | dict" = None,
) -> tuple[float, str]:
    """
    Compute the total buffer percentage to add beyond the PI upper bound.
    Buffer = DEFAULT_BUFFER_PCT x event_multiplier + gex_adjustment
    """
    buffer = DEFAULT_BUFFER_PCT
    reasons = [f"base={DEFAULT_BUFFER_PCT*100:.2f}%"]

    event_flags = {}
    if feature_row is not None:
        for flag in ["has_fomc", "has_cpi", "has_nfp", "has_opex"]:
            val = feature_row.get(flag, 0)
            event_flags[flag] = int(val) if val is not None else 0
    else:
        for flag in ["has_fomc", "has_cpi", "has_nfp", "has_opex"]:
            event_flags[flag] = int(forecast.get(flag, 0))

    active_events = []
    if event_flags.get("has_fomc"):
        active_events.append(("fomc",  EVENT_BUFFER_MULTIPLIERS["fomc"]))
    if event_flags.get("has_cpi"):
        active_events.append(("cpi",   EVENT_BUFFER_MULTIPLIERS["cpi"]))
    if event_flags.get("has_nfp"):
        active_events.append(("nfp",   EVENT_BUFFER_MULTIPLIERS["nfp"]))
    if event_flags.get("has_opex"):
        active_events.append(("opex",  EVENT_BUFFER_MULTIPLIERS["opex"]))

    if active_events:
        top_event, top_mult = max(active_events, key=lambda x: x[1])
        buffer *= top_mult
        reasons.append(f"{top_event}_mult={top_mult}x")

    # Prefer continuous GEX feature if available; fall back to binary flag
    gex_adj = 0.0
    gex_normalized = None
    if feature_row is not None:
        raw_norm = feature_row.get("gex_normalized")
        if raw_norm is not None and not (isinstance(raw_norm, float) and math.isnan(raw_norm)):
            gex_normalized = float(raw_norm)

    if gex_normalized is not None:
        # Continuous: negative gex_normalized widens buffer, positive tightens
        gex_adj = -gex_normalized * GEX_CONTINUOUS_SCALE
        gex_adj = max(-0.002, min(0.005, gex_adj))  # clamp to reasonable range
        if abs(gex_adj) > 0.0001:
            buffer += gex_adj
            reasons.append(f"gex_norm={gex_normalized:.2f}({gex_adj*100:+.3f}%)")
    else:
        gex_flag = None
        if feature_row is not None:
            raw_gex = feature_row.get("gex_flag")
            if raw_gex is not None and not (isinstance(raw_gex, float) and math.isnan(raw_gex)):
                gex_flag = int(raw_gex)

        gex_adj = GEX_BUFFER_ADJUSTMENTS.get(gex_flag, 0.0)
        if gex_adj != 0:
            buffer += gex_adj
            reasons.append(f"gex_flag={gex_flag}({gex_adj*100:+.2f}%)")

    buffer = max(buffer, 0.001)
    reason_str = " | ".join(reasons)

    log.info(f"Buffer computed: {buffer*100:.3f}%  ({reason_str})")
    return buffer, reason_str


# =============================================================================
# STRIKE ROUNDING
# =============================================================================

def round_to_increment(
    price: float,
    increment: float = SPX_STRIKE_INCREMENT,
    direction: str = "away",
) -> float:
    """Round a price to the nearest valid strike increment."""
    if direction == "away":
        return math.ceil(price / increment) * increment
    elif direction == "toward":
        return math.floor(price / increment) * increment
    else:
        return round(price / increment) * increment


def round_call_short(price: float, increment: float = SPX_STRIKE_INCREMENT) -> float:
    """Round call short strike UP to next increment (more OTM = safer)."""
    return round_to_increment(price, increment=increment, direction="away")


def round_put_short(price: float, increment: float = SPX_STRIKE_INCREMENT) -> float:
    """Round put short strike DOWN to next increment (more OTM = safer)."""
    return round_to_increment(price, increment=increment, direction="toward")


# =============================================================================
# CREDIT ESTIMATION
# =============================================================================

def estimate_credit(
    short_strike: float,
    spx_ref: float,
    wing_width: float,
    vix: float,
    dte: int = 5,
    side: str = "put",
) -> float:
    """Rough mid-market credit estimate using simplified BSM approximation."""
    from scipy.stats import norm

    S   = spx_ref
    K1  = short_strike
    K2  = K1 + wing_width if side == "call" else K1 - wing_width
    sig = (vix / 100)
    T   = dte / 252
    r   = 0.05

    def bsm_price(S, K, sig, T, r, side):
        if T <= 0 or sig <= 0:
            return max(0, (S - K) if side == "call" else (K - S))
        d1 = (math.log(S / K) + (r + 0.5 * sig**2) * T) / (sig * math.sqrt(T))
        d2 = d1 - sig * math.sqrt(T)
        if side == "call":
            return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    p_short = bsm_price(S, K1, sig, T, r, side)
    p_long  = bsm_price(S, K2, sig, T, r, side)

    credit = max(0.0, p_short - p_long)
    return round(credit, 2)


# =============================================================================
# SPREAD SIDE BUILDER
# =============================================================================

def build_spread_side(
    side: str,
    short_strike: float,
    wing_widths: list[int],
    spx_ref: float,
    vix: float,
    dte: int = 5,
) -> list[SpreadSide]:
    """Build SpreadSide objects for each offered wing width."""
    results = []

    for width in wing_widths:
        if side == "call":
            long_strike = short_strike + width
        else:
            long_strike = short_strike - width

        est_credit = estimate_credit(
            short_strike, spx_ref, width, vix, dte, side
        )

        max_profit  = round(est_credit * 100, 2)
        max_loss    = round((width - est_credit) * 100, 2)

        if side == "call":
            breakeven = round(short_strike + est_credit, 2)
        else:
            breakeven = round(short_strike - est_credit, 2)

        credit_ratio  = est_credit / width if width > 0 else 0
        meets_minimum = credit_ratio >= MIN_CREDIT_RATIO

        short_pct = abs(short_strike - spx_ref) / spx_ref

        results.append(SpreadSide(
            side            = side,
            short_strike    = short_strike,
            long_strike     = long_strike,
            wing_width      = width,
            short_pct       = round(short_pct, 4),
            max_profit      = max_profit,
            max_loss        = max_loss,
            breakeven       = breakeven,
            credit_ratio    = round(credit_ratio, 4),
            estimated_credit= est_credit,
            meets_min_credit= meets_minimum,
        ))

    return results


# =============================================================================
# MINIMUM WIDTH ENFORCEMENT
# =============================================================================

def get_min_width(event_count: int, has_fomc: int, ticker: str = "SPX") -> float:
    """Return the minimum permissible wing width for this week's event profile."""
    cfg = get_ticker_config(ticker)
    widths = cfg["min_spread_width"]
    if has_fomc:
        return widths["fomc_week"]
    if event_count >= 2:
        return widths["event_2"]
    if event_count >= 1:
        return widths["event_1"]
    return widths["normal"]


def get_recommended_width(
    effective_range_pct: float,
    spx_ref: float,
    event_count: int,
    has_fomc: int,
    ticker: str = "SPX",
) -> float:
    """Suggest a wing width based on the effective range and event profile."""
    cfg = get_ticker_config(ticker)
    standard_widths = cfg["wing_widths"]

    half_range_pts = (effective_range_pct / 2) * spx_ref
    proportional   = half_range_pts * 0.40

    snapped = min(standard_widths, key=lambda w: abs(w - proportional))

    min_width = get_min_width(event_count, has_fomc, ticker=ticker)
    final     = max(snapped, min_width)
    final = min(final, max(standard_widths))

    return final


# =============================================================================
# MAIN BUILDER
# =============================================================================

def build_spread_plan(
    forecast: dict,
    feature_row: "pd.Series | dict" = None,
    week_start: str = None,
    wing_widths: list = None,
    vix_level: float = None,
    spx_open: float = None,
    dte: int = 5,
    ticker: str = "SPX",
) -> SpreadPlan:
    """Build a complete SpreadPlan from a forecast dict."""
    cfg = get_ticker_config(ticker)
    if wing_widths is None:
        wing_widths = cfg["wing_widths"]

    spx_ref = spx_open if spx_open else forecast["spx_ref_close"]

    if vix_level is None and feature_row is not None:
        vix_level = float(feature_row.get("vix_close", 18) or 18)
    if vix_level is None:
        vix_level = 18.0

    # --- Compute buffer ---
    buffer_pct, buffer_reason = compute_buffer(forecast, feature_row)
    buffer_pts = round(buffer_pct * spx_ref, 2)

    # --- Effective range ---
    pi_upper_pct    = forecast["upper_pct"]
    effective_range = pi_upper_pct + buffer_pct
    half_range      = effective_range / 2

    effective_upper = round(spx_ref * (1 + half_range), 2)
    effective_lower = round(spx_ref * (1 - half_range), 2)

    # --- Event flags ---
    def _flag(key, default=0):
        if feature_row is not None:
            v = feature_row.get(key, default)
            return int(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else default
        return int(forecast.get(key, default))

    has_fomc    = _flag("has_fomc")
    has_cpi     = _flag("has_cpi")
    has_nfp     = _flag("has_nfp")
    has_opex    = _flag("has_opex")
    event_count = _flag("event_count")

    gex_raw  = feature_row.get("gex_flag") if feature_row is not None else None
    gex_flag = None
    if gex_raw is not None and not (isinstance(gex_raw, float) and math.isnan(gex_raw)):
        gex_flag = int(gex_raw)

    gex_regime = {1: "positive (suppressive)", 0: "neutral", -1: "negative (amplifying)"}.get(gex_flag, "unknown")

    # --- Short strikes ---
    increment = cfg["strike_increment"]
    call_short = round_call_short(effective_upper, increment=increment)
    put_short  = round_put_short(effective_lower, increment=increment)

    log.info(f"Effective range: +/-{half_range*100:.2f}%  ->  [{effective_lower:.2f}, {effective_upper:.2f}]")
    log.info(f"Short strikes  : call={call_short}  put={put_short}  (ticker={ticker})")

    # --- Minimum width ---
    min_width = get_min_width(event_count, has_fomc, ticker=ticker)

    # --- Build spread sides for ALL widths (flag narrow ones) ---
    call_spreads = build_spread_side("call", call_short, wing_widths, spx_ref, vix_level, dte)
    put_spreads  = build_spread_side("put",  put_short,  wing_widths, spx_ref, vix_level, dte)

    for s in call_spreads + put_spreads:
        s.below_min_width = s.wing_width < min_width

    # --- Recommended width ---
    rec_width = get_recommended_width(effective_range, spx_ref, event_count, has_fomc, ticker=ticker)

    # --- Warnings ---
    warnings = []

    min_floor = cfg["min_spread_width"]["fomc_week"]
    if event_count >= 2:
        warnings.append(f"Multiple macro events this week ({event_count}) — consider reducing size or widening wings further")
    if has_fomc:
        warnings.append(f"FOMC week — minimum width floor raised to {min_floor} pts; gaps through strikes are possible")
    if gex_flag == -1:
        warnings.append("Negative GEX regime — dealer hedging amplifies moves; buffer widened")
    if forecast.get("model_vs_vix", 0) > 0.01:
        warnings.append(f"Model expects wider range than VIX implies ({forecast['model_vs_vix']*100:+.2f}%) — trust the model")
    if vix_level > 25:
        warnings.append(f"Elevated VIX ({vix_level:.1f}) — credit will look attractive but max loss risk is higher than usual")

    if call_spreads and not call_spreads[0].meets_min_credit:
        warnings.append(f"Estimated credit below {MIN_CREDIT_RATIO:.0%} ratio — verify real chain before trading")

    plan = SpreadPlan(
        week_start           = week_start or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        generated_at         = datetime.now(timezone.utc).isoformat(),
        spx_ref_close        = forecast["spx_ref_close"],
        spx_ref_open         = spx_open,
        point_pct            = forecast["point_pct"],
        upper_pct            = forecast["upper_pct"],
        lower_pct            = forecast["lower_pct"],
        vix_implied_pct      = forecast["vix_implied_pct"],
        confidence_level     = forecast["confidence_level"],
        buffer_pct           = round(buffer_pct, 4),
        buffer_pts           = buffer_pts,
        buffer_reason        = buffer_reason,
        effective_range_pct  = round(effective_range, 4),
        effective_upper_px   = effective_upper,
        effective_lower_px   = effective_lower,
        call_spreads         = call_spreads,
        put_spreads          = put_spreads,
        has_fomc             = has_fomc,
        has_cpi              = has_cpi,
        has_nfp              = has_nfp,
        has_opex             = has_opex,
        event_count          = event_count,
        gex_flag             = gex_flag,
        gex_regime           = gex_regime,
        recommended_width    = rec_width,
        warnings             = warnings,
    )

    return plan


# =============================================================================
# PERSISTENCE
# =============================================================================

def init_spread_log_table(conn) -> None:
    """Ensure spread_log table exists.
    Now handled by db.init_all_tables() — kept for backwards compatibility."""
    pass  # Tables created in db.init_all_tables()


def log_spread_plan(
    conn: sqlite3.Connection,
    plan: SpreadPlan,
    wing_width_used: int = None,
) -> None:
    """Persist a SpreadPlan to spread_log."""
    width = wing_width_used or plan.recommended_width

    call = next((s for s in plan.call_spreads if s.wing_width == width), None)
    put  = next((s for s in plan.put_spreads  if s.wing_width == width), None)

    now = datetime.now(timezone.utc).isoformat()

    conn.execute("""
        INSERT INTO spread_log (
            week_start, generated_at,
            spx_ref_close, point_pct, upper_pct, effective_range_pct,
            call_short, call_long, put_short, put_long,
            wing_width_used, buffer_pct, event_count, gex_flag,
            warnings, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(week_start) DO UPDATE SET
            generated_at        = excluded.generated_at,
            spx_ref_close       = excluded.spx_ref_close,
            point_pct           = excluded.point_pct,
            upper_pct           = excluded.upper_pct,
            effective_range_pct = excluded.effective_range_pct,
            call_short          = excluded.call_short,
            call_long           = excluded.call_long,
            put_short           = excluded.put_short,
            put_long            = excluded.put_long,
            wing_width_used     = excluded.wing_width_used,
            buffer_pct          = excluded.buffer_pct,
            event_count         = excluded.event_count,
            gex_flag            = excluded.gex_flag,
            warnings            = excluded.warnings,
            updated_at          = excluded.updated_at
    """, (
        plan.week_start,
        plan.generated_at,
        plan.spx_ref_close,
        plan.point_pct,
        plan.upper_pct,
        plan.effective_range_pct,
        call.short_strike if call else None,
        call.long_strike  if call else None,
        put.short_strike  if put  else None,
        put.long_strike   if put  else None,
        width,
        plan.buffer_pct,
        plan.event_count,
        plan.gex_flag,
        " | ".join(plan.warnings),
        now,
    ))
    conn.commit()
    log.info(f"Spread plan logged for {plan.week_start}")


def update_outcome(
    conn: sqlite3.Connection,
    week_start: str,
    actual_high: float,
    actual_low: float,
    credit_received: float = None,
) -> str:
    """Fill in the actual outcome after the week expires."""
    row = conn.execute(
        "SELECT call_short, put_short, wing_width_used FROM spread_log WHERE week_start = ?",
        (week_start,)
    ).fetchone()

    if not row:
        log.warning(f"No spread_log entry for {week_start}")
        return "not_found"

    call_short, put_short, width = row
    spx_ref = conn.execute(
        "SELECT spx_ref_close FROM spread_log WHERE week_start = ?", (week_start,)
    ).fetchone()[0]

    actual_range_pct = (actual_high - actual_low) / spx_ref if spx_ref else None
    call_breached    = int(actual_high >= call_short) if call_short else 0
    put_breached     = int(actual_low  <= put_short)  if put_short  else 0

    if call_breached or put_breached:
        if credit_received and width:
            pnl_pts = credit_received - width
            outcome = "partial_loss" if pnl_pts > -width * 0.5 else "full_loss"
        else:
            outcome = "full_loss"
            pnl_pts = None
    else:
        outcome = "full_profit"
        pnl_pts = credit_received if credit_received else None

    now = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        UPDATE spread_log SET
            actual_high      = ?,
            actual_low       = ?,
            actual_range_pct = ?,
            call_breached    = ?,
            put_breached     = ?,
            outcome          = ?,
            pnl_pts          = ?,
            updated_at       = ?
        WHERE week_start = ?
    """, (
        actual_high, actual_low, actual_range_pct,
        call_breached, put_breached, outcome,
        pnl_pts, now,
        week_start,
    ))
    conn.commit()
    log.info(f"Outcome updated for {week_start}: {outcome}")
    return outcome


def print_spread_plan(plan: SpreadPlan) -> None:
    """Pretty-print the full spread plan to console."""
    sep = "=" * 70

    print(f"\n{sep}")
    print(f"  WEEKLY SPREAD PLAN  --  Week of {plan.week_start}")
    print(f"  Generated: {plan.generated_at[:19]} UTC")
    print(sep)

    print(f"\n  REFERENCE")
    print(f"    SPX Friday close  : {plan.spx_ref_close:>10,.2f}")
    if plan.spx_ref_open:
        print(f"    SPX Monday open   : {plan.spx_ref_open:>10,.2f}")
    print(f"    VIX implied range : {plan.vix_implied_pct*100:>9.2f}%")

    print(f"\n  FORECAST  ({plan.confidence_level}% CI)")
    print(f"    Point estimate    : +/-{plan.point_pct/2*100:.2f}%  "
          f"({plan.point_pct*100:.2f}% total)")
    print(f"    PI upper bound    :  {plan.upper_pct*100:.2f}%  total range")
    print(f"    Buffer applied    : +{plan.buffer_pct*100:.3f}%  ({plan.buffer_pts:.1f} pts)")
    print(f"    Effective range   :  {plan.effective_range_pct*100:.2f}%  total")
    print(f"    Effective upper   : {plan.effective_upper_px:>10,.2f}")
    print(f"    Effective lower   : {plan.effective_lower_px:>10,.2f}")

    print(f"\n  CONTEXT")
    print(f"    Events this week  : {plan.event_count}  "
          f"(FOMC={plan.has_fomc} CPI={plan.has_cpi} NFP={plan.has_nfp} OPEX={plan.has_opex})")
    print(f"    GEX regime        : {plan.gex_regime}")
    print(f"    Recommended width : {plan.recommended_width} pts")

    if plan.warnings:
        print(f"\n  WARNINGS")
        for w in plan.warnings:
            print(f"    {w}")

    print(f"\n{sep}\n")
