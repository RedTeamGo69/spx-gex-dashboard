# =============================================================================
# spread_levels.py
# Weekly SPX Range Prediction Model — Spread Level Calculator
#
# Takes the forecast dict from har_model.py and converts it into
# actionable credit spread parameters.
# =============================================================================

import logging
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from range_finder.gex_policy import GEX_LIVE_SPREAD_INFLUENCE_ENABLED

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

# Legacy research transformation. BUG-06 keeps the calculation available for
# historical comparison, but GEX_LIVE_SPREAD_INFLUENCE_ENABLED is intentionally
# False until predictive calibration is supported by the pending study.
GEX_CONTINUOUS_SCALE = 0.002  # buffer adjustment per unit of gex_normalized

# The anchored-regime research note emitted when gex_flag is -1. Kept as a
# constant so gex_bridge.reconcile_gex_warnings can replace it with one
# coherent anchored-vs-live message.
GEX_NEGATIVE_REGIME_WARNING = (
    "Negative GEX regime — research context only; not applied to live buffer or strikes"
)

# Legacy module-level SPX defaults. The live per-ticker values come from
# phase1.ticker_config (wing_widths / min_spread_width) — keep these in
# sync with the SPX entry there.
MIN_SPREAD_WIDTH = {
    "normal":     50,
    "event_1":    50,
    "event_2":    100,
    "fomc_week":  100,
}

STANDARD_WING_WIDTHS = [50, 100, 150, 200]

MIN_CREDIT_RATIO = 0.05   # 5% — realistic for far-OTM weekly credit spreads

# Per-ticker configuration — re-exported from the central phase1.ticker_config
# module so any place that historically imported from spread_levels keeps
# working. The full config (display name, yfinance symbol, vol proxy, etc.)
# lives in phase1/ticker_config.py.
from phase1.ticker_config import TICKER_CONFIG, get_config as _get_full_config


def get_ticker_config(ticker: str = "SPX") -> dict:
    """Get spread configuration for a given ticker."""
    return _get_full_config(ticker)


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
    credit_source:    str  = "bsm"   # "market" if from actual bid/ask, "bsm" if theoretical


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

    # Per-side share of the forecast range used to place the bands. 0.5 is
    # the legacy symmetric half-split; the empirical side-share quantile
    # (har_model.estimate_side_share_quantile) is > 0.5 because the weekly
    # open concentrates near one extreme of the realized range.
    side_share_q:     float = 0.5

    warnings:         list[str] = field(default_factory=list)


# =============================================================================
# BUFFER CALCULATION
# =============================================================================

def compute_legacy_gex_buffer_adjustment(
    feature_row: "pd.Series | dict" = None,
) -> float:
    """Return the uncalibrated pre-mitigation GEX adjustment for research.

    This preserves the BUG-06 transformation for audit and historical-study
    comparisons. Production callers must use ``compute_buffer`` instead.
    """
    if feature_row is None:
        return 0.0

    raw_norm = feature_row.get("gex_normalized")
    if raw_norm is None:
        return 0.0
    try:
        gex_normalized = float(raw_norm)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(gex_normalized):
        return 0.0

    gex_adj = -gex_normalized * GEX_CONTINUOUS_SCALE
    return max(-0.002, min(0.005, gex_adj))


def compute_buffer(
    forecast: dict,
    feature_row: "pd.Series | dict" = None,
) -> tuple[float, str]:
    """
    Compute the total buffer percentage to add beyond the PI upper bound.

    BUG-06 mitigation leaves event widening intact while the uncalibrated GEX
    adjustment is excluded from live recommendations.
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

    legacy_gex_adj = compute_legacy_gex_buffer_adjustment(feature_row)
    if GEX_LIVE_SPREAD_INFLUENCE_ENABLED:
        # Retained behind an explicit policy switch for a future, evidence-
        # backed calibration. Do not enable by changing the coefficient.
        gex_adj = legacy_gex_adj
        if abs(gex_adj) > 0.0001:
            buffer += gex_adj
            reasons.append(f"gex_adjustment={gex_adj*100:+.3f}%")
    else:
        reasons.append("gex=disabled_pending_calibration")

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

def _snap_long_strike(short_strike: float, width: float, chain_quotes: dict, side: str):
    """Resolve the actual listed long strike for a target wing width.

    Far-OTM chains coarsen to wider strike increments (SPX goes 25/50-pt,
    QQQ 5-pt), so `short ± width` frequently isn't listed exactly where
    wide wings land. Snap OUTWARD (more OTM — never narrower than asked)
    to the nearest listed strike with quote data. Returns None only when
    no listed strike exists at or beyond the target (true chain edge).
    """
    target = short_strike + width if side == "call" else short_strike - width
    side_key = f"{side}_ask"  # long leg needs an ask price
    row = chain_quotes.get(target)
    if row and side_key in row:
        return target
    listed = sorted(k for k, v in chain_quotes.items() if side_key in v)
    if side == "call":
        cands = [k for k in listed if k >= target]
        return cands[0] if cands else None
    cands = [k for k in listed if k <= target]
    return cands[-1] if cands else None


def _available_wing_widths(
    short_strike: float,
    chain_quotes: dict,
    side: str,
    target_widths: list,
) -> list:
    """Return wing widths whose long leg can be resolved to a real chain
    strike (exactly, or snapped outward — see _snap_long_strike). Keeps
    the original target-list order; falls back to all targets when the
    chain has nothing on this side."""
    available = [
        w for w in target_widths
        if _snap_long_strike(short_strike, w, chain_quotes, side) is not None
    ]
    return available if available else target_widths  # fallback to all if chain has nothing


def _snap_to_chain_strike(target: float, chain_quotes: dict, side: str, direction: str = "up") -> float:
    """Snap a target strike to the nearest actual chain strike.

    direction="up"   → for calls, pick the nearest strike >= target (more OTM)
    direction="down" → for puts, pick the nearest strike <= target (more OTM)
    Returns the original target if the chain has no strikes for this side,
    or when no listed strike exists on the required (more-OTM) side of the
    target. The old fallback in that second case returned the chain's
    opposite extreme — e.g. a put target below the lowest listed strike
    "snapped" to that lowest strike, which is *inside* the requested level.
    Whenever a corrupt reference price pushed targets outside the listed
    range, that fabricated deep-ITM "spreads" at the chain edge instead of
    letting the side honestly come back empty.
    """
    side_key = f"{side}_bid"
    available = sorted(k for k, v in chain_quotes.items() if side_key in v)
    if not available:
        return target

    if direction == "up":
        candidates = [k for k in available if k >= target]
    else:
        candidates = [k for k in available if k <= target]
    if not candidates:
        return target
    return candidates[0] if direction == "up" else candidates[-1]


def _lookup_chain_price(chain_quotes: dict, strike: float, side: str, field: str) -> float | None:
    """Look up bid/ask for a specific strike from chain data.

    chain_quotes: {strike: {"call_bid": .., "call_ask": .., "put_bid": .., "put_ask": ..}}
    Returns None only if the strike/side is not in the chain.
    A $0.00 bid is a valid market price (option is worthless).
    """
    if not chain_quotes:
        return None
    row = chain_quotes.get(strike)
    if not row:
        return None
    key = f"{side}_{field}"
    if key not in row:
        return None
    return float(row[key] or 0.0)


def build_spread_side(
    side: str,
    short_strike: float,
    wing_widths: list[int],
    spx_ref: float,
    vix: float,
    dte: int = 5,
    chain_quotes: dict = None,
) -> list[SpreadSide]:
    """Build SpreadSide objects for each offered wing width.

    If chain_quotes is provided, uses actual market bid/ask (natural prices:
    short leg bid, long leg ask) for credit estimation. Falls back to BSM
    when chain data is missing.
    """
    results = []
    used_longs: set = set()  # dedupe targets that snap to the same listed strike

    for width in wing_widths:
        if side == "call":
            long_strike = short_strike + width
        else:
            long_strike = short_strike - width

        if chain_quotes:
            # Short must exist in the chain (it's snapped upstream, so a
            # miss here means the side genuinely isn't listed).
            short_in_chain = short_strike in chain_quotes and f"{side}_bid" in chain_quotes[short_strike]
            if not short_in_chain:
                continue
            # Long leg: resolve to a REAL listed strike, snapping outward
            # when short±width lands between far-OTM strike increments.
            # Previously these widths were silently dropped, which made
            # the wider rungs of the ladder vanish from the tables at
            # exactly the tiers (far-OTM shorts) where the chain coarsens.
            long_strike = _snap_long_strike(short_strike, width, chain_quotes, side)
            if long_strike is None:
                continue  # true chain edge — nothing listed at/beyond target
            if long_strike in used_longs:
                continue  # two targets snapped to the same strike — keep the narrower
            used_longs.add(long_strike)

        # The economics must use the ACTUAL wing distance, not the target.
        actual_width = abs(long_strike - short_strike)

        # Use actual market prices
        market_credit = None
        if chain_quotes:
            short_bid = _lookup_chain_price(chain_quotes, short_strike, side, "bid")
            long_ask = _lookup_chain_price(chain_quotes, long_strike, side, "ask")
            if short_bid is not None and long_ask is not None:
                market_credit = round(max(0.0, short_bid - long_ask), 2)

        if market_credit is not None:
            est_credit = market_credit
            source = "market"
        else:
            est_credit = estimate_credit(
                short_strike, spx_ref, actual_width, vix, dte, side
            )
            source = "bsm"

        max_profit  = round(est_credit * 100, 2)
        max_loss    = round((actual_width - est_credit) * 100, 2)

        if side == "call":
            breakeven = round(short_strike + est_credit, 2)
        else:
            breakeven = round(short_strike - est_credit, 2)

        credit_ratio  = est_credit / actual_width if actual_width > 0 else 0
        meets_minimum = credit_ratio >= MIN_CREDIT_RATIO

        short_pct = abs(short_strike - spx_ref) / spx_ref

        results.append(SpreadSide(
            side            = side,
            short_strike    = short_strike,
            long_strike     = long_strike,
            wing_width      = actual_width,
            short_pct       = round(short_pct, 4),
            max_profit      = max_profit,
            max_loss        = max_loss,
            breakeven       = breakeven,
            credit_ratio    = round(credit_ratio, 4),
            estimated_credit= est_credit,
            meets_min_credit= meets_minimum,
            credit_source   = source,
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
    chain_quotes: dict = None,
    side_share_q: "float | None" = None,
) -> SpreadPlan:
    """Build a complete SpreadPlan from a forecast dict.

    side_share_q: per-side share of the (buffered) range each band gets,
    from har_model.estimate_side_share_quantile. None → legacy symmetric
    /2 split. Values are clamped to [0.5, 1.0]: below 0.5 would place
    strikes TIGHTER than the legacy behavior.
    """
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

    # --- Effective range → per-side bands ---
    pi_upper_pct    = forecast["upper_pct"]
    effective_range = pi_upper_pct + buffer_pct

    # Per-side share: legacy /2, or the empirical side-share quantile when
    # provided. The range forecast bounds HOW WIDE the week trades; s_up =
    # (H-O)/(H-L) says where the open sits inside that width, and its
    # empirical mass concentrates near the extremes — so giving each side
    # only half the range under-protects the strikes on trending weeks even
    # when the range forecast itself verifies.
    side_share = 0.5
    if side_share_q is not None:
        side_share = min(max(float(side_share_q), 0.5), 1.0)
    half_range      = effective_range * side_share

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

    # Snap short strikes to actual chain strikes if they don't exist
    if chain_quotes:
        call_short = _snap_to_chain_strike(call_short, chain_quotes, "call", direction="up")
        put_short  = _snap_to_chain_strike(put_short, chain_quotes, "put", direction="down")

    log.info(f"Effective range: +/-{half_range*100:.2f}%  ->  [{effective_lower:.2f}, {effective_upper:.2f}]")
    log.info(f"Short strikes  : call={call_short}  put={put_short}  (ticker={ticker})")

    # --- Minimum width ---
    min_width = get_min_width(event_count, has_fomc, ticker=ticker)

    # --- Derive available wing widths from chain strikes ---
    if chain_quotes:
        call_widths = _available_wing_widths(call_short, chain_quotes, "call", wing_widths)
        put_widths  = _available_wing_widths(put_short, chain_quotes, "put", wing_widths)
    else:
        call_widths = wing_widths
        put_widths  = wing_widths

    # --- Build spread sides for available widths (flag narrow ones) ---
    call_spreads = build_spread_side("call", call_short, call_widths, spx_ref, vix_level, dte, chain_quotes=chain_quotes)
    put_spreads  = build_spread_side("put",  put_short,  put_widths,  spx_ref, vix_level, dte, chain_quotes=chain_quotes)

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
        warnings.append(GEX_NEGATIVE_REGIME_WARNING)
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
        side_share_q         = round(side_share, 4),
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
# RISK TIER BUILDER
# =============================================================================

@dataclass
class SpreadTier:
    """One risk tier of spreads (e.g. 'Effective Range', 'PI Upper', 'Point Est')."""
    label:        str
    risk_level:   str    # "conservative", "moderate", "aggressive"
    range_pct:    float  # the range % this tier is based on
    call_short:   float
    put_short:    float
    call_spreads: list[SpreadSide] = field(default_factory=list)
    put_spreads:  list[SpreadSide] = field(default_factory=list)
    # Model-only strikes (before EM floor).  Populated only when the EM floor
    # actually moved a strike; otherwise None → UI skips the extra table.
    model_call_short:   float | None = None
    model_put_short:    float | None = None
    model_call_spreads: list[SpreadSide] = field(default_factory=list)
    model_put_spreads:  list[SpreadSide] = field(default_factory=list)


def build_spread_tiers(
    forecast: dict,
    plan: SpreadPlan,
    spx_ref: float,
    vix_level: float,
    chain_quotes: dict = None,
    wing_widths: list = None,
    dte: int = 5,
    ticker: str = "SPX",
    weekly_em: dict = None,
) -> list[SpreadTier]:
    """Build spread tiers at multiple risk levels from the forecast.

    Returns tiers from most aggressive (Point Estimate) to most conservative
    (Effective Range + buffer), each with their own short strikes and spreads.

    If *weekly_em* is provided (dict with ``upper_level`` / ``lower_level``),
    short strikes that would fall inside the weekly expected-move band are
    pushed out to (at least) the EM boundary.  This prevents selling strikes
    that the market's own straddle pricing says are reachable.
    """
    cfg = get_ticker_config(ticker)
    if wing_widths is None:
        wing_widths = cfg["wing_widths"]
    increment = cfg["strike_increment"]

    # Weekly expected-move boundaries (0 = not available)
    em_upper = float((weekly_em or {}).get("upper_level", 0) or 0)
    em_lower = float((weekly_em or {}).get("lower_level", 0) or 0)
    has_em = em_upper > 0 and em_lower > 0

    tiers_config = [
        ("Lower PI",         "aggressive",  forecast["lower_pct"]),
        ("Point Estimate",   "aggressive",  forecast["point_pct"]),
        (f"{forecast['confidence_level']}% PI Upper", "moderate", forecast["upper_pct"]),
        ("Effective (+buffer)", "conservative", plan.effective_range_pct),
    ]

    # Same per-side share as the plan (legacy 0.5 or the empirical
    # side-share quantile) so every tier's band is placed on the same
    # convention as the headline effective range.
    side_share = min(max(float(getattr(plan, "side_share_q", 0.5) or 0.5), 0.5), 1.0)

    tiers = []
    for label, risk, range_pct in tiers_config:
        half = range_pct * side_share
        raw_call = spx_ref * (1 + half)
        raw_put  = spx_ref * (1 - half)

        model_call_short = round_call_short(raw_call, increment=increment)
        model_put_short  = round_put_short(raw_put, increment=increment)

        if chain_quotes:
            model_call_short = _snap_to_chain_strike(model_call_short, chain_quotes, "call", direction="up")
            model_put_short  = _snap_to_chain_strike(model_put_short, chain_quotes, "put", direction="down")

        # --- Weekly EM floor ---------------------------------------------------
        # If the short strike lands inside the expected-move band, widen it
        # to at least the EM boundary so we never sell inside the straddle range.
        em_adjusted_call = False
        em_adjusted_put  = False
        call_short = model_call_short
        put_short  = model_put_short
        if has_em:
            if call_short < em_upper:
                call_short = round_call_short(em_upper, increment=increment)
                if chain_quotes:
                    call_short = _snap_to_chain_strike(call_short, chain_quotes, "call", direction="up")
                em_adjusted_call = True
            if put_short > em_lower:
                put_short = round_put_short(em_lower, increment=increment)
                if chain_quotes:
                    put_short = _snap_to_chain_strike(put_short, chain_quotes, "put", direction="down")
                em_adjusted_put = True

        # Suppress the "model strikes" ladder when the pre-floor strike sits
        # on the wrong side of the EM band entirely — a call short BELOW the
        # band or a put short ABOVE it. The EM band brackets spot, so such a
        # strike can only come from a corrupt reference price; building its
        # ladder produces deep-ITM rows masquerading as credit spreads (the
        # final strikes above are already EM-floored and stay as they are).
        if em_adjusted_call and model_call_short < em_lower:
            log.warning(
                f"[{label}] model call short {model_call_short:.0f} is below the "
                f"EM band ({em_lower:.0f}-{em_upper:.0f}) — reference price is "
                "suspect; suppressing the model-strike ladder for this side."
            )
            em_adjusted_call = False
        if em_adjusted_put and model_put_short > em_upper:
            log.warning(
                f"[{label}] model put short {model_put_short:.0f} is above the "
                f"EM band ({em_lower:.0f}-{em_upper:.0f}) — reference price is "
                "suspect; suppressing the model-strike ladder for this side."
            )
            em_adjusted_put = False

        em_adjusted = em_adjusted_call or em_adjusted_put

        if em_adjusted:
            sides = []
            if em_adjusted_call:
                sides.append(f"call short {model_call_short:.0f} → {call_short:.0f} (EM upper {em_upper:.0f})")
            if em_adjusted_put:
                sides.append(f"put short {model_put_short:.0f} → {put_short:.0f} (EM lower {em_lower:.0f})")
            log.info(f"[{label}] Weekly EM floor applied: {'; '.join(sides)}")

        if chain_quotes:
            cw = _available_wing_widths(call_short, chain_quotes, "call", wing_widths)
            pw = _available_wing_widths(put_short, chain_quotes, "put", wing_widths)
        else:
            cw = wing_widths
            pw = wing_widths

        call_spreads = build_spread_side("call", call_short, cw, spx_ref, vix_level, dte, chain_quotes=chain_quotes)
        put_spreads  = build_spread_side("put",  put_short,  pw, spx_ref, vix_level, dte, chain_quotes=chain_quotes)

        # Build model-only spreads at original strikes (only when EM floor moved them)
        m_call_spreads = []
        m_put_spreads  = []
        if em_adjusted:
            if em_adjusted_call:
                mcw = _available_wing_widths(model_call_short, chain_quotes, "call", wing_widths) if chain_quotes else wing_widths
                m_call_spreads = build_spread_side("call", model_call_short, mcw, spx_ref, vix_level, dte, chain_quotes=chain_quotes)
            if em_adjusted_put:
                mpw = _available_wing_widths(model_put_short, chain_quotes, "put", wing_widths) if chain_quotes else wing_widths
                m_put_spreads = build_spread_side("put", model_put_short, mpw, spx_ref, vix_level, dte, chain_quotes=chain_quotes)

        tiers.append(SpreadTier(
            label       = label,
            risk_level  = risk,
            range_pct   = range_pct,
            call_short  = call_short,
            put_short   = put_short,
            call_spreads= call_spreads,
            put_spreads = put_spreads,
            model_call_short  = model_call_short if em_adjusted_call else None,
            model_put_short   = model_put_short  if em_adjusted_put  else None,
            model_call_spreads= m_call_spreads,
            model_put_spreads = m_put_spreads,
        ))

    return tiers


# =============================================================================
# PERSISTENCE — re-exported from spread_persistence.py
# =============================================================================

from range_finder.spread_persistence import (  # noqa: F401
    init_spread_log_table,
    log_spread_plan,
    update_outcome,
    update_expiration_outcome,
    get_spread_log,
    print_spread_plan,
)
