from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

# Market calendars
CASH_CALENDAR = "NYSE"
OPTIONS_CALENDAR = "CBOE_Index_Options"

# Main strike filter around spot
STRIKE_RANGE_PCT = 0.05

# Heatmap config
HEATMAP_EXPS = 7
HEATMAP_STRIKES = 13

# Zero-gamma sweep config
ZG_SWEEP_RANGE_PCT = 0.05
ZG_SWEEP_STEP = 5.0
ZG_FINE_STEP = 0.5

# Profile curve config
PROFILE_RANGE_PCT = 0.05
PROFILE_STEP = 1.0

# Time floor: 1 minute in years
T_FLOOR = 1 / (365.25 * 24 * 60)

# Default risk-free rate used when no live rate is passed
DEFAULT_RISK_FREE_RATE = 0.045

# Trading-time T: count only market hours for time-to-expiry
# This significantly improves gamma accuracy for 0DTE/1DTE options
USE_TRADING_TIME = True
TRADING_HOURS_PER_YEAR = 252 * 6.5  # 252 trading days × 6.5 hours per session

# Fetching
MAX_WORKERS = 4
CHAIN_RETRIES = 2
CHAIN_RETRY_SLEEP = 0.6

# Parity / quote quality settings
MAX_PARITY_SPREAD = 2.0
PARITY_NEAR_SPOT_CANDIDATES = 15
PARITY_FINAL_STRIKES = 5
PARITY_RELATIVE_BAND = 0.015
PARITY_HARD_LOW_MULTIPLIER = 0.90
PARITY_HARD_HIGH_MULTIPLIER = 1.10
MIN_PARITY_STRIKES = 2

# Parity estimator method
PARITY_METHOD = "weighted_median"   # options: "median", "weighted_median"

# Weighting controls for parity
PARITY_WEIGHT_EPS = 1e-6
PARITY_SPREAD_WEIGHT_POWER = 1.0
PARITY_DISTANCE_SIGMA_PCT = 0.01

# Synthetic IV bounds and fit acceptance
SYNTH_IV_MIN = 0.01
SYNTH_IV_MAX = 3.00
SYNTH_FIT_MAX_REL_ERROR = 0.08

# Hybrid IV mode
HYBRID_IV_MODE = True

# Liquidity / support scoring
SUPPORT_MAX_SPREAD_FOR_SCORE = 10.0
SUPPORT_HIGH_THRESHOLD = 75.0
SUPPORT_MODERATE_THRESHOLD = 50.0

# Strike support scoring weights (must sum to 1.0)
STRIKE_SUPPORT_W_OI = 0.30
STRIKE_SUPPORT_W_BREADTH = 0.20
STRIKE_SUPPORT_W_CONTRACTS = 0.20
STRIKE_SUPPORT_W_DIRECT_IV = 0.15
STRIKE_SUPPORT_W_SPREAD = 0.10
STRIKE_SUPPORT_W_TWO_SIDED = 0.05

# Expiration support scoring weights (must sum to 1.0)
EXP_SUPPORT_W_OI = 0.35
EXP_SUPPORT_W_CONTRACTS = 0.25
EXP_SUPPORT_W_STRIKE_BREADTH = 0.15
EXP_SUPPORT_W_DIRECT_IV = 0.15
EXP_SUPPORT_W_SPREAD = 0.10

# Stale-data / quote-quality defenses
STALE_FRESHNESS_HIGH_THRESHOLD = 85.0
STALE_FRESHNESS_MODERATE_THRESHOLD = 65.0
STALE_NO_TWO_SIDED_RATIO_WARN = 0.25
STALE_WIDE_SPREAD_RATIO_WARN = 0.25
STALE_CROSSED_RATIO_WARN = 0.05
STALE_MIN_HARD_FILTER_PASS = 3

def build_config_snapshot() -> dict:
    return {
        "cash_calendar": CASH_CALENDAR,
        "options_calendar": OPTIONS_CALENDAR,
        "strike_range_pct": STRIKE_RANGE_PCT,
        "heatmap_exps": HEATMAP_EXPS,
        "heatmap_strikes": HEATMAP_STRIKES,
        "zg_sweep_range_pct": ZG_SWEEP_RANGE_PCT,
        "zg_sweep_step": ZG_SWEEP_STEP,
        "zg_fine_step": ZG_FINE_STEP,
        "profile_range_pct": PROFILE_RANGE_PCT,
        "profile_step": PROFILE_STEP,
        "t_floor": T_FLOOR,
        "default_risk_free_rate": DEFAULT_RISK_FREE_RATE,
        "use_trading_time": USE_TRADING_TIME,
        "trading_hours_per_year": TRADING_HOURS_PER_YEAR,
        "max_workers": MAX_WORKERS,
        "chain_retries": CHAIN_RETRIES,
        "chain_retry_sleep": CHAIN_RETRY_SLEEP,
        "max_parity_spread": MAX_PARITY_SPREAD,
        "parity_near_spot_candidates": PARITY_NEAR_SPOT_CANDIDATES,
        "parity_final_strikes": PARITY_FINAL_STRIKES,
        "parity_relative_band": PARITY_RELATIVE_BAND,
        "parity_hard_low_multiplier": PARITY_HARD_LOW_MULTIPLIER,
        "parity_hard_high_multiplier": PARITY_HARD_HIGH_MULTIPLIER,
        "min_parity_strikes": MIN_PARITY_STRIKES,
        "synth_iv_min": SYNTH_IV_MIN,
        "synth_iv_max": SYNTH_IV_MAX,
        "synth_fit_max_rel_error": SYNTH_FIT_MAX_REL_ERROR,
        "hybrid_iv_mode": HYBRID_IV_MODE,
        "parity_method": PARITY_METHOD,
        "parity_weight_eps": PARITY_WEIGHT_EPS,
        "parity_spread_weight_power": PARITY_SPREAD_WEIGHT_POWER,
        "parity_distance_sigma_pct": PARITY_DISTANCE_SIGMA_PCT,
        "support_max_spread_for_score": SUPPORT_MAX_SPREAD_FOR_SCORE,
        "support_high_threshold": SUPPORT_HIGH_THRESHOLD,
        "support_moderate_threshold": SUPPORT_MODERATE_THRESHOLD,
        "strike_support_weights": {
            "oi": STRIKE_SUPPORT_W_OI,
            "breadth": STRIKE_SUPPORT_W_BREADTH,
            "contracts": STRIKE_SUPPORT_W_CONTRACTS,
            "direct_iv": STRIKE_SUPPORT_W_DIRECT_IV,
            "spread": STRIKE_SUPPORT_W_SPREAD,
            "two_sided": STRIKE_SUPPORT_W_TWO_SIDED,
        },
        "exp_support_weights": {
            "oi": EXP_SUPPORT_W_OI,
            "contracts": EXP_SUPPORT_W_CONTRACTS,
            "strike_breadth": EXP_SUPPORT_W_STRIKE_BREADTH,
            "direct_iv": EXP_SUPPORT_W_DIRECT_IV,
            "spread": EXP_SUPPORT_W_SPREAD,
        },
        "stale_freshness_high_threshold": STALE_FRESHNESS_HIGH_THRESHOLD,
        "stale_freshness_moderate_threshold": STALE_FRESHNESS_MODERATE_THRESHOLD,
        "stale_no_two_sided_ratio_warn": STALE_NO_TWO_SIDED_RATIO_WARN,
        "stale_wide_spread_ratio_warn": STALE_WIDE_SPREAD_RATIO_WARN,
        "stale_crossed_ratio_warn": STALE_CROSSED_RATIO_WARN,
        "stale_min_hard_filter_pass": STALE_MIN_HARD_FILTER_PASS,                
    }
