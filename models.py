"""
Shared data models used across UI modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GEXData:
    spot: float
    spot_source: str
    spot_info: dict
    rfr: float
    rfr_info: dict
    avail: list
    target_exps: list
    gex_df: Any  # pd.DataFrame
    hm_gex: Any
    hm_iv: Any
    stats: dict
    all_options: list
    levels: dict
    profile_df: Any
    sensitivity_df: Any
    scenarios_df: Any
    staleness_info: dict
    confidence_info: dict
    wall_cred: dict
    regime_info: dict
    calendar_snapshot: dict
    run_time: str
    prev_close: float
    spy_quote: dict | None
    dte0_calls: list
    dte0_puts: list
    market_open: bool
    yahoo_es: dict | None
    chain_cache: dict | None
