from __future__ import annotations

import json
from pathlib import Path


def build_run_metadata(
    tool_version: str,
    calendar_snapshot: dict,
    risk_free_info: dict,
    spot_info: dict,
    stats: dict,
    selected_exps: list[str],
    heatmap_exps: list[str],
    config_snapshot: dict,
    confidence_info: dict | None = None,
    sensitivity_rows: list[dict] | None = None,
    strike_support_rows: list[dict] | None = None,
    expiration_support_rows: list[dict] | None = None,
    staleness_info: dict | None = None,
    wall_credibility_info: dict | None = None,
    scenario_rows: list[dict] | None = None,
    expected_move_info: dict | None = None,
) -> dict:
    return {
        "tool_version": tool_version,
        "run_timestamp_ny": calendar_snapshot.get("now_ny"),
        "calendar_snapshot": calendar_snapshot,
        "risk_free": risk_free_info,
        "spot_reference": {
            "spot": spot_info.get("spot"),
            "source": spot_info.get("source"),
            "tradier_spot": spot_info.get("tradier_spot"),
            "implied_spot": spot_info.get("implied_spot"),
            "market_open": spot_info.get("market_open"),
            "parity_attempted": spot_info.get("parity_attempted"),
            "parity_chain_status": spot_info.get("parity_chain_status"),
            "nearest_exp": spot_info.get("nearest_exp"),
            "T_years": spot_info.get("T_years"),
            "parity_diagnostics": spot_info.get("parity_diagnostics"),
            "expiration_close_ny": spot_info.get("expiration_close_ny"),
        },
        "selection": {
            "selected_expirations": selected_exps,
            "selected_expirations_count": len(selected_exps),
            "heatmap_expirations": heatmap_exps,
            "heatmap_expirations_count": len(heatmap_exps),
        },
        "data_quality": {
            "used_option_count": stats.get("used_option_count"),
            "direct_iv_count": stats.get("direct_iv_count"),
            "synthetic_iv_count": stats.get("synthetic_iv_count"),
            "skipped_count": stats.get("skipped_count"),
            "skipped_oi": stats.get("skipped_oi"),
            "coverage_ratio": stats.get("coverage_ratio"),
            "failed_exp_count": stats.get("failed_exp_count"),
            "failed_expirations": stats.get("failed_expirations"),
            "hybrid_iv_mode": stats.get("hybrid_iv_mode"),
            "synthetic_fit_accept_count": stats.get("synthetic_fit_accept_count"),
            "synthetic_fit_reject_count": stats.get("synthetic_fit_reject_count"),
            "no_model_input_count": stats.get("no_model_input_count"),
            "synthetic_fit_avg_rel_error": stats.get("synthetic_fit_avg_rel_error"),
            "synthetic_fit_max_rel_error": stats.get("synthetic_fit_max_rel_error"),
            "strike_support_avg": stats.get("strike_support_avg"),
            "fragile_strike_count": stats.get("fragile_strike_count"),
            "expiration_support_avg": stats.get("expiration_support_avg"),            
        },
        "staleness": staleness_info or {},
        "confidence": confidence_info or {},
        "wall_credibility": wall_credibility_info or {},
        "scenarios": scenario_rows or [],
        "sensitivity": sensitivity_rows or [],
        "support": {
            "strike_rows": strike_support_rows or [],
            "expiration_rows": expiration_support_rows or [],
        },
        "expected_move": expected_move_info or {},
        "config": config_snapshot,
    }


def write_run_metadata_json(metadata: dict, output_path: str) -> str:
    path = Path(output_path)
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return str(path)
