from phase1.config import build_config_snapshot


def test_build_config_snapshot_has_expected_keys():
    snap = build_config_snapshot()

    expected_keys = {
        "cash_calendar",
        "options_calendar",
        "strike_range_pct",
        "heatmap_exps",
        "heatmap_strikes",
        "zg_sweep_range_pct",
        "zg_sweep_step",
        "zg_fine_step",
        "profile_range_pct",
        "profile_step",
        "t_floor",
        "max_workers",
        "chain_retries",
        "chain_retry_sleep",
        "max_parity_spread",
        "parity_near_spot_candidates",
        "parity_final_strikes",
        "parity_relative_band",
        "parity_hard_low_multiplier",
        "parity_hard_high_multiplier",
        "min_parity_strikes",
        "synth_iv_min",
        "synth_iv_max",
        "hybrid_iv_mode",
        "synth_fit_max_rel_error",
        "support_max_spread_for_score",
        "support_high_threshold",
        "support_moderate_threshold",
        "stale_freshness_high_threshold",
        "stale_freshness_moderate_threshold",
        "stale_no_two_sided_ratio_warn",
        "stale_wide_spread_ratio_warn",
        "stale_crossed_ratio_warn",
        "stale_min_hard_filter_pass",                
    }

    assert expected_keys.issubset(set(snap.keys()))
