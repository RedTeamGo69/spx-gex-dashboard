"""BUG-06 production boundary for GEX-derived spread influence."""

import pytest

from range_finder.gex_policy import (
    GEX_LIVE_SPREAD_INFLUENCE_ENABLED,
    live_spread_feature_columns,
    uses_disabled_gex_feature,
)
from range_finder.spread_levels import (
    DEFAULT_BUFFER_PCT,
    build_spread_plan,
    compute_legacy_gex_buffer_adjustment,
)


FORECAST = {
    "point_pct": 0.025,
    "lower_pct": 0.030,
    "upper_pct": 0.040,
    "vix_implied_pct": 0.035,
    "model_vs_vix": -0.010,
    "confidence_level": 80,
    "spx_ref_close": 6000.0,
}


def _plan(gex_normalized: float, gex_flag: int):
    return build_spread_plan(
        forecast=FORECAST,
        feature_row={
            "gex_normalized": gex_normalized,
            "gex_flag": gex_flag,
        },
        week_start="2026-08-31",
        vix_level=18.0,
        ticker="SPX",
    )


def _spread_signature(plan) -> tuple:
    return (
        plan.effective_range_pct,
        tuple((s.short_strike, s.long_strike) for s in plan.call_spreads),
        tuple((s.short_strike, s.long_strike) for s in plan.put_spreads),
    )


def test_live_buffer_and_strikes_ignore_gex_until_calibrated():
    plans = [
        _plan(gex_normalized=100.0, gex_flag=1),
        _plan(gex_normalized=-100.0, gex_flag=-1),
        _plan(gex_normalized=0.0, gex_flag=0),
    ]

    assert {plan.buffer_pct for plan in plans} == {DEFAULT_BUFFER_PCT}
    assert len({_spread_signature(plan) for plan in plans}) == 1
    assert all(
        "gex=disabled_pending_calibration" in plan.buffer_reason
        for plan in plans
    )


def test_legacy_gex_adjustment_remains_available_for_research_only():
    assert compute_legacy_gex_buffer_adjustment(
        {"gex_normalized": 100.0}
    ) == pytest.approx(-0.002)
    assert compute_legacy_gex_buffer_adjustment(
        {"gex_normalized": -100.0}
    ) == pytest.approx(0.005)
    assert compute_legacy_gex_buffer_adjustment(
        {"gex_normalized": 0.0}
    ) == pytest.approx(0.0)


def test_live_model_policy_filters_gex_but_keeps_research_column_detectable():
    original = ["har_d1", "gex_normalized", "vix_close"]

    assert GEX_LIVE_SPREAD_INFLUENCE_ENABLED is False
    assert live_spread_feature_columns(original) == ["har_d1", "vix_close"]
    assert uses_disabled_gex_feature(original) is True
    assert uses_disabled_gex_feature(live_spread_feature_columns(original)) is False
