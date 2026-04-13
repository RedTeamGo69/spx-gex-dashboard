from phase1.rates import (
    parse_fred_rate_response,
    parse_treasury_rate_response,
    interpolate_rate,
    DEFAULT_RISK_FREE_RATE,
)


def test_parse_fred_rate_response_returns_first_valid_observation():
    payload = {
        "observations": [
            {"date": "2026-03-15", "value": "."},
            {"date": "2026-03-14", "value": "4.31"},
        ]
    }

    result = parse_fred_rate_response(payload)

    assert result is not None
    assert abs(result["rate"] - 0.0431) < 1e-12
    assert result["source"] == "fred_dtb3"
    assert result["as_of"] == "2026-03-14"


def test_parse_fred_rate_response_returns_none_if_no_valid_values():
    payload = {
        "observations": [
            {"date": "2026-03-15", "value": "."},
            {"date": "2026-03-14", "value": "."},
        ]
    }

    result = parse_fred_rate_response(payload)
    assert result is None


def test_parse_treasury_rate_response_returns_first_record():
    payload = {
        "data": [
            {"record_date": "2026-02-28", "avg_interest_rate_amt": "4.18"}
        ]
    }

    result = parse_treasury_rate_response(payload)

    assert result is not None
    assert abs(result["rate"] - 0.0418) < 1e-12
    assert result["source"] == "treasury_monthly_average"
    assert result["as_of"] == "2026-02-28"


def test_parse_treasury_rate_response_returns_none_if_no_data():
    payload = {"data": []}
    result = parse_treasury_rate_response(payload)
    assert result is None


# ─── interpolate_rate — term structure helper ─────────────────────────────────
# A realistic short-end curve (slightly inverted, which SPX has seen recently).
SAMPLE_CURVE = {30: 0.0435, 91: 0.0430, 182: 0.0425, 365: 0.0415}


def test_interpolate_rate_scalar_passthrough():
    """A scalar is returned as-is (flat-rate back-compat)."""
    assert interpolate_rate(0.042, 30) == 0.042
    assert interpolate_rate(0.042, 9999) == 0.042
    assert interpolate_rate(0.042, 0) == 0.042


def test_interpolate_rate_none_and_empty_use_fallback():
    """None / empty curve falls through to the fallback value."""
    assert interpolate_rate(None, 30) == DEFAULT_RISK_FREE_RATE
    assert interpolate_rate({}, 30) == DEFAULT_RISK_FREE_RATE
    assert interpolate_rate(None, 30, fallback=0.99) == 0.99
    assert interpolate_rate({}, 30, fallback=0.01) == 0.01


def test_interpolate_rate_exact_hit_on_curve_points():
    """DTE that matches a curve key returns that point exactly."""
    assert abs(interpolate_rate(SAMPLE_CURVE, 30) - 0.0435) < 1e-12
    assert abs(interpolate_rate(SAMPLE_CURVE, 91) - 0.0430) < 1e-12
    assert abs(interpolate_rate(SAMPLE_CURVE, 182) - 0.0425) < 1e-12
    assert abs(interpolate_rate(SAMPLE_CURVE, 365) - 0.0415) < 1e-12


def test_interpolate_rate_linear_interpolation_between_points():
    """Values between curve points linearly interpolate."""
    # Halfway between 30 and 91 DTE
    mid_dte = (30 + 91) / 2  # 60.5
    expected = 0.0435 + (mid_dte - 30) / (91 - 30) * (0.0430 - 0.0435)
    assert abs(interpolate_rate(SAMPLE_CURVE, mid_dte) - expected) < 1e-12

    # 45-day OpEx case — the user's primary concern
    r_45 = interpolate_rate(SAMPLE_CURVE, 45)
    expected_45 = 0.0435 + (45 - 30) / (91 - 30) * (0.0430 - 0.0435)
    assert abs(r_45 - expected_45) < 1e-12
    # Sanity: should sit between 1M and 3M rates
    assert 0.0430 < r_45 < 0.0435


def test_interpolate_rate_clamps_below_shortest_point():
    """DTE shorter than the curve's min point clamps to the min (no
    extrapolation — the yield curve is flatter at the short end anyway
    and extrapolating a nearly-flat curve is a nothing-burger that can
    produce unstable numbers)."""
    assert interpolate_rate(SAMPLE_CURVE, 0) == 0.0435
    assert interpolate_rate(SAMPLE_CURVE, 5) == 0.0435
    assert interpolate_rate(SAMPLE_CURVE, 29.9) == 0.0435


def test_interpolate_rate_clamps_above_longest_point():
    """DTE longer than the curve's max point clamps to the max."""
    assert interpolate_rate(SAMPLE_CURVE, 500) == 0.0415
    assert interpolate_rate(SAMPLE_CURVE, 9999) == 0.0415


def test_interpolate_rate_non_numeric_dte_uses_fallback():
    """Garbage DTE input shouldn't crash — fall back cleanly."""
    assert interpolate_rate(SAMPLE_CURVE, "bogus", fallback=0.5) == 0.5
    assert interpolate_rate(SAMPLE_CURVE, None, fallback=0.5) == 0.5


def test_interpolate_rate_single_point_curve_acts_flat():
    """A curve with only one point clamps everywhere to that point."""
    one_point = {90: 0.0425}
    assert interpolate_rate(one_point, 5) == 0.0425
    assert interpolate_rate(one_point, 90) == 0.0425
    assert interpolate_rate(one_point, 500) == 0.0425
