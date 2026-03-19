from phase1.expected_move import (
    find_atm_straddle,
    compute_expected_move,
    compute_overnight_move,
    classify_session,
    build_expected_move_analysis,
)


def _make_option(strike, bid, ask):
    return {"strike": strike, "bid": bid, "ask": ask, "openInterest": 100}


def test_find_atm_straddle_picks_nearest_strike():
    calls = [_make_option(5640, 12.0, 14.0), _make_option(5650, 8.0, 10.0), _make_option(5660, 5.0, 7.0)]
    puts = [_make_option(5640, 4.0, 6.0), _make_option(5650, 7.0, 9.0), _make_option(5660, 11.0, 13.0)]
    spot = 5651.0

    result = find_atm_straddle(calls, puts, spot)
    assert result is not None
    assert result["strike"] == 5650
    assert result["call_mid"] == 9.0   # (8+10)/2
    assert result["put_mid"] == 8.0    # (7+9)/2
    assert result["straddle_price"] == 17.0


def test_find_atm_straddle_skips_crossed_quotes():
    calls = [_make_option(5650, 10.0, 8.0)]  # crossed: bid > ask
    puts = [_make_option(5650, 7.0, 9.0)]
    spot = 5650.0

    result = find_atm_straddle(calls, puts, spot)
    assert result is None


def test_find_atm_straddle_returns_none_when_no_common_strikes():
    calls = [_make_option(5650, 8.0, 10.0)]
    puts = [_make_option(5660, 7.0, 9.0)]
    spot = 5655.0

    result = find_atm_straddle(calls, puts, spot)
    assert result is None


def test_compute_expected_move_levels():
    straddle = {"straddle_price": 46.0, "strike": 5650}
    spot = 5650.0

    em = compute_expected_move(straddle, spot)
    assert em["expected_move_pts"] == 46.0
    assert em["upper_level"] == 5696.0
    assert em["lower_level"] == 5604.0
    assert em["expected_move_pct"] > 0


def test_compute_expected_move_none_on_no_straddle():
    em = compute_expected_move(None, 5650.0)
    assert em["expected_move_pts"] is None
    assert em["upper_level"] is None


def test_compute_overnight_move_up():
    result = compute_overnight_move(5660.0, 5650.0, source="spx")
    assert result["overnight_move_pts"] == 10.0
    assert result["direction"] == "up"


def test_compute_overnight_move_down():
    result = compute_overnight_move(5640.0, 5650.0, source="spx")
    assert result["overnight_move_pts"] == -10.0
    assert result["direction"] == "down"


def test_compute_overnight_move_handles_zero_prevclose():
    result = compute_overnight_move(5650.0, 0.0)
    assert result["overnight_move_pts"] is None


def test_classify_session_pin_day():
    result = classify_session(
        expected_move_pts=46.0,
        overnight_move_pts=5.0,   # 5/46 ≈ 11%, well below 40%
        gamma_regime="Positive Gamma",
    )
    assert result["classification"] == "Pin Day"
    assert result["move_ratio_label"] == "low"
    assert result["bias"] == "range-bound"


def test_classify_session_trend_day():
    result = classify_session(
        expected_move_pts=46.0,
        overnight_move_pts=10.0,  # 10/46 ≈ 22%, below 40%
        gamma_regime="Negative Gamma",
    )
    assert result["classification"] == "Trend Day"
    assert result["bias"] == "directional"


def test_classify_session_exhaustion_day():
    result = classify_session(
        expected_move_pts=46.0,
        overnight_move_pts=38.0,  # 38/46 ≈ 83%, above 70%
        gamma_regime="Positive Gamma",
    )
    assert result["classification"] == "Exhaustion Day"
    assert result["bias"] == "mean-revert"


def test_classify_session_extension_day():
    result = classify_session(
        expected_move_pts=46.0,
        overnight_move_pts=-40.0,  # abs(40)/46 ≈ 87%, above 70%
        gamma_regime="Negative Gamma",
    )
    assert result["classification"] == "Extension Day"
    assert result["bias"] == "continued-trend"


def test_classify_session_moderate_ratio():
    result = classify_session(
        expected_move_pts=46.0,
        overnight_move_pts=25.0,  # 25/46 ≈ 54%, between 40-70%
        gamma_regime="Positive Gamma",
    )
    assert result["move_ratio_label"] == "moderate"
    assert "Mixed" in result["classification"]


def test_classify_session_handles_none():
    result = classify_session(None, 10.0, "Positive Gamma")
    assert result["classification"] is None


def test_build_expected_move_analysis_full():
    calls = [_make_option(5650, 20.0, 26.0)]
    puts = [_make_option(5650, 18.0, 22.0)]
    spot = 5650.0
    prev_close = 5640.0

    result = build_expected_move_analysis(
        spot=spot,
        prev_close=prev_close,
        zero_gamma=5645.0,
        gamma_regime="Positive Gamma",
        calls_0dte=calls,
        puts_0dte=puts,
    )

    assert result["expected_move"]["expected_move_pts"] == 43.0  # 23 + 20
    assert result["overnight_move"]["overnight_move_pts"] == 10.0
    assert result["classification"]["classification"] is not None
    assert result["level_context"]["zero_gamma_within_em"] is True
    assert result["spy_proxy"] is None


def test_build_expected_move_analysis_with_spy_proxy():
    calls = [_make_option(5650, 20.0, 26.0)]
    puts = [_make_option(5650, 18.0, 22.0)]

    spy_quote = {"last": 565.5, "prevclose": 564.0, "bid": 565.4, "ask": 565.6}

    result = build_expected_move_analysis(
        spot=5650.0,
        prev_close=5640.0,
        zero_gamma=5645.0,
        gamma_regime="Positive Gamma",
        calls_0dte=calls,
        puts_0dte=puts,
        spy_quote=spy_quote,
    )

    assert result["spy_proxy"] is not None
    assert result["spy_proxy"]["spy_move_pct"] > 0
    assert result["spy_proxy"]["implied_spx_move_pts"] > 0
