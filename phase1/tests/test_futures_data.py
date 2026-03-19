from phase1.futures_data import build_futures_context


def test_build_futures_context_basic():
    ctx = build_futures_context(
        es_last=6650.0, es_high=6670.0, es_low=6620.0,
        spx_prevclose=6640.0, source="manual",
    )
    assert ctx is not None
    assert ctx["overnight_move_pts"] == 10.0
    assert ctx["direction"] == "up"
    assert ctx["overnight_range_pts"] == 50.0
    assert ctx["max_overnight_move"] == 30.0  # max(6670-6640, 6640-6620)
    assert ctx["source"] == "manual"


def test_build_futures_context_down_move():
    ctx = build_futures_context(
        es_last=6600.0, es_high=6650.0, es_low=6590.0,
        spx_prevclose=6640.0, source="yahoo_es_f",
    )
    assert ctx["overnight_move_pts"] == -40.0
    assert ctx["direction"] == "down"
    assert ctx["overnight_high_move"] == 10.0    # 6650 - 6640
    assert ctx["overnight_low_move"] == -50.0    # -(6640 - 6590)


def test_build_futures_context_no_high_low():
    ctx = build_futures_context(
        es_last=6650.0, es_high=None, es_low=None,
        spx_prevclose=6640.0, source="manual",
    )
    assert ctx is not None
    assert ctx["overnight_move_pts"] == 10.0
    assert ctx["overnight_range_pts"] is None


def test_build_futures_context_returns_none_on_bad_input():
    assert build_futures_context(None, None, None, 6640.0) is None
    assert build_futures_context(0, None, None, 6640.0) is None
    assert build_futures_context(6650.0, None, None, 0) is None
