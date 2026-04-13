from phase1.futures_data import build_futures_context


def test_build_futures_context_legacy_fallback_no_es_prevclose():
    """When es_prevclose is missing, fall back to es_last - spx_prevclose.
    Still correct enough for manual-entry users who don't know ES's prior
    close; carries the basis as a small systematic bias (see docstring)."""
    ctx = build_futures_context(
        es_last=6650.0, es_high=6670.0, es_low=6620.0,
        spx_prevclose=6640.0, source="manual",
    )
    assert ctx is not None
    assert ctx["overnight_move_pts"] == 10.0  # 6650 - 6640
    assert ctx["direction"] == "up"
    assert ctx["overnight_range_pts"] == 50.0
    assert ctx["max_overnight_move"] == 30.0
    assert ctx["source"] == "manual"
    assert ctx["basis_clean"] is False


def test_build_futures_context_basis_clean_with_es_prevclose():
    """When es_prevclose is provided, compute the move as es_last - es_prevclose
    (a pure ES-to-ES move) which eliminates the ES-SPX basis."""
    # ES trades at a +3pt premium to SPX (typical basis from dividends/carry).
    # SPX closed at 6640, ES settled at 6643. Overnight, ES moves up to 6650.
    # The true overnight move is +7 (6650 - 6643), NOT +10 (6650 - 6640).
    ctx = build_futures_context(
        es_last=6650.0, es_high=6655.0, es_low=6641.0,
        spx_prevclose=6640.0, es_prevclose=6643.0,
        source="yahoo_es_f",
    )
    assert ctx["overnight_move_pts"] == 7.0      # 6650 - 6643, basis removed
    assert ctx["direction"] == "up"
    assert ctx["overnight_high_move"] == 12.0    # 6655 - 6643
    assert ctx["overnight_low_move"] == -2.0     # -(6643 - 6641)
    assert ctx["basis_clean"] is True
    assert ctx["es_prevclose"] == 6643.0


def test_build_futures_context_basis_clean_down_move():
    # Symmetric check: ES basis +3, ES moves DOWN overnight.
    ctx = build_futures_context(
        es_last=6600.0, es_high=6650.0, es_low=6590.0,
        spx_prevclose=6640.0, es_prevclose=6643.0,
        source="yahoo_es_f",
    )
    assert ctx["overnight_move_pts"] == -43.0    # 6600 - 6643
    assert ctx["direction"] == "down"
    assert ctx["overnight_high_move"] == 7.0     # 6650 - 6643
    assert ctx["overnight_low_move"] == -53.0    # -(6643 - 6590)
    assert ctx["basis_clean"] is True


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
