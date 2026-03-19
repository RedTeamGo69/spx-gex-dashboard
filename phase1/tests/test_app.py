from phase1.app import validate_runtime_inputs, select_heatmap_exps


def test_validate_runtime_inputs_rejects_empty():
    assert validate_runtime_inputs("") is False


def test_validate_runtime_inputs_rejects_placeholder():
    assert validate_runtime_inputs("YOUR_TOKEN_HERE") is False


def test_validate_runtime_inputs_accepts_realish_value():
    assert validate_runtime_inputs("abc123") is True


def test_select_heatmap_exps_returns_first_future_expirations():
    avail = ["2026-03-13", "2026-03-20", "2026-03-23", "2026-03-27"]
    out = select_heatmap_exps(avail, "2026-03-20", count=2)
    assert out == ["2026-03-20", "2026-03-23"]
