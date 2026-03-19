import phase1.gex_engine as gex_engine
from phase1.scenarios import build_default_scenarios, run_scenario_engine


def test_build_default_scenarios_has_base():
    scenarios = build_default_scenarios()
    assert len(scenarios) >= 5
    assert scenarios[0]["name"] == "Base"


def test_compute_strike_gex_from_all_options_returns_df():
    all_options = [
        (5000, 100, 0.20, 1, 1/365),
        (5000, 100, 0.20, -1, 1/365),
        (5050, 50, 0.18, 1, 1/365),
    ]

    df = gex_engine.compute_strike_gex_from_all_options(all_options, spot=5000, r=0.04)

    assert not df.empty
    assert "strike" in df.columns
    assert "net_gex" in df.columns


def test_run_scenario_engine_returns_expected_columns():
    all_options = [
        (4950, 120, 0.22, -1, 1/365),
        (5000, 140, 0.20, 1, 1/365),
        (5050, 110, 0.19, 1, 1/365),
    ]

    df = run_scenario_engine(all_options, base_spot=5000, base_r=0.04)

    assert not df.empty
    assert "scenario" in df.columns
    assert "spot" in df.columns
    assert "rate" in df.columns
    assert "call_wall" in df.columns
    assert "put_wall" in df.columns
    assert "zero_gamma" in df.columns
    assert "zero_gamma_type" in df.columns
    assert "gamma_regime" in df.columns
    assert "zero_gamma_move" in df.columns