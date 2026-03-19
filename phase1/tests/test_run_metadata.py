from phase1.run_metadata import build_run_metadata, write_run_metadata_json


def test_build_run_metadata_has_expected_sections():
    metadata = build_run_metadata(
        tool_version="v5",
        calendar_snapshot={"now_ny": "2026-03-14T12:00:00-04:00"},
        risk_free_info={"rate": 0.045, "source": "fallback_default", "label": "fallback default 4.50%"},
        spot_info={"spot": 5000.0, "source": "tradier (forced, market closed)"},
        stats={"used_option_count": 100, "failed_exp_count": 0, "hybrid_iv_mode": True},
        selected_exps=["2026-03-20"],
        heatmap_exps=["2026-03-20", "2026-03-23"],
        config_snapshot={"profile_range_pct": 0.05},
        confidence_info={"score": 88.0, "label": "High"},
        sensitivity_rows=[{"shock_pct": 0.0, "zero_gamma": 5000.0}],
        strike_support_rows=[{"strike": 5000, "support_score": 88.0}],
        expiration_support_rows=[{"expiration": "2026-03-20", "support_score": 91.0}],
        staleness_info={"freshness_score": 82.0, "freshness_label": "Moderate"},
        wall_credibility_info={"call_wall": {"score": 82.0, "label": "Moderate"}},
        scenario_rows=[{"scenario": "Base", "zero_gamma": 5000.0}],                                
    )

    assert metadata["tool_version"] == "v5"
    assert "calendar_snapshot" in metadata
    assert "risk_free" in metadata
    assert "spot_reference" in metadata
    assert "selection" in metadata
    assert "data_quality" in metadata
    assert "config" in metadata
    assert "confidence" in metadata
    assert "sensitivity" in metadata
    assert "support" in metadata
    assert "strike_rows" in metadata["support"]
    assert "expiration_rows" in metadata["support"]
    assert "staleness" in metadata
    assert "wall_credibility" in metadata
    assert "scenarios" in metadata                  
    

def test_write_run_metadata_json_creates_file(tmp_path):
    path = tmp_path / "run_metadata.json"
    metadata = {"hello": "world"}

    out = write_run_metadata_json(metadata, str(path))

    assert path.exists()
    assert out.endswith("run_metadata.json")
