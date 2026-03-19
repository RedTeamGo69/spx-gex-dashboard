from phase1.confidence import build_run_confidence


def test_build_run_confidence_high_case():
    stats = {
        "coverage_ratio": 1.0,
        "failed_exp_count": 0,
        "synthetic_fit_reject_count": 0,
        "synthetic_iv_count": 0,
        "used_option_count": 100,
        "synthetic_fit_max_rel_error": None,
    }
    spot_info = {
        "source": "implied weighted median (5 strikes)",
        "parity_attempted": True,
        "parity_diagnostics": {
            "common_usable_strikes": 10,
            "hard_filter_pass_count": 5,
            "simple_median_spot": 5000.0,
            "weighted_median_spot": 5001.0,
        },
    }

    out = build_run_confidence(stats, spot_info)
    assert out["label"] == "High"
    assert out["score"] >= 85


def test_build_run_confidence_penalizes_failed_expirations():
    stats = {
        "coverage_ratio": 1.0,
        "failed_exp_count": 2,
        "synthetic_fit_reject_count": 0,
        "synthetic_iv_count": 0,
        "used_option_count": 100,
        "synthetic_fit_max_rel_error": None,
    }
    spot_info = {
        "source": "implied weighted median (5 strikes)",
        "parity_attempted": True,
        "parity_diagnostics": {},
    }

    out = build_run_confidence(stats, spot_info)
    assert out["score"] < 100
    assert out["failed_exp_count"] == 2


def test_build_run_confidence_penalizes_tradier_spot():
    stats = {
        "coverage_ratio": 1.0,
        "failed_exp_count": 0,
        "synthetic_fit_reject_count": 0,
        "synthetic_iv_count": 0,
        "used_option_count": 100,
        "synthetic_fit_max_rel_error": None,
    }
    spot_info = {
        "source": "tradier (forced, market closed)",
        "parity_attempted": False,
        "parity_diagnostics": {},
    }

    out = build_run_confidence(stats, spot_info)
    assert out["score"] < 100
    assert out["spot_source"].startswith("tradier")


def test_build_run_confidence_penalizes_synthetic_rejections():
    stats = {
        "coverage_ratio": 1.0,
        "failed_exp_count": 0,
        "synthetic_fit_reject_count": 3,
        "synthetic_iv_count": 20,
        "used_option_count": 100,
        "synthetic_fit_max_rel_error": 0.06,
    }
    spot_info = {
        "source": "implied weighted median (5 strikes)",
        "parity_attempted": True,
        "parity_diagnostics": {},
    }

    out = build_run_confidence(stats, spot_info)
    assert out["score"] < 100
    assert any("synthetic" in reason.lower() for reason in out["reasons"])

def test_build_run_confidence_penalizes_low_freshness():
    stats = {
        "coverage_ratio": 1.0,
        "failed_exp_count": 0,
        "synthetic_fit_reject_count": 0,
        "synthetic_iv_count": 0,
        "used_option_count": 100,
        "synthetic_fit_max_rel_error": None,
    }
    spot_info = {
        "source": "implied weighted median (5 strikes)",
        "parity_attempted": True,
        "parity_diagnostics": {},
    }
    staleness_info = {
        "freshness_score": 55.0,
        "freshness_label": "Low",
    }

    out = build_run_confidence(stats, spot_info, staleness_info=staleness_info)
    assert out["score"] < 100