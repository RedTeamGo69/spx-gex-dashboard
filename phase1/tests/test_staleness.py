from phase1.staleness import build_staleness_info


def test_staleness_info_high_case():
    calendar_snapshot = {"now_ny": "2026-03-15T12:00:00-04:00"}
    spot_info = {
        "market_open": True,
        "source": "implied weighted median (5 strikes)",
        "parity_attempted": True,
        "parity_chain_status": "ok",
        "parity_diagnostics": {
            "hard_filter_pass_count": 5,
            "call_quality": {"total": 100, "no_two_sided_quote": 2, "wide_spread": 2, "crossed": 0, "locked": 0, "crossed_or_locked": 0},
            "put_quality": {"total": 100, "no_two_sided_quote": 3, "wide_spread": 2, "crossed": 0, "locked": 0, "crossed_or_locked": 0},
        },
    }
    stats = {
        "failed_exp_count": 0,
        "strike_support_avg": 80.0,
        "fragile_strike_count": 0,
    }

    out = build_staleness_info(calendar_snapshot, spot_info, stats)
    assert out["freshness_label"] == "High"


def test_staleness_info_penalizes_closed_market():
    calendar_snapshot = {}
    spot_info = {
        "market_open": False,
        "source": "vendor (forced, market closed)",
        "parity_attempted": False,
        "parity_chain_status": None,
        "parity_diagnostics": None,
    }
    stats = {
        "failed_exp_count": 0,
        "strike_support_avg": 80.0,
        "fragile_strike_count": 0,
    }

    out = build_staleness_info(calendar_snapshot, spot_info, stats)
    assert out["freshness_score"] < 100
    assert any("closed" in r.lower() for r in out["reasons"])


def test_staleness_info_penalizes_quote_quality_problems():
    calendar_snapshot = {}
    spot_info = {
        "market_open": True,
        "source": "implied weighted median (3 strikes)",
        "parity_attempted": True,
        "parity_chain_status": "ok",
        "parity_diagnostics": {
            "hard_filter_pass_count": 2,
            "call_quality": {"total": 100, "no_two_sided_quote": 40, "wide_spread": 35, "crossed": 10, "locked": 0, "crossed_or_locked": 10},
            "put_quality": {"total": 100, "no_two_sided_quote": 30, "wide_spread": 30, "crossed": 8, "locked": 0, "crossed_or_locked": 8},
        },
    }
    stats = {
        "failed_exp_count": 1,
        "strike_support_avg": 40.0,
        "fragile_strike_count": 4,
    }

    out = build_staleness_info(calendar_snapshot, spot_info, stats)
    assert out["freshness_label"] in ("Moderate", "Low")
    assert len(out["defenses_triggered"]) > 0