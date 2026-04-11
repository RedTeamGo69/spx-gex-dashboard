import json
import pandas as pd

import phase1.dashboard as dashboard


def test_build_bar_chart_json_returns_valid_json():
    df = pd.DataFrame({"strike": [5000, 5005], "net_gex": [1000, -500]})
    js = dashboard.build_bar_chart_json(
        df=df,
        strikes=df["strike"].values,
        net_gex=df["net_gex"].values,
        levels={"call_wall": 5005, "put_wall": 5000, "zero_gamma": 5002.5},
        spot=5001,
    )
    parsed = json.loads(js)
    assert "data" in parsed
    assert "layout" in parsed
    assert len(parsed["data"]) == 1


def test_build_status_html_includes_provenance():
    stats = {
        "direct_iv_count": 10,
        "synthetic_iv_count": 5,
        "skipped_count": 0,
        "skipped_oi": 0.0,
        "failed_expirations": [],
        "hybrid_iv_mode": True,
    }
    run_metadata = {
        "calendar_snapshot": {
            "now_ny": "2026-03-14T12:00:00-04:00",
            "cash_market_open": False,
            "options_market_open": False,
        },
        "risk_free": {"label": "fallback default 4.50%"},
        "selection": {"selected_expirations_count": 1, "heatmap_expirations_count": 3},
        "config": {"strike_range_pct": 0.05, "profile_step": 1.0, "hybrid_iv_mode": True},
        "staleness": {
            "freshness_score": 72.0,
            "freshness_label": "Moderate",
            "trading_guidance": "Use caution: market-data quality is mixed.",
            "reasons": ["Test reason"],
        },
    }

    html = dashboard.build_status_html(stats, spot_info=None, run_metadata=run_metadata)

    assert "Run provenance" in html
    assert "fallback default 4.50%" in html
    assert "Direct IV contracts" in html
    assert "Market-data freshness" in html


def test_build_wall_credibility_html_contains_title():
    info = {
        "call_wall": {
            "level_name": "call_wall",
            "level_value": 5050.0,
            "score": 84.0,
            "label": "Moderate",
            "reasons": ["Test reason 1"],
        },
        "put_wall": {
            "level_name": "put_wall",
            "level_value": 4950.0,
            "score": 90.0,
            "label": "High",
            "reasons": ["Test reason 2"],
        },
        "zero_gamma": {
            "level_name": "zero_gamma",
            "level_value": 5000.0,
            "score": 76.0,
            "label": "Moderate",
            "reasons": ["Test reason 3"],
        },
    }

    html = dashboard.build_wall_credibility_html(info)
    assert "Wall Credibility" in html
    assert "Call Wall" in html
    assert "Zero Gamma" in html
