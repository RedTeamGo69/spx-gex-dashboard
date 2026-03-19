import json
import pandas as pd

import phase1.dashboard as dashboard


def test_build_heatmap_html_handles_empty_df():
    html = dashboard.build_heatmap_html(pd.DataFrame(), spot=5000, title="Test Heatmap", is_iv=False)
    assert "No data" in html
    assert "Test Heatmap" in html


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


def test_build_profile_chart_json_returns_valid_json():
    profile_df = pd.DataFrame({
        "price": [4990, 5000, 5010],
        "total_gex": [-100, 0, 120],
    })
    regime_info = {
        "regime": "Negative Gamma",
        "color": "#ff5252",
        "distance_text": "-10.00 pts from zero gamma",
        "note": "test note",
    }

    js = dashboard.build_profile_chart_json(
        profile_df=profile_df,
        levels={"zero_gamma": 5000},
        spot=4990,
        regime_info=regime_info,
    )
    parsed = json.loads(js)
    assert "data" in parsed
    assert "layout" in parsed
    assert len(parsed["data"]) == 1

def test_build_sensitivity_html_contains_title():
    df = pd.DataFrame(
        {
            "shock_pct": [-0.005, 0.0, 0.005],
            "shocked_spot": [4975.0, 5000.0, 5025.0],
            "zero_gamma": [5010.0, 5005.0, 5000.0],
            "spot_minus_zero_gamma": [-35.0, -5.0, 25.0],
            "zero_gamma_type": ["Fallback node", "True crossing", "True crossing"],
            "regime": ["Negative Gamma", "Negative Gamma", "Positive Gamma"],
        }
    )

    html = dashboard.build_sensitivity_html(df)
    assert "Zero Gamma Sensitivity" in html
    assert "Positive Gamma" in html
    assert "True crossing" in html    

def test_build_strike_support_html_contains_title():
    df = pd.DataFrame(
        {
            "strike": [5000, 5050],
            "support_score": [82.0, 44.0],
            "support_label": ["High", "Low"],
            "supporting_expirations": [3, 1],
            "total_oi": [1200, 80],
        }
    )

    html = dashboard.build_strike_support_html(df)
    assert "Strike Support" in html
    assert "High" in html
    assert "Low" in html


def test_build_expiration_support_html_contains_title():
    df = pd.DataFrame(
        {
            "expiration": ["2026-03-20", "2026-03-21"],
            "support_score": [88.0, 57.0],
            "support_label": ["High", "Moderate"],
            "strikes_used": [12, 5],
            "total_oi": [5000, 800],
        }
    )

    html = dashboard.build_expiration_support_html(df)
    assert "Expiration Support" in html
    assert "Moderate" in html

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

def test_build_scenarios_html_contains_title():
    df = pd.DataFrame(
        {
            "scenario": ["Base", "Spot +0.50%"],
            "call_wall": [5050.0, 5060.0],
            "put_wall": [4950.0, 4960.0],
            "zero_gamma": [5000.0, 5010.0],
            "gamma_regime": ["Negative Gamma", "Positive Gamma"],
        }
    )

    html = dashboard.build_scenarios_html(df)
    assert "Scenario Engine" in html
    assert "Spot +0.50%" in html
    assert "Positive Gamma" in html