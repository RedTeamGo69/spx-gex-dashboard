import pandas as pd

from phase1.wall_credibility import build_wall_credibility


def test_build_wall_credibility_returns_all_sections():
    levels = {
        "call_wall": 5050.0,
        "put_wall": 4950.0,
        "zero_gamma": 5000.0,
    }

    strike_support_df = pd.DataFrame(
        {
            "strike": [4950, 5000, 5050],
            "support_score": [72.0, 88.0, 81.0],
            "support_label": ["Moderate", "High", "High"],
            "supporting_expirations": [2, 3, 2],
            "total_oi": [900, 1500, 1200],
            "abs_net_gex": [800, 1200, 1000],
        }
    )

    sensitivity_df = pd.DataFrame(
        {
            "shock_pct": [-0.005, 0.0, 0.005],
            "shocked_spot": [4975, 5000, 5025],
            "zero_gamma": [4998, 5000, 5002],
            "spot_minus_zero_gamma": [-23, 0, 23],
            "regime": ["Negative Gamma", "At Zero Gamma", "Positive Gamma"],
        }
    )

    confidence_info = {"score": 88.0, "label": "High"}
    staleness_info = {"freshness_score": 82.0, "freshness_label": "Moderate"}

    out = build_wall_credibility(
        levels=levels,
        strike_support_df=strike_support_df,
        sensitivity_df=sensitivity_df,
        confidence_info=confidence_info,
        staleness_info=staleness_info,
    )

    assert "call_wall" in out
    assert "put_wall" in out
    assert "zero_gamma" in out
    assert "score" in out["call_wall"]
    assert "score" in out["zero_gamma"]


def test_zero_gamma_credibility_penalizes_large_range():
    levels = {
        "call_wall": 5050.0,
        "put_wall": 4950.0,
        "zero_gamma": 5000.0,
    }

    strike_support_df = pd.DataFrame(
        {
            "strike": [5000],
            "support_score": [80.0],
            "support_label": ["High"],
            "supporting_expirations": [3],
            "total_oi": [1500],
            "abs_net_gex": [1200],
        }
    )

    sensitivity_df = pd.DataFrame(
        {
            "shock_pct": [-0.005, 0.0, 0.005],
            "shocked_spot": [4975, 5000, 5025],
            "zero_gamma": [4970, 5000, 5035],
            "spot_minus_zero_gamma": [5, 0, -10],
            "regime": ["Positive Gamma", "At Zero Gamma", "Negative Gamma"],
        }
    )

    confidence_info = {"score": 90.0, "label": "High"}
    staleness_info = {"freshness_score": 90.0, "freshness_label": "High"}

    out = build_wall_credibility(
        levels=levels,
        strike_support_df=strike_support_df,
        sensitivity_df=sensitivity_df,
        confidence_info=confidence_info,
        staleness_info=staleness_info,
    )

    assert out["zero_gamma"]["zg_range"] > 40
    assert out["zero_gamma"]["score"] < 85