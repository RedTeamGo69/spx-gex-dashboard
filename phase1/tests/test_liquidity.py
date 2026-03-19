from phase1.liquidity import (
    label_support_score,
    build_strike_support_df,
    build_expiration_support_df,
)


def test_label_support_score_thresholds():
    assert label_support_score(80) == "High"
    assert label_support_score(60) == "Moderate"
    assert label_support_score(40) == "Low"


def test_build_strike_support_df_prefers_better_supported_strike():
    records = [
        {
            "expiration": "2026-03-20",
            "strike": 5000,
            "oi": 500,
            "iv_source": "direct_iv",
            "spread": 0.5,
            "is_call": True,
            "is_put": False,
            "net_gex": 1000,
            "abs_gex": 1000,
        },
        {
            "expiration": "2026-03-21",
            "strike": 5000,
            "oi": 600,
            "iv_source": "direct_iv",
            "spread": 0.6,
            "is_call": False,
            "is_put": True,
            "net_gex": -900,
            "abs_gex": 900,
        },
        {
            "expiration": "2026-03-20",
            "strike": 5050,
            "oi": 50,
            "iv_source": "synthetic_iv",
            "spread": 4.0,
            "is_call": True,
            "is_put": False,
            "net_gex": 200,
            "abs_gex": 200,
        },
    ]

    df = build_strike_support_df(records, selected_exp_count=2)

    score_5000 = float(df.loc[df["strike"] == 5000, "support_score"].iloc[0])
    score_5050 = float(df.loc[df["strike"] == 5050, "support_score"].iloc[0])

    assert score_5000 > score_5050


def test_build_expiration_support_df_returns_rows():
    records = [
        {
            "expiration": "2026-03-20",
            "strike": 5000,
            "oi": 500,
            "iv_source": "direct_iv",
            "spread": 0.5,
            "is_call": True,
            "is_put": False,
            "net_gex": 1000,
            "abs_gex": 1000,
        },
        {
            "expiration": "2026-03-21",
            "strike": 5050,
            "oi": 100,
            "iv_source": "synthetic_iv",
            "spread": 2.0,
            "is_call": False,
            "is_put": True,
            "net_gex": -300,
            "abs_gex": 300,
        },
    ]

    df = build_expiration_support_df(records)

    assert len(df) == 2
    assert "support_score" in df.columns
    assert "support_label" in df.columns