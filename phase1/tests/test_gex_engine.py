import pandas as pd

import phase1.gex_engine as gex_engine


def test_bs_gamma_vec_returns_expected_shape():
    out = gex_engine.bs_gamma_vec(
        S_arr=[5000, 5005],
        K_arr=[5000, 5010],
        T_arr=[1/365, 1/365],
        r=0.04,
        sigma_arr=[0.20, 0.22],
    )
    assert out.shape == (2, 2)


def test_get_gamma_regime_text_positive():
    info = gex_engine.get_gamma_regime_text(spot=5050, zero_gamma=5000)
    assert info["regime"] == "Positive Gamma"


def test_get_gamma_regime_text_negative():
    info = gex_engine.get_gamma_regime_text(spot=4950, zero_gamma=5000)
    assert info["regime"] == "Negative Gamma"


def test_find_key_levels_empty_df_returns_spot():
    empty = pd.DataFrame(columns=["strike", "net_gex"])
    levels = gex_engine.find_key_levels(empty, spot=5000)
    assert levels["call_wall"] == 5000
    assert levels["put_wall"] == 5000
    assert levels["zero_gamma"] == 5000


def test_calculate_all_basic_fake_client():
    class FakeClient:
        def prefetch_chains(self, ticker, expirations):
            return None

        def get_chain_cached(self, ticker, exp):
            return {
                "status": "ok",
                "calls": [
                    {
                        "strike": 5000,
                        "openInterest": 100,
                        "impliedVolatility": 0.20,
                        "vendorGamma": 0.0,
                        "bid": 10.0,
                        "ask": 10.5,
                        "mid": 10.25,
                    }
                ],
                "puts": [
                    {
                        "strike": 5000,
                        "openInterest": 100,
                        "impliedVolatility": 0.20,
                        "vendorGamma": 0.0,
                        "bid": 10.0,
                        "ask": 10.5,
                        "mid": 10.25,
                    }
                ],
                "error": None,
            }

    client = FakeClient()

    gex_df, stats, all_options, strike_support_df, expiration_support_df = gex_engine.calculate_all(
        client=client,
        ticker="SPX",
        target_exps=["2026-03-20"],
        spot=5000,
        r=0.04,
    )

    assert not gex_df.empty
    assert stats["used_option_count"] == 2
    assert len(all_options) == 2
    assert not strike_support_df.empty
    assert not expiration_support_df.empty


def test_zero_gamma_sweep_details_flags_fallback_when_no_crossing():
    details = gex_engine.zero_gamma_sweep_details(
        all_options=[(5000, 100, 0.20, 1, 1/365)],
        spot=5000,
        r=0.04,
    )

    assert details["is_true_crossing"] is False
    assert details["zero_gamma_type"] == "Fallback node"
    assert details["method"] == "min_abs_fallback"


def test_zero_gamma_sweep_details_detects_true_crossing(monkeypatch):
    import numpy as np

    def fake_sweep(_all_options, prices, _r):
        # _sweep_gex_at_prices returns total GEX (already Spot²-scaled)
        return np.array([float(p - 100.0) for p in prices])

    import phase1.zero_gamma as zero_gamma_mod
    monkeypatch.setattr(zero_gamma_mod, "_sweep_gex_at_prices", fake_sweep)

    details = gex_engine.zero_gamma_sweep_details(
        all_options=[(1, 1, 1, 1, 1)],
        spot=95.0,
        r=0.04,
    )

    assert details["is_true_crossing"] is True
    assert details["zero_gamma_type"] == "True crossing"
    assert details["method"].startswith("crossing")