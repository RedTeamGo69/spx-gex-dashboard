from datetime import datetime

import pandas as pd
import pytest

import phase1.gex_engine as gex_engine
from phase1.config import NY_TZ


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
    # Regression: the bridge reads levels["net_gex"]; on an empty chain it
    # must be a concrete 0.0, never absent (a KeyError would crash the
    # force-mode/empty path) and never a stale None.
    assert levels["net_gex"] == 0.0


def test_find_key_levels_exposes_true_chain_wide_net_gex():
    """Regression for the gex_bridge net_gex bug: find_key_levels must return
    the chain-wide net GEX (identical to calculate_all's stats["net_gex"]),
    NOT leave the key absent — otherwise gex_bridge.extract_gex_context reads
    None and silently persists the call_wall_gex + put_wall_gex two-strike
    proxy, which can carry the WRONG SIGN when one wall dominates an
    oppositely-signed book."""
    class FakeClient:
        def prefetch_chains(self, ticker, expirations):
            return None

        def get_chain_cached(self, ticker, exp):
            # Two call strikes (net-positive book) plus a single very heavy
            # put wall. The put wall is the largest-magnitude single strike,
            # so put_wall_gex alone is a poor proxy for the true net.
            return {
                "status": "ok",
                "calls": [
                    {"strike": 5050, "openInterest": 400, "impliedVolatility": 0.20,
                     "vendorGamma": 0.0, "bid": 5.0, "ask": 5.5, "mid": 5.25},
                    {"strike": 5100, "openInterest": 400, "impliedVolatility": 0.20,
                     "vendorGamma": 0.0, "bid": 4.0, "ask": 4.5, "mid": 4.25},
                ],
                "puts": [
                    {"strike": 4950, "openInterest": 300, "impliedVolatility": 0.20,
                     "vendorGamma": 0.0, "bid": 5.0, "ask": 5.5, "mid": 5.25},
                ],
                "error": None,
            }

    fake_now = datetime(2026, 3, 10, 10, 0, tzinfo=NY_TZ)
    gex_df, stats, all_options, _, _ = gex_engine.calculate_all(
        client=FakeClient(),
        ticker="SPX",
        target_exps=["2026-03-20"],
        spot=5000,
        r=0.04,
        now=fake_now,
    )

    levels = gex_engine.find_key_levels(gex_df, 5000, all_options=all_options, r=0.04)

    # The exposed net must equal both calculate_all's stat and the raw sum.
    assert "net_gex" in levels
    assert levels["net_gex"] == stats["net_gex"]
    assert abs(levels["net_gex"] - float(gex_df["net_gex"].sum())) < 1e-6
    # And it must genuinely differ from the two-strike wall proxy the bridge
    # used to fall back on — proving the fix changes the persisted value.
    proxy = float(levels["call_wall_gex"]) + float(levels["put_wall_gex"])
    assert abs(levels["net_gex"] - proxy) > 1e-6


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

    # Pin `now` to a date before the 2026-03-20 expiration so the
    # expired-exp filter in calculate_all doesn't drop it when the test
    # is run on a wall clock that's past March 2026.
    fake_now = datetime(2026, 3, 10, 10, 0, tzinfo=NY_TZ)

    gex_df, stats, all_options, strike_support_df, expiration_support_df = gex_engine.calculate_all(
        client=client,
        ticker="SPX",
        target_exps=["2026-03-20"],
        spot=5000,
        r=0.04,
        now=fake_now,
    )

    assert not gex_df.empty
    assert stats["used_option_count"] == 2
    assert len(all_options) == 2
    assert not strike_support_df.empty
    assert not expiration_support_df.empty
    # Volume amplification diagnostics: this fixture has OI=100, volume=0
    # on both legs, so size == OI everywhere → amplification ratio is 1.0
    # and no strikes are volume-dominated.
    assert stats["vol_amplification_ratio"] == 1.0
    assert stats["vol_dominated_strike_count"] == 0
    assert stats["vol_dominated_pct"] == 0.0


def test_calculate_all_flags_volume_amplification():
    """When today's volume dwarfs yesterday's OI on most strikes, the
    amplification ratio should exceed 1.0 and vol_dominated_strike_count
    should reflect the number of volume-dominated strikes."""
    class FakeClient:
        def prefetch_chains(self, ticker, expirations):
            return None

        def get_chain_cached(self, ticker, exp):
            # OI = 10, volume = 500 — volume dominates by 50× on both legs
            return {
                "status": "ok",
                "calls": [{
                    "strike": 5000,
                    "openInterest": 10,
                    "volume": 500,
                    "impliedVolatility": 0.20,
                    "vendorGamma": 0.0,
                    "bid": 10.0, "ask": 10.5, "mid": 10.25,
                }],
                "puts": [{
                    "strike": 5000,
                    "openInterest": 10,
                    "volume": 500,
                    "impliedVolatility": 0.20,
                    "vendorGamma": 0.0,
                    "bid": 10.0, "ask": 10.5, "mid": 10.25,
                }],
                "error": None,
            }

    fake_now = datetime(2026, 3, 10, 10, 0, tzinfo=NY_TZ)
    _, stats, _, _, _ = gex_engine.calculate_all(
        client=FakeClient(),
        ticker="SPX",
        target_exps=["2026-03-20"],
        spot=5000,
        r=0.04,
        now=fake_now,
    )
    assert stats["vol_amplification_ratio"] == 50.0
    assert stats["vol_dominated_strike_count"] == 2
    assert stats["vol_dominated_pct"] == 1.0


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

    def fake_sweep(_all_options, prices, _r, r_curve=None, q=0.0):
        # _sweep_gex_at_prices returns total GEX (already Spot²-scaled).
        # r_curve/q are accepted but ignored — this test pins the crossing
        # location, not the rate/dividend path.
        return np.array([float(p - 100.0) for p in prices])

    import phase1.zero_gamma as zero_gamma_mod
    monkeypatch.setattr(zero_gamma_mod, "_sweep_gex_at_prices", fake_sweep)

    # spot=98 puts the synthetic crossing at 100 INSIDE the ±5% sweep window
    # (93.1..102.9). The old fixture's spot=95 only worked because the legacy
    # $5 coarse grid overshot the declared sweep-high by a full step — the
    # spot-relative grid honors the window boundary exactly.
    details = gex_engine.zero_gamma_sweep_details(
        all_options=[(1, 1, 1, 1, 1)],
        spot=98.0,
        r=0.04,
    )

    assert details["is_true_crossing"] is True
    assert details["zero_gamma_type"] == "True crossing"
    assert details["method"].startswith("crossing")

def test_calculate_all_iv_stats_fall_back_when_first_exp_expired():
    """When the nearest selected expiration has already settled, the
    call_iv / put_iv stats must come from the first LIVE expiration
    instead of silently reading 0.0 (the old behavior keyed the IV
    sample on target_exps[0] even when that exp was skipped)."""
    def _opt(iv):
        return {
            "strike": 5000,
            "openInterest": 100,
            "impliedVolatility": iv,
            "vendorGamma": 0.0,
            "bid": 10.0,
            "ask": 10.5,
            "mid": 10.25,
        }

    class FakeClient:
        def prefetch_chains(self, ticker, expirations):
            return None

        def get_chain_cached(self, ticker, exp):
            # Expired Wednesday chain carries a poison IV that must NOT
            # leak into stats; the live Friday chain carries 15%/17%.
            if exp == "2026-03-18":
                return {"status": "ok", "calls": [_opt(0.99)], "puts": [_opt(0.99)], "error": None}
            return {"status": "ok", "calls": [_opt(0.15)], "puts": [_opt(0.17)], "error": None}

    # Thursday 10am ET — the 2026-03-18 (Wed) expiration settled yesterday.
    fake_now = datetime(2026, 3, 19, 10, 0, tzinfo=NY_TZ)

    gex_df, stats, all_options, _, _ = gex_engine.calculate_all(
        client=FakeClient(),
        ticker="SPX",
        target_exps=["2026-03-18", "2026-03-20"],
        spot=5000,
        r=0.04,
        now=fake_now,
    )

    assert stats["expired_exp_count"] == 1
    assert abs(stats["call_iv"] - 15.0) < 1e-9
    assert abs(stats["put_iv"] - 17.0) < 1e-9


def test_calculate_all_after_hours_all_expirations_expired():
    """0DTE selected after the close: every expiration is dropped by the
    expired-exp guard. calculate_all must return an EMPTY-but-schema'd
    gex_df (plus coherent stats) instead of raising KeyError('net_gex')
    on the column-less frame — that crash masked the friendly 'all
    expirations have settled' notice in the app."""
    class FakeClient:
        def prefetch_chains(self, ticker, expirations):
            return None

        def get_chain_cached(self, ticker, exp):
            raise AssertionError("expired expirations must be skipped before any chain fetch")

    # 8 PM ET on expiration day — the 2026-03-20 session settled at 4 PM.
    fake_now = datetime(2026, 3, 20, 20, 0, tzinfo=NY_TZ)

    gex_df, stats, all_options, strike_support_df, expiration_support_df = gex_engine.calculate_all(
        client=FakeClient(),
        ticker="SPX",
        target_exps=["2026-03-20"],
        spot=5000,
        r=0.04,
        now=fake_now,
    )

    assert gex_df.empty
    assert "net_gex" in gex_df.columns  # schema present even when empty
    assert stats["expired_exp_count"] == 1
    assert stats["used_option_count"] == 0
    assert stats["net_gex"] == 0.0
    assert all_options == []

    # The downstream pipeline (mirroring fetch_all_data) must also cope.
    levels = gex_engine.find_key_levels(gex_df, 5000, all_options=all_options, r=0.04)
    assert levels["zero_gamma"] == 5000
    regime = gex_engine.get_gamma_regime_text(5000, levels["zero_gamma"])
    assert regime["regime"] == "At Zero Gamma"


@pytest.mark.parametrize(
    ("ticker", "am_root", "pm_root", "spot"),
    [
        ("SPX", "SPX", "SPXW", 5000),
        ("NDX", "NDX", "NDXP", 18000),
    ],
)
def test_calculate_all_filters_expired_root_inside_mixed_third_friday_chain(
        ticker, am_root, pm_root, spot):
    """At 3:59 Friday, stale AM rows are dead but PM rows remain live."""
    def _opt(strike, root):
        return {
            "strike": strike,
            "openInterest": 100,
            "volume": 0,
            "impliedVolatility": 0.20,
            "vendorGamma": 0.0,
            "bid": 10.0,
            "ask": 10.5,
            "mid": 10.25,
            "root": root,
        }

    class FakeClient:
        def prefetch_chains(self, ticker, expirations):
            return None

        def get_chain_cached(self, ticker, exp):
            return {
                "status": "ok",
                "calls": [_opt(spot - 10, am_root), _opt(spot, pm_root)],
                "puts": [_opt(spot - 10, am_root), _opt(spot, pm_root)],
                "error": None,
            }

    gex_df, stats, all_options, _, _ = gex_engine.calculate_all(
        client=FakeClient(),
        ticker=ticker,
        target_exps=["2026-03-20"],
        spot=spot,
        r=0.04,
        now=datetime(2026, 3, 20, 15, 59, tzinfo=NY_TZ),
    )

    assert not gex_df.empty
    assert {row[0] for row in all_options} == {spot}
    assert stats["used_option_count"] == 2
    assert stats["expired_option_count"] == 2


@pytest.mark.parametrize(
    ("ticker", "root", "spot"),
    [
        ("SPX", "SPXW", 5000),
        ("XSP", "XSP", 500),
        ("NDX", "NDXP", 18000),
    ],
)
@pytest.mark.parametrize("minute", [0, 5])
def test_calculate_all_drops_pm_root_at_and_after_four(
        ticker, root, spot, minute):
    def _opt(root):
        return {
            "strike": spot,
            "openInterest": 100,
            "volume": 0,
            "impliedVolatility": 0.20,
            "vendorGamma": 0.0,
            "bid": 10.0,
            "ask": 10.5,
            "mid": 10.25,
            "root": root,
        }

    class FakeClient:
        def prefetch_chains(self, ticker, expirations):
            return None

        def get_chain_cached(self, ticker, exp):
            return {
                "status": "ok",
                "calls": [_opt(root)],
                "puts": [_opt(root)],
                "error": None,
            }

    gex_df, stats, all_options, _, _ = gex_engine.calculate_all(
        client=FakeClient(),
        ticker=ticker,
        target_exps=["2026-03-20"],
        spot=spot,
        r=0.04,
        now=datetime(2026, 3, 20, 16, minute, tzinfo=NY_TZ),
    )

    assert gex_df.empty
    assert all_options == []
    assert stats["expired_exp_count"] == 1
    assert stats["expired_option_count"] == 2


# ── dividend yield in BS gamma (audit G20) ───────────────────────────────────

def test_bs_gamma_vec_dividend_yield_matches_closed_form():
    """Gamma with continuous yield q = e^{-qT}·φ(d1_q)/(S·σ·√T), where the
    d1 drift is (r - q + σ²/2). Verify the vectorized kernel against a direct
    computation and confirm q actually changes the value (q=0 is the legacy
    path)."""
    import math
    from scipy.stats import norm
    S, K, T, r, sig, q = 5000.0, 5100.0, 0.25, 0.04, 0.20, 0.02
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / (sig * math.sqrt(T))
    expected = math.exp(-q * T) * norm.pdf(d1) / (S * sig * math.sqrt(T))
    gq = gex_engine.bs_gamma_vec([S], [K], [T], r, [sig], q=q)
    assert gq[0, 0] == pytest.approx(expected, rel=1e-9)
    g0 = gex_engine.bs_gamma_vec([S], [K], [T], r, [sig], q=0.0)
    assert gq[0, 0] != g0[0, 0]           # q genuinely threads through
