from phase1.model_inputs import (
    bs_gamma,
    bs_charm,
    infer_iv_from_gamma,
    fit_synthetic_iv,
    prepare_option_for_model,
    normalize_option_for_model,
)


def test_bs_gamma_positive_for_valid_inputs():
    g = bs_gamma(S=5000, K=5000, T=1/365, r=0.04, sigma=0.20)
    assert g > 0


def test_bs_gamma_zero_for_invalid_inputs():
    assert bs_gamma(S=0, K=5000, T=1/365, r=0.04, sigma=0.20) == 0.0
    assert bs_gamma(S=5000, K=0, T=1/365, r=0.04, sigma=0.20) == 0.0
    assert bs_gamma(S=5000, K=5000, T=0, r=0.04, sigma=0.20) == 0.0
    assert bs_gamma(S=5000, K=5000, T=1/365, r=0.04, sigma=0) == 0.0


def test_bs_charm_zero_for_invalid_inputs():
    assert bs_charm(S=0, K=5000, T=1/365, r=0.04, sigma=0.20, sign=+1) == 0.0
    assert bs_charm(S=5000, K=0, T=1/365, r=0.04, sigma=0.20, sign=+1) == 0.0
    assert bs_charm(S=5000, K=5000, T=0, r=0.04, sigma=0.20, sign=+1) == 0.0
    assert bs_charm(S=5000, K=5000, T=1/365, r=0.04, sigma=0, sign=+1) == 0.0


def test_bs_charm_matches_numerical_delta_derivative():
    """
    bs_charm should approximate ∂Δ/∂t for a standard case.

    Uses the BS call delta formula directly (N(d1)) and differentiates
    numerically against T, then compares to bs_charm's analytic value.
    With q=0 the charm is ∂Δ_call/∂t where t advances forward in time,
    i.e. -∂Δ_call/∂T (since T = expiry - t). Sign handling is checked
    here end-to-end.
    """
    import numpy as np
    from scipy.stats import norm

    S, K, r, sigma = 5000.0, 5000.0, 0.04, 0.20

    def delta_call(T):
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1)

    T = 5 / 365.0
    # ∂Δ/∂t ≈ -∂Δ/∂T  (forward time advance shrinks T)
    dT = 1e-5
    d_delta_dt_numeric = -(delta_call(T + dT) - delta_call(T - dT)) / (2 * dT)

    analytic = bs_charm(S, K, T, r, sigma, sign=+1)
    assert abs(analytic - d_delta_dt_numeric) / abs(d_delta_dt_numeric) < 5e-3


def test_prepare_option_for_model_emits_charm():
    opt = {"strike": 5000.0, "openInterest": 1000, "impliedVolatility": 0.20}
    prep = prepare_option_for_model(opt, sign=+1, T=1/365, spot=5000.0, r=0.04)
    assert prep["accepted"]
    assert "charm_now" in prep["normalized"]
    # near-ATM 1DTE charm should be a meaningfully non-zero number
    assert prep["normalized"]["charm_now"] != 0.0


def test_infer_iv_from_gamma_round_trip_reasonable():
    true_iv = 0.25
    target_gamma = bs_gamma(S=5000, K=5000, T=1/365, r=0.04, sigma=true_iv)

    inferred = infer_iv_from_gamma(
        target_gamma=target_gamma,
        S=5000,
        K=5000,
        T=1/365,
        r=0.04,
    )

    assert inferred > 0
    assert abs(inferred - true_iv) < 0.05


def test_fit_synthetic_iv_reports_good_fit():
    true_iv = 0.30
    target_gamma = bs_gamma(S=5000, K=5000, T=1/365, r=0.04, sigma=true_iv)

    fit = fit_synthetic_iv(target_gamma, S=5000, K=5000, T=1/365, r=0.04)

    assert fit["accepted"] is True
    assert fit["iv"] > 0
    assert fit["rel_error"] is not None
    assert fit["rel_error"] < 0.08


def test_fit_synthetic_iv_rejects_below_economic_floor():
    """A deep OTM strike with a near-zero vendor gamma can round-trip
    to a numerically tight but economically nonsense sub-1% IV. The
    fitter must refuse it rather than poisoning the stats.call_iv /
    put_iv averages downstream."""
    # Build a target gamma from a very low sigma (0.02 = 2% annualized).
    # This should produce a fit that numerically round-trips but sits
    # below the 0.05 economic floor.
    tiny_sigma = 0.02
    target = bs_gamma(S=5000, K=4750, T=5/252, r=0.04, sigma=tiny_sigma)
    fit = fit_synthetic_iv(target, S=5000, K=4750, T=5/252, r=0.04)
    assert fit["accepted"] is False
    assert fit["reason"] == "below_economic_floor"


def test_prepare_option_prefers_direct_iv():
    opt = {
        "strike": 5000,
        "openInterest": 100,
        "impliedVolatility": 0.22,
        "vendorGamma": 0.0,
    }

    result = prepare_option_for_model(opt, sign=1, T=1/365, spot=5000, r=0.04)

    assert result["accepted"] is True
    assert result["reason"] == "direct_iv"
    assert result["normalized"]["iv_source"] == "direct_iv"


def test_prepare_option_uses_synthetic_iv_when_direct_missing():
    target_gamma = bs_gamma(S=5000, K=5000, T=1/365, r=0.04, sigma=0.30)

    opt = {
        "strike": 5000,
        "openInterest": 100,
        "impliedVolatility": 0.0,
        "vendorGamma": target_gamma,
    }

    result = prepare_option_for_model(opt, sign=-1, T=1/365, spot=5000, r=0.04)

    assert result["accepted"] is True
    assert result["reason"] == "synthetic_iv"
    assert result["synthetic_fit_rel_error"] is not None
    assert result["normalized"]["iv_source"] == "synthetic_iv"


def test_prepare_option_returns_no_model_input_when_nothing_available():
    opt = {
        "strike": 5000,
        "openInterest": 100,
        "impliedVolatility": 0.0,
        "vendorGamma": 0.0,
    }

    result = prepare_option_for_model(opt, sign=1, T=1/365, spot=5000, r=0.04)

    assert result["accepted"] is False
    assert result["reason"] == "no_model_input"


def test_normalize_option_for_model_backward_wrapper():
    opt = {
        "strike": 5000,
        "openInterest": 100,
        "impliedVolatility": 0.22,
        "vendorGamma": 0.0,
    }

    norm = normalize_option_for_model(opt, sign=1, T=1/365, spot=5000, r=0.04)
    assert norm is not None
    assert norm["iv_source"] == "direct_iv"
