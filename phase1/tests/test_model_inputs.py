from phase1.model_inputs import (
    bs_gamma,
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
