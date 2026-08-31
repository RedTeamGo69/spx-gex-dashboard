"""Validation/production separation for weekly HAR fits."""

import numpy as np
import pandas as pd

import range_finder.har_model as hm


def _model_frame(n_complete: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2025-01-06", periods=n_complete + 2, freq="W-MON")
    pos = np.arange(len(idx), dtype=float)
    d1 = 0.02 + pos * 0.0002
    wk = 0.025 + np.sin(pos / 4.0) * 0.003
    mo = 0.03 + np.cos(pos / 7.0) * 0.002
    df = pd.DataFrame(
        {
            "har_d1": d1,
            "har_w": wk,
            "har_m": mo,
            "log_range": np.log(0.01 + 0.5 * d1 + 0.3 * wk + 0.2 * mo),
        },
        index=idx,
    )
    # Current-week and next-week scaffold rows are feature-complete but do
    # not have realized targets and must never enter either fit.
    df.loc[idx[-2]:, "log_range"] = np.nan
    return df


def test_production_refit_uses_all_completed_rows_after_holdout_evaluation():
    df = _model_frame()
    cols = ["har_d1", "har_w", "har_m"]

    X_train, X_test, y_train, y_test = hm.time_series_split(
        df, feature_cols=cols
    )
    validation = hm.fit_model(X_train, y_train, model_name="validation")
    metrics = hm.evaluate_oos(
        validation, X_test, y_test, model_name="validation"
    )
    metrics_before_refit = dict(metrics)

    production = hm.fit_production_model(
        df, feature_cols=cols, model_name="production"
    )

    assert int(validation.nobs) == 48
    assert int(production.nobs) == 60
    assert metrics == metrics_before_refit
    assert production.model.data.row_labels[-1] == df.index[-3]
    assert df.index[-2] not in production.model.data.row_labels
    assert df.index[-1] not in production.model.data.row_labels


def test_production_rows_equal_validation_train_plus_holdout():
    df = _model_frame(n_complete=50)
    cols = ["har_d1", "har_w", "har_m"]
    X_train, X_test, _, _ = hm.time_series_split(df, feature_cols=cols)

    production = hm.fit_production_model(
        df, feature_cols=cols, model_name="production"
    )

    assert list(production.model.data.row_labels) == list(
        X_train.index.append(X_test.index)
    )


def test_validation_metrics_are_computed_before_and_from_train_only_fit(monkeypatch):
    df = _model_frame(n_complete=50)
    cols = ["har_d1", "har_w", "har_m"]
    fitted_lengths = []
    validation_token = object()
    production_token = object()

    def fake_fit(X, y, model_name="HAR"):
        fitted_lengths.append(len(y))
        return validation_token if len(y) == 40 else production_token

    def fake_evaluate(result, X_test, y_test, model_name="HAR"):
        assert result is validation_token
        assert len(X_test) == len(y_test) == 10
        assert fitted_lengths == [40]
        return {"n_test": 10, "sentinel": "holdout"}

    monkeypatch.setattr(hm, "fit_model", fake_fit)
    monkeypatch.setattr(hm, "evaluate_oos", fake_evaluate)

    validation, production, metrics = hm.fit_validation_and_production(
        df, feature_cols=cols, model_name="test"
    )

    assert validation is validation_token
    assert production is production_token
    assert metrics == {"n_test": 10, "sentinel": "holdout"}
    assert fitted_lengths == [40, 50]
