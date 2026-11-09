"""Tests for src/evaluate.py."""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

from src.config import load_config
from src.evaluate import DEFAULT_METRICS, evaluate, metrics_from_config, predict, scores
from src.clean_data import DEFAULT_CONFIG_PATH


@pytest.fixture
def data():
    rng = np.random.default_rng(47)
    X = pd.DataFrame({"age": rng.uniform(18, 64, 200), "bmi": rng.uniform(16, 50, 200)})
    y = 250 * X["age"] + 300 * X["bmi"] + 1000 + rng.normal(0, 500, 200)
    return X, y


def test_perfect_predictions():
    result = scores([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert result == {"r2": 1.0, "mae": 0.0, "rmse": 0.0}


def test_known_values():
    result = scores([0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 3.0, -3.0], metrics=["mae", "rmse"])
    assert result["mae"] == pytest.approx(2.0)
    assert result["rmse"] == pytest.approx(np.sqrt(5.0))
    assert "r2" not in result


def test_returns_plain_floats():
    result = scores(np.array([1, 2, 3]), pd.Series([1.5, 2.5, 2.0]))
    assert set(result) == set(DEFAULT_METRICS)
    assert all(type(v) is float for v in result.values())


@pytest.mark.parametrize(
    "y_true, y_pred, message",
    [
        ([1.0, 2.0], [1.0], "Shape mismatch"),
        ([], [], "empty"),
        ([1.0, np.nan], [1.0, 2.0], "NaN"),
    ],
)
def test_bad_inputs_raise(y_true, y_pred, message):
    with pytest.raises(ValueError, match=message):
        scores(y_true, y_pred)


def test_unknown_metric_raises():
    with pytest.raises(ValueError, match="pr_auc"):
        scores([1.0, 2.0], [1.0, 2.0], metrics=["pr_auc"])


def test_statsmodels_ols_gets_intercept(data):
    X, y = data
    model = sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()
    expected = model.predict(sm.add_constant(X, has_constant="add"))
    np.testing.assert_allclose(predict(model, X), expected)


def test_evaluate_matches_notebook_scores(data):
    X, y = data
    X_train, X_eval, y_train, y_eval = X[:150], X[150:], y[:150], y[150:]
    model = sm.OLS(y_train, sm.add_constant(X_train, has_constant="add")).fit()

    result = evaluate(model, X_eval, y_eval)
    pred = model.predict(sm.add_constant(X_eval, has_constant="add"))
    assert result["r2"] == pytest.approx(1 - ((y_eval - pred) ** 2).sum() / ((y_eval - y_eval.mean()) ** 2).sum())
    assert result["mae"] == pytest.approx(np.abs(y_eval - pred).mean())
    assert result["rmse"] == pytest.approx(np.sqrt(((y_eval - pred) ** 2).mean()))
    assert result["r2"] > 0.9


def test_evaluate_works_with_sklearn_models(data):
    X, y = data
    model = LinearRegression().fit(X[:150], y[:150])
    result = evaluate(model, X[150:], y[150:])
    assert result["r2"] > 0.9


def test_metrics_from_config():
    assert metrics_from_config({}) == list(DEFAULT_METRICS)
    assert metrics_from_config({"evaluation": {"metrics": ["mae"]}}) == ["mae"]
    repo_metrics = metrics_from_config(load_config(DEFAULT_CONFIG_PATH))
    assert scores([1.0, 2.0, 3.0], [1.0, 2.0, 2.5], repo_metrics)


def test_prediction_report_preserves_rows_and_input():
    from src.evaluate import prediction_report

    frame = pd.DataFrame({"charges": [10., 20.]}, index=[8, 3])
    original = frame.copy(deep=True)
    report = prediction_report(frame, [12., 17.])
    pd.testing.assert_frame_equal(frame, original)
    assert report.index.tolist() == [8, 3]
    assert report["residual"].tolist() == [-2., 3.]
    assert report["absolute_error"].tolist() == [2., 3.]
    with pytest.raises(ValueError, match="one prediction"):
        prediction_report(frame, [1.])


@pytest.mark.parametrize("bundle", [False, True])
def test_cli_saved_models_and_exports(tmp_path, capsys, bundle):
    import json
    import joblib
    import yaml
    from sklearn.impute import SimpleImputer
    from src.evaluate import main

    train = pd.DataFrame({"age": [20., 30., 40., 50.], "sex": ["f", "m", "f", "m"]})
    encoded = pd.get_dummies(train, columns=["sex"], dtype=float).drop(columns="sex_f")
    imputer = SimpleImputer(strategy="median").fit(encoded)
    y = 10 * train.age + 50 * encoded.sex_m
    model = sm.OLS(y, sm.add_constant(encoded, has_constant="add")).fit()
    held_out = pd.DataFrame({"age": [25., np.nan], "sex": ["f", "f"], "charges": [250., 350.]})
    if bundle:
        artifact = dict(model=model, imputer=imputer, predictors=list(train.columns),
                        categorical=["sex"], feature_columns=list(encoded.columns), target="charges")
    else:
        model = LinearRegression().fit(train[["age"]], 10 * train.age)
        artifact = model
        held_out = pd.DataFrame({"age": [25., 35.], "charges": [250., 350.]})
    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact, model_path)
    data_path = tmp_path / "held_out.csv"
    held_out.to_csv(data_path, index=False)
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"paths": {"model": str(model_path)},
                                        "evaluation": {"metrics": ["mae", "rmse"]}}))
    output = tmp_path / "reports"
    result = main(["--config", str(cfg_path), "--data", str(data_path),
                   "--output-dir", str(output)])
    assert result == pytest.approx({"mae": 0., "rmse": 0.}, abs=1e-9)
    assert json.loads(capsys.readouterr().out) == result
    saved = pd.read_csv(output / "evaluation_predictions.csv")
    np.testing.assert_allclose(saved.predicted_charges, held_out.charges)
    assert list(pd.read_csv(output / "evaluation_scores.csv")) == ["mae", "rmse"]


def test_cli_requires_explicit_held_out_data():
    from src.evaluate import main

    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
