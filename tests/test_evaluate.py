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
