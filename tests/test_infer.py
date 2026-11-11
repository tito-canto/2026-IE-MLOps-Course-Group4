"""Inference parity with the notebook and saved-artifact CLI integration."""
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import yaml
from sklearn.impute import SimpleImputer

from src.infer import infer, main


@pytest.fixture
def bundle():
    features = pd.DataFrame({"age": [20., 30., 40., 50.], "sex_m": [0., 1., 0., 1.]})
    imputer = SimpleImputer(strategy="median").fit(features)
    y = 10 * features.age + 50 * features.sex_m + 100
    model = sm.OLS(y, sm.add_constant(features, has_constant="add")).fit()
    model.remove_data()
    return dict(model=model, imputer=imputer, predictors=["age", "sex"],
                categorical=["sex"], feature_columns=["age", "sex_m"], target="charges")


def test_notebook_parity_cleaning_and_saved_imputation(bundle):
    frame = pd.DataFrame({"sex": [" M ", "f", "new", None],
                          "age": ["25", "bad", 45., np.nan],
                          "id": [1, 2, 3, 4]}, index=[7, 2, 2, 9])
    original = frame.copy(deep=True)
    statistics = bundle["imputer"].statistics_.copy()
    expected_X = pd.DataFrame({"age": [25., 35., 45., 35.], "sex_m": [1., 0., 0., 0.]})
    expected = bundle["model"].predict(sm.add_constant(expected_X, has_constant="add"))
    result = infer(bundle, frame)
    np.testing.assert_allclose(result.prediction, expected)
    assert result.index.tolist() == frame.index.tolist()
    assert list(result) == ["prediction"]
    pd.testing.assert_frame_equal(frame, original)
    np.testing.assert_array_equal(bundle["imputer"].statistics_, statistics)


def test_single_row_and_unused_target(bundle):
    result = infer(bundle, pd.DataFrame({"age": [25.], "sex": ["m"], "charges": [np.nan]}))
    assert result.prediction.iloc[0] == pytest.approx(400.)


@pytest.mark.parametrize("frame,message", [
    (pd.DataFrame({"age": [25.]}), "Missing required predictors"),
    (pd.DataFrame(columns=["age", "sex"]), "empty dataset"),
])
def test_invalid_input(bundle, frame, message):
    with pytest.raises(ValueError, match=message):
        infer(bundle, frame)


def test_nonfinite_model_output_rejected(bundle, monkeypatch):
    monkeypatch.setattr(bundle["model"], "predict", lambda X: np.full(len(X), np.inf))
    with pytest.raises(ValueError, match="finite prediction"):
        infer(bundle, pd.DataFrame({"age": [25.], "sex": ["m"]}))


@pytest.fixture
def files(tmp_path, bundle):
    model = tmp_path / "model.joblib"
    joblib.dump(bundle, model)
    data = tmp_path / "new.csv"
    pd.DataFrame({"age": [25., 35.], "sex": ["m", "f"]}).to_csv(data, index=False)
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"paths": {"model": "model.joblib", "predictions": "output"}}))
    return model, data, config


def test_cli_config_defaults_and_roundtrip(files, tmp_path):
    model, data, config = files
    original_model = model.read_bytes()
    original_data = data.read_bytes()
    result = main(["--config", str(config), "--root", str(tmp_path), "--data", str(data)])
    saved = pd.read_csv(tmp_path / "output" / "predictions.csv")
    pd.testing.assert_frame_equal(saved, result)
    np.testing.assert_allclose(saved.prediction, [400., 450.])
    assert model.read_bytes() == original_model
    assert data.read_bytes() == original_data


def test_module_entrypoint_with_overrides(files, tmp_path):
    model, data, config = files
    output = tmp_path / "nested" / "results.csv"
    completed = subprocess.run(
        [sys.executable, "-m", "src.infer", "--config", str(config),
         "--data", str(data), "--model", str(model), "--output", str(output)],
        cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, check=True,
    )
    assert "Wrote 2 predictions" in completed.stdout
    np.testing.assert_allclose(pd.read_csv(output).prediction, [400., 450.])


@pytest.mark.parametrize("protected", [0, 1, 2])
def test_cli_does_not_overwrite_inputs(files, protected):
    model, data, config = files
    path = files[protected]
    before = path.read_bytes()
    with pytest.raises(ValueError, match="must not overwrite"):
        main(["--config", str(config), "--data", str(data),
              "--model", str(model), "--output", str(path)])
    assert path.read_bytes() == before


def test_cli_requires_explicit_data():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code == 2
