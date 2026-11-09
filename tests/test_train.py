"""Tests for src/train.py."""
import joblib
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

from src.train import (
    DEFAULT_CONFIG_PATH,
    fit_model,
    main,
    predict_ols,
    prepare_features,
    run,
    select_feature_columns,
    split_data,
    train_model,
)

TARGET = "charges"
NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]
PREDICTORS = NUMERIC + CATEGORICAL
SEED = 47
EXPECTED_COLUMNS = [
    "age",
    "bmi",
    "children",
    "sex_male",
    "smoker_yes",
    "region_northwest",
    "region_southeast",
    "region_southwest",
]


def make_clean(rows: int = 200, seed: int = 0) -> pd.DataFrame:
    """Build clean data whose charges follow a known formula plus a little noise."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "age": rng.integers(18, 65, rows).astype(float),
            "sex": rng.choice(["female", "male"], rows),
            "bmi": rng.uniform(18, 45, rows),
            "children": rng.integers(0, 5, rows).astype(float),
            "smoker": rng.choice(["no", "yes"], rows),
            "region": rng.choice(["northeast", "northwest", "southeast", "southwest"], rows),
        }
    )
    df[TARGET] = (
        1000
        + 250 * df["age"]
        + 300 * df["bmi"]
        + 20000 * (df["smoker"] == "yes")
        + rng.normal(0, 10, rows)
    )
    return df


def run_train(df: pd.DataFrame):
    return train_model(df, TARGET, NUMERIC, CATEGORICAL, SEED)


def test_split_sizes_and_rows_do_not_overlap():
    df = make_clean()
    splits = split_data(df, SEED)

    assert len(splits.reserved) == 20
    assert len(splits.test) == 45
    assert len(splits.train) == 135
    assert len(splits.development) == 180

    assert set(splits.train.index) | set(splits.test.index) == set(splits.development.index)
    assert not set(splits.train.index) & set(splits.test.index)
    assert not set(splits.development.index) & set(splits.reserved.index)


def test_split_is_repeatable_and_depends_on_the_seed():
    df = make_clean()
    first = split_data(df, SEED)
    second = split_data(df, SEED)
    other = split_data(df, SEED + 1)

    pdt.assert_frame_equal(first.train, second.train)
    pdt.assert_frame_equal(first.reserved, second.reserved)
    assert set(first.reserved.index) != set(other.reserved.index)


def test_first_level_of_each_category_is_dropped():
    assert select_feature_columns(make_clean(), PREDICTORS, CATEGORICAL) == EXPECTED_COLUMNS


def test_prepare_features_ignores_unseen_levels_and_fills_absent_ones():
    frame = make_clean(rows=3)
    frame["region"] = ["southeast", "atlantis", "northeast"]
    frame["smoker"] = "no"

    X = prepare_features(frame, PREDICTORS, CATEGORICAL, EXPECTED_COLUMNS)

    assert X.columns.tolist() == EXPECTED_COLUMNS
    assert X["region_southeast"].tolist() == [1.0, 0.0, 0.0]
    assert X["region_southwest"].tolist() == [0.0, 0.0, 0.0]
    assert X["smoker_yes"].tolist() == [0.0, 0.0, 0.0]
    assert (X.dtypes == float).all()


def test_fit_recovers_known_coefficients():
    model, _ = fit_model(make_clean(), TARGET, PREDICTORS, CATEGORICAL, EXPECTED_COLUMNS)

    assert model.params["const"] == pytest.approx(1000, abs=20)
    assert model.params["age"] == pytest.approx(250, abs=1)
    assert model.params["bmi"] == pytest.approx(300, abs=1)
    assert model.params["smoker_yes"] == pytest.approx(20000, abs=10)


def test_imputer_uses_medians_of_the_frame_it_was_fitted_on():
    df = make_clean()
    df.loc[df.index[:5], "bmi"] = np.nan

    model, imputer = fit_model(df, TARGET, PREDICTORS, CATEGORICAL, EXPECTED_COLUMNS)

    position = EXPECTED_COLUMNS.index("bmi")
    assert imputer.statistics_[position] == pytest.approx(df["bmi"].median())
    assert model.nobs == len(df)


def test_train_model_fits_each_model_on_the_right_rows():
    df = make_clean()
    result = run_train(df)

    assert result.train_model.nobs == len(result.splits.train)
    assert result.final_model.nobs == len(result.splits.development)
    assert result.feature_columns == EXPECTED_COLUMNS


def test_imputers_do_not_learn_from_held_out_rows():
    df = make_clean()
    splits = split_data(df, SEED)
    # An extreme value in rows outside the fit must not move the train-only median.
    df.loc[splits.test.index, "bmi"] = 1000.0
    df.loc[splits.reserved.index, "bmi"] = 1000.0

    result = run_train(df)

    position = EXPECTED_COLUMNS.index("bmi")
    assert result.train_imputer.statistics_[position] == pytest.approx(
        result.splits.train["bmi"].median()
    )
    assert result.train_imputer.statistics_[position] < 100


def test_final_model_ignores_reserved_rows():
    df = make_clean()
    baseline = run_train(df)

    changed = df.copy()
    changed.loc[baseline.splits.reserved.index, TARGET] = 1e9
    result = run_train(changed)

    pdt.assert_series_equal(result.final_model.params, baseline.final_model.params)
    pdt.assert_series_equal(result.train_model.params, baseline.train_model.params)


def test_training_is_repeatable():
    df = make_clean()
    pdt.assert_series_equal(run_train(df).final_model.params, run_train(df).final_model.params)


def test_input_is_not_modified():
    df = make_clean()
    original = df.copy()
    run_train(df)
    pdt.assert_frame_equal(df, original)


def write_project(tmp_path, clean: pd.DataFrame):
    (tmp_path / "data" / "processed").mkdir(parents=True)
    clean.to_csv(tmp_path / "data" / "processed" / "processed.csv", index=False)
    cfg = {
        "paths": {
            "processed_data": "data/processed/processed.csv",
            "model": "models/model.joblib",
        },
        "data": {"target": TARGET, "numeric": NUMERIC, "categorical": CATEGORICAL},
        "variables": {"seed": SEED},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return config_path


def test_run_saves_a_bundle_that_reproduces_predictions(tmp_path):
    config_path = write_project(tmp_path, make_clean())
    result = run(config_path, root=tmp_path)

    bundle = joblib.load(tmp_path / "models" / "model.joblib")
    assert list(bundle) == [
        "model",
        "imputer",
        "predictors",
        "categorical",
        "feature_columns",
        "target",
    ]
    assert bundle["predictors"] == PREDICTORS
    assert bundle["categorical"] == CATEGORICAL
    assert bundle["feature_columns"] == EXPECTED_COLUMNS
    assert bundle["target"] == TARGET

    reserved = result.splits.reserved
    X = pd.DataFrame(
        bundle["imputer"].transform(
            prepare_features(reserved, PREDICTORS, CATEGORICAL, EXPECTED_COLUMNS)
        ),
        columns=EXPECTED_COLUMNS,
        index=reserved.index,
    )
    assert np.allclose(predict_ols(bundle["model"], X), predict_ols(result.final_model, X))


def test_saved_model_does_not_contain_training_rows(tmp_path):
    config_path = write_project(tmp_path, make_clean())
    run(config_path, root=tmp_path)

    bundle = joblib.load(tmp_path / "models" / "model.joblib")
    assert bundle["model"].model.exog is None
    assert bundle["model"].model.endog is None


def test_main_cli(tmp_path):
    config_path = write_project(tmp_path, make_clean())
    main(["--config", str(config_path), "--root", str(tmp_path)])
    assert (tmp_path / "models" / "model.joblib").exists()


def test_main_exits_with_an_error_when_processed_data_is_missing(tmp_path):
    config_path = write_project(tmp_path, make_clean())
    (tmp_path / "data" / "processed" / "processed.csv").unlink()

    with pytest.raises(SystemExit) as excinfo:
        main(["--config", str(config_path), "--root", str(tmp_path)])
    assert excinfo.value.code == 1


def test_repo_config_has_training_settings():
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    assert cfg["paths"]["model"]
    assert cfg["paths"]["processed_data"]
    assert isinstance(cfg["variables"]["seed"], int)
    assert cfg["data"]["target"] == TARGET
    assert set(cfg["data"]["numeric"] + cfg["data"]["categorical"]) == set(cfg["data"]["feature"])
