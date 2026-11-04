"""Tests for src/clean_data.py."""
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

from src.clean_data import clean_data, main, run

TARGET = "charges"
NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]


def make_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [19, "18", "abc", 33],
            "sex": [" Female", "MALE ", "male", None],
            "bmi": [27.9, "33.77", 33.0, None],
            "children": [0, 1, "3", 0],
            "smoker": ["Yes", " no", "NO", "no"],
            "region": ["SouthWest", "southeast ", "SOUTHEAST", "northwest"],
            "charges": [16884.92, "1725.55", None, "not a number"],
        }
    )


def run_clean(df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(df, TARGET, NUMERIC, CATEGORICAL)


def test_numeric_columns_are_coerced():
    clean = run_clean(make_raw())
    for col in NUMERIC + [TARGET]:
        assert pd.api.types.is_numeric_dtype(clean[col]), col
    assert clean.loc[1, "age"] == 18
    assert clean.loc[1, "bmi"] == pytest.approx(33.77)


def test_unparseable_predictor_becomes_nan_but_row_is_kept():
    df = make_raw()
    df.loc[2, "charges"] = 4449.46  # give the "abc" age row a valid target
    clean = run_clean(df)
    assert len(clean) == 3
    assert np.isnan(clean.loc[clean["charges"] == 4449.46, "age"]).all()


def test_categoricals_are_stripped_and_lowercased():
    clean = run_clean(make_raw())
    assert clean["sex"].tolist() == ["female", "male"]
    assert clean["smoker"].tolist() == ["yes", "no"]
    assert clean["region"].tolist() == ["southwest", "southeast"]


def test_missing_categorical_stays_missing():
    df = make_raw()
    df["charges"] = [1.0, 2.0, 3.0, 4.0]
    clean = run_clean(df)
    assert pd.isna(clean.loc[3, "sex"])
    assert pd.isna(clean.loc[3, "bmi"])


def test_rows_with_missing_or_invalid_target_are_dropped():
    clean = run_clean(make_raw())
    assert len(clean) == 2
    assert clean[TARGET].notna().all()
    assert clean.index.tolist() == [0, 1]


def test_input_is_not_modified():
    df = make_raw()
    original = df.copy()
    run_clean(df)
    pdt.assert_frame_equal(df, original)


def test_cleaning_is_idempotent():
    once = run_clean(make_raw())
    twice = run_clean(once)
    pdt.assert_frame_equal(once, twice)


def test_duplicates_and_extra_columns_are_kept():
    df = pd.concat([make_raw().iloc[[0]]] * 2, ignore_index=True)
    df["extra"] = ["a", "b"]
    clean = run_clean(df)
    assert len(clean) == 2
    assert "extra" in clean.columns


def test_missing_required_column_raises():
    with pytest.raises(KeyError, match="smoker"):
        run_clean(make_raw().drop(columns="smoker"))


def write_project(tmp_path, raw: pd.DataFrame):
    (tmp_path / "data" / "raw").mkdir(parents=True)
    raw.to_csv(tmp_path / "data" / "raw" / "insurance.csv", index=False)
    cfg = {
        "paths": {
            "raw_data": "data/raw/insurance.csv",
            "processed_data": "data/processed/processed.csv",
        },
        "data": {"target": TARGET, "numeric": NUMERIC, "categorical": CATEGORICAL},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return config_path


def test_run_reads_config_and_writes_processed_csv(tmp_path):
    config_path = write_project(tmp_path, make_raw())
    result = run(config_path, root=tmp_path)

    out = tmp_path / "data" / "processed" / "processed.csv"
    assert out.exists()
    written = pd.read_csv(out)
    assert len(written) == len(result) == 2
    assert written["sex"].tolist() == ["female", "male"]


def test_main_cli(tmp_path):
    config_path = write_project(tmp_path, make_raw())
    main(["--config", str(config_path), "--root", str(tmp_path)])
    written = pd.read_csv(tmp_path / "data" / "processed" / "processed.csv")
    assert len(written) == 2


def test_repo_config_has_cleaning_settings():
    from src.clean_data import DEFAULT_CONFIG_PATH

    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    data = cfg["data"]
    assert data["target"] == TARGET
    assert set(data["numeric"] + data["categorical"]) == set(data["feature"])
