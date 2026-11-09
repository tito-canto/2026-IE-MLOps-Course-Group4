"""Tests for src/validate.py."""
import logging

import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

from src.validate import DEFAULT_CONFIG_PATH, main, run, validate_data

TARGET = "charges"
NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]
REQUIRED = NUMERIC + CATEGORICAL + [TARGET]


def make_clean() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [19, 18, 28],
            "sex": ["female", "male", "male"],
            "bmi": [27.9, 33.77, 33.0],
            "children": [0, 1, 3],
            "smoker": ["yes", "no", "no"],
            "region": ["southwest", "southeast", "southeast"],
            "charges": [16884.92, 1725.55, 4449.46],
        }
    )


def run_validate(df: pd.DataFrame, expected_rows: int | None = None) -> bool:
    return validate_data(df, REQUIRED, TARGET, NUMERIC, expected_rows=expected_rows)


def test_valid_data_returns_true():
    assert run_validate(make_clean()) is True


def test_matching_row_count_passes():
    assert run_validate(make_clean(), expected_rows=3) is True


def test_row_count_is_not_checked_when_not_set():
    assert run_validate(make_clean().head(1)) is True


def test_wrong_row_count_raises():
    with pytest.raises(ValueError, match="Expected 1338 rows, found 3"):
        run_validate(make_clean(), expected_rows=1338)


def test_empty_data_raises():
    with pytest.raises(ValueError, match="no rows"):
        run_validate(make_clean().head(0))


@pytest.mark.parametrize("column", REQUIRED)
def test_missing_required_column_raises(column):
    with pytest.raises(ValueError, match=f"Missing required columns: .*{column}"):
        run_validate(make_clean().drop(columns=column))


def test_target_and_numeric_columns_are_required_even_if_not_listed():
    with pytest.raises(ValueError, match="Missing required columns: .*charges"):
        validate_data(make_clean().drop(columns=TARGET), [], TARGET, NUMERIC)
    with pytest.raises(ValueError, match="Missing required columns: .*bmi"):
        validate_data(make_clean().drop(columns="bmi"), [], TARGET, NUMERIC)


@pytest.mark.parametrize("column", NUMERIC + [TARGET])
def test_text_in_numeric_column_raises(column):
    df = make_clean().astype({column: "object"})
    df.loc[0, column] = "not a number"
    with pytest.raises(ValueError, match=f"Columns must be numeric: .*{column}"):
        run_validate(df)


def test_missing_target_value_raises():
    df = make_clean()
    df.loc[[0, 2], TARGET] = None
    with pytest.raises(ValueError, match="'charges' has 2 missing values"):
        run_validate(df)


def test_all_failures_are_reported_together():
    df = make_clean().drop(columns="bmi").astype({"age": "object"})
    df.loc[0, "age"] = "old"
    df.loc[1, TARGET] = None
    with pytest.raises(ValueError) as excinfo:
        run_validate(df, expected_rows=1338)
    message = str(excinfo.value)
    assert "Missing required columns: ['bmi']" in message
    assert "Expected 1338 rows, found 3" in message
    assert "Columns must be numeric: ['age']" in message
    assert "'charges' has 1 missing values" in message


def test_duplicates_are_reported_but_do_not_fail(caplog):
    caplog.set_level(logging.INFO, logger="src.validate")
    df = pd.concat([make_clean(), make_clean().head(1)], ignore_index=True)
    assert run_validate(df) is True
    assert "Duplicate rows: 1" in caplog.text


def test_missing_predictors_are_reported_but_do_not_fail(caplog):
    caplog.set_level(logging.INFO, logger="src.validate")
    df = make_clean()
    df.loc[0, "bmi"] = None
    df.loc[1, "sex"] = None
    assert run_validate(df) is True
    assert "'bmi': 1" in caplog.text
    assert "'sex': 1" in caplog.text


def test_extra_columns_are_allowed():
    assert run_validate(make_clean().assign(extra="a")) is True


@pytest.mark.parametrize("expected_rows", [None, 1338])
def test_input_is_not_modified(expected_rows):
    df = make_clean()
    original = df.copy()
    try:
        run_validate(df, expected_rows=expected_rows)
    except ValueError:
        pass
    pdt.assert_frame_equal(df, original)


def write_project(tmp_path, processed: pd.DataFrame | None, expected_rows: int = 3):
    if processed is not None:
        (tmp_path / "data" / "processed").mkdir(parents=True)
        processed.to_csv(tmp_path / "data" / "processed" / "processed.csv", index=False)
    cfg = {
        "paths": {"processed_data": "data/processed/processed.csv"},
        "data": {"required_columns": REQUIRED, "target": TARGET, "numeric": NUMERIC},
        "validation": {"expected_rows": expected_rows},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    return config_path


def test_run_reads_config_and_validates_processed_csv(tmp_path):
    config_path = write_project(tmp_path, make_clean())
    assert run(config_path, root=tmp_path) is True


def test_run_uses_expected_rows_from_config(tmp_path):
    config_path = write_project(tmp_path, make_clean(), expected_rows=2)
    with pytest.raises(ValueError, match="Expected 2 rows, found 3"):
        run(config_path, root=tmp_path)


def test_run_does_not_change_the_processed_csv(tmp_path):
    config_path = write_project(tmp_path, make_clean())
    processed = tmp_path / "data" / "processed" / "processed.csv"
    before = processed.read_bytes()
    run(config_path, root=tmp_path)
    assert processed.read_bytes() == before


def test_main_cli_passes_on_valid_data(tmp_path):
    config_path = write_project(tmp_path, make_clean())
    main(["--config", str(config_path), "--root", str(tmp_path)])


def test_main_cli_exits_with_error_on_invalid_data(tmp_path, caplog):
    config_path = write_project(tmp_path, make_clean().drop(columns="bmi"))
    with pytest.raises(SystemExit) as excinfo:
        main(["--config", str(config_path), "--root", str(tmp_path)])
    assert excinfo.value.code == 1
    assert "Missing required columns: ['bmi']" in caplog.text


def test_main_cli_exits_with_error_when_processed_csv_is_missing(tmp_path, caplog):
    config_path = write_project(tmp_path, None)
    with pytest.raises(SystemExit) as excinfo:
        main(["--config", str(config_path), "--root", str(tmp_path)])
    assert excinfo.value.code == 1
    assert "processed.csv" in caplog.text


def test_repo_config_has_validation_settings():
    cfg = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    data = cfg["data"]
    assert cfg["validation"]["expected_rows"] == 1338
    assert data["target"] in data["required_columns"]
    assert set(data["numeric"]).issubset(data["required_columns"])
