"""Tests for src/load_data.py."""
import logging
import sys

import pandas as pd
import pytest
import yaml

from src import load_data as ld
from src.clean_data import clean_data

COLUMNS = ["age", "sex", "bmi", "children", "smoker", "region", "charges"]
CSV_TEXT = (
    "age,sex,bmi,children,smoker,region,charges\n"
    "19,female,27.9,0,yes,southwest,16884.92\n"
    "18,male,33.77,1,no,southeast,1725.55\n"
)


@pytest.fixture
def raw_csv(tmp_path):
    path = tmp_path / "insurance.csv"
    path.write_text(CSV_TEXT)
    return path


# ---------- load_data: happy path ----------

def test_returns_dataframe_with_all_rows_and_columns(raw_csv):
    df = ld.load_data(raw_csv)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 7)
    assert list(df.columns) == COLUMNS


def test_data_is_untouched(raw_csv):
    df = ld.load_data(raw_csv)
    pd.testing.assert_frame_equal(df, pd.read_csv(raw_csv))


def test_does_not_modify_file_on_disk(raw_csv):
    before = raw_csv.read_bytes()
    ld.load_data(raw_csv)
    assert raw_csv.read_bytes() == before


def test_accepts_string_path(raw_csv):
    assert ld.load_data(str(raw_csv)).shape == (2, 7)


def test_passes_when_required_columns_present(raw_csv):
    df = ld.load_data(raw_csv, required_columns=COLUMNS)
    assert df.shape == (2, 7)


# ---------- load_data: fail fast ----------

def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="not found"):
        ld.load_data(tmp_path / "nope.csv")


def test_folder_raises(tmp_path):
    with pytest.raises(IsADirectoryError, match="folder"):
        ld.load_data(tmp_path)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(ValueError, match="empty"):
        ld.load_data(path)


def test_header_only_file_raises(tmp_path):
    path = tmp_path / "header.csv"
    path.write_text(",".join(COLUMNS) + "\n")
    with pytest.raises(ValueError, match="no rows"):
        ld.load_data(path)


def test_missing_required_columns_raises(tmp_path):
    path = tmp_path / "partial.csv"
    path.write_text("age,sex\n19,female\n")
    with pytest.raises(ValueError, match="missing required columns") as exc:
        ld.load_data(path, required_columns=COLUMNS)
    assert "charges" in str(exc.value)
    assert "age" not in str(exc.value)


# ---------- audit_data ----------

def test_audit_logs_shape_duplicates_and_missing(caplog):
    df = pd.DataFrame({"a": [1, 1, None], "b": ["x", "x", "y"]})
    with caplog.at_level(logging.INFO, logger=ld.logger.name):
        ld.audit_data(df)
    text = caplog.text
    assert "Shape: (3, 2)" in text
    assert "Duplicate rows: 1" in text
    assert "Missing values" in text
    assert "Data types" in text


# ---------- main (script entry point) ----------

def test_main_reads_path_from_config(tmp_path, raw_csv, monkeypatch):
    cfg = {
        "paths": {"raw_data": str(raw_csv)},  # absolute path overrides PROJECT_ROOT
        "data": {"required_columns": COLUMNS},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["load_data", "--config", str(cfg_path)])

    df = ld.main()
    assert df.shape == (2, 7)


def test_main_fails_when_config_points_to_missing_file(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump({"paths": {"raw_data": str(tmp_path / "missing.csv")}}))
    monkeypatch.setattr(sys, "argv", ["load_data", "--config", str(cfg_path)])

    with pytest.raises(FileNotFoundError):
        ld.main()


# ---------- integration: load_data -> clean_data ----------

def test_loaded_messy_data_is_cleaned(project, schema, column_kwargs,
                                      expected_clean_rows):
    raw_path = project.parent / "data" / "raw" / "insurance.csv"
    raw = ld.load_data(raw_path, required_columns=schema["required_columns"])

    cleaned = clean_data(raw, **column_kwargs)

    assert len(cleaned) == expected_clean_rows
    for column in schema["numeric"] + [schema["target"]]:
        assert pd.api.types.is_numeric_dtype(cleaned[column]), column
    for column in schema["categorical"]:
        values = cleaned[column].dropna()
        assert (values == values.str.strip().str.lower()).all(), column
    assert cleaned[schema["target"]].notna().all()


def test_cleaning_does_not_change_the_loaded_frame(project, column_kwargs):
    raw = ld.load_data(project.parent / "data" / "raw" / "insurance.csv")
    before = raw.copy(deep=True)
    clean_data(raw, **column_kwargs)
    pd.testing.assert_frame_equal(raw, before)


def test_loaded_tidy_data_keeps_every_row_through_cleaning(
        tmp_path, raw_frame, schema, column_kwargs):
    path = tmp_path / "insurance.csv"
    raw_frame.to_csv(path, index=False)

    raw = ld.load_data(path, required_columns=schema["required_columns"])
    cleaned = clean_data(raw, **column_kwargs)

    pd.testing.assert_frame_equal(cleaned, raw_frame, check_dtype=False)


def test_column_missing_from_loaded_data_stops_cleaning(
        tmp_path, raw_frame, column_kwargs):
    path = tmp_path / "insurance.csv"
    raw_frame.drop(columns="smoker").to_csv(path, index=False)

    raw = ld.load_data(path)  # no required_columns: the gap reaches cleaning
    with pytest.raises(KeyError, match="smoker"):
        clean_data(raw, **column_kwargs)
