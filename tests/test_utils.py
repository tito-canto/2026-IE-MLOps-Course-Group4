"""Tests for src/utils.py."""
import joblib
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from src.infer import infer
from src.load_data import load_data
from src.train import save_model, train_model
from src.utils import load_model, write_csv


# ---------- write_csv ----------

def test_write_csv_creates_missing_parent_folders(tmp_path):
    path = tmp_path / "nested" / "deeper" / "out.csv"
    write_csv(pd.DataFrame({"value": [3]}), path)
    assert pd.read_csv(path).to_dict("list") == {"value": [3]}


def test_write_csv_does_not_write_the_index(tmp_path):
    path = tmp_path / "out.csv"
    write_csv(pd.DataFrame({"value": [1, 2]}, index=[10, 20]), path)
    assert path.read_text().splitlines() == ["value", "1", "2"]


def test_write_csv_overwrites_an_existing_file(tmp_path):
    path = tmp_path / "out.csv"
    path.write_text("old,content\n1,2\n")
    write_csv(pd.DataFrame({"value": [5]}), path)
    assert pd.read_csv(path).to_dict("list") == {"value": [5]}


def test_write_csv_accepts_a_string_path(tmp_path):
    path = tmp_path / "sub" / "out.csv"
    write_csv(pd.DataFrame({"value": [7]}), str(path))
    assert path.exists()


def test_write_csv_raises_when_parent_is_a_file(tmp_path):
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("")
    with pytest.raises(OSError):
        write_csv(pd.DataFrame({"value": [1]}), blocker / "out.csv")


# ---------- load_model ----------

def test_load_model_returns_the_saved_object(tmp_path):
    path = tmp_path / "bundle.joblib"
    joblib.dump({"model": "m", "feature_columns": ["age"]}, path)
    assert load_model(path) == {"model": "m", "feature_columns": ["age"]}


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model(tmp_path / "missing.joblib")


# ---------- integration: utils -> load_data and utils -> infer ----------

def test_load_data_reads_write_csv_output_unchanged(tmp_path, raw_frame,
                                                    schema):
    path = tmp_path / "data" / "raw" / "insurance.csv"
    write_csv(raw_frame, path)

    loaded = load_data(path, required_columns=schema["required_columns"])

    pdt.assert_frame_equal(loaded, raw_frame, check_dtype=False)


def test_load_data_rejects_empty_frame_from_write_csv(tmp_path, schema):
    path = tmp_path / "empty.csv"
    write_csv(pd.DataFrame(columns=schema["required_columns"]), path)
    with pytest.raises(ValueError, match="no rows"):
        load_data(path, required_columns=schema["required_columns"])


def test_load_model_bundle_feeds_infer(tmp_path, raw_frame, schema,
                                       column_kwargs):
    result = train_model(raw_frame, **column_kwargs, seed=47)
    path = tmp_path / "models" / "model.joblib"
    save_model(result, path, **column_kwargs)
    reserved = result.splits.reserved.drop(columns=schema["target"])

    predictions = infer(load_model(path), reserved)

    assert predictions.index.equals(reserved.index)
    assert np.isfinite(predictions["prediction"]).all()
