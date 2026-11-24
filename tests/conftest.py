"""Shared fixtures for tests that chain one pipeline stage into the next.

The raw dataset is personal data and is not tracked in Git, so integration
tests use a generated dataset with the same schema as ``config.yaml``.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

TARGET = "charges"
NUMERIC = ["age", "bmi", "children"]
CATEGORICAL = ["sex", "smoker", "region"]
ROWS = 80
SEED = 47
REGIONS = ["northeast", "northwest", "southeast", "southwest"]
# Rows of the messy frame whose target cannot be used (text and missing).
BAD_TARGET_ROWS = [3, 4]


def make_raw(rows: int = ROWS, seed: int = SEED) -> pd.DataFrame:
    """Generate tidy insurance-like data with a known linear relationship.

    Args:
        rows: Number of rows to generate.
        seed: Random seed, so every test run sees the same data.

    Returns:
        A DataFrame with the numeric, categorical and target columns.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 65, rows)
    bmi = rng.uniform(16, 45, rows).round(2)
    children = rng.integers(0, 5, rows)
    smoker = rng.choice(["yes", "no"], rows, p=[0.3, 0.7])
    charges = (
        250 * age + 300 * bmi + 500 * children + 20000 * (smoker == "yes")
        + rng.normal(0, 1000, rows)
    ).round(2)
    return pd.DataFrame({
        "age": age,
        "sex": rng.choice(["female", "male"], rows),
        "bmi": bmi,
        "children": children,
        "smoker": smoker,
        "region": rng.choice(REGIONS, rows),
        "charges": charges,
    })


@pytest.fixture
def schema() -> dict:
    """Column roles shared by every stage, as named in ``config.yaml``."""
    return {
        "target": TARGET,
        "numeric": list(NUMERIC),
        "categorical": list(CATEGORICAL),
        "required_columns": NUMERIC + CATEGORICAL + [TARGET],
    }


@pytest.fixture
def column_kwargs(schema: dict) -> dict:
    """Keyword arguments for clean_data, train_model and save_model."""
    return {key: schema[key] for key in ("target", "numeric", "categorical")}


@pytest.fixture
def validate_kwargs(schema: dict) -> dict:
    """Keyword arguments for ``validate_data``."""
    keys = ("required_columns", "target", "numeric")
    return {key: schema[key] for key in keys}


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """Tidy raw data that every stage should accept unchanged in shape."""
    return make_raw()


@pytest.fixture
def messy_raw_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """Raw data with the problems ``clean_data`` must fix before validation.

    Numbers stored as text, inconsistent category spelling, an unparseable
    predictor (kept as NaN) and two unusable targets (rows dropped).
    """
    messy = raw_frame.astype(object)
    messy.loc[0, "age"] = str(raw_frame.loc[0, "age"])
    messy.loc[1, "sex"] = f"  {str(raw_frame.loc[1, 'sex']).upper()} "
    messy.loc[1, "smoker"] = f"{str(raw_frame.loc[1, 'smoker']).title()} "
    messy.loc[2, "region"] = str(raw_frame.loc[2, "region"]).upper()
    messy.loc[2, "bmi"] = "unknown"
    messy.loc[BAD_TARGET_ROWS[0], "charges"] = "not a number"
    messy.loc[BAD_TARGET_ROWS[1], "charges"] = None
    return messy


@pytest.fixture
def expected_clean_rows() -> int:
    """Rows left after cleaning ``messy_raw_frame``."""
    return ROWS - len(BAD_TARGET_ROWS)


@pytest.fixture
def project(tmp_path: Path, messy_raw_frame: pd.DataFrame, schema: dict,
            expected_clean_rows: int) -> Path:
    """Lay out a project folder with the messy raw CSV and a full config.

    Args:
        tmp_path: pytest temporary folder used as the project root.
        messy_raw_frame: Raw data written to ``data/raw/insurance.csv``.
        schema: Column roles written to the ``data`` section.
        expected_clean_rows: Row count the validation section expects.

    Returns:
        Path to the written ``config.yaml``; relative paths resolve against
        ``tmp_path``.
    """
    raw_path = tmp_path / "data" / "raw" / "insurance.csv"
    raw_path.parent.mkdir(parents=True)
    messy_raw_frame.to_csv(raw_path, index=False)
    cfg = {
        "paths": {
            "raw_data": "data/raw/insurance.csv",
            "processed_data": "data/processed/processed.csv",
            "model": "models/model.joblib",
            "predictions": "data/inference",
        },
        "data": schema,
        "validation": {"expected_rows": expected_clean_rows},
        "variables": {"seed": SEED},
        "evaluation": {"metrics": ["mae"]},
        "quality_gate": {"metric": "mae", "threshold": 1e9},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return config_path
