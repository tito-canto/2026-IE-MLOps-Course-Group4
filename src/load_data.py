"""
load_data.py  [Step 1: LOAD]
============================
Purpose   Read the raw dataset into a DataFrame, exactly as it is on disk
Input     Path to the raw file (config: paths.raw_data)
Output    Raw pandas DataFrame, untouched
Called by main.py          Uses   utils.py (file reading)
Next      clean_data.py

Why it matters
- One entry point for data: swap CSV for a database or API later without touching other files
- Read only: the raw file is never modified, so every run is reproducible
- Fail fast with a clear message if the file is missing, a folder, or empty

Source    notebooks/Insurance_Single_OLS_Adapted.ipynb, section 2 "Load and audit the data"
Run       python -m src.load_data            (uses config.yaml at the project root)
          python -m src.load_data --config path/to/config.yaml

Tests     tests/test_load_data.py (stage 2)
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_data(path: Path, required_columns: list[str] | None = None) -> pd.DataFrame:
    """Read the raw CSV at `path` and return it untouched.

    Raises a clear error if the path is missing, is a folder, is empty,
    or (when `required_columns` is given) lacks any expected column.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")
    if path.is_dir():
        raise IsADirectoryError(f"Expected a file but got a folder: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Raw data file is empty: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Raw data file has no rows: {path}")

    if required_columns:
        missing = sorted(set(required_columns) - set(df.columns))
        if missing:
            raise ValueError(f"Raw data is missing required columns: {missing}")

    return df


def audit_data(df: pd.DataFrame) -> None:
    """Log a quick audit of the raw data (shape, duplicates, missing values, dtypes)."""
    logger.info("Shape: %s", df.shape)
    logger.info("Duplicate rows: %d", df.duplicated().sum())
    logger.info("Missing values:\n%s", df.isna().sum().to_string())
    logger.info("Data types:\n%s", df.dtypes.to_string())
    logger.info("Head:\n%s", df.head().to_string())


def main() -> pd.DataFrame:
    parser = argparse.ArgumentParser(description="Load and audit the raw dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    raw_path = PROJECT_ROOT / cfg["paths"]["raw_data"]
    required = cfg.get("data", {}).get("required_columns")

    df = load_data(raw_path, required_columns=required)
    audit_data(df)
    return df


if __name__ == "__main__":
    main()
