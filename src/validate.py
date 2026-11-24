"""
validate.py  [Step 3: VALIDATE, the data quality gate]
======================================================
Purpose   Check schema, types and business rules BEFORE expensive steps run
Input     Clean DataFrame + the rules (required columns, allowed target values, ranges)
Output    True if valid, otherwise a clear error that stops the pipeline
Called by main.py (training) and api.py (stage 3, every live request)
Next      the split in main.py

Why it matters
- Garbage in, garbage out: stop bad data at the door, not deep inside training
- Validate, never fix. Fixing belongs in clean_data.py
- Rules live in config, so they are versioned and easy to change

Tests     tests/test_validate.py (stage 2)

Extracted from notebooks/Insurance_Single_OLS_Adapted.ipynb:
- section 2 "Load and audit the data": required columns are present; shape, duplicates and
  missing values are reported
- section 3 "Conservative cleaning": the results it should guarantee are checked, namely numeric
  columns hold numbers and no target is missing
- section 11 "Reproducibility checks": the dataset has the expected number of rows

Duplicates and missing predictors are reported but do not fail, same as the notebook: it keeps
duplicate rows and imputes missing predictors later, using training data only.

Run as a script (from the repo root, after src.clean_data):
    python -m src.validate [--config config.yaml]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.config import load_config

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def validate_data(
    df: pd.DataFrame,
    required_columns: Sequence[str],
    target: str,
    numeric: Sequence[str],
    expected_rows: int | None = None,
) -> bool:
    """Check cleaned data against the schema and business rules before training.

    All rules are checked and every failure is reported together, so one run
    shows everything that needs fixing. Duplicates and missing predictors are
    only logged, as in the notebook. ``df`` is never modified.

    Args:
        df: Cleaned data returned by :func:`src.clean_data.clean_data`.
        required_columns: Columns that must be present.
        target: Target column; it must be numeric and have no missing values.
        numeric: Columns that must have a numeric dtype.
        expected_rows: Exact row count required. Use ``None`` for live
            requests, where a fixed count does not apply.

    Returns:
        ``True`` when every rule passes.

    Raises:
        ValueError: Listing every failed rule (missing columns, no rows, wrong
            row count, non-numeric columns, missing target values).
    """
    numeric_columns = list(numeric) + [target]
    expected_columns = list(dict.fromkeys(list(required_columns) + numeric_columns))
    errors = []

    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {missing_columns}")

    if df.empty:
        errors.append("Dataset has no rows")
    if expected_rows is not None and len(df) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, found {len(df)}")

    non_numeric = [
        col
        for col in numeric_columns
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        errors.append(f"Columns must be numeric: {non_numeric}")

    if target in df.columns:
        missing_targets = int(df[target].isna().sum())
        if missing_targets:
            errors.append(f"Target '{target}' has {missing_targets} missing values")

    if errors:
        raise ValueError("Data validation failed: " + "; ".join(errors))

    missing_values = df.isna().sum()
    logger.info("Shape: %d rows, %d columns", *df.shape)
    logger.info("Duplicate rows: %d", df.duplicated().sum())
    logger.info("Missing values: %s", missing_values[missing_values > 0].to_dict())
    return True


def run(config_path: Path = DEFAULT_CONFIG_PATH, root: Path = ROOT) -> bool:
    """Validate the configured processed CSV.

    Args:
        config_path: YAML settings file with ``paths``, ``data`` and ``validation``.
        root: Folder that the relative paths in the config resolve against.

    Returns:
        ``True`` when the processed data passes every rule.

    Raises:
        FileNotFoundError: If the config or processed CSV does not exist.
        ValueError: If any validation rule fails.
    """
    cfg = load_config(Path(config_path))
    processed_path = root / cfg["paths"]["processed_data"]
    data_cfg = cfg["data"]

    df = pd.read_csv(processed_path)
    valid = validate_data(
        df,
        required_columns=data_cfg["required_columns"],
        target=data_cfg["target"],
        numeric=data_cfg["numeric"],
        expected_rows=cfg["validation"]["expected_rows"],
    )
    logger.info("Validation passed for %s", processed_path)
    return valid


def main(argv: Sequence[str] | None = None) -> None:
    """Validate the processed dataset from the command line.

    Run: ``python -m src.validate [--config config.yaml] [--root .]``.

    Args:
        argv: Command line arguments; ``None`` reads them from ``sys.argv``.

    Returns:
        None.

    Raises:
        SystemExit: With code 1 if a file is missing or validation fails.
    """
    parser = argparse.ArgumentParser(description="Validate the processed insurance dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Base folder for the paths in the config"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        run(args.config, root=args.root)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
