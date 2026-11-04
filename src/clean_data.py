"""
clean_data.py  [Step 2: CLEAN]
==============================
Purpose   Make raw data consistent so every later step behaves predictably
Input     Raw DataFrame + target column name
Output    Clean DataFrame, guaranteed to contain the target
Called by main.py (training) and api.py (stage 3, same cleaning for live requests)
Next      validate.py

Why it matters
- Stateless fixes only: column names, duplicates, rows without a target, index reset
- Nothing that LEARNS from data (means, bins, scaling): that belongs in features.py, after the split
- Same input always gives the same output, so it is safe to rerun
- Report how many rows were dropped: a sudden jump signals an upstream problem

Tests     tests/test_data_clean.py (stage 2)

Extracted from notebooks/Insurance_Single_OLS_Adapted.ipynb, section 3 "Conservative cleaning":
- numeric columns (and the target) are coerced to numbers; unparseable values become NaN
- categorical text is stripped and lower-cased
- rows with a missing target are dropped
- predictor missingness is left alone (imputed later on training data only)
- no outlier removal and no duplicate removal (same as the notebook)

Run as a script (from the repo root):
    python -m src.clean_data [--config config.yaml]
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


def clean_data(
    df: pd.DataFrame,
    target: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> pd.DataFrame:
    """Return a cleaned copy of ``df``; the input DataFrame is never modified."""
    required = list(numeric) + list(categorical) + [target]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    clean = df.copy()
    for col in list(numeric) + [target]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    for col in categorical:
        clean[col] = clean[col].astype("string").str.strip().str.lower()

    rows_before = len(clean)
    clean = clean.dropna(subset=[target]).reset_index(drop=True)
    logger.info(
        "Dropped %d of %d rows with a missing target", rows_before - len(clean), rows_before
    )
    return clean


def run(config_path: Path = DEFAULT_CONFIG_PATH, root: Path = ROOT) -> pd.DataFrame:
    """Read the raw CSV named in the config, clean it, and write the processed CSV."""
    cfg = load_config(Path(config_path))
    raw_path = root / cfg["paths"]["raw_data"]
    processed_path = root / cfg["paths"]["processed_data"]
    data_cfg = cfg["data"]

    df = pd.read_csv(raw_path)
    clean = clean_data(
        df,
        target=data_cfg["target"],
        numeric=data_cfg["numeric"],
        categorical=data_cfg["categorical"],
    )

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(processed_path, index=False)
    logger.info("Wrote %d clean rows to %s", len(clean), processed_path)
    return clean


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Clean the raw insurance dataset.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="Base folder for the paths in the config"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run(args.config, root=args.root)


if __name__ == "__main__":
    main()
