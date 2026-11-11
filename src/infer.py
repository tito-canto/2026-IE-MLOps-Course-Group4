"""
infer.py  [Step 7: INFER]
=========================
Purpose   Turn new, unseen records into predictions with the saved pipeline
Input     Fitted pipeline + new data (DataFrame, same feature columns, no target)
Output    DataFrame with a fixed schema: "prediction" (+ optional "probability"), same index as input
Called by main.py (batch, writes reports/predictions.csv) and api.py (stage 3, live requests)

Why it matters
- No training here: inference is reused unchanged by the API, so batch and live results match
- Fixed output columns: downstream teams can rely on them
- Keep the input index (and ID) so predictions join back to source records

Tests     tests/test_infer.py (stage 2)

Run       python -m src.infer --data new_records.csv [--output reports/predictions.csv]
          Only load trusted joblib artifacts. Unseen categories follow the notebook:
          their dummy columns are discarded when aligning the saved feature design.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd

from src.features import transform_saved_features
from src.train import predict_ols
from src.config import load_config

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def infer(bundle: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Predict from the notebook's saved bundle (sections 6, 9 and 11).

    Reuse the frozen design and fitted imputer without fitting on new rows.
    Apply the notebook's stateless predictor cleaning without requiring a target.
    """
    missing = sorted(set(bundle["predictors"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required predictors: {missing}")
    if frame.empty:
        raise ValueError("Cannot infer on an empty dataset")
    clean = frame[list(bundle["predictors"])].copy()
    for column in bundle["predictors"]:
        if column in bundle["categorical"]:
            clean[column] = clean[column].astype("string").str.strip().str.lower()
        else:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")
    X = transform_saved_features(bundle, clean)
    predicted = predict_ols(bundle["model"], X)
    if predicted.shape != (len(frame),) or not np.isfinite(predicted).all():
        raise ValueError("Model must return one finite prediction per row")
    return pd.DataFrame({"prediction": predicted}, index=frame.index)


def main(argv: Sequence[str] | None = None) -> pd.DataFrame:
    """Load a trusted saved bundle and predict every row of an explicit input CSV."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--root", type=Path, default=ROOT, help="Base folder for config paths")
    parser.add_argument("--data", type=Path, required=True, help="CSV of new predictor rows")
    parser.add_argument("--model", type=Path, help="Override the configured model path")
    parser.add_argument("--output", type=Path, help="Override the prediction CSV path")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    model_path = args.model or args.root / cfg["paths"]["model"]
    output_path = args.output or args.root / cfg["paths"]["predictions"] / "predictions.csv"
    if output_path.resolve() in {args.data.resolve(), model_path.resolve(), args.config.resolve()}:
        raise ValueError("Output must not overwrite input data, model, or configuration")
    bundle = joblib.load(model_path)
    result = infer(bundle, pd.read_csv(args.data))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Wrote {len(result)} predictions to {output_path}")
    return result


if __name__ == "__main__":
    main()
