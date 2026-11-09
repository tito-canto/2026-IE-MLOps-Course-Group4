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
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.train import prepare_features, predict_ols


def infer(bundle: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Predict from the notebook's saved bundle (sections 6, 9 and 11).

    Reuse the frozen design and fitted imputer without fitting on new rows.
    Input predictors should already be cleaned, as in the notebook.
    """
    missing = sorted(set(bundle["predictors"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required predictors: {missing}")
    if frame.empty:
        raise ValueError("Cannot infer on an empty dataset")
    features = prepare_features(
        frame, bundle["predictors"], bundle["categorical"], bundle["feature_columns"]
    )
    X = pd.DataFrame(
        bundle["imputer"].transform(features),
        columns=bundle["feature_columns"], index=frame.index,
    )
    predicted = predict_ols(bundle["model"], X)
    if predicted.shape != (len(frame),) or not np.isfinite(predicted).all():
        raise ValueError("Model must return one finite prediction per row")
    return pd.DataFrame({"prediction": predicted}, index=frame.index)
