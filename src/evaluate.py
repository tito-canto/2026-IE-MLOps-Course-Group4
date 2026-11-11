"""
evaluate.py  [Step 6: EVALUATE]
===============================
Purpose   Score the fitted pipeline on data it has never seen
Input     Fitted pipeline + X_eval, y_eval (validation split; test split once, at the end)
Output    Dictionary of plain numbers, e.g. {"r2": 0.78, "mae": 4180.5, "rmse": 5920.3}
Called by main.py          Feeds  stage 3: the quality gate and experiment tracking

Why it matters
- Never score on training data: it hides overfitting
- Metric choice follows the business goal and problem type (set in config: evaluation.metrics)
- Plain numbers are easy to log, compare between runs and turn into a go/no-go decision

Tests     tests/test_evaluate.py (stage 2)

Extracted from notebooks/Insurance_Single_OLS_Adapted.ipynb (sections 6, 8 and 9):
- scores(): R2, MAE and RMSE of actual vs predicted charges
- predict(): adds the intercept column for statsmodels OLS results, as predict_ols() did
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.regression.linear_model import RegressionResultsWrapper

from src.features import transform_saved_features

METRICS = {
    "r2": r2_score,
    "mae": mean_absolute_error,
    "rmse": lambda y, p: float(np.sqrt(mean_squared_error(y, p))),
}
DEFAULT_METRICS = ("r2", "mae", "rmse")


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted model; statsmodels OLS results get the intercept added."""
    if isinstance(model, RegressionResultsWrapper):
        X = sm.add_constant(X, has_constant="add")
    return np.asarray(model.predict(X), dtype=float)


def scores(
    y_true: Sequence[float], y_pred: Sequence[float], metrics: Sequence[str] = DEFAULT_METRICS
) -> dict[str, float]:
    """Compare actual and predicted values; returns {metric_name: plain float}."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    if y_true.size == 0:
        raise ValueError("Cannot evaluate on an empty set")
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        raise ValueError("y_true and y_pred must not contain NaN")

    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; choose from {sorted(METRICS)}")
    return {name: float(METRICS[name](y_true, y_pred)) for name in metrics}


def evaluate(
    model: Any,
    X_eval: pd.DataFrame,
    y_eval: Sequence[float],
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> dict[str, float]:
    """Score a fitted model on held-out data (never the training split)."""
    return scores(y_eval, predict(model, X_eval), metrics)


def metrics_from_config(cfg: Mapping[str, Any]) -> list[str]:
    """Read evaluation.metrics from the config, falling back to the defaults."""
    return list(cfg.get("evaluation", {}).get("metrics", DEFAULT_METRICS))


def prediction_report(
    frame: pd.DataFrame, predicted: Sequence[float], target: str = "charges"
) -> pd.DataFrame:
    """Extract the row-level error report from notebook section 9.

    Predictions follow row order, preserving the held-out frame's index.
    """
    predicted = np.asarray(predicted, dtype=float)
    if predicted.shape != (len(frame),):
        raise ValueError("Expected one prediction per evaluation row")
    report = frame.copy()
    report["predicted_charges"] = predicted
    report["residual"] = report[target] - report["predicted_charges"]
    report["absolute_error"] = report["residual"].abs()
    return report


def main(argv: Sequence[str] | None = None) -> dict[str, float]:
    """Evaluate an explicitly supplied held-out CSV using a saved model.

    Run: python -m src.evaluate --data held_out.csv [--model model.joblib]
    Notebook bundles receive their saved encoding and imputation; ordinary
    fitted pipelines receive raw predictor columns. No preprocessing is fitted.
    Only load trusted joblib artifacts.
    """
    from src.config import load_config

    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--config", type=Path, default=root / "config.yaml")
    parser.add_argument("--data", type=Path, required=True, help="Held-out CSV; never training data")
    parser.add_argument("--model", type=Path, help="Trusted joblib model or notebook bundle")
    parser.add_argument("--output-dir", type=Path, help="Optionally save metrics and prediction CSVs")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    artifact = joblib.load(args.model or root / cfg["paths"]["model"])
    frame = pd.read_csv(args.data)
    if isinstance(artifact, dict):
        target = artifact["target"]
        X = transform_saved_features(artifact, frame)
        model = artifact["model"]
    else:
        target = cfg.get("data", {}).get("target", "charges")
        X = frame.drop(columns=[target])
        model = artifact
    predicted = predict(model, X)
    result = scores(frame[target], predicted, metrics_from_config(cfg))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([result]).to_csv(args.output_dir / "evaluation_scores.csv", index=False)
        prediction_report(frame, predicted, target).to_csv(
            args.output_dir / "evaluation_predictions.csv", index=False,
        )
    print(json.dumps(result, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
