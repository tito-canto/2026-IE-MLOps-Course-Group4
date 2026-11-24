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
# Whether a smaller ("lower") or larger ("higher") value means a better model.
GATE_DIRECTIONS = {"r2": "higher", "mae": "lower", "rmse": "lower"}


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict with any fitted model, adding the intercept for statsmodels OLS.

    Lets evaluation score both the notebook's OLS results and ordinary
    scikit-learn pipelines through one call.

    Args:
        model: A statsmodels OLS result or any object with ``predict(X)``.
        X: Features in the layout the model was fitted on.

    Returns:
        One float prediction per row of ``X``.

    Raises:
        ValueError: If ``X`` does not match the fitted design.
    """
    if isinstance(model, RegressionResultsWrapper):
        X = sm.add_constant(X, has_constant="add")
    return np.asarray(model.predict(X), dtype=float)


def scores(
    y_true: Sequence[float], y_pred: Sequence[float], metrics: Sequence[str] = DEFAULT_METRICS
) -> dict[str, float]:
    """Score predictions against actual values with the requested metrics.

    Plain floats are easy to log, compare between runs and feed to
    :func:`check_quality_gate`.

    Args:
        y_true: Actual target values.
        y_pred: Predicted values, same length as ``y_true``.
        metrics: Metric names from ``METRICS`` (``"r2"``, ``"mae"``, ``"rmse"``).

    Returns:
        ``{metric_name: value}``, e.g. ``{"mae": 4035.4}``.

    Raises:
        ValueError: If the inputs differ in shape, are empty or contain NaN,
            or a metric name is unknown.
    """
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
    """Score a fitted model on held-out data.

    Never pass training rows: scores on training data hide overfitting.

    Args:
        model: A statsmodels OLS result or any object with ``predict(X)``.
        X_eval: Held-out features in the layout the model was fitted on.
        y_eval: Actual target values for ``X_eval``.
        metrics: Metric names from ``METRICS``.

    Returns:
        ``{metric_name: value}`` for the held-out rows.

    Raises:
        ValueError: If predictions and targets do not line up, or a metric is unknown.
    """
    return scores(y_eval, predict(model, X_eval), metrics)


def metrics_from_config(cfg: Mapping[str, Any]) -> list[str]:
    """Read which metrics to compute from the config.

    Args:
        cfg: Loaded configuration; ``evaluation.metrics`` is optional.

    Returns:
        The configured metric names, or ``["r2", "mae", "rmse"]`` if not set.
    """
    return list(cfg.get("evaluation", {}).get("metrics", DEFAULT_METRICS))


def check_quality_gate(
    metrics: Mapping[str, float], cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """Decide whether a model's held-out metrics clear the configured quality gate.

    The gate is read from ``cfg["quality_gate"]`` with keys ``metric`` (one of
    METRICS) and ``threshold`` (a finite number). Lower-is-better metrics pass
    when ``value <= threshold``; higher-is-better metrics pass when
    ``value >= threshold``.

    Args:
        metrics: Scores returned by :func:`scores`, e.g. ``{"mae": 4035.4}``.
        cfg: Loaded configuration containing a ``quality_gate`` section.

    Returns:
        ``{"metric", "value", "threshold", "direction", "passed"}`` describing the decision.

    Raises:
        ValueError: If the gate section is missing or malformed, or its metric
            was not computed for this run.
    """
    gate = cfg.get("quality_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("Config needs a 'quality_gate' section with 'metric' and 'threshold'")
    metric, threshold = gate.get("metric"), gate.get("threshold")
    if metric not in GATE_DIRECTIONS:
        raise ValueError(f"quality_gate.metric must be one of {sorted(GATE_DIRECTIONS)}, got {metric!r}")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold):
        raise ValueError(f"quality_gate.threshold must be a finite number, got {threshold!r}")
    if metric not in metrics:
        raise ValueError(
            f"Quality gate metric {metric!r} was not computed; add it to evaluation.metrics"
        )
    value = float(metrics[metric])
    direction = GATE_DIRECTIONS[metric]
    passed = value <= threshold if direction == "lower" else value >= threshold
    return {"metric": metric, "value": value, "threshold": float(threshold),
            "direction": direction, "passed": bool(passed)}


def prediction_report(
    frame: pd.DataFrame, predicted: Sequence[float], target: str = "charges"
) -> pd.DataFrame:
    """Build the row-level error report from notebook section 9.

    Row-level errors show which customers the model under- or overestimates,
    which a single MAE value hides.

    Args:
        frame: Held-out rows including the target; not modified.
        predicted: One prediction per row of ``frame``, in row order.
        target: Target column name.

    Returns:
        A copy of ``frame`` (same index) with ``predicted_charges``,
        ``residual`` (actual minus predicted) and ``absolute_error`` columns.

    Raises:
        ValueError: If there is not exactly one prediction per row.
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
    """Evaluate a saved model on an explicitly supplied held-out CSV.

    Run: python -m src.evaluate --data held_out.csv [--model model.joblib]
    Notebook bundles receive their saved encoding and imputation; ordinary
    fitted pipelines receive raw predictor columns. No preprocessing is fitted.
    Only load trusted joblib artifacts.

    Args:
        argv: Command line arguments; ``None`` reads them from ``sys.argv``.

    Returns:
        ``{metric_name: value}``, also printed as JSON. With ``--output-dir``,
        scores and row-level predictions are written as CSVs.

    Raises:
        FileNotFoundError: If the config, model or data file does not exist.
        KeyError: If the data lacks the target or a predictor column.
        ValueError: If scoring fails (see :func:`scores`).
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
