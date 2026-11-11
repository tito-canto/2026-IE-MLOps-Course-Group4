"""
train.py  [Step 5: TRAIN]
=========================
Purpose   Bundle the feature recipe + the algorithm into ONE artifact and fit it
Input     Clean DataFrame + column lists and random seed (from config)
Output    One fitted bundle (imputer + OLS model + column lists), saved as models/model.joblib
Called by main.py          Next   evaluate.py

Why it matters
- Training fits on the train split only, so validation and test metrics stay honest
- One artifact for preprocessing + model: what you test is exactly what you deploy
- Column lists and the seed come from config
- Fixed random seed: same data, same result

Tests     tests/test_train.py (stage 2)

Extracted from notebooks/Insurance_Single_OLS_Adapted.ipynb:
- section 5 "Freeze development and reserved samples": the train / test / reserved split
- section 6 "Encode predictors and define the single OLS model": dummy encoding, median
  imputation and the OLS fit on the train split
- section 9 "Refit the same model": the same specification refit on all development rows
- section 11 "Reproducibility checks and exports": saving the bundle and checking that the saved
  file reproduces the predictions

Scoring, diagnostics and the exported score and coefficient tables (sections 7, 8, 9 and 10) are
not training, so they are left for evaluate.py. Plots are not produced here.

Run as a script (from the repo root, after src.clean_data):
    python -m src.train [--config config.yaml]
"""
from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import NamedTuple, Sequence

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from src.config import load_config

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

RESERVED_SIZE = 0.10  # share of all rows set aside as the final reserved sample
TEST_SIZE = 0.25  # share of the development rows used as the test split


class Splits(NamedTuple):
    development: pd.DataFrame  # train + test
    train: pd.DataFrame
    test: pd.DataFrame
    reserved: pd.DataFrame


class TrainingResult(NamedTuple):
    splits: Splits
    feature_columns: list[str]
    train_model: object  # OLS fit on the train split, for evaluating on the test split
    train_imputer: SimpleImputer
    final_model: object  # same specification, refit on all development rows
    final_imputer: SimpleImputer


def split_data(clean: pd.DataFrame, seed: int) -> Splits:
    """Set aside the reserved sample, then split the remaining rows into train and test."""
    development, reserved = train_test_split(clean, test_size=RESERVED_SIZE, random_state=seed)
    train, test = train_test_split(development, test_size=TEST_SIZE, random_state=seed)
    logger.info("Train: %d | Test: %d | Reserved: %d", len(train), len(test), len(reserved))
    return Splits(development, train, test, reserved)


def raw_features(
    frame: pd.DataFrame, predictors: Sequence[str], categorical: Sequence[str]
) -> pd.DataFrame:
    """Return the predictors with every categorical level expanded into a dummy column."""
    return pd.get_dummies(
        frame[list(predictors)], columns=list(categorical), drop_first=False, dtype=float
    ).astype(float)


def select_feature_columns(
    train: pd.DataFrame, predictors: Sequence[str], categorical: Sequence[str]
) -> list[str]:
    """Freeze the model's columns from the train split, dropping each variable's first level."""
    train_all = raw_features(train, predictors, categorical)
    reference = {}
    for col in categorical:
        dummy_cols = sorted(c for c in train_all.columns if c.startswith(f"{col}_"))
        reference[col] = dummy_cols[0] if dummy_cols else None
    logger.info("Reference categories: %s", reference)
    return [c for c in train_all.columns if c not in reference.values()]


def prepare_features(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    categorical: Sequence[str],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Encode ``frame`` into the fixed design; levels unseen in training are ignored."""
    return (
        raw_features(frame, predictors, categorical)
        .reindex(columns=list(feature_columns), fill_value=0)
        .astype(float)
    )


def fit_ols(X: pd.DataFrame, y: pd.Series):
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()


def predict_ols(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(sm.add_constant(X, has_constant="add")))


def fit_model(
    frame: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    categorical: Sequence[str],
    feature_columns: Sequence[str],
):
    """Fit a median imputer and the OLS model on ``frame``; return ``(model, imputer)``."""
    features = prepare_features(frame, predictors, categorical, feature_columns)
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(features), columns=feature_columns, index=frame.index)
    return fit_ols(X, frame[target]), imputer


def train_model(
    clean: pd.DataFrame,
    target: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
    seed: int,
) -> TrainingResult:
    """Split ``clean``, fit the model on the train split, and refit it on all development rows."""
    predictors = list(numeric) + list(categorical)
    splits = split_data(clean, seed)
    feature_columns = select_feature_columns(splits.train, predictors, categorical)
    logger.info("Predictors: %s", feature_columns)

    train_split_model, train_imputer = fit_model(
        splits.train, target, predictors, categorical, feature_columns
    )
    logger.info("OLS summary (train split):\n%s", train_split_model.summary())

    final_model, final_imputer = fit_model(
        splits.development, target, predictors, categorical, feature_columns
    )
    logger.info("Final OLS summary (all development rows):\n%s", final_model.summary())

    return TrainingResult(
        splits, feature_columns, train_split_model, train_imputer, final_model, final_imputer
    )


def save_model(
    result: TrainingResult,
    model_path: Path,
    target: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> None:
    """Save the final model with its preprocessing, then check the file predicts identically."""
    predictors = list(numeric) + list(categorical)
    saved_model = copy.deepcopy(result.final_model)
    saved_model.remove_data()  # Keep the training rows out of the saved file.

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": saved_model,
            "imputer": result.final_imputer,
            "predictors": predictors,
            "categorical": list(categorical),
            "feature_columns": result.feature_columns,
            "target": target,
        },
        model_path,
    )

    reserved = result.splits.reserved
    X_reserved = pd.DataFrame(
        result.final_imputer.transform(
            prepare_features(reserved, predictors, categorical, result.feature_columns)
        ),
        columns=result.feature_columns,
        index=reserved.index,
    )
    loaded = joblib.load(model_path)
    if not np.allclose(
        predict_ols(loaded["model"], X_reserved), predict_ols(result.final_model, X_reserved)
    ):
        raise ValueError(f"Saved model at {model_path} does not reproduce the fitted model")
    logger.info("Saved model to %s", model_path)


def run(config_path: Path = DEFAULT_CONFIG_PATH, root: Path = ROOT) -> TrainingResult:
    """Read the processed CSV named in the config, train the model, and save it."""
    cfg = load_config(Path(config_path))
    processed_path = root / cfg["paths"]["processed_data"]
    model_path = root / cfg["paths"]["model"]
    data_cfg = cfg["data"]

    clean = pd.read_csv(processed_path)
    result = train_model(
        clean,
        target=data_cfg["target"],
        numeric=data_cfg["numeric"],
        categorical=data_cfg["categorical"],
        seed=cfg["variables"]["seed"],
    )
    save_model(
        result,
        model_path,
        target=data_cfg["target"],
        numeric=data_cfg["numeric"],
        categorical=data_cfg["categorical"],
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the insurance charges OLS model.")
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
