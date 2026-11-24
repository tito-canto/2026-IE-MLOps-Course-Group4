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
from sklearn.compose import ColumnTransformer

from src.config import load_config
from src.features import build_feature_recipe, prepare_features, select_feature_columns

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"

RESERVED_SIZE = 0.10  # share of all rows set aside as the final reserved sample
TEST_SIZE = 0.25  # share of the development rows used as the test split


class Splits(NamedTuple):
    """The frozen data splits used for training and evaluation.

    Attributes:
        development: Train and test rows together (90% of the data).
        train: Rows the train-split model is fitted on.
        test: Rows for checking the train-split model.
        reserved: Final held-out sample (10%), scored once by the quality gate.
    """

    development: pd.DataFrame  # train + test
    train: pd.DataFrame
    test: pd.DataFrame
    reserved: pd.DataFrame


class TrainingResult(NamedTuple):
    """Everything :func:`train_model` produces, ready to save or evaluate.

    Attributes:
        splits: The frozen data splits.
        feature_columns: Frozen feature design chosen on the train split.
        train_model: OLS fitted on the train split only.
        train_imputer: Median imputer fitted on the train split.
        final_model: Same specification refitted on all development rows.
        final_imputer: Median imputer fitted on all development rows.
        final_preprocessor: Fitted encode-and-impute recipe for the final model.
    """

    splits: Splits
    feature_columns: list[str]
    train_model: object  # OLS fit on the train split, for evaluating on the test split
    train_imputer: SimpleImputer
    final_model: object  # same specification, refit on all development rows
    final_imputer: SimpleImputer
    final_preprocessor: ColumnTransformer


def split_data(clean: pd.DataFrame, seed: int) -> Splits:
    """Set aside the reserved sample, then split the rest into train and test.

    The reserved sample (10% of rows) is never used for fitting, so the
    quality gate is scored on data the final model has not seen.

    Args:
        clean: Cleaned, validated data.
        seed: Random seed, so the same data always gives the same splits.

    Returns:
        A :class:`Splits` with development, train (75% of development), test
        (25% of development) and reserved rows.

    Raises:
        ValueError: If there are too few rows to split.
    """
    development, reserved = train_test_split(clean, test_size=RESERVED_SIZE, random_state=seed)
    train, test = train_test_split(development, test_size=TEST_SIZE, random_state=seed)
    logger.info("Train: %d | Test: %d | Reserved: %d", len(train), len(test), len(reserved))
    return Splits(development, train, test, reserved)


def fit_ols(X: pd.DataFrame, y: pd.Series):
    """Fit ordinary least squares with an intercept, as in notebook section 6.

    OLS minimises squared error. Model quality is still judged with MAE by the
    quality gate (see ``quality_gate`` in ``config.yaml`` and the README).

    Args:
        X: Numeric features without an intercept column.
        y: Target values aligned with ``X``.

    Returns:
        The fitted statsmodels ``RegressionResultsWrapper``.

    Raises:
        ValueError: If ``X`` and ``y`` have different lengths.
    """
    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit()


def predict_ols(model, X: pd.DataFrame) -> np.ndarray:
    """Predict with a fitted OLS model, adding the intercept column it expects.

    Args:
        model: Result returned by :func:`fit_ols`.
        X: Features with the same columns used for fitting, without an intercept.

    Returns:
        One predicted target value per row of ``X``.

    Raises:
        ValueError: If ``X`` does not match the fitted design.
    """
    return np.asarray(model.predict(sm.add_constant(X, has_constant="add")))


def fit_model(
    frame: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    categorical: Sequence[str],
    feature_columns: Sequence[str],
):
    """Fit the feature recipe and OLS model on ``frame``.

    Args:
        frame: Rows to fit on, including the target.
        target: Target column name.
        predictors: Predictor columns, numeric and categorical.
        categorical: Subset of ``predictors`` that are categorical.
        feature_columns: Frozen design from :func:`select_feature_columns`.

    Returns:
        ``(model, imputer)``: the fitted OLS result and the median imputer
        fitted on ``frame``.

    Raises:
        KeyError: If a predictor or the target is missing from ``frame``.
    """
    model, recipe = _fit_model_with_recipe(frame, target, predictors, categorical, feature_columns)
    return model, recipe.named_transformers_["features"].named_steps["impute"]


def _fit_model_with_recipe(
    frame: pd.DataFrame,
    target: str,
    predictors: Sequence[str],
    categorical: Sequence[str],
    feature_columns: Sequence[str],
):
    """Fit the feature recipe and OLS model, keeping the whole fitted recipe.

    Args:
        frame: Rows to fit on, including the target.
        target: Target column name.
        predictors: Predictor columns, numeric and categorical.
        categorical: Subset of ``predictors`` that are categorical.
        feature_columns: Frozen design from :func:`select_feature_columns`.

    Returns:
        ``(model, recipe)``: the fitted OLS result and the fitted
        ``ColumnTransformer`` from :func:`build_feature_recipe`.

    Raises:
        KeyError: If a predictor or the target is missing from ``frame``.
    """
    recipe = build_feature_recipe(predictors, categorical, feature_columns)
    transformed = recipe.fit_transform(frame)
    X = pd.DataFrame(transformed, columns=feature_columns, index=frame.index)
    return fit_ols(X, frame[target]), recipe


def train_model(
    clean: pd.DataFrame,
    target: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
    seed: int,
) -> TrainingResult:
    """Split the data, fit OLS on the train split, then refit on all development rows.

    The train-split model shows how the specification generalises; the refit
    uses more data for the model that is saved and gated on the reserved sample.

    Args:
        clean: Cleaned, validated data including the target.
        target: Target column name, e.g. ``"charges"``.
        numeric: Numeric predictor columns.
        categorical: Categorical predictor columns.
        seed: Random seed for the splits.

    Returns:
        A :class:`TrainingResult` with the splits, feature design and both models.

    Raises:
        KeyError: If a predictor or the target is missing from ``clean``.
        ValueError: If there are too few rows to split.
    """
    predictors = list(numeric) + list(categorical)
    splits = split_data(clean, seed)
    feature_columns = select_feature_columns(splits.train, predictors, categorical)
    logger.info("Predictors: %s", feature_columns)

    train_split_model, train_imputer = fit_model(
        splits.train, target, predictors, categorical, feature_columns
    )
    logger.info("OLS summary (train split):\n%s", train_split_model.summary())

    final_model, final_preprocessor = _fit_model_with_recipe(
        splits.development, target, predictors, categorical, feature_columns
    )
    final_imputer = final_preprocessor.named_transformers_["features"].named_steps["impute"]
    logger.info("Final OLS summary (all development rows):\n%s", final_model.summary())

    return TrainingResult(
        splits, feature_columns, train_split_model, train_imputer, final_model, final_imputer,
        final_preprocessor,
    )


def save_model(
    result: TrainingResult,
    model_path: Path,
    target: str,
    numeric: Sequence[str],
    categorical: Sequence[str],
) -> None:
    """Save the final model and its preprocessing as one bundle, then verify it.

    Training rows are stripped from the saved model to keep personal data out
    of the file. The saved bundle is reloaded and must reproduce the in-memory
    predictions on the reserved sample.

    Args:
        result: Output of :func:`train_model`.
        model_path: Destination ``.joblib`` file; parent folders are created.
        target: Target column name, stored in the bundle.
        numeric: Numeric predictor columns.
        categorical: Categorical predictor columns.

    Returns:
        None. The bundle is written to ``model_path``.

    Raises:
        ValueError: If the saved bundle does not reproduce the fitted model.
        OSError: If the file cannot be written.
    """
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
            "preprocessor": result.final_preprocessor,
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
    loaded_X = pd.DataFrame(
        loaded['preprocessor'].transform(reserved), columns=result.feature_columns, index=reserved.index
    )
    if not np.allclose(
        predict_ols(loaded["model"], loaded_X), predict_ols(result.final_model, X_reserved)
    ):
        raise ValueError(f"Saved model at {model_path} does not reproduce the fitted model")
    logger.info("Saved model to %s", model_path)


def run(config_path: Path = DEFAULT_CONFIG_PATH, root: Path = ROOT) -> TrainingResult:
    """Train on the configured processed CSV and save the model bundle.

    Args:
        config_path: YAML settings file with ``paths``, ``data`` and ``variables``.
        root: Folder that the relative paths in the config resolve against.

    Returns:
        The :class:`TrainingResult` of the run.

    Raises:
        FileNotFoundError: If the config or processed CSV does not exist.
        ValueError: If the saved model does not reproduce the fitted model.
    """
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
    """Train and save the model from the command line.

    Run: ``python -m src.train [--config config.yaml] [--root .]``.

    Args:
        argv: Command line arguments; ``None`` reads them from ``sys.argv``.

    Returns:
        None.

    Raises:
        SystemExit: With code 1 if a file is missing or training fails.
    """
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
