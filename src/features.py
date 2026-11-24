"""Notebook-compatible feature encoding and an unfitted preprocessing recipe.

Choose feature columns from the training split only. Fit the returned recipe
on training data, or on development data using that same frozen design.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)


def raw_features(frame: pd.DataFrame, predictors: Sequence[str], categorical: Sequence[str]) -> pd.DataFrame:
    """One-hot encode every level of each categorical predictor.

    Nothing is learned, so the result depends only on the rows given. Which
    dummy columns are kept is decided later by :func:`select_feature_columns`.

    Args:
        frame: Data containing every column in ``predictors``; not modified.
        predictors: Predictor columns to keep, numeric and categorical.
        categorical: Subset of ``predictors`` to one-hot encode.

    Returns:
        A float DataFrame with numeric predictors and one ``<column>_<level>``
        dummy per categorical level, on the same index as ``frame``.

    Raises:
        KeyError: If a predictor column is missing from ``frame``.
    """
    return pd.get_dummies(
        frame[list(predictors)], columns=list(categorical), drop_first=False, dtype=float
    ).astype(float)


def select_feature_columns(
    train: pd.DataFrame, predictors: Sequence[str], categorical: Sequence[str]
) -> list[str]:
    """Freeze the model's feature columns using the training split only.

    The alphabetically first dummy of each categorical becomes the reference
    level and is dropped, which avoids perfect collinearity with the OLS
    intercept. Choosing it on training rows keeps test data out of the design.

    Args:
        train: Training split rows.
        predictors: Predictor columns, numeric and categorical.
        categorical: Subset of ``predictors`` that are categorical.

    Returns:
        Ordered feature column names, e.g. ``["age", ..., "smoker_yes", ...]``.

    Raises:
        KeyError: If a predictor column is missing from ``train``.
    """
    encoded = raw_features(train, predictors, categorical)
    reference = {}
    for column in categorical:
        dummy_columns = sorted(name for name in encoded.columns if name.startswith(f"{column}_"))
        reference[column] = dummy_columns[0] if dummy_columns else None
    logger.info("Reference categories: %s", reference)
    return [name for name in encoded.columns if name not in reference.values()]


def prepare_features(
    frame: pd.DataFrame,
    predictors: Sequence[str],
    categorical: Sequence[str],
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    """Encode rows into the frozen feature design chosen on the training split.

    New rows may lack a level or contain one never seen in training; both are
    aligned to the frozen columns so the model always receives the same shape.

    Args:
        frame: Rows to encode; not modified.
        predictors: Predictor columns, numeric and categorical.
        categorical: Subset of ``predictors`` that are categorical.
        feature_columns: Column order from :func:`select_feature_columns`.

    Returns:
        A float DataFrame with exactly ``feature_columns``. Absent levels are 0
        and dummies for unseen levels are dropped.

    Raises:
        KeyError: If a predictor column is missing from ``frame``.
    """
    return (
        raw_features(frame, predictors, categorical)
        .reindex(columns=list(feature_columns), fill_value=0)
        .astype(float)
    )


class NotebookEncoder(TransformerMixin, BaseEstimator):
    """scikit-learn transformer that applies :func:`prepare_features`.

    Wrapping the encoding as a transformer lets it be saved in one pipeline
    with the imputer, so inference repeats exactly what training did.

    Args:
        predictors: Predictor columns, numeric and categorical.
        categorical: Subset of ``predictors`` that are categorical.
        feature_columns: Frozen column order from :func:`select_feature_columns`.
    """

    def __init__(self, predictors: Sequence[str], categorical: Sequence[str], feature_columns: Sequence[str]):
        self.predictors = predictors
        self.categorical = categorical
        self.feature_columns = feature_columns

    def fit(self, X: pd.DataFrame, y=None):
        """Check that ``X`` can be encoded and mark the encoder as fitted.

        The encoder learns nothing; the design was frozen beforehand.

        Args:
            X: Rows containing every predictor column.
            y: Ignored; present for scikit-learn compatibility.

        Returns:
            This encoder, so calls can be chained.

        Raises:
            KeyError: If a predictor column is missing from ``X``.
        """
        prepare_features(X, self.predictors, self.categorical, self.feature_columns)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode rows into the frozen feature design.

        Args:
            X: Rows containing every predictor column.

        Returns:
            A float DataFrame with exactly ``feature_columns``.

        Raises:
            sklearn.exceptions.NotFittedError: If :meth:`fit` was not called.
            KeyError: If a predictor column is missing from ``X``.
        """
        check_is_fitted(self, 'is_fitted_')
        return prepare_features(X, self.predictors, self.categorical, self.feature_columns)


def build_feature_recipe(
    predictors: Sequence[str], categorical: Sequence[str], feature_columns: Sequence[str]
) -> ColumnTransformer:
    """Build the shared, unfitted preprocessing recipe: encode, then median-impute.

    Training fits this recipe and saves it inside the model bundle, so
    inference reuses the same encoding and the medians learned on fit data.

    Args:
        predictors: Predictor columns, numeric and categorical.
        categorical: Subset of ``predictors`` that are categorical.
        feature_columns: Frozen column order from :func:`select_feature_columns`.

    Returns:
        An unfitted ``ColumnTransformer`` that outputs ``feature_columns``
        and drops every other input column (including the target).
    """
    encoder = NotebookEncoder(predictors, categorical, feature_columns)
    pipeline = Pipeline([('encode', encoder), ('impute', SimpleImputer(strategy='median'))])
    return ColumnTransformer([('features', pipeline, list(predictors))], remainder='drop')


def transform_saved_features(bundle: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Turn rows into model-ready features with a saved bundle's fitted recipe.

    Bundles with a ``preprocessor`` use it directly. Older bundles, which only
    hold the fitted ``imputer`` and frozen ``feature_columns``, are still
    supported so previously saved models keep working.

    Args:
        bundle: Saved bundle from :func:`src.train.save_model`.
        frame: Cleaned rows containing every predictor column.

    Returns:
        Imputed features with columns ``bundle["feature_columns"]`` and the
        same index as ``frame``.

    Raises:
        KeyError: If the bundle lacks a required entry or ``frame`` lacks a predictor.
    """
    if 'preprocessor' in bundle:
        transformed = bundle['preprocessor'].transform(frame)
    else:
        encoded = prepare_features(frame, bundle['predictors'], bundle['categorical'], bundle['feature_columns'])
        transformed = bundle['imputer'].transform(encoded)
    return pd.DataFrame(transformed, columns=bundle['feature_columns'], index=frame.index)
