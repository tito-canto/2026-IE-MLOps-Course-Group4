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
    """Encode categoricals without learning or modifying the input frame."""
    return pd.get_dummies(
        frame[list(predictors)], columns=list(categorical), drop_first=False, dtype=float
    ).astype(float)


def select_feature_columns(
    train: pd.DataFrame, predictors: Sequence[str], categorical: Sequence[str]
) -> list[str]:
    """Choose the fixed design on training rows, dropping the first level of each category."""
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
    """Apply the frozen design; unseen and absent categorical levels map to zeros."""
    return (
        raw_features(frame, predictors, categorical)
        .reindex(columns=list(feature_columns), fill_value=0)
        .astype(float)
    )


class NotebookEncoder(TransformerMixin, BaseEstimator):
    """Align dummy variables to the design selected from the training split."""

    def __init__(self, predictors: Sequence[str], categorical: Sequence[str], feature_columns: Sequence[str]):
        self.predictors = predictors
        self.categorical = categorical
        self.feature_columns = feature_columns

    def fit(self, X: pd.DataFrame, y=None):
        """Check the input and mark the stateless encoder as fitted."""
        prepare_features(X, self.predictors, self.categorical, self.feature_columns)
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode rows with the fixed training design."""
        check_is_fitted(self, 'is_fitted_')
        return prepare_features(X, self.predictors, self.categorical, self.feature_columns)


def build_feature_recipe(
    predictors: Sequence[str], categorical: Sequence[str], feature_columns: Sequence[str]
) -> ColumnTransformer:
    """Return an unfitted encoder and median imputer for a frozen feature design."""
    encoder = NotebookEncoder(predictors, categorical, feature_columns)
    pipeline = Pipeline([('encode', encoder), ('impute', SimpleImputer(strategy='median'))])
    return ColumnTransformer([('features', pipeline, list(predictors))], remainder='drop')


def transform_saved_features(bundle: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Transform rows using a new recipe or the legacy imputer-and-columns bundle."""
    if 'preprocessor' in bundle:
        transformed = bundle['preprocessor'].transform(frame)
    else:
        encoded = prepare_features(frame, bundle['predictors'], bundle['categorical'], bundle['feature_columns'])
        transformed = bundle['imputer'].transform(encoded)
    return pd.DataFrame(transformed, columns=bundle['feature_columns'], index=frame.index)
