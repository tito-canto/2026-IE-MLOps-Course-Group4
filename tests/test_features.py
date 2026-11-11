"""Tests for the notebook-compatible feature recipe."""
import joblib
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
from sklearn.exceptions import NotFittedError

from src.features import build_feature_recipe, prepare_features, select_feature_columns


def test_recipe_matches_notebook_and_keeps_training_medians(tmp_path):
    train = pd.DataFrame({'age': [20., 30., np.nan], 'sex': ['f', 'm', 'f']}, index=[5, 2, 8])
    held_out = pd.DataFrame({'age': [np.nan, 1000., 40.], 'sex': ['m', 'new', None]}, index=[4, 7, 9])
    before = train.copy(deep=True)
    columns = select_feature_columns(train, ['age', 'sex'], ['sex'])
    assert columns == ['age', 'sex_m']
    recipe = build_feature_recipe(['age', 'sex'], ['sex'], columns)
    with pytest.raises(NotFittedError):
        recipe.transform(held_out)
    np.testing.assert_allclose(recipe.fit_transform(train), [[20., 0.], [30., 1.], [25., 0.]])
    expected = [[25., 1.], [1000., 0.], [40., 0.]]
    np.testing.assert_allclose(recipe.transform(held_out), expected)
    pdt.assert_frame_equal(train, before)
    path = tmp_path / 'recipe.joblib'
    joblib.dump(recipe, path)
    np.testing.assert_allclose(joblib.load(path).transform(held_out), expected)


def test_encoding_keeps_index_and_original_data():
    data = pd.DataFrame({'age': [31., 40.], 'sex': ['f', 'new']}, index=[9, 3])
    original = data.copy(deep=True)
    result = prepare_features(data, ['age', 'sex'], ['sex'], ['age', 'sex_m'])
    pdt.assert_frame_equal(result, pd.DataFrame({'age': [31., 40.], 'sex_m': [0., 0.]}, index=[9, 3]))
    pdt.assert_frame_equal(data, original)


def test_recipe_rejects_missing_columns():
    recipe = build_feature_recipe(['age', 'sex'], ['sex'], ['age', 'sex_m'])
    with pytest.raises((KeyError, ValueError), match='sex'):
        recipe.fit(pd.DataFrame({'age': [20.]}))
