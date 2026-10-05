"""Verify feature and target separation independently from data cleaning."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from purchase_propensity.features.build_features import split_features_and_target


def test_split_features_and_target_preserves_values_and_index() -> None:
    dataframe = pd.DataFrame({"page_views": [3, 1], "purchase": [1, 0]}, index=[5, 9])
    original = dataframe.copy(deep=True)

    features, target = split_features_and_target(dataframe, "purchase")

    assert_frame_equal(features, original[["page_views"]])
    assert_series_equal(target, original["purchase"])
    features.iloc[0, 0] = 99
    target.iloc[0] = 0
    assert_frame_equal(dataframe, original)


def test_split_features_and_target_rejects_missing_target() -> None:
    dataframe = pd.DataFrame({"page_views": [3]})

    with pytest.raises(ValueError, match="Target column not found: purchase"):
        split_features_and_target(dataframe, "purchase")


def test_split_features_and_target_rejects_duplicate_columns() -> None:
    dataframe = pd.DataFrame([[1, 0]], columns=["purchase", "purchase"])

    with pytest.raises(ValueError, match="Column names must be unique"):
        split_features_and_target(dataframe, "purchase")
