"""Verify cleaning behavior and preservation of the caller's data."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from purchase_propensity.data.preprocessing import (
    normalize_column_names,
    remove_duplicate_rows,
    remove_empty_rows,
)


def test_normalize_column_names_converts_labels_without_mutation() -> None:
    dataframe = pd.DataFrame(
        [[3, 10, 1]], columns=[" Page Views ", "TIME--ON SITE", "Purchase"]
    )
    original = dataframe.copy(deep=True)
    expected = pd.DataFrame(
        [[3, 10, 1]], columns=["page_views", "time_on_site", "purchase"]
    )

    normalized = normalize_column_names(dataframe)

    assert_frame_equal(normalized, expected)
    normalized.iloc[0, 0] = 99
    assert_frame_equal(dataframe, original)


def test_normalize_column_names_converts_numeric_labels_to_strings() -> None:
    dataframe = pd.DataFrame([[1]], columns=[123])

    assert normalize_column_names(dataframe).columns.tolist() == ["123"]


def test_normalize_column_names_rejects_collisions() -> None:
    dataframe = pd.DataFrame([[1, 2]], columns=["Page Views", "page-views"])

    with pytest.raises(ValueError, match="unique after normalization"):
        normalize_column_names(dataframe)


@pytest.mark.parametrize("column_name", ["", "  ", "!!!"])
def test_normalize_column_names_rejects_empty_labels(column_name: str) -> None:
    dataframe = pd.DataFrame([[1]], columns=[column_name])

    with pytest.raises(ValueError, match="must not be empty"):
        normalize_column_names(dataframe)


def test_remove_duplicate_rows_keeps_first_occurrence_and_index() -> None:
    dataframe = pd.DataFrame(
        {"page_views": [3, 3, 1], "purchase": [1, 1, 0]}, index=[5, 7, 9]
    )
    original = dataframe.copy(deep=True)

    cleaned = remove_duplicate_rows(dataframe)

    assert_frame_equal(cleaned, original.loc[[5, 9]])
    cleaned.iloc[0, 0] = 99
    assert_frame_equal(dataframe, original)


def test_remove_empty_rows_preserves_partial_rows_and_index() -> None:
    dataframe = pd.DataFrame(
        {"page_views": [3.0, None, None], "purchase": [1.0, None, 0.0]}
    )
    original = dataframe.copy(deep=True)

    cleaned = remove_empty_rows(dataframe)

    assert_frame_equal(cleaned, original.loc[[0, 2]])
    cleaned.iloc[0, 0] = 99
    assert_frame_equal(dataframe, original)


def test_remove_empty_rows_preserves_empty_strings_and_whitespace() -> None:
    dataframe = pd.DataFrame({"page_views": ["", " "], "purchase": [None, None]})

    assert_frame_equal(remove_empty_rows(dataframe), dataframe)
