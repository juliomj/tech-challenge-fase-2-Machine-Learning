"""Tests for dataset preprocessing."""

import pandas as pd
import pytest

from purchase_propensity.data.preprocessing import (
    normalize_boolean_columns,
    normalize_categorical_values,
    normalize_column_names,
    preprocess,
    remove_duplicate_rows,
    remove_empty_rows,
    validate_columns,
)


def test_normalize_column_names_returns_copy() -> None:
    dataframe = pd.DataFrame({"Column Name": [1]})

    result = normalize_column_names(dataframe)

    assert result is not dataframe
    assert list(result.columns) == ["column_name"]
    assert list(dataframe.columns) == ["Column Name"]


@pytest.mark.parametrize(
    ("columns", "expected_message"),
    [
        (["!!!"], "empty"),
        (["Column Name", "column_name"], "unique"),
    ],
)
def test_normalize_column_names_rejects_invalid_names(
    columns: list[str],
    expected_message: str,
) -> None:
    dataframe = pd.DataFrame([[1] * len(columns)], columns=columns)

    with pytest.raises(ValueError, match=expected_message):
        normalize_column_names(dataframe)


def test_remove_empty_rows_preserves_index() -> None:
    dataframe = pd.DataFrame(
        {"value": [1, None, 3], "other": [None, None, 4]},
        index=[10, 20, 30],
    )

    result = remove_empty_rows(dataframe)

    assert list(result.index) == [10, 30]
    assert len(result) == 2  # noqa: PLR2004


def test_remove_duplicate_rows_preserves_first_occurrence() -> None:
    dataframe = pd.DataFrame(
        {"value": [1, 1, 2]},
        index=[10, 20, 30],
    )

    result = remove_duplicate_rows(dataframe)

    assert result["value"].tolist() == [1, 2]
    assert list(result.index) == [10, 30]


def test_normalize_categorical_values_removes_surrounding_whitespace(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.copy()
    dataframe["Month"] = [" May ", "Jun", " Jul "]
    dataframe["VisitorType"] = [
        " Returning_Visitor ",
        "New_Visitor",
        " Other ",
    ]

    normalized = normalize_column_names(dataframe)
    result = normalize_categorical_values(normalized)

    assert result["month"].tolist() == ["May", "Jun", "Jul"]
    assert result["visitortype"].tolist() == [
        "Returning_Visitor",
        "New_Visitor",
        "Other",
    ]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (["true", "false"], [True, False]),
        (["TRUE", "FALSE"], [True, False]),
        (["1", "0"], [True, False]),
        (["yes", "no"], [True, False]),
    ],
)
def test_normalize_boolean_columns(
    values: list[str],
    expected: list[bool],
) -> None:
    dataframe = pd.DataFrame(
        {
            "weekend": values,
            "revenue": values,
        },
    )

    result = normalize_boolean_columns(dataframe)

    assert result["weekend"].tolist() == expected
    assert result["revenue"].tolist() == expected


def test_normalize_boolean_columns_preserves_missing_values() -> None:
    dataframe = pd.DataFrame(
        {
            "weekend": ["true", None],
            "revenue": ["false", None],
        },
    )

    result = normalize_boolean_columns(dataframe)

    assert result["weekend"].tolist() == [True, pd.NA]
    assert result["revenue"].tolist() == [False, pd.NA]


def test_normalize_boolean_columns_rejects_invalid_values() -> None:
    dataframe = pd.DataFrame(
        {
            "weekend": ["true", "maybe"],
            "revenue": [False, True],
        },
    )

    with pytest.raises(ValueError, match="Invalid boolean"):
        normalize_boolean_columns(dataframe)


def test_validate_columns_accepts_expected_schema(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = normalize_column_names(valid_dataframe)

    validate_columns(dataframe)


@pytest.mark.parametrize(
    "column",
    [
        "Month",
        "Revenue",
        "VisitorType",
    ],
)
def test_validate_columns_rejects_missing_columns(
    valid_dataframe: pd.DataFrame,
    column: str,
) -> None:
    dataframe = normalize_column_names(valid_dataframe)
    dataframe = dataframe.drop(columns=[column.lower()])

    with pytest.raises(ValueError, match="Missing expected columns"):
        validate_columns(dataframe)


def test_validate_columns_rejects_unexpected_columns(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = normalize_column_names(valid_dataframe)
    dataframe["unexpected"] = 1

    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_columns(dataframe)


def test_preprocess_returns_clean_copy(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.copy()
    dataframe.loc[0, "Month"] = " May "
    dataframe = pd.concat([dataframe, dataframe.iloc[[0]]])

    original = dataframe.copy(deep=True)

    result = preprocess(dataframe)

    pd.testing.assert_frame_equal(dataframe, original)
    assert result is not dataframe
    assert len(result) == 3  # noqa: PLR2004
    assert result.loc[0, "month"] == "May"


def test_preprocess_removes_empty_rows(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = pd.concat(
        [
            valid_dataframe,
            pd.DataFrame([dict.fromkeys(valid_dataframe)]),
        ],
        ignore_index=True,
    )

    result = remove_empty_rows(dataframe)

    assert len(result) == len(valid_dataframe)
