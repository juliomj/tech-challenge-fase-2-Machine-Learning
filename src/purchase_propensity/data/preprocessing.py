"""Apply basic cleaning without changing the input DataFrame."""

import re

import pandas as pd


def normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Trim and lowercase labels, replacing punctuation and spaces with underscores.

    Args:
        dataframe: Dataset whose column labels will be converted to strings.

    Returns:
        A copy with normalized column names and unchanged data and index.

    Raises:
        ValueError: If normalization produces empty or duplicate column names.
    """
    normalized = dataframe.rename(
        columns=lambda column: re.sub(r"\W+", "_", str(column).strip().lower()).strip(
            "_"
        )
    )
    if "" in normalized.columns:
        raise ValueError("Column names must not be empty after normalization.")
    if not normalized.columns.is_unique:
        raise ValueError("Column names must be unique after normalization.")
    return normalized.copy()


def remove_duplicate_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Keep the first occurrence of each row, comparing all columns.

    Args:
        dataframe: Dataset that may contain repeated rows.

    Returns:
        A copy without duplicate rows, preserving the original index.
    """
    return dataframe.drop_duplicates().copy()


def remove_empty_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove rows only when every value is missing according to pandas.

    Args:
        dataframe: Dataset that may contain completely missing rows.

    Returns:
        A copy preserving partially filled rows and the original index.
        Empty strings and whitespace are not treated as missing by this function.
    """
    return dataframe.dropna(how="all").copy()
