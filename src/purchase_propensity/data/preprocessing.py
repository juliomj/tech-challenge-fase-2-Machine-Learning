"""Preprocess and validate the raw dataset."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from src.purchase_propensity.data.loader import load_dataset

EXPECTED_COLUMNS = {
    "administrative",
    "administrative_duration",
    "informational",
    "informational_duration",
    "productrelated",
    "productrelated_duration",
    "bouncerates",
    "exitrates",
    "pagevalues",
    "specialday",
    "month",
    "operatingsystems",
    "browser",
    "region",
    "traffictype",
    "visitortype",
    "weekend",
    "revenue",
}

BOOLEAN_COLUMNS = (
    "weekend",
    "revenue",
)

CATEGORICAL_COLUMNS = (
    "month",
    "operatingsystems",
    "browser",
    "region",
    "traffictype",
    "visitortype",
)


def normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to lowercase snake_case.

    Args:
        dataframe: Input DataFrame.

    Returns:
        A copy of the DataFrame with normalized column names.

    Raises:
        ValueError: If a normalized column name is empty or duplicated.
    """
    normalized = dataframe.rename(
        columns=lambda column: re.sub(
            r"\W+",
            "_",
            str(column).strip().lower(),
        ).strip("_"),
    )

    if "" in normalized.columns:
        msg = "Column names must not be empty after normalization."
        raise ValueError(msg)

    if not normalized.columns.is_unique:
        msg_0 = "Column names must be unique after normalization."
        raise ValueError(msg_0)

    return normalized.copy()


def validate_columns(dataframe: pd.DataFrame) -> None:
    """Validate the expected dataset schema.

    Args:
        dataframe: DataFrame whose columns should be validated.

    Raises:
        ValueError: If expected columns are missing or unexpected columns
            are present.
    """
    actual_columns = set(dataframe.columns)

    missing_columns = EXPECTED_COLUMNS - actual_columns
    unexpected_columns = actual_columns - EXPECTED_COLUMNS

    if missing_columns:
        msg = f"Missing expected columns: {sorted(missing_columns)}"
        raise ValueError(msg)

    if unexpected_columns:
        msg_0 = f"Unexpected columns found: {sorted(unexpected_columns)}"
        raise ValueError(msg_0)


def remove_empty_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove rows containing no values."""
    return dataframe.dropna(how="all").copy()


def remove_duplicate_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    return dataframe.drop_duplicates().copy()


def normalize_categorical_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove surrounding whitespace from string categorical values."""
    normalized = dataframe.copy()

    for column in CATEGORICAL_COLUMNS:
        if column not in normalized.columns:
            continue

        if pd.api.types.is_object_dtype(normalized[column]):
            normalized[column] = normalized[column].str.strip()

    return normalized


def normalize_boolean_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize boolean columns to pandas BooleanDtype.

    Args:
        dataframe: Input DataFrame.

    Returns:
        A copy with normalized boolean columns.

    Raises:
        ValueError: If a non-null value cannot be interpreted as boolean.
    """
    normalized = dataframe.copy()

    boolean_mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }

    for column in BOOLEAN_COLUMNS:
        if column not in normalized.columns:
            continue

        if pd.api.types.is_bool_dtype(normalized[column]):
            continue

        converted = (
            normalized[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .map(boolean_mapping)
        )

        invalid_mask = normalized[column].notna() & converted.isna()

        if invalid_mask.any():
            invalid_values = (
                normalized.loc[invalid_mask, column].drop_duplicates().tolist()
            )
            msg = f"Invalid boolean values found in '{column}': {invalid_values}"
            raise ValueError(
                msg,
            )

        normalized[column] = converted.astype("Int64")

    return normalized


def preprocess(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Apply dataset-level preprocessing.

    Args:
        dataframe: Raw input DataFrame.

    Returns:
        A cleaned copy of the input DataFrame.

    Raises:
        ValueError: If the dataset schema is invalid or boolean values
            are inconsistent.
    """
    processed = normalize_column_names(dataframe)
    validate_columns(processed)
    processed = remove_empty_rows(processed)
    processed = remove_duplicate_rows(processed)
    processed = normalize_categorical_values(processed)
    return normalize_boolean_columns(processed)


def save_csv(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Save a DataFrame to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)


def preprocess_file(input_path: Path, output_path: Path) -> None:
    """Load, preprocess, and save a dataset.

    Args:
        input_path: Path to the raw CSV file.
        output_path: Path to the processed CSV file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If the input dataset is invalid.
    """
    dataframe = load_dataset(input_path)
    processed = preprocess(dataframe)
    save_csv(processed, output_path)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments and return it."""
    parser = argparse.ArgumentParser(description="Preprocess the raw dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the raw CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the processed CSV file.",
    )

    return parser.parse_args()


def main() -> None:
    """Run the preprocessing command."""
    args = parse_arguments()
    preprocess_file(args.input, args.output)


if __name__ == "__main__":
    main()
