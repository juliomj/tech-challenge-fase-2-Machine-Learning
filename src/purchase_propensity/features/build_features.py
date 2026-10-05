"""Prepare features and target without engineering new features."""

import pandas as pd


def split_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate the target while preserving row alignment and data types.

    Args:
        dataframe: Dataset with unique column names.
        target_column: Exact target label, after any column normalization.

    Returns:
        Independent copies of the feature DataFrame and target Series.

    Raises:
        ValueError: If the target is absent or column names are duplicated.
    """
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column not found: {target_column}")
    if not dataframe.columns.is_unique:
        raise ValueError("Column names must be unique before splitting the target.")
    features = dataframe.drop(columns=[target_column]).copy()
    target = dataframe[target_column].copy()
    return features, target
