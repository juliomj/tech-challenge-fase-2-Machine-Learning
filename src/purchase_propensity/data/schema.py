"""Column groups of the dataset and the feature/target split shared by stages."""

import pandas as pd

from purchase_propensity.utils.config import CONFIG

TARGET_COLUMN = CONFIG["model"]["target_column"]

NUMERICAL_COLUMNS = (
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
)

CATEGORICAL_COLUMNS = (
    "month",
    "operatingsystems",
    "browser",
    "region",
    "traffictype",
    "visitortype",
)

BINARY_COLUMNS = ("weekend",)

BOOLEAN_COLUMNS = (*BINARY_COLUMNS, TARGET_COLUMN)

EXPECTED_COLUMNS = frozenset(
    {*NUMERICAL_COLUMNS, *CATEGORICAL_COLUMNS, *BOOLEAN_COLUMNS},
)


def split_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target.

    Args:
        dataframe: Dataset containing the target column.

    Returns:
        Copies of the feature DataFrame and of the target Series.

    Raises:
        ValueError: If the target column is missing.
    """
    if TARGET_COLUMN not in dataframe.columns:
        msg = f"Target column not found: {TARGET_COLUMN}"
        raise ValueError(msg)

    features = dataframe.drop(columns=[TARGET_COLUMN])
    return features, dataframe[TARGET_COLUMN].copy()
