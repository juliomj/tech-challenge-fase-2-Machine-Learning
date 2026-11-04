"""Prepare model features and train/test datasets."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

from purchase_propensity.data.loader import load_dataset
from purchase_propensity.data.schema import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
    split_features_and_target,
)
from purchase_propensity.utils.config import CONFIG

if TYPE_CHECKING:
    import numpy as np

# Disable considering that we opted-in and are using the suggested: .fillna(0)
pd.set_option("future.no_silent_downcasting", True)  # noqa: FBT003

RANDOM_SEED = CONFIG["project"]["random_seed"]
TEST_SIZE = CONFIG["model"]["test_size"]

PAGE_GROUPS = ("administrative", "informational", "productrelated")

ENGINEERED_NUMERICAL_COLUMNS = (
    "total_pages",
    "total_duration",
    "administrative_avg_duration",
    "informational_avg_duration",
    "productrelated_avg_duration",
    "has_administrative_activity",
    "has_informational_activity",
)


def validate_columns(dataframe: pd.DataFrame) -> None:
    """Validate the columns required for feature preparation.

    Args:
        dataframe: Input DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {
        *NUMERICAL_COLUMNS,
        *CATEGORICAL_COLUMNS,
        *BINARY_COLUMNS,
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        msg = f"Missing required columns: {sorted(missing_columns)}"
        raise ValueError(msg)


def average_duration(dataframe: pd.DataFrame, page_group: str) -> pd.Series:
    """Compute the mean time per page of a page group.

    Args:
        dataframe: DataFrame with ``<group>`` and ``<group>_duration`` columns.
        page_group: Page group name, for example ``"administrative"``.

    Returns:
        Duration divided by page count, or 0 when no page was visited.
    """
    pages = dataframe[page_group].replace(0, pd.NA)
    return dataframe[f"{page_group}_duration"].div(pages).fillna(0)


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add domain-driven session-level features.

    The original variables are preserved. Engineered variables summarize
    browsing volume and duration.

    Args:
        dataframe: Input DataFrame.

    Returns:
        A copy containing the original and engineered features.
    """
    features = dataframe.copy()
    duration_columns = [f"{group}_duration" for group in PAGE_GROUPS]

    # skipna=False: um valor ausente gera total ausente, imputado depois.
    features["total_pages"] = features[list(PAGE_GROUPS)].sum(axis=1, skipna=False)
    features["total_duration"] = features[duration_columns].sum(axis=1, skipna=False)

    for group in PAGE_GROUPS:
        features[f"{group}_avg_duration"] = average_duration(features, group)

    for group in ("administrative", "informational"):
        features[f"has_{group}_activity"] = (features[group] > 0).astype(int)

    return features


def to_float(values: pd.DataFrame) -> pd.DataFrame:
    """Cast values to float so nullable integer columns can be imputed."""
    return values.astype("float64")


def build_numerical_pipeline() -> Pipeline:
    """Build the median imputation applied to numerical variables."""
    return Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))])


def build_categorical_pipeline() -> Pipeline:
    """Build the most-frequent imputation and one-hot encoding pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ],
    )


def build_binary_pipeline() -> Pipeline:
    """Build the float casting and most-frequent imputation pipeline."""
    return Pipeline(
        steps=[
            (
                "to_numeric",
                FunctionTransformer(to_float, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ],
    )


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build the feature transformation pipeline.

    Numerical variables receive median imputation.
    Categorical variables receive most-frequent imputation and one-hot
    encoding. Binary variables receive most-frequent imputation.

    Returns:
        An unfitted ColumnTransformer.
    """
    numerical_columns = [*NUMERICAL_COLUMNS, *ENGINEERED_NUMERICAL_COLUMNS]

    return ColumnTransformer(
        transformers=[
            ("numerical", build_numerical_pipeline(), numerical_columns),
            ("categorical", build_categorical_pipeline(), list(CATEGORICAL_COLUMNS)),
            ("binary", build_binary_pipeline(), list(BINARY_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_train_test(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into stratified train and test sets.

    Args:
        features: Feature DataFrame.
        target: Target Series.

    Returns:
        Training features, test features, training target, and test target.
    """
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=target,
    )
    return train_features, test_features, train_target, test_target


def normalize_feature_names(feature_names: list[str]) -> list[str]:
    """Normalize generated feature names to lowercase snake_case."""
    return [
        re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower())
        for name in feature_names
    ]


def transform_datasets(
    pipeline: ColumnTransformer,
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    train_target: pd.Series,
    test_target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit transformations on training data and transform both datasets.

    Args:
        pipeline: Unfitted sklearn preprocessing pipeline.
        train_features: Training features.
        test_features: Test features.
        train_target: Training target.
        test_target: Test target.

    Returns:
        Prepared training and test DataFrames.
    """
    # O OneHotEncoder usa sparse_output=False, entao a saida e sempre densa.
    train_transformed = cast("np.ndarray", pipeline.fit_transform(train_features))
    test_transformed = cast("np.ndarray", pipeline.transform(test_features))

    feature_names = normalize_feature_names(list(pipeline.get_feature_names_out()))

    train_prepared = pd.DataFrame(
        train_transformed,
        columns=feature_names,
        index=train_features.index,
    )
    test_prepared = pd.DataFrame(
        test_transformed,
        columns=feature_names,
        index=test_features.index,
    )

    train_prepared[TARGET_COLUMN] = train_target
    test_prepared[TARGET_COLUMN] = test_target

    return train_prepared, test_prepared


def save_prepared_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    output_directory: Path,
) -> None:
    """Save prepared training and test datasets.

    Args:
        train_data: Prepared training dataset.
        test_data: Prepared test dataset.
        output_directory: Directory for prepared datasets.
    """
    output_directory.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(
        output_directory / CONFIG["data"]["train_dataset_file"],
        index=False,
    )

    test_data.to_csv(
        output_directory / CONFIG["data"]["test_dataset_file"],
        index=False,
    )


def build_features(input_path: Path, output_directory: Path) -> None:
    """Prepare features and persist train/test datasets.

    Args:
        input_path: Path to the processed CSV.
        output_directory: Directory for prepared datasets.
    """
    dataframe = load_dataset(input_path)
    validate_columns(dataframe)

    features, target = split_features_and_target(add_engineered_features(dataframe))
    train_features, test_features, train_target, test_target = split_train_test(
        features,
        target,
    )

    train_prepared, test_prepared = transform_datasets(
        pipeline=build_preprocessing_pipeline(),
        train_features=train_features,
        test_features=test_features,
        train_target=train_target,
        test_target=test_target,
    )
    save_prepared_data(train_prepared, test_prepared, output_directory)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Prepare features and train/test datasets.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the processed CSV file.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Directory for prepared datasets.",
    )

    return parser.parse_args()


def main() -> None:
    """Run feature preparation."""
    args = parse_arguments()

    build_features(
        input_path=args.input,
        output_directory=args.output,
    )


if __name__ == "__main__":
    main()
