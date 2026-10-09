"""Tests for feature preparation."""

from pathlib import Path

import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from purchase_propensity.features.build_features import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    ENGINEERED_NUMERICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    TARGET_COLUMN,
    add_engineered_features,
    build_features,
    build_preprocessing_pipeline,
    split_features_and_target,
    split_train_test,
    transform_datasets,
    validate_columns,
)


def test_validate_columns_accepts_valid_dataframe(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)

    validate_columns(dataframe)


@pytest.mark.parametrize(
    "column",
    [
        "administrative",
        "month",
        "weekend",
        "revenue",
    ],
)
def test_validate_columns_rejects_missing_columns(
    valid_dataframe: pd.DataFrame,
    column: str,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)
    dataframe = dataframe.drop(columns=[column])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(dataframe)


def test_add_engineered_features_preserves_original_columns(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)

    engineered = add_engineered_features(dataframe)

    for column in dataframe.columns:
        assert column in engineered.columns


def test_add_engineered_features_creates_expected_columns(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)

    engineered = add_engineered_features(dataframe)

    for column in ENGINEERED_NUMERICAL_COLUMNS:
        assert column in engineered.columns


def test_add_engineered_features_calculates_total_pages(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)

    engineered = add_engineered_features(dataframe)

    expected = (
        dataframe["administrative"]
        + dataframe["informational"]
        + dataframe["productrelated"]
    )

    pd.testing.assert_series_equal(
        engineered["total_pages"],
        expected,
        check_names=False,
    )


def test_add_engineered_features_calculates_total_duration(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)

    engineered = add_engineered_features(dataframe)

    expected = (
        dataframe["administrative_duration"]
        + dataframe["informational_duration"]
        + dataframe["productrelated_duration"]
    )

    pd.testing.assert_series_equal(
        engineered["total_duration"],
        expected,
        check_names=False,
    )


def test_add_engineered_features_handles_zero_page_counts(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)
    dataframe.loc[:, "administrative"] = 0
    dataframe.loc[:, "informational"] = 0
    dataframe.loc[:, "productrelated"] = 0

    engineered = add_engineered_features(dataframe)

    assert (engineered["administrative_avg_duration"] == 0).all()
    assert (engineered["informational_avg_duration"] == 0).all()
    assert (engineered["productrelated_avg_duration"] == 0).all()


def test_add_engineered_features_creates_activity_flags(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)

    engineered = add_engineered_features(dataframe)

    expected_administrative = (dataframe["administrative"] > 0).astype(int)

    expected_informational = (dataframe["informational"] > 0).astype(int)

    pd.testing.assert_series_equal(
        engineered["has_administrative_activity"],
        expected_administrative,
        check_names=False,
    )

    pd.testing.assert_series_equal(
        engineered["has_informational_activity"],
        expected_informational,
        check_names=False,
    )


def test_split_features_and_target_returns_copies(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)
    original = dataframe.copy(deep=True)

    features, target = split_features_and_target(dataframe)

    assert TARGET_COLUMN not in features.columns
    assert target.name == TARGET_COLUMN

    pd.testing.assert_frame_equal(dataframe, original)

    assert features is not dataframe
    assert target is not dataframe


def test_split_features_and_target_preserves_index(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)
    dataframe.index = [10, 20, 30]

    features, target = split_features_and_target(dataframe)

    assert list(features.index) == [10, 20, 30]
    assert list(target.index) == [10, 20, 30]


def test_split_features_and_target_rejects_missing_target(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)
    dataframe = dataframe.drop(columns=[TARGET_COLUMN])

    with pytest.raises(ValueError, match="Target column"):
        split_features_and_target(dataframe)


def test_build_preprocessing_pipeline_returns_column_transformer() -> None:
    pipeline = build_preprocessing_pipeline()

    assert isinstance(pipeline, ColumnTransformer)


def test_build_preprocessing_pipeline_is_unfitted() -> None:
    pipeline = build_preprocessing_pipeline()

    assert not hasattr(pipeline, "transformers_")
    assert not hasattr(pipeline, "feature_names_in_")


def test_feature_groups_are_disjoint() -> None:
    numerical_columns = set(NUMERICAL_COLUMNS) | set(ENGINEERED_NUMERICAL_COLUMNS)

    feature_groups = [
        numerical_columns,
        set(CATEGORICAL_COLUMNS),
        set(BINARY_COLUMNS),
    ]

    for index, current_group in enumerate(feature_groups):
        for other_group in feature_groups[index + 1 :]:
            assert current_group.isdisjoint(other_group)


def test_target_is_not_a_feature() -> None:
    all_features = (
        set(NUMERICAL_COLUMNS)
        | set(CATEGORICAL_COLUMNS)
        | set(BINARY_COLUMNS)
        | set(ENGINEERED_NUMERICAL_COLUMNS)
    )

    assert TARGET_COLUMN not in all_features


def test_split_train_test_preserves_all_rows(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)
    dataframe = pd.concat([dataframe] * 10, ignore_index=True)

    features, target = split_features_and_target(dataframe)

    (
        train_features,
        test_features,
        train_target,
        test_target,
    ) = split_train_test(features, target)

    assert len(train_features) + len(test_features) == len(features)
    assert len(train_target) + len(test_target) == len(target)


def test_split_train_test_preserves_feature_columns(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)
    dataframe = pd.concat([dataframe] * 10, ignore_index=True)

    features, target = split_features_and_target(dataframe)

    (
        train_features,
        test_features,
        _,
        _,
    ) = split_train_test(features, target)

    assert list(train_features.columns) == list(features.columns)
    assert list(test_features.columns) == list(features.columns)


def test_transform_datasets_applies_one_hot_encoding(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)
    dataframe = pd.concat([dataframe] * 10, ignore_index=True)

    dataframe = add_engineered_features(dataframe)

    features, target = split_features_and_target(dataframe)

    (
        train_features,
        test_features,
        train_target,
        test_target,
    ) = split_train_test(features, target)

    pipeline = build_preprocessing_pipeline()

    train_prepared, test_prepared = transform_datasets(
        pipeline=pipeline,
        train_features=train_features,
        test_features=test_features,
        train_target=train_target,
        test_target=test_target,
    )

    assert len(train_prepared) == len(train_features)
    assert len(test_prepared) == len(test_features)

    assert TARGET_COLUMN in train_prepared.columns
    assert TARGET_COLUMN in test_prepared.columns

    assert len(train_prepared.columns) > len(features.columns)
    assert len(test_prepared.columns) == len(train_prepared.columns)


def test_transform_datasets_fits_pipeline_only_on_training_data(
    valid_dataframe: pd.DataFrame,
) -> None:
    dataframe = valid_dataframe.rename(columns=str.lower)
    dataframe = pd.concat([dataframe] * 10, ignore_index=True)

    dataframe = add_engineered_features(dataframe)

    features, target = split_features_and_target(dataframe)

    (
        train_features,
        test_features,
        train_target,
        test_target,
    ) = split_train_test(features, target)

    pipeline = build_preprocessing_pipeline()

    transform_datasets(
        pipeline=pipeline,
        train_features=train_features,
        test_features=test_features,
        train_target=train_target,
        test_target=test_target,
    )

    assert hasattr(pipeline, "transformers_")
    assert hasattr(pipeline, "feature_names_in_")


def test_build_features_creates_train_and_test_files(
    tmp_path: Path,
    valid_dataframe: pd.DataFrame,
) -> None:
    input_path = tmp_path / "processed.csv"
    output_directory = tmp_path / "prepared"

    dataframe = valid_dataframe.rename(columns=str.lower)
    dataframe = pd.concat([dataframe] * 10, ignore_index=True)
    dataframe.to_csv(input_path, index=False)

    build_features(
        input_path=input_path,
        output_directory=output_directory,
    )

    train_path = output_directory / "train.csv"
    test_path = output_directory / "test.csv"

    assert train_path.is_file()
    assert test_path.is_file()


def test_build_features_outputs_valid_prepared_datasets(
    tmp_path: Path,
    valid_dataframe: pd.DataFrame,
) -> None:
    input_path = tmp_path / "processed.csv"
    output_directory = tmp_path / "prepared"

    dataframe = valid_dataframe.rename(columns=str.lower)
    dataframe = pd.concat([dataframe] * 10, ignore_index=True)
    dataframe.to_csv(input_path, index=False)

    build_features(
        input_path=input_path,
        output_directory=output_directory,
    )

    train = pd.read_csv(output_directory / "train.csv")
    test = pd.read_csv(output_directory / "test.csv")

    assert not train.empty
    assert not test.empty

    assert TARGET_COLUMN in train.columns
    assert TARGET_COLUMN in test.columns

    assert list(train.columns) == list(test.columns)


def test_build_features_preserves_total_row_count(
    tmp_path: Path,
    valid_dataframe: pd.DataFrame,
) -> None:
    input_path = tmp_path / "processed.csv"
    output_directory = tmp_path / "prepared"

    dataframe = valid_dataframe.rename(columns=str.lower)
    dataframe = pd.concat([dataframe] * 10, ignore_index=True)
    dataframe.to_csv(input_path, index=False)

    build_features(
        input_path=input_path,
        output_directory=output_directory,
    )

    train = pd.read_csv(output_directory / "train.csv")
    test = pd.read_csv(output_directory / "test.csv")

    assert len(train) + len(test) == len(dataframe)


def test_build_features_rejects_missing_input(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "missing.csv"
    output_directory = tmp_path / "prepared"

    with pytest.raises(FileNotFoundError, match="not found"):
        build_features(
            input_path=input_path,
            output_directory=output_directory,
        )


def test_build_features_rejects_empty_input(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "empty.csv"
    output_directory = tmp_path / "prepared"

    pd.DataFrame().to_csv(input_path, index=False)

    with pytest.raises(ValueError, match="empty"):
        build_features(
            input_path=input_path,
            output_directory=output_directory,
        )
