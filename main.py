"""Demonstrate the initial data preparation modules with a user-supplied CSV."""

import argparse
from pathlib import Path

import yaml

from purchase_propensity.data.loader import load_dataset
from purchase_propensity.data.preprocessing import (
    normalize_column_names,
    remove_duplicate_rows,
    remove_empty_rows,
)
from purchase_propensity.features.build_features import split_features_and_target
from purchase_propensity.utils.logger import get_logger

PROJECT_ROOT = Path(__file__).resolve().parent


def _parse_dataset_path(default_path: Path) -> Path:
    """Read the optional CSV path from the command line.

    Args:
        default_path: Dataset path to use when --dataset is omitted.

    Returns:
        The user-supplied path or the configured default path.
    """
    parser = argparse.ArgumentParser(
        description="Demonstração da Etapa 1: preparo básico de dados."
    )
    parser.add_argument(
        "--dataset", type=Path, default=default_path, help="Caminho do CSV de entrada."
    )
    return parser.parse_args().dataset


def main() -> None:
    """Load, clean and split the CSV defined in config or passed with --dataset.

    Paths in the configuration are relative to the project root. A command-line
    dataset path is relative to the current working directory, unless absolute.
    The demonstration logs dimensions and leaves all files untouched.

    Raises:
        FileNotFoundError: If the configuration or dataset is missing.
        ValueError: If the dataset is empty or its column labels are invalid.
    """
    logger = get_logger(__name__)
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    default_path = (
        PROJECT_ROOT / config["data"]["raw_path"] / config["data"]["dataset_file"]
    )
    dataframe = load_dataset(_parse_dataset_path(default_path))
    cleaned = normalize_column_names(dataframe)
    cleaned = remove_empty_rows(cleaned)
    cleaned = remove_duplicate_rows(cleaned)
    features, target = split_features_and_target(
        cleaned, config["model"]["target_column"]
    )
    logger.info(
        "Etapa 1 concluída: %d linhas de entrada, %d linhas limpas, "
        "%d features, target '%s'.",
        len(dataframe),
        len(features),
        features.shape[1],
        target.name,
    )


if __name__ == "__main__":
    main()
