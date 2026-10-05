"""Read tabular data without applying transformations."""

from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Read a comma-separated CSV with at least one data row.

    Args:
        file_path: Path to a CSV file with a header row.

    Returns:
        The dataset as read by pandas, without cleaning or inference rules.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no columns or no data rows.
        pandas.errors.ParserError: If the CSV is malformed.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    try:
        dataframe = pd.read_csv(file_path)
    except EmptyDataError as error:
        raise ValueError(f"Dataset is empty: {file_path}") from error

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {file_path}")
    return dataframe
