"""Verify CSV loading and meaningful errors for unusable input files."""

from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from purchase_propensity.data.loader import load_dataset


def test_load_dataset_reads_valid_csv(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("page_views,purchase\n3,1\n1,0\n", encoding="utf-8")
    expected = pd.DataFrame({"page_views": [3, 1], "purchase": [1, 0]})

    assert_frame_equal(load_dataset(dataset_path), expected)


def test_load_dataset_rejects_missing_file(tmp_path: Path) -> None:
    dataset_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset file not found") as error:
        load_dataset(dataset_path)

    assert str(dataset_path) in str(error.value)


@pytest.mark.parametrize("contents", ["", " \n\n", "page_views,purchase\n"])
def test_load_dataset_rejects_empty_csv(tmp_path: Path, contents: str) -> None:
    dataset_path = tmp_path / "empty.csv"
    dataset_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset is empty"):
        load_dataset(dataset_path)
