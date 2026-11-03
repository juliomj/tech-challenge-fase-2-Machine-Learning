"""Download a ZIP archive and extract its CSV."""

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


def download_dataset(url: str, output_path: Path) -> None:
    """Download a ZIP archive and extract its CSV to output_path.

    Args:
        url: Address of the ZIP archive.
        output_path: Destination of the extracted CSV.

    Raises:
        RuntimeError: If the archive does not contain exactly one CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response:  # noqa: S310
        archive = BytesIO(response.read())

    with ZipFile(archive) as zip_file:
        csv_files = [
            name for name in zip_file.namelist() if name.lower().endswith(".csv")
        ]

        if len(csv_files) != 1:
            msg = f"Expected exactly one CSV file, found: {csv_files}"
            raise RuntimeError(msg)

        with (
            zip_file.open(csv_files[0]) as source,
            output_path.open("wb") as target,
        ):
            target.write(source.read())


def main() -> None:
    """Run the dataset download."""
    parser = argparse.ArgumentParser(description="Download the raw dataset.")
    parser.add_argument("--url", required=True, help="URL of the ZIP archive.")
    parser.add_argument("--output", required=True, help="Path of the raw CSV.")
    args = parser.parse_args()

    download_dataset(
        url=args.url,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
