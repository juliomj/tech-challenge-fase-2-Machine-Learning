"""Download a ZIP archive and extract its CSV."""

import argparse
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile


def download_dataset(url: str, output_path: Path) -> None:
    """Download a ZIP archive and extract its CSV to output_path."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with urlopen(url) as response:
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


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    download_dataset(
        url=args.url,
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
