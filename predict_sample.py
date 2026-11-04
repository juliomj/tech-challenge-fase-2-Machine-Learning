"""Send sessions from the test set to the served model and compare the answers."""

import json
from urllib.request import Request, urlopen

import pandas as pd

from purchase_propensity.data.loader import load_dataset
from purchase_propensity.data.schema import split_features_and_target
from purchase_propensity.utils.config import CONFIG, PROJECT_ROOT
from purchase_propensity.utils.logger import get_logger

SERVING_URL = "http://localhost:8000/invocations"
FIRST_SESSION = 29
SESSION_COUNT = 5


def load_sample() -> tuple[pd.DataFrame, pd.Series]:
    """Load a few sessions from the prepared test set.

    Returns:
        The features and the real outcome of the selected sessions.
    """
    data = CONFIG["data"]
    test_data = load_dataset(
        PROJECT_ROOT / data["prepared_path"] / data["test_dataset_file"],
    )
    sample = test_data.iloc[FIRST_SESSION : FIRST_SESSION + SESSION_COUNT]
    return split_features_and_target(sample)


def request_predictions(features: pd.DataFrame) -> list[bool]:
    """Ask the served model whether each session ends in a purchase.

    Args:
        features: Sessions to classify.

    Returns:
        One prediction per session.
    """
    payload = {
        "dataframe_split": features.astype("float64").to_dict(
            orient="split",
            index=False,
        ),
    }
    request = Request(
        SERVING_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urlopen(request) as response:  # noqa: S310
        return json.load(response)["predictions"]


def main() -> None:
    """Log the predicted and the real outcome of each sample session."""
    logger = get_logger(__name__)
    features, target = load_sample()
    predictions = request_predictions(features)

    for predicted, real in zip(predictions, target, strict=True):
        logger.info("Previsto: %-5s | Real: %s", predicted, real)


if __name__ == "__main__":
    main()
