"""Provide named loggers with a simple console configuration."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Configure default console logging when no root handlers exist.

    Args:
        name: Logger name, typically the calling module's ``__name__``.

    Returns:
        A named logger using the existing application configuration, if present.
        Repeated calls do not add duplicate handlers.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
