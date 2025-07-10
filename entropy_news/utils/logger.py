# entropy_news/utils/logger.py

import logging
import os


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a logger that writes to ``log_file``.

    This helper ensures the log directory exists and avoids adding duplicate
    handlers when invoked multiple times.

    Args:
        name: Identifier for the logger.
        log_file: Path to the desired log file.
        level: Logging level applied to the logger.

    Returns:
        The configured logger instance.
    """

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    abs_log_file = os.path.abspath(log_file)

    # add file handler only if the same file isn't already handled
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == abs_log_file
        for h in logger.handlers
    ):
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # add console handler if none exists
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(logging.StreamHandler())

    return logger
