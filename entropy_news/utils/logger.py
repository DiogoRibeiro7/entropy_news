# entropy_news/utils/logger.py

import logging
import os

def setup_logger(name: str, log_file: str, level=logging.INFO):
    """Return a configured ``logging.Logger``.

    The function creates the directory for ``log_file`` if needed and adds a
    ``FileHandler`` and ``StreamHandler`` to ``name``. If either handler already
    exists (e.g. when ``setup_logger`` is called multiple times), it will not be
    added again. This prevents duplicated messages when scripts run the logger
    setup more than once.
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
