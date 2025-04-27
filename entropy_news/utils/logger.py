# entropy_news/utils/logger.py

import logging
import os

def setup_logger(name: str, log_file: str, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    # Check if a StreamHandler is already attached to avoid duplicates
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(logging.StreamHandler())  # Console output

    return logger
