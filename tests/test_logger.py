import logging
from pathlib import Path

from entropy_news.utils.logger import setup_logger


def test_setup_logger_no_duplicate_handlers(tmp_path: Path):
    log_file = tmp_path / "logs" / "test.log"
    logger = setup_logger("test_logger", str(log_file))

    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert log_file.exists()

    initial_handler_count = len(logger.handlers)
    logger2 = setup_logger("test_logger", str(log_file))
    assert logger2 is logger
    assert len(logger.handlers) == initial_handler_count
