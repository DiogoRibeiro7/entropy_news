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


def test_setup_logger_respects_level(tmp_path: Path) -> None:
    """Ensure the returned logger uses the requested log level."""
    log_file = tmp_path / "logs" / "level.log"
    logger = setup_logger("level_logger", str(log_file), level=logging.DEBUG)

    assert logger.level == logging.DEBUG


def test_setup_logger_without_file() -> None:
    """No file handler should be added when ``log_file`` is ``None``."""

    logger = setup_logger("no_file_logger", None)

    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)
