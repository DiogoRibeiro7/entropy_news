# entropy_news/utils/io.py

"""Basic text file utilities."""

import os
from typing import List


def save_texts(texts: List[str], file_path: str) -> None:
    """Write a list of strings to ``file_path``.

    Empty or whitespace-only entries are skipped.

    Args:
        texts: Sequence of strings to persist.
        file_path: Destination text file.

    Raises:
        OSError: If writing to ``file_path`` fails.
    """

    with open(file_path, "w", encoding="utf-8") as f:
        for line in texts:
            stripped = line.strip()
            if stripped:
                f.write(f"{stripped}\n")


def load_texts(file_path: str) -> List[str]:
    """Return non-empty stripped lines from ``file_path``.

    Args:
        file_path: Path to the source text file.

    Returns:
        List of lines without surrounding whitespace.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        ValueError: If the file contains no valid lines.
        OSError: For other I/O related errors.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Text file not found: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except OSError as exc:  # pragma: no cover - hard to trigger
        raise OSError(f"Failed to read {file_path}: {exc}") from exc

    if not lines:
        raise ValueError(f"Text file is empty: {file_path}")

    return lines
