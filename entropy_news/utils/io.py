# entropy_news/utils/io.py

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
    """Return non-empty stripped lines from a text file.

    Args:
        file_path: Path to the source text file.

    Returns:
        List of lines without surrounding whitespace.
    """

    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
