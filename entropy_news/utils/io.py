# entropy_news/utils/io.py

from typing import List


def load_texts(file_path: str) -> List[str]:
    """Return non-empty stripped lines from ``file_path``."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]
