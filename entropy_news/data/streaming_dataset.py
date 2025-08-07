"""Streaming dataset for memory-efficient text loading."""

from __future__ import annotations

from collections import OrderedDict
from typing import List
import gzip

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .preprocessor import TextPreprocessor


class LRUCache:
    """Simple least-recently-used (LRU) cache."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._store: OrderedDict[int, List[torch.Tensor]] = OrderedDict()

    def __contains__(self, key: int) -> bool:
        return key in self._store

    def __getitem__(self, key: int) -> List[torch.Tensor]:
        value = self._store.pop(key)
        self._store[key] = value
        return value

    def __setitem__(self, key: int, value: List[torch.Tensor]) -> None:
        if key in self._store:
            self._store.pop(key)
        elif len(self._store) >= self.capacity:
            self._store.popitem(last=False)
        self._store[key] = value


class StreamingNewsDataset(Dataset):
    """Lazily load and tokenize news texts from a file.

    The dataset reads the source file in fixed-size chunks and caches a limited
    number of them, enabling iteration over large corpora without loading them
    entirely into memory.
    """

    def __init__(
        self,
        file_path: str,
        preprocessor: TextPreprocessor,
        seq_len: int = 100,
        *,
        chunk_size: int = 1000,
        cache_size: int = 100,
    ) -> None:
        """Instantiate the streaming dataset.

        Args:
            file_path: Path to the newline-delimited text file.
            preprocessor: Text preprocessor used to encode lines.
            seq_len: Maximum length of each sequence in tokens.
            chunk_size: Number of lines to load per file read.
            cache_size: Number of chunks to keep in the LRU cache.
        """
        self.file_path = file_path
        self.preprocessor = preprocessor
        self.seq_len = seq_len
        self.chunk_size = chunk_size
        self.cache = LRUCache(cache_size)
        self._index_file()

    def _open(self):
        """Open the dataset file, supporting gzip compression."""
        if self.file_path.lower().endswith(".gz"):
            return gzip.open(self.file_path, "rt", encoding="utf-8")
        return open(self.file_path, "r", encoding="utf-8")

    # ------------------------------------------------------------------
    def _index_file(self) -> None:
        """Create a lightweight index of chunk starting positions."""
        self.chunk_positions: List[int] = []
        self.total_lines = 0
        with self._open() as f:
            position = f.tell()
            for i, _ in enumerate(f):
                if i % self.chunk_size == 0:
                    self.chunk_positions.append(position)
                position = f.tell()
            self.total_lines = i + 1 if "i" in locals() else 0

    # ------------------------------------------------------------------
    def _load_chunk(self, chunk_idx: int) -> List[torch.Tensor]:
        """Load and preprocess a chunk of lines."""
        start_pos = self.chunk_positions[chunk_idx]
        lines: List[torch.Tensor] = []
        with self._open() as f:
            f.seek(start_pos)
            for _ in range(self.chunk_size):
                line = f.readline()
                if not line:
                    break
                encoded = self.preprocessor.encode(line.strip())
                tensor = torch.tensor(encoded[: self.seq_len + 1], dtype=torch.long)
                if tensor.size(0) < self.seq_len + 1:
                    pad_width = self.seq_len + 1 - tensor.size(0)
                    tensor = F.pad(tensor, (0, pad_width))
                lines.append(tensor)
        return lines

    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.total_lines

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_lines:
            raise IndexError("index out of range")
        chunk_idx = idx // self.chunk_size
        if chunk_idx in self.cache:
            chunk = self.cache[chunk_idx]
        else:
            chunk = self._load_chunk(chunk_idx)
            self.cache[chunk_idx] = chunk
        item_idx = idx % self.chunk_size
        full_seq = chunk[item_idx]
        return full_seq[:-1], full_seq[1:]

