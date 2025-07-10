# entropy_news/data/dataset.py

import torch
from torch.utils.data import Dataset
from typing import List

class NewsDataset(Dataset):
    def __init__(self, encoded_texts: List[List[int]], seq_len: int = 100):
        """Store padded token sequences for language-model training.

        Args:
            encoded_texts: Tokenized texts to be padded.
            seq_len: Maximum length of each sequence in tokens.
        """
        self.seq_len = seq_len
        self.data = [self.pad_sequence(seq) for seq in encoded_texts]

    def pad_sequence(self, seq: List[int]) -> torch.Tensor:
        """Pad or truncate ``seq`` to ``seq_len + 1`` tokens."""
        if len(seq) < self.seq_len + 1:
            seq += [0] * (self.seq_len + 1 - len(seq))
        else:
            seq = seq[:self.seq_len + 1]
        return torch.Tensor(seq).long()

    def __len__(self) -> int:
        """Return the number of stored sequences."""
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        """Return input and target tensors for ``idx``."""
        full_seq = self.data[idx]
        return full_seq[:-1], full_seq[1:]

