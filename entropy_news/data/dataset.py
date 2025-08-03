# entropy_news/data/dataset.py

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class NewsDataset(Dataset):
    def __init__(self, encoded_texts: list[list[int]], seq_len: int = 100):
        """Store padded token sequences for language-model training.

        Args:
            encoded_texts: Tokenized texts to be padded.
            seq_len: Maximum length of each sequence in tokens.
        """
        self.seq_len = seq_len
        seq_tensors = [torch.tensor(seq[: seq_len + 1], dtype=torch.long) for seq in encoded_texts]
        if seq_tensors:
            padded = pad_sequence(seq_tensors, batch_first=True, padding_value=0)
            if padded.size(1) < seq_len + 1:
                pad_width = seq_len + 1 - padded.size(1)
                padded = F.pad(padded, (0, pad_width))
            self.data = padded
        else:
            self.data = torch.zeros(0, seq_len + 1, dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of stored sequences."""
        return self.data.size(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve one training example.

        Args:
            idx: Index of the desired sequence.

        Returns:
            Tuple ``(x, y)`` where ``x`` is the input sequence and ``y`` is the
            target sequence shifted by one token.
        """
        full_seq = self.data[idx]
        # Targets are inputs shifted by one token
        return full_seq[:-1], full_seq[1:]

