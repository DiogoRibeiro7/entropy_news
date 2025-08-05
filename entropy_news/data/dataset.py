# entropy_news/data/dataset.py

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    """Dataset of tokenised news sequences for language modelling."""

    def __init__(
        self,
        encoded_texts: Sequence[Sequence[int]],
        seq_len: int = 100,
        in_memory: bool = True,
    ) -> None:
        """Store token sequences for language-model training.

        Args:
            encoded_texts: Tokenized texts to be padded.
            seq_len: Maximum length of each sequence in tokens.
            in_memory: Whether to pre-pad all sequences and keep them in
                memory. For large datasets set to ``False`` to pad on demand.
        """
        self.seq_len = seq_len
        self.encoded_texts = [list(seq) for seq in encoded_texts]
        if in_memory:
            seq_tensors = [
                torch.tensor(seq[: seq_len + 1], dtype=torch.long)
                for seq in self.encoded_texts
            ]
            if seq_tensors:
                padded = pad_sequence(seq_tensors, batch_first=True, padding_value=0)
                if padded.size(1) < seq_len + 1:
                    pad_width = seq_len + 1 - padded.size(1)
                    padded = F.pad(padded, (0, pad_width))
                self.data = padded
            else:
                self.data = torch.zeros(0, seq_len + 1, dtype=torch.long)
        else:
            self.data = None

    def __len__(self) -> int:
        """Return the number of stored sequences."""
        return len(self.encoded_texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve one training example.

        Args:
            idx: Index of the desired sequence.

        Returns:
            Tuple ``(x, y)`` where ``x`` is the input sequence and ``y`` is the
            target sequence shifted by one token.
        """
        if self.data is not None:
            full_seq = self.data[idx]
        else:
            seq = self.encoded_texts[idx][: self.seq_len + 1]
            full_seq = torch.tensor(seq, dtype=torch.long)
            if full_seq.size(0) < self.seq_len + 1:
                pad_width = self.seq_len + 1 - full_seq.size(0)
                full_seq = F.pad(full_seq, (0, pad_width))
        return full_seq[:-1], full_seq[1:]

