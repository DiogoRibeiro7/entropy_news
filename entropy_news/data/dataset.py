# entropy_news/data/dataset.py

import torch
from torch.utils.data import Dataset
from typing import List

class NewsDataset(Dataset):
    def __init__(self, encoded_texts: List[List[int]], seq_len: int = 100):
        self.seq_len = seq_len
        self.data = [self.pad_sequence(seq) for seq in encoded_texts]

    def pad_sequence(self, seq: List[int]) -> torch.Tensor:
        if len(seq) < self.seq_len + 1:
            seq += [0] * (self.seq_len + 1 - len(seq))
        else:
            seq = seq[:self.seq_len + 1]
        return torch.Tensor(seq).long()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        full_seq = self.data[idx]
        return full_seq[:-1], full_seq[1:]

# Exemplo de uso:
# preprocessor = TextPreprocessor()
# preprocessor.build_vocab(texts)
# encoded = [preprocessor.encode(t) for t in texts]
# dataset = NewsDataset(encoded, seq_len=100)
