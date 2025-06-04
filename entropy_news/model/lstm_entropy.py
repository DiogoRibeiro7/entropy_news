# entropy_news/model/lstm_entropy.py

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Any

class EntropyLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        embedding_matrix: Any | None = None,
    ) -> None:
        """Simple LSTM model used for entropy estimation.

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary.
        embed_dim : int, optional
            Dimension of the embedding vectors, by default ``100``.
        hidden_dim : int, optional
            Hidden dimension of the LSTM, by default ``16``.
        num_layers : int, optional
            Number of stacked LSTM layers, by default ``1``.
        dropout : float, optional
            Dropout probability between LSTM layers, by default ``0.0``.
        embedding_matrix : np.ndarray | None, optional
            Pre-trained embedding matrix. If provided the weights are frozen.
        """

        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.Tensor(embedding_matrix))
            self.embedding.weight.requires_grad = False  # Freezes embeddings

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for a batch of token indices."""
        x = x.to(self.device)
        emb = self.embedding(x)
        output, _ = self.lstm(emb)
        logits = self.fc(output)
        return logits

# Exemplo de uso:
# model = EntropyLSTM(vocab_size=len(vocab), embed_dim=100, hidden_dim=16, embedding_matrix=preprocessor.embedding_matrix)
# model = model.to(model.device)
