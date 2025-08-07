# entropy_news/model/lstm_entropy.py

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
from entropy_news.types import EmbeddingMatrix

class EntropyLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        embedding_matrix: EmbeddingMatrix | None = None,
    ) -> None:
        """Construct a lightweight LSTM model for entropy estimation.

        Args:
            vocab_size: Size of the vocabulary used for embedding lookup.
            embed_dim: Dimension of the embedding vectors. Defaults to ``100``.
            hidden_dim: Number of hidden units in the LSTM. Defaults to ``16``.
            num_layers: How many LSTM layers to stack. Defaults to ``1``.
            dropout: Dropout probability applied between layers. Defaults to
                ``0.0``.
            embedding_matrix: Optional ``numpy`` array containing pre-trained
                word vectors. When provided, the embedding layer weights are
                frozen.
        """

        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # Simple embedding layer for token lookups
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.Tensor(embedding_matrix))
            self.embedding.weight.requires_grad = False  # Freezes embeddings

        # Recurrent layer that processes sequences token by token
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
            batch_first=True,
        )
        # Final linear layer projecting to vocabulary size
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of token indices.

        Args:
            x: Tensor of token IDs with shape ``(batch, seq_len)``.

        Returns:
            Tensor containing raw predictions for each token position.
        """

        # Embed tokens then process through the LSTM
        emb = self.embedding(x)
        output, _ = self.lstm(emb)
        logits = self.fc(output)
        return logits
