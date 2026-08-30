from __future__ import annotations

import torch
import torch.nn as nn

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
        """Construct the LSTM language model used for entropy estimation."""
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed = self.embedding
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.Tensor(embedding_matrix))
            self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return token logits and recurrent state for sequential chunk scoring."""
        emb = self.embedding(x)
        output, next_state = self.lstm(emb, state)
        return self.fc(output), next_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits while resetting recurrent state for each batch."""
        logits, _ = self.forward_with_state(x)
        return logits
