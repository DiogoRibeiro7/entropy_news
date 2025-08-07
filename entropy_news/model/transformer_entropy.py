from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from entropy_news.types import EmbeddingMatrix


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding module."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x``.

        Args:
            x: Tensor with shape ``(batch, seq_len, d_model)``.
        """

        return x + self.pe[:, : x.size(1)]


class EntropyTransformer(nn.Module):
    """Minimal Transformer encoder model for entropy estimation."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        num_heads: int = 2,
        ff_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        embedding_matrix: Optional[EmbeddingMatrix] = None,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight.data.copy_(torch.tensor(embedding_matrix))
            self.embedding.weight.requires_grad = False

        self.pos_encoder = PositionalEncoding(embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for a batch of token indices."""

        emb = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        emb = self.pos_encoder(emb)
        encoded = self.encoder(emb)
        return self.fc(encoded)
