from __future__ import annotations

import torch
import torch.nn as nn
from entropy_news.types import EmbeddingMatrix


class EntropyLSTMAttention(nn.Module):
    """LSTM model enhanced with multi-head self-attention."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        hidden_dim: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_heads: int = 2,
        embedding_matrix: EmbeddingMatrix | None = None,
    ) -> None:
        """Initialize the hybrid LSTM-attention model.

        Args:
            vocab_size: Size of the token vocabulary.
            embed_dim: Dimension of the word embeddings.
            hidden_dim: Hidden dimension of the LSTM and attention modules.
            num_layers: Number of stacked LSTM layers.
            dropout: Dropout probability applied between LSTM layers.
            num_heads: Number of attention heads.
            embedding_matrix: Optional pre-trained embedding weights.
        """
        super().__init__()
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
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute vocabulary logits for ``x``.

        Args:
            x: Tensor of token indices with shape ``(batch, seq_len)``.

        Returns:
            Tensor of shape ``(batch, seq_len, vocab_size)`` containing logits.
        """
        emb = self.embedding(x)
        lstm_out, _ = self.lstm(emb)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        logits = self.fc(attn_out)
        return logits
